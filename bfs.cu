/* Tim Zuijderduijn (s3620166) 2025
   main.cc
*/

#include <cstdio>
#include <iostream>
#include <string>
#include <cuda.h>
#include <chrono>
#include <cmath>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "bfs_kernels.cu"
#include "digraph.h"

void checkOutput(std::vector<int> &distance, std::vector<int> &expectedDistance, Digraph &G) {
  for (int i = 0; i < G.numVertices; i++) {
      if (distance[i] != expectedDistance[i]) {
          printf("%d %d %d\n", i, distance[i], expectedDistance[i]);
          printf("Wrong output!\n");
          exit(1);
      }
  }

  printf("Output OK!\n\n");
}

void initializeCudaBfs(int startVertex, std::vector<int> &distance, std::vector<int> &parent, Digraph &G,
                       thrust::device_vector<int> &d_distance,
                       thrust::device_vector<int> &d_parent,
                       thrust::device_vector<int> &d_currentQueue) {
  //initialize values
  std::fill(distance.begin(), distance.end(), std::numeric_limits<int>::max());
  std::fill(parent.begin(), parent.end(), std::numeric_limits<int>::max());
  distance[startVertex] = 0;
  parent[startVertex] = 0;

  d_distance = distance;
  d_parent = parent;

  thrust::fill_n(d_currentQueue.begin(), 1, startVertex);

}

void initializeCudaBfsAug(std::vector<int> &startVertices, std::vector<int> &distance,
                          std::vector<int> &parent, Digraph &G,
                          thrust::device_vector<int> &d_distance,
                          thrust::device_vector<int> &d_parent,
                          thrust::device_vector<int> &d_currentQueue,
                          thrust::device_vector<int> &d_IDTagList) {
  //initialize values
  std::fill(distance.begin(), distance.end(), std::numeric_limits<int>::max());
  std::fill(parent.begin(), parent.end(), std::numeric_limits<int>::max());
  for (auto v : startVertices) {
    distance[v] = 0;
    parent[v] = 0;
  }

  d_distance = distance;
  d_parent = parent;

  // copy starting nodes to frontier
  thrust::copy(startVertices.begin(), startVertices.end(), d_currentQueue.begin());

  // Init the ID tag list
  /*
  for (auto v : startVertices) {
    int temp = v * std::ceil(std::log(G.numVertices)); 
    IDtagList[temp] = v;
  }

  checkError(cuMemcpyHtoD(d_IDtagList, IDtagList.data(), 
                          std::ceil(std::log(G.numVertices)) * G.numVertices * sizeof(int)),
                          "cannot copy to d_IDtagList");
  */

}

void runCudaBfs(int startVertex, Digraph &G, std::vector<int> &distance,
                std::vector<int> &parent, int numVertices,
                thrust::device_vector<int> &d_adjacencyList,
                thrust::device_vector<int> &d_edgesOffset,
                thrust::device_vector<int> &d_edgesSize,
                thrust::device_vector<int> &d_distance,
                thrust::device_vector<int> &d_parent,
                thrust::device_vector<int> &d_currentQueue,
                thrust::device_vector<int> &d_nextQueue,
                thrust::device_vector<int> &d_degrees) {
  
  initializeCudaBfs(startVertex, distance, parent, G,
                    d_distance, d_parent, d_currentQueue);

  //launch kernel
  printf("Starting standard parallel bfs.\n");
  auto start = std::chrono::steady_clock::now();

  int queueSize = 1;
  int nextQueueSize = 0;
  int level = 0;
  int maxLevel = std::ceil(std::pow(std::cbrt(10), 2) * (std::log(numVertices) / std::log(19)));
  bool reachedEnd = true;

  while (queueSize) {
      if (level >= maxLevel) {
        reachedEnd = false;
        break;
      }

      thrust::fill(d_degrees.begin(), d_degrees.end(), 0);

      // next layer phase
      nextLayer<<<queueSize / 1024 + 1, 1024>>>
                (level,
                thrust::raw_pointer_cast(d_adjacencyList.data()),
                thrust::raw_pointer_cast(d_edgesOffset.data()),
                thrust::raw_pointer_cast(d_edgesSize.data()),
                thrust::raw_pointer_cast(d_distance.data()),
                thrust::raw_pointer_cast(d_parent.data()),
                queueSize,
                thrust::raw_pointer_cast(d_currentQueue.data()));
      // counting degrees phase
      countDegrees<<<queueSize / 1024 + 1, 1024>>>
                  (thrust::raw_pointer_cast(d_adjacencyList.data()),
                  thrust::raw_pointer_cast(d_edgesOffset.data()),
                  thrust::raw_pointer_cast(d_edgesSize.data()),
                  thrust::raw_pointer_cast(d_parent.data()),
                  queueSize,
                  thrust::raw_pointer_cast(d_currentQueue.data()),
                  thrust::raw_pointer_cast(d_degrees.data()));
      
      // doing scan on degrees
      thrust::inclusive_scan(d_degrees.begin(), d_degrees.end(), d_degrees.begin());
      nextQueueSize = d_degrees.back();

      // assigning vertices to nextQueue
      assignVerticesNextQueue<<<queueSize / 1024 + 1, 1024>>>
                              (thrust::raw_pointer_cast(d_adjacencyList.data()),
                               thrust::raw_pointer_cast(d_edgesOffset.data()),
                               thrust::raw_pointer_cast(d_edgesSize.data()),
                               thrust::raw_pointer_cast(d_parent.data()),
                               queueSize,
                               thrust::raw_pointer_cast(d_currentQueue.data()),
                               thrust::raw_pointer_cast(d_nextQueue.data()),
                               thrust::raw_pointer_cast(d_degrees.data()),
                               nextQueueSize);

      level++;
      queueSize = nextQueueSize;
      d_currentQueue.swap(d_nextQueue);
  }


  auto end = std::chrono::steady_clock::now();
  long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  printf("Elapsed time in milliseconds : %li ms.\n", duration);
  if (!reachedEnd) {
    printf("Did not reach end.\n");
  }
  
  //std::cout << "\n" << level << " " << maxLevel << "\n";

  // TODO: kopieer d_parent en d_distance naar host
}

void runCudaBfsAug(std::vector<int> startVertices, Digraph &G, std::vector<int> &distance,
                std::vector<int> &parent, int numVertices,
                thrust::device_vector<int> &d_adjacencyList,
                thrust::device_vector<int> &d_edgesOffset,
                thrust::device_vector<int> &d_edgesSize,
                thrust::device_vector<int> &d_distance,
                thrust::device_vector<int> &d_parent,
                thrust::device_vector<int> &d_currentQueue,
                thrust::device_vector<int> &d_nextQueue,
                thrust::device_vector<int> &d_degrees,
                thrust::device_vector<int> &d_IDTagList) {

  initializeCudaBfsAug(startVertices, distance, parent, G,
                       d_distance, d_parent, d_currentQueue, d_IDTagList);

  //launch kernel
  printf("Starting augmented parallel bfs.\n");
  auto start = std::chrono::steady_clock::now();

  int queueSize = startVertices.size();
  int nextQueueSize = 0;
  int level = 0;

  while (queueSize) {
    
      thrust::fill(d_degrees.begin(), d_degrees.end(), 0);

      // next layer phase
      nextLayer<<<queueSize / 1024 + 1, 1024>>>
                (level,
                thrust::raw_pointer_cast(d_adjacencyList.data()),
                thrust::raw_pointer_cast(d_edgesOffset.data()),
                thrust::raw_pointer_cast(d_edgesSize.data()),
                thrust::raw_pointer_cast(d_distance.data()),
                thrust::raw_pointer_cast(d_parent.data()),
                queueSize,
                thrust::raw_pointer_cast(d_currentQueue.data()));
      // counting degrees phase
      countDegrees<<<queueSize / 1024 + 1, 1024>>>
                  (thrust::raw_pointer_cast(d_adjacencyList.data()),
                  thrust::raw_pointer_cast(d_edgesOffset.data()),
                  thrust::raw_pointer_cast(d_edgesSize.data()),
                  thrust::raw_pointer_cast(d_parent.data()),
                  queueSize,
                  thrust::raw_pointer_cast(d_currentQueue.data()),
                  thrust::raw_pointer_cast(d_degrees.data()));
      
      // doing scan on degrees
      thrust::inclusive_scan(d_degrees.begin(), d_degrees.end(), d_degrees.begin());
      nextQueueSize = d_degrees.back();

      // assigning vertices to nextQueue
      assignVerticesNextQueue<<<queueSize / 1024 + 1, 1024>>>
                              (thrust::raw_pointer_cast(d_adjacencyList.data()),
                               thrust::raw_pointer_cast(d_edgesOffset.data()),
                               thrust::raw_pointer_cast(d_edgesSize.data()),
                               thrust::raw_pointer_cast(d_parent.data()),
                               queueSize,
                               thrust::raw_pointer_cast(d_currentQueue.data()),
                               thrust::raw_pointer_cast(d_nextQueue.data()),
                               thrust::raw_pointer_cast(d_degrees.data()),
                               nextQueueSize);

      queueSize = nextQueueSize;
      d_currentQueue.swap(d_nextQueue);
  }

  auto end = std::chrono::steady_clock::now();
  long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  printf("Elapsed time in milliseconds : %li ms.\n", duration);

  // TODO: kopieer d_parent en d_distance naar host
}

void initDevVector(Digraph &G,
                   thrust::device_vector<int> &d_adjacencyList,
                   thrust::device_vector<int> &d_edgesOffset,
                   thrust::device_vector<int> &d_edgesSize,
                   thrust::device_vector<int> &d_distance,
                   thrust::device_vector<int> &d_parent,
                   thrust::device_vector<int> &d_currentQueue,
                   thrust::device_vector<int> &d_nextQueue,
                   thrust::device_vector<int> &d_degrees) {

  d_adjacencyList = G.adjacencyList;
  d_edgesOffset = G.edgesOffset;
  d_edgesSize = G.edgesSize;
  thrust::fill(d_distance.begin(), d_distance.end(), 0);
  thrust::fill(d_parent.begin(), d_parent.end(), 0);
  thrust::fill(d_currentQueue.begin(), d_currentQueue.end(), 0);
  thrust::fill(d_nextQueue.begin(), d_nextQueue.end(), 0);
  thrust::fill(d_degrees.begin(), d_degrees.end(), 0);

}

int startBFS(Digraph &G, int startVertex,
             std::vector<int> &distance, std::vector<int> &parent) {

  // device vectors for kernels
  thrust::device_vector<int> d_adjacencyList(G.adjacencyList);
  thrust::device_vector<int> d_edgesOffset(G.edgesOffset);
  thrust::device_vector<int> d_edgesSize(G.edgesSize);
  thrust::device_vector<int> d_distance(G.numVertices, 0);
  thrust::device_vector<int> d_parent(G.numVertices, 0);
  thrust::device_vector<int> d_currentQueue(G.numVertices, 0);
  thrust::device_vector<int> d_nextQueue(G.numVertices, 0);
  thrust::device_vector<int> d_degrees(G.numVertices, 0);

  // normal BFS
  runCudaBfs(startVertex, G, distance, parent, G.numVertices,
             d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance,
             d_parent, d_currentQueue, d_nextQueue, d_degrees);


  std::vector<int> startVertices = {0,1,2,3};
  thrust::device_vector<int> d_IDTagList(G.numVertices * std::ceil(std::log(G.numVertices)), -1);

  initDevVector(G, d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance,
                d_parent, d_currentQueue, d_nextQueue, d_degrees);

  // augmented BFS
  runCudaBfsAug(startVertices, G, distance, parent, G.numVertices,
                d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance,
                d_parent, d_currentQueue, d_nextQueue, d_degrees,
                d_IDTagList);

  return 0;

} // main
