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
                       thrust::device_vector<int> &d_degrees) {
  //initialize values
  std::fill(distance.begin(), distance.end(), std::numeric_limits<int>::max());
  std::fill(parent.begin(), parent.end(), std::numeric_limits<int>::max());
  distance[startVertex] = 0;
  parent[startVertex] = 0;

  /*checkError(cuMemcpyHtoD(d_distance, distance.data(), G.numVertices * sizeof(int)),
              "cannot copy to d)distance");
  checkError(cuMemcpyHtoD(d_parent, parent.data(), G.numVertices * sizeof(int)),
              "cannot copy to d_parent");
  */

  d_distance = distance;
  d_parent = parent;

  int firstElementQueue = startVertex;
  //cuMemcpyHtoD(d_currentQueue, &firstElementQueue, sizeof(int));
  std::vector<int> temp_degrees(1, firstElementQueue);
  d_degrees = temp_degrees;

}
/*
void initializeCudaBfsAug(std::vector<int> &startVertices, std::vector<int> &distance,
                          std::vector<int> &parent, std::vector<int> &IDtagList, Digraph &G) {
  //initialize values
  std::fill(distance.begin(), distance.end(), std::numeric_limits<int>::max());
  std::fill(parent.begin(), parent.end(), std::numeric_limits<int>::max());
  for (auto v : startVertices) {
    distance[v] = 0;
    parent[v] = 0;
  }

  checkError(cuMemcpyHtoD(d_distance, distance.data(), G.numVertices * sizeof(int)),
              "cannot copy to d)distance");
  checkError(cuMemcpyHtoD(d_parent, parent.data(), G.numVertices * sizeof(int)),
              "cannot copy to d_parent");

  // Copy nodes to frontier
  cuMemcpyHtoD(d_currentQueue, startVertices.data(), startVertices.size() * sizeof(int));

  // Init the ID tag list
  for (auto v : startVertices) {
    int temp = v * std::ceil(std::log(G.numVertices)); 
    IDtagList[temp] = v;
  }

  checkError(cuMemcpyHtoD(d_IDtagList, IDtagList.data(), 
                          std::ceil(std::log(G.numVertices)) * G.numVertices * sizeof(int)),
                          "cannot copy to d_IDtagList");

}*/

void finalizeCudaBfs(std::vector<int> &distance, std::vector<int> &parent, Digraph &G) {
  //copy memory from device
  /*checkError(cuMemcpyDtoH(distance.data(), d_distance, G.numVertices * sizeof(int)),
              "cannot copy d_distance to host");
  checkError(cuMemcpyDtoH(parent.data(), d_parent, G.numVertices * sizeof(int)), "cannot copy d_parent to host");
  */
}
/*
void finalizeCudaBfsAug(std::vector<int> &distance, std::vector<int> &parent,
                        std::vector<int> &IDtagList, Digraph &G) {
  //copy memory from device
  checkError(cuMemcpyDtoH(distance.data(), d_distance, G.numVertices * sizeof(int)),
              "cannot copy d_distance to host");
  checkError(cuMemcpyDtoH(parent.data(), d_parent, G.numVertices * sizeof(int)), "cannot copy d_parent to host");
  checkError(cuMemcpyDtoH(IDtagList.data(), d_IDtagList,
             std::ceil(std::log(G.numVertices)) * G.numVertices * sizeof(int)), "cannot copy d_IDtagList to host");
}
*/

void runCudaBfs(int startVertex, Digraph &G, std::vector<int> &distance,
                std::vector<int> &parent, int numVertices,
                thrust::device_vector<int> &d_adjacencyList,
                thrust::device_vector<int> &d_edgesOffset,
                thrust::device_vector<int> &d_edgesSize,
                thrust::device_vector<int> &d_distance,
                thrust::device_vector<int> &d_parent,
                thrust::device_vector<int> &d_currentQueue,
                thrust::device_vector<int> &d_nextQueue,
                thrust::device_vector<int> &d_degrees,
                thrust::host_vector<int> &zincrDegrees) {
  
  initializeCudaBfs(startVertex, distance, parent, G,
                    d_distance, d_parent, d_degrees);
  
  thrust::device_vector<int> d_incrDegrees;// = incrDegrees;

  //launch kernel
  printf("Starting standards parallel bfs.\n");
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
      /*scanDegrees<<<queueSize / 1024 + 1, 1024>>>
                  (queueSize,
                  thrust::raw_pointer_cast(d_degrees.data()),
                  thrust::raw_pointer_cast(d_incrDegrees.data()));

      //thrust::copy(d_incrDegrees.begin(), d_incrDegrees.end(), incrDegrees.begin());
      int* incrDegrees = thrust::raw_pointer_cast(d_incrDegrees.data());
      incrDegrees[0] = 0;
      for (int i = 1024; i < queueSize + 1024; i += 1024) {
          incrDegrees[i / 1024] += incrDegrees[i / 1024 - 1];
      }
      nextQueueSize = incrDegrees[(queueSize - 1) / 1024 + 1];
      //d_incrDegrees = incrDegrees;*/

      thrust::inclusive_scan(d_degrees.begin(), d_degrees.end(), d_degrees.begin());

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
                               //thrust::raw_pointer_cast(d_incrDegrees.data()),
                               nextQueueSize);

      level++;
      queueSize = nextQueueSize;
      //std::swap(d_currentQueue, d_nextQueue);
      d_currentQueue.swap(d_nextQueue);
  }


  auto end = std::chrono::steady_clock::now();
  long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  printf("Elapsed time in milliseconds : %li ms.\n", duration);
  if (!reachedEnd) {
    printf("Did not reach end.\n");
  }
  
  std::cout << level << " " << maxLevel << "\n";

  finalizeCudaBfs(distance, parent, G);
}
/*
void runCudaBfsAug(std::vector<int> startVertices, Digraph &G, std::vector<int> &distance,
                   std::vector<int> &parent, std::vector<int> &IDtagList, int numVertices) {

  initializeCudaBfsAug(startVertices, distance, parent, IDtagList, G);

  //launch kernel
  printf("Starting augmented parallel bfs.\n");
  auto start = std::chrono::steady_clock::now();

  int queueSize = startVertices.size();
  int nextQueueSize = 0;
  int level = 0;

  while (queueSize) {
      // next layer phase
      nextLayer(level, queueSize);
      // counting degrees phase
      countDegrees(level, queueSize);
      // doing scan on degrees
      scanDegrees(queueSize);
      nextQueueSize = incrDegrees[(queueSize - 1) / 1024 + 1];
      // assigning vertices to nextQueue
      assignVerticesNextQueue(queueSize, nextQueueSize);

      level++;
      queueSize = nextQueueSize;
      std::swap(d_currentQueue, d_nextQueue);
  }

  auto end = std::chrono::steady_clock::now();
  long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  printf("Elapsed time in milliseconds : %li ms.\n", duration);

  finalizeCudaBfs(distance, parent, G);

}*/

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

  thrust::host_vector<int> incrDegrees;

  thrust::device_vector<int> d_IDtagList;

  runCudaBfs(startVertex, G, distance, parent, G.numVertices,
             d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance,
             d_parent, d_currentQueue, d_nextQueue, d_degrees, incrDegrees);

  //std::vector<int> startVertices = {0,1,2,3};
  //std::vector<int> IDtagList(G.numVertices * std::ceil(std::log(G.numVertices)), -1);

  //runCudaBfsAug(startVertices, G, distance, parent, IDtagList, G.numVertices);

  return 0;

} // main
