/* Tim Zuijderduijn (s3620166) 2025
   bfs.cu
*/

#include <cstdio>
#include <iostream>
#include <string>
#include <cuda.h>
#include <chrono>
#include <cmath>
#include <vector>
#include <queue>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include "bfs_kernels.cu"
#include "digraph.h"

struct printf_functor
{
  __host__ __device__
  void operator()(int x)
  {
    // note that using printf in a __device__ function requires
    // code compiled for a GPU with compute capability 2.0 or
    // higher (nvcc --arch=sm_20)
    printf("%d ", x);
  }
};

// Used in augmented BFS to sort by vertex first and pivot ID second
struct sort_vertex_ID
{
  __host__ __device__
  bool operator()(thrust::tuple<int, int> const &a, thrust::tuple<int, int> const &b)
  {
    if (thrust::get<0>(a) < thrust::get<0>(b)) // vertex in a smaller than vertex in b
      return true;
    if (thrust::get<0>(a) > thrust::get<0>(b)) // vertex in a larger than vertex in b
      return false;
      
    return thrust::get<1>(a) < thrust::get<1>(b); // if vertices are equal, check for pivot ID
  }
};

// Used to sort IDTagList by lowest ID tag first, -1 is sorted to the back of the subarrays
struct sort_lowest_IDTag
{
  __host__ __device__
  bool operator()(int a, int b)
  {
    if (a < b && a != -1) // a smaller than b, a is not -1
      return true;
    
    if (b == -1) // if a is larger/equal to b, check if b is not -1
      return true;
    
    // else if a >= b and b != -1
    return false;
  }
};

void bfsCPU(int start, Digraph &G, std::vector<int> &distance,
            std::vector<int> &parent, std::vector<bool> &visited) {
    distance[start] = 0;
    parent[start] = start;
    visited[start] = true;
    std::queue<int> Q;
    Q.push(start);

    while (!Q.empty()) {
        int u = Q.front();
        Q.pop();

        for (int i = G.edgesOffset[u]; i < G.edgesOffset[u] + G.edgesSize[u]; i++) {
            int v = G.adjacencyList[i];
            if (!visited[v]) {
                visited[v] = true;
                distance[v] = distance[u] + 1;
                parent[v] = i;
                Q.push(v);
            }
        }
    }
}

void runSeqBFS(int startVertex, Digraph &G, std::vector<int> &distance,
               std::vector<int> &parent, std::vector<bool> &visited) {
    printf("Starting sequential bfs.\n");
    auto start = std::chrono::steady_clock::now();
    bfsCPU(startVertex, G, distance, parent, visited);
    auto end = std::chrono::steady_clock::now();
    long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Elapsed time in milliseconds : %li ms.\n\n", duration);
}

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

void initializeCudaBfsAug(std::vector<int> &startVertices, Digraph &G,
                          thrust::device_vector<int> &d_currentQueue,
                          thrust::device_vector<int> &d_IDTagList,
                          thrust::device_vector<int> &d_queueID,
                          int IDTagSize) {

  // copy starting nodes to frontier
  thrust::copy(startVertices.begin(), startVertices.end(), d_currentQueue.begin());
  thrust::copy(startVertices.begin(), startVertices.end(), d_queueID.begin());

  // Init the ID tag list
  // G.numVertices * std::ceil(std::log(G.numVertices) entries in IDTagList
  std::vector<int> tempList(d_IDTagList.size(), -1);
  for (auto v : startVertices) {
    int temp = v * IDTagSize; 
    tempList[temp] = v;
  }

  thrust::copy(tempList.begin(), tempList.end(), d_IDTagList.begin());

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
  int maxLevel = std::ceil(std::pow(std::cbrt(numVertices), 2) * std::log(numVertices));
  bool reachedEnd = true;

  while (queueSize) {
      /*if (level >= maxLevel) {
        reachedEnd = false;
        break;
      }*/

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
      thrust::inclusive_scan(d_degrees.begin(), d_degrees.begin() + queueSize, d_degrees.begin());
      nextQueueSize = d_degrees[queueSize - 1];

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
  
  if (!reachedEnd) {
    printf("Did not reach end.\n");
  }
  printf("Elapsed time in milliseconds : %li ms.\n\n", duration);

  thrust::copy(d_distance.begin(), d_distance.end(), distance.begin());
  thrust::copy(d_parent.begin(), d_parent.end(), parent.begin());

  //std::cout << "\n" << level << " " << maxLevel << "\n";

}

bool runCudaBfsAug(std::vector<int> startVertices, Digraph &G,
                   int distance, int numVertices, int IDTagSize,
                   thrust::device_vector<int> &d_adjacencyList,
                   thrust::device_vector<int> &d_edgesOffset,
                   thrust::device_vector<int> &d_edgesSize,
                   thrust::device_vector<int> &d_currentQueue,
                   thrust::device_vector<int> &d_nextQueue,
                   thrust::device_vector<int> &d_degrees,
                   thrust::device_vector<int> &d_IDTagList,
                   thrust::device_vector<int> &d_queueID,
                   thrust::device_vector<int> &d_nextQueueID) {

  initializeCudaBfsAug(startVertices, G, d_currentQueue,
                       d_IDTagList, d_queueID, IDTagSize);

  //launch kernel
  printf("Starting augmented parallel bfs.\n");
  auto start = std::chrono::steady_clock::now();

  int queueSize = startVertices.size();
  int nextQueueSize = 0;
  int level = 0;
  bool reachedEnd = true;

  // Used as a flag by assignPivotID()
  thrust::device_vector<bool> IDTagListOverflow(1, false);

  while (queueSize) {

      // Stops the BFS if a given distance has been reached
      if (distance > -1 && level >= distance) {
        reachedEnd = false;
        break;
      }

      if (IDTagListOverflow[0]) {
        break;
      }

      // Counting degrees phase
      augCountDegrees<<<queueSize / 1024 + 1, 1024>>>
                      (thrust::raw_pointer_cast(d_adjacencyList.data()),
                      thrust::raw_pointer_cast(d_edgesOffset.data()),
                      thrust::raw_pointer_cast(d_edgesSize.data()),
                      queueSize,
                      thrust::raw_pointer_cast(d_currentQueue.data()),
                      thrust::raw_pointer_cast(d_degrees.data()),
                      thrust::raw_pointer_cast(d_IDTagList.data()),
                      thrust::raw_pointer_cast(d_queueID.data()),
                      IDTagSize);
      
      // Doing scan on degrees
      thrust::inclusive_scan(d_degrees.begin(), d_degrees.begin() + queueSize, d_degrees.begin());
      nextQueueSize = d_degrees[queueSize - 1];

      // Assigning vertices to nextQueue
      // Also checks the ID tag list
      augAssignVNQ<<<queueSize / 1024 + 1, 1024>>>
                    (thrust::raw_pointer_cast(d_adjacencyList.data()),
                    thrust::raw_pointer_cast(d_edgesOffset.data()),
                    thrust::raw_pointer_cast(d_edgesSize.data()),
                    queueSize,
                    thrust::raw_pointer_cast(d_currentQueue.data()),
                    thrust::raw_pointer_cast(d_nextQueue.data()),
                    thrust::raw_pointer_cast(d_degrees.data()),
                    nextQueueSize,
                    thrust::raw_pointer_cast(d_IDTagList.data()),
                    thrust::raw_pointer_cast(d_queueID.data()),
                    thrust::raw_pointer_cast(d_nextQueueID.data()),
                    IDTagSize);

      // Sorts values in d_nextQueue and d_nextQueueID
      // Sorts by vertex first, pivot ID second (so d_nextQueue first, d_nextQueueID second)
      auto iterSortFirst = thrust::make_zip_iterator(thrust::make_tuple(d_nextQueue.begin(), d_nextQueueID.begin()));
      auto iterSortLast = thrust::make_zip_iterator(thrust::make_tuple(d_nextQueue.begin() + nextQueueSize,
                                                                       d_nextQueueID.begin() + nextQueueSize));

      thrust::stable_sort(iterSortFirst, iterSortLast, sort_vertex_ID());

      // Compaction pass, duplication removal using d_nextQueue and d_nextQueueID
      auto iterUnique = thrust::unique(iterSortFirst, iterSortLast);
      if (iterUnique != iterSortLast) {
        thrust::fill(iterUnique, iterSortLast, thrust::make_tuple(0, -1));
        nextQueueSize -= thrust::distance(iterUnique, iterSortLast);
      }

      // Assigns pivot IDS to d_IDTagList
      assignPivotID<<<nextQueueSize / 1024 + 1, 1024>>>
                    (thrust::raw_pointer_cast(d_nextQueue.data()),
                     thrust::raw_pointer_cast(d_nextQueueID.data()),
                     thrust::raw_pointer_cast(d_IDTagList.data()),
                     nextQueueSize,
                     IDTagSize,
                     thrust::raw_pointer_cast(IDTagListOverflow.data()));
      
      /*thrust::for_each(d_IDTagList.begin(), d_IDTagList.end(), printf_functor());
      std::cout << "\n";
      thrust::for_each(d_currentQueue.begin(), d_currentQueue.end(), printf_functor());
      std::cout << "\n";
      thrust::for_each(d_queueID.begin(), d_queueID.end(), printf_functor());
      std::cout << "\n";
      thrust::for_each(d_nextQueue.begin(), d_nextQueue.end(), printf_functor());
      std::cout << "\n";
      thrust::for_each(d_nextQueueID.begin(), d_nextQueueID.end(), printf_functor());
      std::cout << "\n" << nextQueueSize;
      std::cout << "\n" << IDTagSize;
      std::cout << "\n\n";*/

      level++;
      queueSize = nextQueueSize;
      d_currentQueue.swap(d_nextQueue);
      d_queueID.swap(d_nextQueueID);
  }

  auto end = std::chrono::steady_clock::now();
  long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  printf("Elapsed time in milliseconds : %li ms.\n", duration);

  if (!reachedEnd) {
    printf("Did not reach end.\n");
  }
  if (IDTagListOverflow[0]) {
    printf("ID Tag list has overflow.\n");
  }

  // Returns true if there was no IDTagList overflow
  // Returns false otherwise
  return !IDTagListOverflow[0];

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

void startBFS(Digraph &G, int startVertex) {

  std::vector<int> distance(G.numVertices, std::numeric_limits<int>::max());
  std::vector<int> parent(G.numVertices, std::numeric_limits<int>::max());

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

}

bool startAugBFS(Digraph &G, std::vector<int> &startVertices, int dist,
                 thrust::device_vector<int> &d_IDTagVertices) {

  thrust::device_vector<int> d_adjacencyList(G.adjacencyList);
  thrust::device_vector<int> d_edgesOffset(G.edgesOffset);
  thrust::device_vector<int> d_edgesSize(G.edgesSize);
  thrust::device_vector<int> d_distance(G.numVertices, 0);
  thrust::device_vector<int> d_parent(G.numVertices, 0);
  thrust::device_vector<int> d_currentQueue(G.numVertices, 0);
  thrust::device_vector<int> d_nextQueue(G.numVertices, 0);
  thrust::device_vector<int> d_degrees(G.numVertices, 0);

  int IDTagSize = std::ceil(std::log(G.numVertices));
  thrust::device_vector<int> d_IDTagList(G.numVertices * IDTagSize);
  thrust::device_vector<int> d_queueID(G.numVertices, -1);
  thrust::device_vector<int> d_nextQueueID(G.numVertices, -1);

  // augmented BFS
  if (!runCudaBfsAug(startVertices, G, dist, G.numVertices, IDTagSize,
                    d_adjacencyList, d_edgesOffset, d_edgesSize,
                    d_currentQueue, d_nextQueue, d_degrees,
                    d_IDTagList, d_queueID, d_nextQueueID)) {
    //std::cout << "Augmented BFS returned false.\n";
    return false;
  }

  // Return lowest IDTag per vertex in d_IDTagList
  for (int i = 0; i < G.numVertices; i++) {
    thrust::stable_sort(d_IDTagList.begin() + i * IDTagSize,
                        d_IDTagList.begin() + i * IDTagSize + IDTagSize,
                        sort_lowest_IDTag());
    d_IDTagVertices[i] = d_IDTagList[i * IDTagSize];
  }
  
  return true;

}

int testBFS(Digraph &G, int startVertex,
             std::vector<int> &distance, std::vector<int> &parent) {

  // sequential BFS
  std::vector<bool> visited(G.numVertices, false);
  runSeqBFS(startVertex, G, distance, parent, visited);

  // distance and parent for parallel BFS
  std::vector<int> distanceP(G.numVertices, std::numeric_limits<int>::max());
  std::vector<int> parentP(G.numVertices, std::numeric_limits<int>::max());

  // device vectors for kernels
  thrust::device_vector<int> d_adjacencyList(G.adjacencyList);
  thrust::device_vector<int> d_edgesOffset(G.edgesOffset);
  thrust::device_vector<int> d_edgesSize(G.edgesSize);
  thrust::device_vector<int> d_distance(G.numVertices, 0);
  thrust::device_vector<int> d_parent(G.numVertices, 0);
  thrust::device_vector<int> d_currentQueue(G.numVertices, 0);
  thrust::device_vector<int> d_nextQueue(G.numVertices, 0);
  thrust::device_vector<int> d_degrees(G.numVertices, 0);

  // normal parallel BFS
  runCudaBfs(startVertex, G, distanceP, parentP, G.numVertices,
             d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance,
             d_parent, d_currentQueue, d_nextQueue, d_degrees);

  checkOutput(distanceP, distance, G);

  int dD = -1; // -1 means no distance bound
  std::vector<int> startVertices = {startVertex};
  int IDTagSize = std::ceil(std::log(G.numVertices));
  thrust::device_vector<int> d_IDTagList(G.numVertices * IDTagSize);
  thrust::device_vector<int> d_queueID(G.numVertices, -1);
  thrust::device_vector<int> d_nextQueueID(G.numVertices, -1);

  initDevVector(G, d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance,
                d_parent, d_currentQueue, d_nextQueue, d_degrees);

  // augmented parallel BFS
  if (!runCudaBfsAug(startVertices, G, dD, G.numVertices, IDTagSize,
                    d_adjacencyList, d_edgesOffset, d_edgesSize,
                    d_currentQueue, d_nextQueue, d_degrees,
                    d_IDTagList, d_queueID, d_nextQueueID)) {
    //std::cout << "Augmented BFS returned false.\n";
  } 

  return 0;

} // main
