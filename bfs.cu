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
#include <set>

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
#include <thrust/random/linear_congruential_engine.h>

#include "bfs_kernels.cu"
#include "digraph.h"

struct printf_functor
{
  __host__ __device__
  void operator()(int x)
  {
    printf("%d ", x);
  }
};

// Used in modified BFS to sort by vertex first and pivot ID second
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

void checkIDTagList(std::vector<std::vector<int>> &distanceVectors,
                    Digraph &G,
                    thrust::device_vector<int> d_modDistance,
                    thrust::device_vector<int> d_IDTagList,
                    int IDTagSize, std::vector<int> startVertices) {

  std::vector<int> modDistanceVertex;
  size_t svCount = 0;
  for (auto v : startVertices) {
    modDistanceVertex.clear();
    int temp;
    for (int i = 0; i < G.numVertices; i++) {
      temp = std::numeric_limits<int>::max();
      for (int j = 0; j < IDTagSize; j++) {
        if (d_IDTagList[i*IDTagSize + j] == v) {
          temp = d_modDistance[i*IDTagSize + j];
          break;
        }
      }
      modDistanceVertex.push_back(temp);
    }
    for (size_t j = 0; j < modDistanceVertex.size(); j++) {
      if (modDistanceVertex[j] != distanceVectors[svCount][j]) {
        std::cout << j << " " << distanceVectors[svCount][j] << " " << modDistanceVertex[j] << "\n";
        std::cout << "Wrong output for vertex " << v << "\n";
        exit(1);
      }
    }
    svCount++;
  }
  std::cout << "Output OK!\n";

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

void initializeCudaBfsMod(std::vector<int> &startVertices, Digraph &G,
                          thrust::device_vector<int> &d_currentQueue,
                          thrust::device_vector<int> &d_IDTagList,
                          thrust::device_vector<int> &d_queueID,
                          int IDTagSize,
                          thrust::device_vector<int> &d_modDistance,
                          thrust::device_vector<int> &d_IDTagCount) {

  // copy starting nodes to frontier
  thrust::copy(startVertices.begin(), startVertices.end(), d_currentQueue.begin());
  thrust::copy(startVertices.begin(), startVertices.end(), d_queueID.begin());

  // Init the ID tag list
  // G.numVertices * std::ceil(std::log(G.numVertices) entries in IDTagList
  std::vector<int> tempList(d_IDTagList.size(), -1);
  std::vector<int> modDistance(d_modDistance.size(), std::numeric_limits<int>::max());

  for (auto v : startVertices) {
    int temp = v * IDTagSize; 
    tempList[temp] = v;

    modDistance[v*IDTagSize] = 0;
    d_IDTagCount[v] = 1;
  }

  thrust::copy(tempList.begin(), tempList.end(), d_IDTagList.begin());
  thrust::copy(modDistance.begin(), modDistance.end(), d_modDistance.begin());

}

void runCudaBfs(int startVertex, Digraph &G, long &tempTimePBFS,
                std::vector<int> &distance,
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

  int queueSize = 1;
  int nextQueueSize = 0;
  int level = 0;
  int maxLevel = std::ceil(std::pow(std::cbrt(numVertices), 2) * std::log(numVertices));
  bool reachedEnd = true;

  //launch kernel
  //printf("Starting standard parallel bfs.\n");
  auto start = std::chrono::steady_clock::now();

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
  //printf("Elapsed time in milliseconds : %li ms.\n\n", duration);
  tempTimePBFS += duration;

  thrust::copy(d_distance.begin(), d_distance.end(), distance.begin());
  thrust::copy(d_parent.begin(), d_parent.end(), parent.begin());

}

bool runCudaBfsMod(std::vector<int> startVertices, Digraph &G, long &tempTimeMBFS,
                   int distance, int numVertices, int IDTagSize,
                   thrust::device_vector<int> &d_adjacencyList,
                   thrust::device_vector<int> &d_edgesOffset,
                   thrust::device_vector<int> &d_edgesSize,
                   thrust::device_vector<int> &d_currentQueue,
                   thrust::device_vector<int> &d_nextQueue,
                   thrust::device_vector<int> &d_degrees,
                   thrust::device_vector<int> &d_IDTagList,
                   thrust::device_vector<int> &d_queueID,
                   thrust::device_vector<int> &d_nextQueueID,
                   thrust::device_vector<int> &d_modDistance,
                   thrust::device_vector<int> &d_IDTagCount,
                   thrust::device_vector<int> &d_nextQueueIDListNum) {

  initializeCudaBfsMod(startVertices, G, d_currentQueue,
                       d_IDTagList, d_queueID, IDTagSize,
                       d_modDistance, d_IDTagCount);

  int queueSize = startVertices.size();
  int nextQueueSize = 0;
  int level = 0;
  bool reachedEnd = true;
  int const maxQueueSize = G.numVertices * IDTagSize;

  // Used as a flag by assignPivotID()
  thrust::device_vector<bool> IDTagListOverflow(1, false);

  //launch kernel
  //printf("Starting modified parallel bfs.\n");
  auto start = std::chrono::steady_clock::now();

  while (queueSize) {

      // Stops the BFS if a given distance has been reached
      if (distance > -1 && level >= distance) {
        reachedEnd = false;
        break;
      }

      // Counting degrees phase
      modCountDegrees<<<queueSize / 1024 + 1, 1024>>>
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

      // If nextQueue is too large, there is ID tag list overflow
      if (nextQueueSize > maxQueueSize) {
        IDTagListOverflow[0] = true;
        break;
      }

      // Assigning vertices to nextQueue
      // Also checks the ID tag list
      modAssignVNQ<<<queueSize / 1024 + 1, 1024>>>
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
 
      // Counts number of pivot IDs per vertex
      // Assigns spot for vertex in ID tag list
      numberPivotID<<<nextQueueSize / 1024 + 1, 1024>>>
                    (thrust::raw_pointer_cast(d_nextQueue.data()),
                     thrust::raw_pointer_cast(d_IDTagList.data()),
                     nextQueueSize,
                     IDTagSize,
                     thrust::raw_pointer_cast(IDTagListOverflow.data()),
                     thrust::raw_pointer_cast(d_IDTagCount.data()),
                     thrust::raw_pointer_cast(d_nextQueueIDListNum.data()));

      if (IDTagListOverflow[0]) {
        break;
      }

      // Assigns pivot IDS to d_IDTagList
      assignPivotID<<<nextQueueSize / 1024 + 1, 1024>>>
                    (thrust::raw_pointer_cast(d_nextQueue.data()),
                     thrust::raw_pointer_cast(d_nextQueueID.data()),
                     thrust::raw_pointer_cast(d_IDTagList.data()),
                     nextQueueSize,
                     IDTagSize,
                     thrust::raw_pointer_cast(d_IDTagCount.data()),
                     thrust::raw_pointer_cast(d_nextQueueIDListNum.data()),
                     thrust::raw_pointer_cast(d_modDistance.data()),
                     level);
      
      /*thrust::for_each(d_IDTagList.begin(), d_IDTagList.end(), printf_functor());
      std::cout << "\n";
      thrust::for_each(d_currentQueue.begin(), d_currentQueue.end(), printf_functor());
      std::cout << "\n";
      thrust::for_each(d_queueID.begin(), d_queueID.end(), printf_functor());
      std::cout << "\n";
      thrust::for_each(d_nextQueue.begin(), d_nextQueue.end(), printf_functor());
      std::cout << "\n";
      thrust::for_each(d_nextQueueID.begin(), d_nextQueueID.end(), printf_functor());*/
      //std::cout << "\n" << nextQueueSize;
      //std::cout << "\n" << IDTagSize;
      //std::cout << "\n\n";

      level++;
      queueSize = nextQueueSize;
      d_currentQueue.swap(d_nextQueue);
      d_queueID.swap(d_nextQueueID);
  }

  auto end = std::chrono::steady_clock::now();
  long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  //printf("Elapsed time in milliseconds : %li ms.\n", duration);
  tempTimeMBFS = duration;

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

void initDevVectorPBFS(Digraph &G,
                       thrust::device_vector<int> &d_adjacencyList,
                       thrust::device_vector<int> &d_edgesSize,
                       thrust::device_vector<int> &d_edgesOffset,
                       thrust::device_vector<int> &d_currentQueue,
                       thrust::device_vector<int> &d_nextQueue,
                       thrust::device_vector<int> &d_degrees) {

  d_adjacencyList = G.adjacencyList;
  d_edgesSize = G.edgesSize;
  d_edgesOffset = G.edgesOffset;
  thrust::fill(d_currentQueue.begin(), d_currentQueue.end(), 0);
  thrust::fill(d_nextQueue.begin(), d_nextQueue.end(), 0);
  thrust::fill(d_degrees.begin(), d_degrees.end(), 0);

}

void initDevVectorMBFS(Digraph &G,
                       thrust::device_vector<int> &d_adjacencyList,
                       thrust::device_vector<int> &d_edgesSize,
                       thrust::device_vector<int> &d_edgesOffset,
                       thrust::device_vector<int> &d_currentQueue,
                       thrust::device_vector<int> &d_nextQueue,
                       thrust::device_vector<int> &d_degrees,
                       thrust::device_vector<int> &d_IDTagList,
                       thrust::device_vector<int> &d_queueID,
                       thrust::device_vector<int> &d_nextQueueID,
                       thrust::device_vector<int> &d_modDistance,
                       thrust::device_vector<int> &d_IDTagCount,
                       thrust::device_vector<int> &d_nextQueueIDListNum) {

  d_adjacencyList = G.adjacencyList;
  d_edgesOffset = G.edgesOffset;
  d_edgesSize = G.edgesSize;
  thrust::fill(d_currentQueue.begin(), d_currentQueue.end(), 0);
  thrust::fill(d_nextQueue.begin(), d_nextQueue.end(), 0);
  thrust::fill(d_degrees.begin(), d_degrees.end(), 0);
  thrust::fill(d_IDTagList.begin(), d_IDTagList.end(), -1);
  thrust::fill(d_queueID.begin(), d_queueID.end(), -1);
  thrust::fill(d_nextQueueID.begin(), d_nextQueueID.end(), -1);
  thrust::fill(d_modDistance.begin(), d_modDistance.end(), 0);
  thrust::fill(d_IDTagCount.begin(), d_IDTagCount.end(), 0);
  thrust::fill(d_nextQueueIDListNum.begin(), d_nextQueueIDListNum.end(), -1);

}

void startBFS(Digraph &G, int startVertex) {

  long temp = 0; // placeholder for time variable
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
  runCudaBfs(startVertex, G, temp, distance, parent, G.numVertices,
             d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance,
             d_parent, d_currentQueue, d_nextQueue, d_degrees);

}

bool startModBFS(Digraph &G, std::vector<int> &startVertices, int dist,
                 thrust::device_vector<int> &d_IDTagVertices) {

  int IDTagSize = std::ceil(std::log(G.numVertices));
  long temp = 0; // placeholder for time variable

  thrust::device_vector<int> d_adjacencyList(G.adjacencyList);
  thrust::device_vector<int> d_edgesOffset(G.edgesOffset);
  thrust::device_vector<int> d_edgesSize(G.edgesSize);
  thrust::device_vector<int> d_currentQueue(G.numVertices * IDTagSize, 0);
  thrust::device_vector<int> d_nextQueue(G.numVertices * IDTagSize, 0);
  thrust::device_vector<int> d_degrees(G.numVertices * IDTagSize, 0);

  thrust::device_vector<int> d_IDTagList(G.numVertices * IDTagSize);
  thrust::device_vector<int> d_queueID(G.numVertices * IDTagSize, -1);
  thrust::device_vector<int> d_nextQueueID(G.numVertices * IDTagSize, -1);
  thrust::device_vector<int> d_modDistance(G.numVertices * IDTagSize, 0);
  thrust::device_vector<int> d_IDTagCount(G.numVertices, 0);
  thrust::device_vector<int> d_nextQueueIDListNum(d_nextQueue.size(), -1);

  // modified BFS
  if (!runCudaBfsMod(startVertices, G, temp, dist, G.numVertices, IDTagSize,
                    d_adjacencyList, d_edgesOffset, d_edgesSize,
                    d_currentQueue, d_nextQueue, d_degrees,
                    d_IDTagList, d_queueID, d_nextQueueID,
                    d_modDistance, d_IDTagCount, d_nextQueueIDListNum)) {
    //std::cout << "Modified BFS returned false.\n";
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

// Used for experiment
int testBFS(Digraph &G, int startVertex,
             std::vector<int> &distance, std::vector<int> &parent) {

  // sequential BFS
  std::vector<bool> visited(G.numVertices, false);
  runSeqBFS(startVertex, G, distance, parent, visited);

  int dD = -1; // -1 means no distance bound
  int IDTagSize = std::ceil(std::log(G.numVertices));

  long avgTimePBFS = 0;
  long avgTimeMBFS = 0;
  long tempTimePBFS = 0;
  long tempTimeMBFS = 0;
  size_t epochs = 100;
  size_t startVerticesSize = IDTagSize;
  std::set<int> svSet;
  thrust::minstd_rand svRNG;

  std::cout << "Experiment with multiple start vertices.\n";
  std::cout << "Amount of epochs: " << epochs << "\n";
  std::cout << "Amount of start vertices: " << startVerticesSize << "\n\n";
  std::cout << "normal PBFS - modified PBFS\n";

  // distance and parent for parallel BFS
  std::vector<int> distanceP(G.numVertices, std::numeric_limits<int>::max());
  std::vector<int> parentP(G.numVertices, std::numeric_limits<int>::max());

  // device vectors for kernels
  thrust::device_vector<int> d_adjacencyList(G.adjacencyList);
  thrust::device_vector<int> d_edgesOffset(G.edgesOffset);
  thrust::device_vector<int> d_edgesSize(G.edgesSize);
  thrust::device_vector<int> d_distance(G.numVertices, 0);
  thrust::device_vector<int> d_parent(G.numVertices, 0);
  thrust::device_vector<int> d_currentQueue(G.numVertices * IDTagSize, 0);
  thrust::device_vector<int> d_nextQueue(G.numVertices * IDTagSize, 0);
  thrust::device_vector<int> d_degrees(G.numVertices * IDTagSize, 0);

  thrust::device_vector<int> d_IDTagList(G.numVertices * IDTagSize);
  thrust::device_vector<int> d_queueID(G.numVertices * IDTagSize, -1);
  thrust::device_vector<int> d_nextQueueID(G.numVertices * IDTagSize, -1);
  thrust::device_vector<int> d_modDistance(G.numVertices * IDTagSize, 0);
  thrust::device_vector<int> d_IDTagCount(G.numVertices, 0);
  thrust::device_vector<int> d_nextQueueIDListNum(d_nextQueue.size(), -1);

  size_t e = 0;
  while (e < epochs) {
  
    // Generate new start vertices
    svSet.clear();
    while (svSet.size() < startVerticesSize) {
      svSet.insert(svRNG() % G.numVertices);
    }
    std::vector<int> startVertices(svSet.begin(), svSet.end());
    
    // Run parallel BFS k times for each start vertex
    for (auto v : startVertices) {
    
      initDevVectorPBFS(G, d_adjacencyList, d_edgesSize, d_edgesOffset,
                        d_currentQueue, d_nextQueue, d_degrees);

      // normal parallel BFS
      runCudaBfs(v, G, tempTimePBFS, distanceP, parentP, G.numVertices,
                d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance,
                d_parent, d_currentQueue, d_nextQueue, d_degrees);

    }

    initDevVectorMBFS(G, d_adjacencyList, d_edgesSize, d_edgesOffset, d_currentQueue,
                      d_nextQueue, d_degrees, d_IDTagList, d_queueID, d_nextQueueID,
                      d_modDistance, d_IDTagCount, d_nextQueueIDListNum);

    // modified parallel BFS
    if (!runCudaBfsMod(startVertices, G, tempTimeMBFS, dD, G.numVertices, IDTagSize,
                      d_adjacencyList, d_edgesOffset, d_edgesSize,
                      d_currentQueue, d_nextQueue, d_degrees,
                      d_IDTagList, d_queueID, d_nextQueueID,
                      d_modDistance, d_IDTagCount, d_nextQueueIDListNum)) {
      //std::cout << "Modified BFS returned false.\n";
      tempTimePBFS = 0;
      tempTimeMBFS = 0;
      continue;
    }

    if (tempTimePBFS == 0 && tempTimeMBFS == 0) {
      continue;
    }

    std::cout << tempTimePBFS << "  " << tempTimeMBFS << "\n";
    avgTimePBFS += tempTimePBFS;
    avgTimeMBFS += tempTimeMBFS;
    tempTimePBFS = 0;
    tempTimeMBFS = 0;
    e++;

  }

  avgTimePBFS /= epochs;
  avgTimeMBFS /= epochs;

  std::cout << "\n\nAmount of epochs: " << epochs << "\n";
  std::cout << "Amount of start vertices: " << startVerticesSize << "\n\n";
  std::cout << "Average time normal parallel BFS: " << avgTimePBFS << "ms\n";
  std::cout << "Average time modified parallel BFS: " << avgTimeMBFS << "ms\n";

  return 0;

} // main

// Used for validation
int testBFSVal(Digraph &G, int startVertex) {

  int IDTagSize = std::ceil(std::log(G.numVertices));
  std::cout << IDTagSize << "\n";
  std::vector<int> startVertices = {1,2,3,4};

  // distance and parent for parallel BFS
  std::vector<std::vector<int>> distanceVectors;

  // device vectors for kernels
  thrust::device_vector<int> d_adjacencyList(G.adjacencyList);
  thrust::device_vector<int> d_edgesOffset(G.edgesOffset);
  thrust::device_vector<int> d_edgesSize(G.edgesSize);
  thrust::device_vector<int> d_distance(G.numVertices, 0);
  thrust::device_vector<int> d_parent(G.numVertices, 0);
  thrust::device_vector<int> d_currentQueue(G.numVertices * IDTagSize, 0);
  thrust::device_vector<int> d_nextQueue(G.numVertices * IDTagSize, 0);
  thrust::device_vector<int> d_degrees(G.numVertices * IDTagSize, 0);
  long temp = 0;

  for (auto v : startVertices) {

    // sequential BFS
    std::vector<int> distance(G.numVertices, std::numeric_limits<int>::max());
    std::vector<int> parent(G.numVertices, std::numeric_limits<int>::max());
    std::vector<bool> visited(G.numVertices, false);
    runSeqBFS(v, G, distance, parent, visited);  
    
    std::vector<int> distanceP(G.numVertices, std::numeric_limits<int>::max());
    std::vector<int> parentP(G.numVertices, std::numeric_limits<int>::max());
    initDevVectorPBFS(G, d_adjacencyList, d_edgesSize, d_edgesOffset,
                      d_currentQueue, d_nextQueue, d_degrees);

    // normal parallel BFS
    printf("Starting normal parallel bfs.\n");
    runCudaBfs(v, G, temp, distanceP, parentP, G.numVertices,
                d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance,
                d_parent, d_currentQueue, d_nextQueue, d_degrees);

    checkOutput(distanceP, distance, G);
    distanceVectors.push_back(distanceP);
      
  }

  // device vectors for modified bfs
  thrust::device_vector<int> d_IDTagList(G.numVertices * IDTagSize);
  thrust::device_vector<int> d_queueID(G.numVertices * IDTagSize, -1);
  thrust::device_vector<int> d_nextQueueID(G.numVertices * IDTagSize, -1);
  thrust::device_vector<int> d_modDistance(G.numVertices * IDTagSize, 0);
  thrust::device_vector<int> d_IDTagCount(G.numVertices, 0);
  thrust::device_vector<int> d_nextQueueIDListNum(d_nextQueue.size(), -1);
  int dD = -1; // -1 means no distance bound

  initDevVectorMBFS(G, d_adjacencyList, d_edgesSize, d_edgesOffset, d_currentQueue,
                    d_nextQueue, d_degrees, d_IDTagList, d_queueID, d_nextQueueID,
                    d_modDistance, d_IDTagCount, d_nextQueueIDListNum);

  printf("Starting modified parallel bfs.\n");
  if (!runCudaBfsMod(startVertices, G, temp, dD, G.numVertices, IDTagSize,
                      d_adjacencyList, d_edgesOffset, d_edgesSize,
                      d_currentQueue, d_nextQueue, d_degrees,
                      d_IDTagList, d_queueID, d_nextQueueID, d_modDistance,
                      d_IDTagCount, d_nextQueueIDListNum)) {
      std::cout << "Modified BFS returned false.\n";
  }

  checkIDTagList(distanceVectors, G, d_modDistance, d_IDTagList, IDTagSize, startVertices);

  return 0;

}
