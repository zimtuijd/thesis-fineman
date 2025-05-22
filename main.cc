/* Tim Zuijderduijn (s3620166) 2025
   main.cc
*/

#include <cstdio>
#include <iostream>
#include <string>
#include <cuda.h>
#include <chrono>
#include <cmath>
#include "digraph.h"

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;

CUfunction cuNextLayer;
CUfunction cuCountDegrees;
CUfunction cuScanDegrees;
CUfunction cuAssignVerticesNextQueue;

CUdeviceptr d_adjacencyList;
CUdeviceptr d_edgesOffset;
CUdeviceptr d_edgesSize;
CUdeviceptr d_distance;
CUdeviceptr d_parent;
CUdeviceptr d_currentQueue;
CUdeviceptr d_nextQueue;
CUdeviceptr d_degrees;

CUdeviceptr d_IDtagList;


int *incrDegrees;

void checkError(CUresult error, std::string msg) {
  if (error != CUDA_SUCCESS) {
    printf("%s: %d\n", msg.c_str(), error);
    exit(1);
  }
}

void initCuda(Digraph &G) {
  //initialize CUDA
  cuInit(0);
  checkError(cuDeviceGet(&cuDevice, 1), "cannot get device 0");
  checkError(cuCtxCreate(&cuContext, 0, cuDevice), "cannot create context");
  checkError(cuModuleLoad(&cuModule, "bfs.ptx"), "cannot load module");
  checkError(cuModuleGetFunction(&cuNextLayer, cuModule, "nextLayer"), "cannot get kernel handle");
  checkError(cuModuleGetFunction(&cuCountDegrees, cuModule, "countDegrees"), "cannot get kernel handle");
  checkError(cuModuleGetFunction(&cuScanDegrees, cuModule, "scanDegrees"), "cannot get kernel handle");
  checkError(cuModuleGetFunction(&cuAssignVerticesNextQueue, cuModule, "assignVerticesNextQueue"),
              "cannot get kernel handle");

  
  //copy memory to device
  checkError(cuMemAlloc(&d_adjacencyList, G.numEdges * sizeof(int)), "cannot allocate d_adjacencyList");
  checkError(cuMemAlloc(&d_edgesOffset, G.numVertices * sizeof(int)), "cannot allocate d_edgesOffset");
  checkError(cuMemAlloc(&d_edgesSize, G.numVertices * sizeof(int)), "cannot allocate d_edgesSize");
  checkError(cuMemAlloc(&d_distance, G.numVertices * sizeof(int)), "cannot allocate d_distance");
  checkError(cuMemAlloc(&d_parent, G.numVertices * sizeof(int)), "cannot allocate d_parent");
  checkError(cuMemAlloc(&d_currentQueue, G.numVertices * sizeof(int)), "cannot allocate d_currentQueue");
  checkError(cuMemAlloc(&d_nextQueue, G.numVertices * sizeof(int)), "cannot allocate d_nextQueue");
  checkError(cuMemAlloc(&d_degrees, G.numVertices * sizeof(int)), "cannot allocate d_degrees");
  checkError(cuMemAllocHost((void **) &incrDegrees, sizeof(int) * G.numVertices), "cannot allocate memory");

  // ID Tag list for augmented BFS
  checkError(cuMemAlloc(&d_IDtagList, std::ceil(std::log(G.numVertices)) * G.numVertices * sizeof(int)),
                        "cannot allocate ID tag list");

  checkError(cuMemcpyHtoD(d_adjacencyList, G.adjacencyList.data(), G.numEdges * sizeof(int)),
              "cannot copy to d_adjacencyList");
  checkError(cuMemcpyHtoD(d_edgesOffset, G.edgesOffset.data(), G.numVertices * sizeof(int)),
              "cannot copy to d_edgesOffset");
  checkError(cuMemcpyHtoD(d_edgesSize, G.edgesSize.data(), G.numVertices * sizeof(int)),
              "cannot copy to d_edgesSize");


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

void initializeCudaBfs(int startVertex, std::vector<int> &distance, std::vector<int> &parent, Digraph &G) {
  //initialize values
  std::fill(distance.begin(), distance.end(), std::numeric_limits<int>::max());
  std::fill(parent.begin(), parent.end(), std::numeric_limits<int>::max());
  distance[startVertex] = 0;
  parent[startVertex] = 0;

  checkError(cuMemcpyHtoD(d_distance, distance.data(), G.numVertices * sizeof(int)),
              "cannot copy to d)distance");
  checkError(cuMemcpyHtoD(d_parent, parent.data(), G.numVertices * sizeof(int)),
              "cannot copy to d_parent");

  int firstElementQueue = startVertex;
  cuMemcpyHtoD(d_currentQueue, &firstElementQueue, sizeof(int));
}

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
  for (auto v : startVertices) {
    cuMemcpyHtoD(d_currentQueue, &v, sizeof(int));
  }

  // Init the ID tag list
  for (auto v : startVertices) {
    int temp = v * std::ceil(std::log(G.numVertices)); 
    IDtagList[temp] = v;
  }

  checkError(cuMemcpyHtoD(d_IDtagList, IDtagList.data(), 
                          std::ceil(std::log(G.numVertices)) * G.numVertices * sizeof(int)),
                          "cannot copy to d_IDtagList");

}

void finalizeCudaBfs(std::vector<int> &distance, std::vector<int> &parent, Digraph &G) {
  //copy memory from device
  checkError(cuMemcpyDtoH(distance.data(), d_distance, G.numVertices * sizeof(int)),
              "cannot copy d_distance to host");
  checkError(cuMemcpyDtoH(parent.data(), d_parent, G.numVertices * sizeof(int)), "cannot copy d_parent to host");
}

void finalizeCudaBfsAug(std::vector<int> &distance, std::vector<int> &parent,
                        std::vector<int> &IDtagList, Digraph &G) {
  //copy memory from device
  checkError(cuMemcpyDtoH(distance.data(), d_distance, G.numVertices * sizeof(int)),
              "cannot copy d_distance to host");
  checkError(cuMemcpyDtoH(parent.data(), d_parent, G.numVertices * sizeof(int)), "cannot copy d_parent to host");
  checkError(cuMemcpyDtoH(IDtagList.data(), d_IDtagList,
             std::ceil(std::log(G.numVertices)) * G.numVertices * sizeof(int)), "cannot copy d_IDtagList to host");
}

void finalizeCuda() {
  //free memory
  checkError(cuMemFree(d_adjacencyList), "cannot free memory for d_adjacencyList");
  checkError(cuMemFree(d_edgesOffset), "cannot free memory for d_edgesOffset");
  checkError(cuMemFree(d_edgesSize), "cannot free memory for d_edgesSize");
  checkError(cuMemFree(d_distance), "cannot free memory for d_distance");
  checkError(cuMemFree(d_parent), "cannot free memory for d_parent");
  checkError(cuMemFree(d_currentQueue), "cannot free memory for d_parent");
  checkError(cuMemFree(d_nextQueue), "cannot free memory for d_parent");
  checkError(cuMemFreeHost(incrDegrees), "cannot free memory for incrDegrees");
}

void nextLayer(int level, int queueSize) {
  void *args[] = {&level, &d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_distance, &d_parent, &queueSize,
                  &d_currentQueue};
  checkError(cuLaunchKernel(cuNextLayer, queueSize / 1024 + 1, 1, 1,
                            1024, 1, 1, 0, 0, args, 0),
              "cannot run kernel cuNextLayer");
  cuCtxSynchronize();
}

void countDegrees(int level, int queueSize) {
  void *args[] = {&d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_parent, &queueSize,
                  &d_currentQueue, &d_degrees};
  checkError(cuLaunchKernel(cuCountDegrees, queueSize / 1024 + 1, 1, 1,
                            1024, 1, 1, 0, 0, args, 0),
              "cannot run kernel cuNextLayer");
  cuCtxSynchronize();
}

void scanDegrees(int queueSize) {
  //run kernel so every block in d_currentQueue has prefix sums calculated
  void *args[] = {&queueSize, &d_degrees, &incrDegrees};
  checkError(cuLaunchKernel(cuScanDegrees, queueSize / 1024 + 1, 1, 1,
                            1024, 1, 1, 0, 0, args, 0), "cannot run kernel scanDegrees");
  cuCtxSynchronize();

  //count prefix sums on CPU for ends of blocks exclusive
  //already written previous block sum
  incrDegrees[0] = 0;
  for (int i = 1024; i < queueSize + 1024; i += 1024) {
      incrDegrees[i / 1024] += incrDegrees[i / 1024 - 1];
  }
}

void assignVerticesNextQueue(int queueSize, int nextQueueSize) {
  void *args[] = {&d_adjacencyList, &d_edgesOffset, &d_edgesSize, &d_parent, &queueSize, &d_currentQueue,
                  &d_nextQueue, &d_degrees, &incrDegrees, &nextQueueSize};
  checkError(cuLaunchKernel(cuAssignVerticesNextQueue, queueSize / 1024 + 1, 1, 1,
                            1024, 1, 1, 0, 0, args, 0),
              "cannot run kernel assignVerticesNextQueue");
  cuCtxSynchronize();
}

void runCudaBfs(int startVertex, Digraph &G, std::vector<int> &distance,
                std::vector<int> &parent, int numVertices) {
  initializeCudaBfs(startVertex, distance, parent, G);

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
  if (!reachedEnd) {
    printf("Did not reach end.\n");
  }
  
  std::cout << level << " " << maxLevel << "\n";

  finalizeCudaBfs(distance, parent, G);
}

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

}

int main(int argc, char** argv) {

  Digraph G;
  int startVertex = atoi(argv[1]);
  readGraph(G, argc, argv);

  printf("Number of vertices %d\n", G.numVertices);
  printf("Number of edges %d\n\n", G.numEdges);

  //vectors for results
  std::vector<int> distance(G.numVertices, std::numeric_limits<int>::max());
  std::vector<int> parent(G.numVertices, std::numeric_limits<int>::max());

  initCuda(G);

  runCudaBfs(startVertex, G, distance, parent, G.numVertices);

  std::vector<int> startVertices = {0,1,2,3};
  std::vector<int> IDtagList(G.numVertices * std::ceil(std::log(G.numVertices)), -1);

  runCudaBfsAug(startVertices, G, distance, parent, IDtagList, G.numVertices);
  
  finalizeCuda();

  return 0;

} // main