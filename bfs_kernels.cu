/* Tim Zuijderduijn (s3620166) 2025
   bfs.cu
*/

#include <device_launch_parameters.h>
#include <cstdio>

extern "C" {

  /* Standard parallel BFS using scan
  */
  __global__
  void nextLayer(int level, int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_distance, int *d_parent,
                int queueSize, int *d_currentQueue) {
      int thid = blockIdx.x * blockDim.x + threadIdx.x;

      if (thid < queueSize) {
          int u = d_currentQueue[thid];
          for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
              int v = d_adjacencyList[i];
              if (level + 1 < d_distance[v]) {
                  d_distance[v] = level + 1;
                  d_parent[v] = i;
              }
          }
      }
  }

  __global__
  void countDegrees(int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_parent,
                    int queueSize, int *d_currentQueue, int *d_degrees) {
      int thid = blockIdx.x * blockDim.x + threadIdx.x;

      if (thid < queueSize) {
          int u = d_currentQueue[thid];
          int degree = 0;
          for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
              int v = d_adjacencyList[i];
              if (d_parent[v] == i && v != u) {
                  ++degree;
              }
          }
          d_degrees[thid] = degree;
      }
  }

  __global__
  void scanDegrees(int size, int *d_degrees, int *incrDegrees) {
      int thid = blockIdx.x * blockDim.x + threadIdx.x;

      if (thid < size) {
          //write initial values to shared memory
          __shared__ int prefixSum[1024];
          int modulo = threadIdx.x;
          prefixSum[modulo] = d_degrees[thid];
          __syncthreads();

          //calculate scan on this block
          //go up
          for (int nodeSize = 2; nodeSize <= 1024; nodeSize <<= 1) {
              if ((modulo & (nodeSize - 1)) == 0) {
                  if (thid + (nodeSize >> 1) < size) {
                      int nextPosition = modulo + (nodeSize >> 1);
                      prefixSum[modulo] += prefixSum[nextPosition];
                  }
              }
              __syncthreads();
          }

          //write information for increment prefix sums
          if (modulo == 0) {
              int block = thid >> 10;
              incrDegrees[block + 1] = prefixSum[modulo];
          }

          //go down
          for (int nodeSize = 1024; nodeSize > 1; nodeSize >>= 1) {
              if ((modulo & (nodeSize - 1)) == 0) {
                  if (thid + (nodeSize >> 1) < size) {
                      int next_position = modulo + (nodeSize >> 1);
                      int tmp = prefixSum[modulo];
                      prefixSum[modulo] -= prefixSum[next_position];
                      prefixSum[next_position] = tmp;

                  }
              }
              __syncthreads();
          }
          d_degrees[thid] = prefixSum[modulo];
      }

  }

  __global__
  void assignVerticesNextQueue(int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_parent, int queueSize,
                              int *d_currentQueue, int *d_nextQueue, int *d_degrees, int nextQueueSize) {
      int thid = blockIdx.x * blockDim.x + threadIdx.x;

      if (thid < queueSize) {
          /*__shared__ int sharedIncrement;
          if (!threadIdx.x) {
              sharedIncrement = incrDegrees[thid >> 10];
          }
          __syncthreads();
          */
          int sum = 0;
          if (threadIdx.x) {
              sum = d_degrees[thid - 1];
          }

          int u = d_currentQueue[thid];
          int counter = 0;
          for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
              int v = d_adjacencyList[i];
              if (d_parent[v] == i && v != u) {
                  int nextQueuePlace = sum + counter; // no sharedIncrement
                  d_nextQueue[nextQueuePlace] = v;
                  counter++;
              }
          }
      }
  }

  /*************************************************************
   * Augmented BFS
  */

  __global__
  void augNextLayer(int level, int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int *d_distance, int *d_parent,
                    int queueSize, int *d_currentQueue) {
      int thid = blockIdx.x * blockDim.x + threadIdx.x;

      if (thid < queueSize) {
          int u = d_currentQueue[thid];
          for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
              int v = d_adjacencyList[i];
              if (level + 1 < d_distance[v]) {
                  d_distance[v] = level + 1;
                  d_parent[v] = i;
              }
          }
      }
  }

    __global__
  void augCountDegrees(int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize,
                       int queueSize, int *d_currentQueue, int *d_degrees) {
      int thid = blockIdx.x * blockDim.x + threadIdx.x;

      if (thid < queueSize) {
          int u = d_currentQueue[thid];
          int degree = 0;
          for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
              int v = d_adjacencyList[i];
              if (v != u) {
                  ++degree;
              }
          }
          d_degrees[thid] = degree;
      }
  }

  __device__
  bool isVisitedByPivotID(int v, int u_pivotID, int IDTagSize, int *d_IDTagList) {
    
    for (int tagIdx = IDTagSize * v; tagIdx < IDTagSize * v + IDTagSize; tagIdx++) {
      if (d_IDTagList[tagIdx] == u_pivotID) {
        return true;
      }
    }
    return false;
  
  }

  __global__
  void augAssignVNQ(int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int queueSize,
                    int *d_currentQueue, int *d_nextQueue, int *d_degrees, int nextQueueSize,
                    int* d_IDTagList, int* d_queueID, int* d_nextQueueID, int IDTagSize) {
      int thid = blockIdx.x * blockDim.x + threadIdx.x;

      if (thid < queueSize) {
          int sum = 0;
          if (threadIdx.x) {
              sum = d_degrees[thid - 1];
          }

          int u = d_currentQueue[thid];
          int u_pivotID = d_queueID[thid]; // current node's pivot ID
          int counter = 0;
          for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
              int v = d_adjacencyList[i];
              if (v != u) {
                  
                  // scan through d_IDTagList to check if node has been reached by pivot ID 
                  // if already visited by pivot id, do not add to queue
                  if (isVisitedByPivotID(v, u_pivotID, IDTagSize, d_IDTagList)) {
                    break;
                  }

                  int nextQueuePlace = sum + counter;
                  d_nextQueue[nextQueuePlace] = v;
                  d_nextQueueID[nextQueuePlace] = u_pivotID; // adds corresponding pivot ID to next queue
                  counter++;
              }
          }
      }
  }

  __device__
  bool updateIDTagList(int *d_IDTagList, int IDTagSize, int v, int v_pivotID) {

    // Find first empty slot in d_IDTagList for some v
    // If found, pivot ID is included and returns true
    // If not found, returns false
    for (int i = v * IDTagSize; i < v * IDTagSize + IDTagSize; i++) {
      if (d_IDTagList[i] == -1) {
        d_IDTagList[i] = v_pivotID;
        return true;
      }
    }
    return false;

  }

  __global__
  void assignPivotID(int *d_nextQueue, int *d_nextQueueID, int *d_IDTagList,
                     int nextQueueSize, int IDTagSize, bool *IDTagListOverflow) {
      
      int thid = blockIdx.x * blockDim.x + threadIdx.x;

      if (thid < nextQueueSize) {
        
        // Check whether this is the first occurence of the vertex in the frontier
        if (thid == 0 || d_nextQueue[thid] != d_nextQueue[thid - 1]) {
          
          int v = d_nextQueue[thid];
          
          // Check next log(n) places in d_nextQueue for vertex v
          // Add pivot ID to d_IDTagList, if index i in d_nextQueue matches v
          for (int i = thid; i < thid + IDTagSize; i++) {
            if (d_nextQueue[i] == v && !updateIDTagList(d_IDTagList, IDTagSize,
                                                        v, d_nextQueueID[i])) {
              IDTagListOverflow[0] = 1;
              return;
            }
          }
        }
      }
  }

}