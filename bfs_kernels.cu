/* Tim Zuijderduijn (s3620166) 2025
   bfs_kernels.cu
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

          int sum = 0;
          if (thid > 0) {
              sum = d_degrees[thid - 1];
          }

          int u = d_currentQueue[thid];
          int counter = 0;
          for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
              int v = d_adjacencyList[i];
              if (d_parent[v] == i && v != u) {
                  int nextQueuePlace = sum + counter;
                  d_nextQueue[nextQueuePlace] = v;
                  counter++;
              }
          }
      }
  }

  /*************************************************************
   * Augmented BFS
  */

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
  void augCountDegrees(int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize,
                       int queueSize, int *d_currentQueue, int *d_degrees,
                       int* d_IDTagList, int* d_queueID, int IDTagSize) {
      int thid = blockIdx.x * blockDim.x + threadIdx.x;

      if (thid < queueSize) {
          int u = d_currentQueue[thid];
          int u_pivotID = d_queueID[thid]; // current node's pivot ID
          int degree = 0;

          // Counts the total degrees of vertex u
          // Excludes itself and vertices already visited by u_pivotID
          for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
              int v = d_adjacencyList[i];
              if (v != u && !isVisitedByPivotID(v, u_pivotID, IDTagSize, d_IDTagList)) {
                ++degree;
              }
          }
          d_degrees[thid] = degree;
      }
  }

  __global__
  void augAssignVNQ(int *d_adjacencyList, int *d_edgesOffset, int *d_edgesSize, int queueSize,
                    int *d_currentQueue, int *d_nextQueue, int *d_degrees, int nextQueueSize,
                    int *d_IDTagList, int *d_queueID, int *d_nextQueueID, int IDTagSize) {
      int thid = blockIdx.x * blockDim.x + threadIdx.x;

      if (thid < queueSize) {
          int sum = 0;
          if (thid > 0) {
              sum = d_degrees[thid - 1];
          }

          int u = d_currentQueue[thid];
          int u_pivotID = d_queueID[thid]; // current node's pivot ID
          int counter = 0;
          for (int i = d_edgesOffset[u]; i < d_edgesOffset[u] + d_edgesSize[u]; i++) {
              int v = d_adjacencyList[i];
              // scan through d_IDTagList to check if node has been reached by pivot ID 
              // if already visited by pivot id, do not add to queue
              if (v != u && !isVisitedByPivotID(v, u_pivotID, IDTagSize, d_IDTagList)) {
                int nextQueuePlace = sum + counter;
                d_nextQueue[nextQueuePlace] = v;
                d_nextQueueID[nextQueuePlace] = u_pivotID; // adds corresponding pivot ID to next queue
                counter++;
              }
          }
      }
  }

  __global__
  void numberPivotID(int *d_nextQueue, int *d_IDTagList,
                     int nextQueueSize, int IDTagSize, bool *IDTagListOverflow,
                     int *d_IDTagCount, int *d_nextQueueIDListNum) {
    int thid = blockIdx.x * blockDim.x + threadIdx.x;

      if (thid < nextQueueSize) {
        
        // Check whether this is the first occurence of the vertex in the frontier
        if (thid == 0 || d_nextQueue[thid] != d_nextQueue[thid - 1]) {
          
          int v = d_nextQueue[thid];
          
          // Check next log(n) places in d_nextQueue for vertex v
          // If vertex i in d_nextQueue matches v, assign index for i in ID tag list
          for (int i = thid; i < thid + IDTagSize; i++) {
            if (i < nextQueueSize && d_nextQueue[i] == v) {
              d_IDTagCount[v] += 1;
              if (d_IDTagCount[v] > IDTagSize) {
                IDTagListOverflow[0] = 1;
                return;
              }
              d_nextQueueIDListNum[i] = d_IDTagCount[v] - 1;
            }
          }
        }
      }

  }

  __global__
  void assignPivotID(int *d_nextQueue, int *d_nextQueueID, int *d_IDTagList,
                     int nextQueueSize, int IDTagSize,
                     int *d_IDTagCount, int *d_nextQueueIDListNum,
                     int *d_augDistance, int level) {
      
      int thid = blockIdx.x * blockDim.x + threadIdx.x;

      if (thid < nextQueueSize) {
        
        int u = d_nextQueue[thid];
        int u_pivotID = d_nextQueueID[thid];
        int IDTagLoc = u * IDTagSize + d_nextQueueIDListNum[thid];

        // Add pivot ID to tag list and corresponding distance
        d_IDTagList[IDTagLoc] = u_pivotID;
        if (level + 1 < d_augDistance[IDTagLoc]) {
          d_augDistance[IDTagLoc] = level + 1;
        }

      }
  }

}