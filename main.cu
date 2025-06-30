/* Tim Zuijderduijn (s3620166) 2025
   main.cu
*/

#include "fineman.cu"
#include <cuda.h>

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;

void checkError(CUresult error, std::string msg) {
  if (error != CUDA_SUCCESS) {
    printf("%s: %d\n", msg.c_str(), error);
    exit(1);
  }
}

void initCuda(Digraph &G) {
  //initialize
  cuInit(0);
  checkError(cuDeviceGet(&cuDevice, 0), "cannot get device 0");
  checkError(cuCtxCreate(&cuContext, 0, cuDevice), "cannot create context");
}

int main(int argc, char** argv) {

  Digraph G;
  int startVertex = atoi(argv[1]);
  readGraph(G, argc, argv);

  printf("Number of vertices %d\n", G.numVertices);
  printf("Number of edges %d\n\n", G.numEdges);

  // vectors for results
  std::vector<int> distance(G.numVertices, std::numeric_limits<int>::max());
  std::vector<int> parent(G.numVertices, std::numeric_limits<int>::max());

  initCuda(G);

  // runs the bfs implementations
  // startBFS(G, startVertex, distance, parent);

  startFineman(G);

  return 0;

} // main