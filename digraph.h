#ifndef BFS_CUDA_GRAPH_H
#define BFS_CUDA_GRAPH_H

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <iostream>

struct Digraph {
    std::vector<int> adjacencyList; // all edges
    std::vector<int> edgesOffset; // offset to adjacencyList for every vertex
    std::vector<int> edgesSize; //number of edges for every vertex
    int numVertices = 0;
    int numEdges = 0;
};

void readGraph(Digraph &G, int argc, char **argv);
Digraph transposeGraph(Digraph &G);

#endif //BFS_CUDA_GRAPH_H
