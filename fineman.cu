/* Tim Zuijderduijn (s3620166) 2025
   fineman.cu
*/

#include "bfs.cu"

void parSC(Digraph &G, float eps_pi,
           thrust::device_vector<int> &d_shortcutEdges) {

  int k = std::log(1 - (G.numVertices / (2*(1 + eps_pi))) + 0.5*G.numVertices)
          / std::log(1 + eps_pi);
  
  // TODO: randomly permute list of vertices
  
  // TODO: split list of vertices into subsequences

  for (int i = 0; i < 2*k; i++) {
    
    // TODO: calculate max and min distance and choose random distance

    // TODO: make use of augBFS

    // TODO: vertices reached by searches, new shortcuts, adjust graph accordingly using dead vertices

  }

  // TODO: finishing parSC?

}

void parDiam(Digraph &G, float failure_prob, float eps_pi) {

  thrust::device_vector<int> d_shortcutEdges;

  for (int i = 0; i < std::log(G.numVertices); i++) {
    for (int j = 0; j < failure_prob * std::log(G.numVertices); j++) {
      parSC(G, eps_pi, d_shortcutEdges);
    }
    // add shortcuts to graph found in for loop j
  }
  // return augmented graph?

}

void startFineman(Digraph &G) {
  
  float failure_prob = 1.0;
  float eps_pi = 1.0;
  parDiam(G, failure_prob, eps_pi);

}