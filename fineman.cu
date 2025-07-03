/* Tim Zuijderduijn (s3620166) 2025
   fineman.cu
*/

#include "bfs.cu"

#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

void parSC(Digraph &G, int maxLevel, float eps_pi,
           thrust::device_vector<int> &d_shortcutEdges) {

  int k = std::ceil(std::log(1 - (G.numVertices / (2*(1 + eps_pi))) + 0.5*G.numVertices)
                    / std::log(1 + eps_pi));
  int numLayers = std::log(G.numVertices) / std::log(6.5); // N_L in the paper
  int maxK = 2*k; // N_k in the paper
  int level = 0;
  int partitionSize = 0;

  // TODO: randomly permute list of vertices
  thrust::device_vector<int> d_vertPermutation(G.numVertices, 0);
  thrust::shuffle_copy(thrust::counting_iterator<int>(0),
                       thrust::counting_iterator<int>(G.numVertices),
                       d_vertPermutation.begin(),
                       thrust::default_random_engine(0));

  //thrust::for_each(d_vertPermutation.begin(), d_vertPermutation.end(), printf_functor());
  //std::cout << "\n";

  for (int i = 0; i < 2*k; i++) {
    
    // Sets size of partition Xi
    if (i < k) {
      partitionSize = std::pow(1 + eps_pi, i);
    }
    else {
      partitionSize = std::pow(1 + eps_pi, 2*k - i + 1);
    }
    std::vector<int> partitionVertices(d_vertPermutation.begin(), d_vertPermutation.begin() + partitionSize);

    if (level >= maxLevel) {
      break;
    }

    // TODO: calculate max and min distance and choose random distance
    int min_d = 1 + (maxLevel - level) * maxK * numLayers - i * numLayers;
    int max_d = min_d + numLayers - 1;
    //std::cout << min_d << " " << max_d << "\n";
    int dist = min_d;

    // TODO: make use of augBFS
    startAugBFS(G, partitionVertices, dist);

    // TODO: vertices reached by searches, new shortcuts, adjust graph accordingly using dead vertices
    
    level++;
  }

  // TODO: finishing parSC?

}

void parDiam(Digraph &G, float failure_prob, float eps_pi) {

  thrust::device_vector<int> d_shortcutEdges;

  for (int i = 0; i < std::log(G.numVertices); i++) {
    for (int j = 0; j < failure_prob * std::log(G.numVertices); j++) {
      parSC(G, std::log(G.numVertices), eps_pi, d_shortcutEdges);
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