/* Tim Zuijderduijn (s3620166) 2025
   fineman.cu
*/

#include "bfs.cu"

#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>
#include <thrust/set_operations.h>

bool parSC(Digraph &G, int maxLevel, float eps_pi,
           thrust::device_vector<int> &d_shortcutEdgesAll) {

  int k = std::ceil(std::log(1 - (G.numVertices / (2*(1 + eps_pi))) + 0.5*G.numVertices)
                    / std::log(1 + eps_pi));
  int numLayers = std::log(G.numVertices) / std::log(6.5); // N_L in the paper?
  int maxK = 2*k; // N_k in the paper?
  int level = 0;
  int partitionSize = 0;

  // Rj+, Rj-, Fj+, Fj-
  thrust::device_vector<int> d_coreForward(G.numVertices, 0);
  thrust::device_vector<int> d_coreBack(G.numVertices, 0);
  thrust::device_vector<int> d_fringeForward(G.numVertices, 0);
  thrust::device_vector<int> d_fringeBack(G.numVertices, 0);

  thrust::device_vector<int> d_forwardVert(G.numVertices, 0);
  thrust::device_vector<int> d_backVert(G.numVertices, 0);  

  thrust::device_vector<int> d_deadVertices(G.numVertices, 0);
  //thrust::device_vector<thrust::tuple<int, int>> d_shortcutEdges();

  // TODO: random seed
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

    // Contains vertices which will be searched from in current Xi
    std::vector<int> partitionVertices(d_vertPermutation.begin(), d_vertPermutation.begin() + partitionSize);

    if (level >= maxLevel) {
      break;
    }
 
    int min_d = 1 + (maxLevel - level) * maxK * numLayers - i * numLayers;
    int max_d = min_d + numLayers - 1;
    //std::cout << min_d << " " << max_d << "\n";
    
    // TODO: choose random distance
    int dist = min_d;

    // TODO: variant of G for back search
    if (!startAugBFS(G, partitionVertices, dist, d_coreForward) &&
        !startAugBFS(G, partitionVertices, dist, d_coreBack) &&
        !startAugBFS(G, partitionVertices, dist + 1, d_fringeForward) &&
        !startAugBFS(G, partitionVertices, dist + 1, d_fringeBack)) {
      return false;
    }

    // Fringe sets should not contain core vertices
    /* COMMENTED FOR THE COMPILER
    auto fringeForwardLast = thrust::set_difference(d_fringeForward.begin(), d_fringeForward.end(),
                                                    d_coreForward.begin(), d_coreForward.end(),
                                                    d_fringeForward.begin()); 
    auto fringeBackLast = thrust::set_difference(d_fringeBack.begin(), d_fringeBack.end(),
                                                 d_coreBack.begin(), d_coreBack.end(),
                                                 d_fringeBack.begin());
    */

    // Calculating sets indicated by V_F,j and V_B,j in paper
    // Returns iterator referring to the end of output sets
    /* COMMENTED FOR THE COMPILER
    auto forwardLast = thrust::set_difference(d_coreForward.begin(), d_coreForward.end(),
                                              d_coreBack.begin(), d_coreBack.end(), d_forwardVert.begin());
    auto backLast = thrust::set_difference(d_coreBack.begin(), d_coreBack.end(),
                                           d_coreForward.begin(), d_coreForward.end(), d_backVert.begin());
    */

    // TODO: new shortcuts, set union of existing shortcuts
    

    // TODO: set dead vertices
    

    level++;
  }

  // TODO: finishing parSC?

  return true;

}

void parDiam(Digraph &G, float failure_prob, float eps_pi) {

  thrust::device_vector<int> d_shortcutEdgesAll;

  for (int i = 0; i < std::log(G.numVertices); i++) {
    for (int j = 0; j < failure_prob * std::log(G.numVertices); j++) {
      parSC(G, std::log(G.numVertices), eps_pi, d_shortcutEdgesAll);
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