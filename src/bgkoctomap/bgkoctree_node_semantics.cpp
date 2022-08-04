#include "bgkoctree_node_semantics.h"
#include <cmath>

#include <assert.h>
#include <algorithm>
#include <set>
#include <random>

namespace la3dm {

    /// Default static values
    int Semantics::num_class = 2;
    float Semantics::sf2 = 1.0f;
    float Semantics::ell = 1.0f;
    float Semantics::free_thresh = 0.3f;
    float Semantics::occupied_thresh = 0.7f;
    float Semantics::var_thresh = 1000.0f;
    float Semantics::prior_A = 0.5f;
    float Semantics::prior_B = 0.5f;
    float Semantics::prior = 0.5f;

    void Semantics::get_probs(std::vector<float>& probs) const {
        assert (probs.size() == num_class);
        float sum = 0;
        for (auto m : ms)
          sum += m;
        for (int i = 0; i < num_class; ++i)
          probs[i] = ms[i] / sum;
    }

    void Semantics::get_vars(std::vector<float>& vars) const {
      assert (vars.size() == num_class);
      float sum = 0;
      for (auto m : ms)
        sum += m;
      for (int i = 0; i < num_class; ++i)
        vars[i] = ((ms[i] / sum) - (ms[i] / sum) * (ms[i] / sum)) / (sum + 1.0f);
    }

    float Semantics::get_prob_traversability() const {
        return tm_A / (tm_A + tm_B);
    }

    float Semantics::get_var_traversability() const {
        return (tm_A * tm_B) / ( (tm_A + tm_B) * (tm_A + tm_B) * (tm_A + tm_B + 1.0f));
    }

    // Convert Dirichlet to beta distribution
    void Semantics::get_semantic_traversability() {
        // Toy example
	//std::set<int> untraversable = {2, 3};
	//std::set<int> traversable = {1};
        
	// Cassie exp
	std::set<int> untraversable = {1, 5, 6, 7, 8, 9, 10, 12, 13};
	std::set<int> traversable = {2, 3, 4, 11};
        
        // KITTI exp
	//std::set<int> traversable = {1, 2, 10};
	//std::set<int> unsure = {0};

	// TartanAir exp
	//std::set<int> untraversable = {13, 96, 110, 129, 137, 153, 164, 167, 178, 184, 196, 199, 200, 220, 227, 245, 246, 250};
	//std::set<int> traversable = {64, 197, 205, 207, 222};
	//std::set<int> unsure = {152, 160, 163, 226, 230, 244, 252};
	
	// TartanAir neighborhood
        //std::set<int> untraversable = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 15, 16};
	//std::set<int> traversable = {12, 13, 17};
	
	for (std::set<int>::iterator it = traversable.begin(); it != traversable.end(); ++it)
	  stm_A += ms[*it];
	for (std::set<int>::iterator it = untraversable.begin(); it != untraversable.end(); ++it)
	  stm_B += ms[*it];
	//std::cout << stm_A << " "<<stm_B << " " << get_prob_semantic_traversability() << std::endl;
    }

    // Compute mode
    float Semantics::get_prob_semantic_traversability() const {
        if (stm_A > 1 && stm_B > 1)
          return (stm_A - 1) / (stm_A + stm_B - 2);
	else if (stm_A <= 1 && stm_B > 1)
	  return 0;
	else if (stm_A > 1 && stm_B <= 1)
	  return 1;
	else
	  return 0.5;
    }

    // Sample from Bernoulli distribution
    int Semantics::get_meas_semantic_traversability() const {
        const float range_from  = 0;
        const float range_to    = 1;
        std::random_device rand_dev;
        std::mt19937 generator(rand_dev());
        std::uniform_real_distribution<float> distr(range_from, range_to);
        if (distr(generator) <= get_prob_semantic_traversability())
	  return 1;
        else
          return 0;
    }

    void Semantics::update(std::vector<float>& ybars) {
        assert (ybars.size() == num_class);
        classified = true;
        
	for (int k = 0; k < num_class; ++k)
          ms[k] += ybars[k];
	
	std::vector<float> probs(num_class);
	this->get_probs(probs);
	
	semantics = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
	state = semantics == 0 ? State::FREE : State::OCCUPIED;
    }


    void Semantics::update_traversability(float ybar, float kbar) {
        tm_A += ybar;
        tm_B += kbar - ybar;
    }
}
