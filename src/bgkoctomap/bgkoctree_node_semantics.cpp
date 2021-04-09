#include "bgkoctree_node_semantics.h"
#include <cmath>

#include <assert.h>
#include <algorithm>
#include <list>

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

    /*Occupancy::Occupancy(float A, float B) : m_A(Occupancy::prior_A + A), m_B(Occupancy::prior_B + B) {
        classified = false;
        float var = get_var();
        if (var > Occupancy::var_thresh)
            state = State::UNKNOWN;
        else {
            float p = get_prob();
            state = p > Occupancy::occupied_thresh ? State::OCCUPIED : (p < Occupancy::free_thresh ? State::FREE
                                                                                                   : State::UNKNOWN);
        }
    }*/

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

    void Semantics::update(std::vector<float>& ybars) {
        assert (ybars.size() == num_class);
        classified = true;
        
	// TartanAir exp
	std::list<int> untraversable = {13, 96, 110, 129, 137, 153, 164, 167, 178, 184, 196, 199, 200, 220, 227, 245, 246, 250};
	std::list<int> traversable = {64, 197, 205, 207, 222};
	std::list<int> unsure = {152, 160, 163, 226, 230, 244, 252};

	for (int k = 0; k < num_class; ++k) {
          ms[k] += ybars[k];
	  if (find(untraversable.begin(), untraversable.end(), k) != untraversable.end())
	    tm_B += ybars[k];
	}

        //float var = get_var();
        //if (var > Occupancy::var_thresh)
        //    state = State::UNKNOWN;
        //else {
            
            std::vector<float> probs(num_class);
            this->get_probs(probs);

            semantics = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));

            state = semantics == 0 ? State::FREE : State::OCCUPIED;

            //state = p > Occupancy::occupied_thresh ? State::OCCUPIED : (p < Occupancy::free_thresh ? State::FREE
            //                                                                                       : State::UNKNOWN);
        //}
    }

    void Semantics::update_traversability_with_semantics() {
        // Toy example
        //if (semantics == 1)
          //tm_A += 1;
        //if (semantics > 1)
          //tm_B += 1;
        
        // Cassie exp
        //if (semantics == 2 || semantics == 3 || semantics == 4 || semantics == 11)
          //tm_A += 50;
        //if (semantics == 1 || semantics == 5 || semantics == 6 || semantics == 7 || semantics == 8 ||
           // semantics == 9 || semantics == 10 || semantics == 12 || semantics == 13)
          //tm_B += 50;
        
        // KITTI exp
        /*if (semantics == 1 || semantics == 2 || semantics == 10)
          tm_A += 50;
        else if (semantics == 0) {}
        else
          tm_B += 50;*/

	// TartanAir exp
	std::list<int> untraversable = {13, 96, 110, 129, 137, 153, 164, 167, 178, 184, 196, 199, 200, 220, 227, 245, 246, 250};
	std::list<int> traversable = {64, 197, 205, 207, 222};
	std::list<int> unsure = {152, 160, 163, 226, 230, 244, 252};
	if (find(untraversable.begin(), untraversable.end(), semantics) != untraversable.end()) {
	  tm_B += 1;
	} else if (find(traversable.begin(), traversable.end(), semantics) != traversable.end()) {
	  tm_A += 1;
	}
    }

    void Semantics::update_traversability(float ybar, float kbar) {
        //classified = true;
        tm_A += ybar;
        tm_B += kbar - ybar;
    }


    /*std::ofstream &operator<<(std::ofstream &os, const Occupancy &oc) {
        os.write((char *) &oc.m_A, sizeof(oc.m_A));
        os.write((char *) &oc.m_B, sizeof(oc.m_B));
        return os;
    }

    std::ifstream &operator>>(std::ifstream &is, Occupancy &oc) {
        float m_A, m_B;
        is.read((char *) &m_A, sizeof(m_A));
        is.read((char *) &m_B, sizeof(m_B));
        oc = OcTreeNode(m_A, m_B);
        return is;
    }

    std::ostream &operator<<(std::ostream &os, const Occupancy &oc) {
        return os << '(' << oc.m_A << ' ' << oc.m_B << ' ' << oc.get_prob() << ')';
    }*/
}
