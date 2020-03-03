#include "bgkoctree_node_semantics.h"
#include <cmath>

#include <assert.h>
#include <algorithm>

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

    float Semantics::get_prob_traversability() const {
        return tm_A / (tm_A + tm_B);
    }

    void Semantics::update(std::vector<float>& ybars) {
        assert (ybars.size() == num_class);
        classified = true;
        for (int k = 0; k < num_class; ++k)
          ms[k] += ybars[k];

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
        if (semantics == 1 || semantics == 2 || semantics == 10)
          tm_A += 100;
        else if (semantics == 0) {}
        else
          tm_B += 100;
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
