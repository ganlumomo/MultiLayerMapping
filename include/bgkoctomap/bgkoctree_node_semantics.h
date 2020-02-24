#ifndef LA3DM_BGK_SEMANTICS_H
#define LA3DM_BGK_SEMANTICS_H

#include <iostream>
#include <fstream>
#include <vector>

#include "bgkoctree_node.h"

namespace la3dm {

    /*
     * @brief Inference ouputs and occupancy state.
     *
     * Occupancy has member variables: m_A and m_B (kernel densities of positive
     * and negative class, respectively) and State.
     * Before using this class, set the static member variables first.
     */
    class Semantics {

        //friend std::ostream &operator<<(std::ostream &os, const Occupancy &oc);

        //friend std::ofstream &operator<<(std::ofstream &os, const Occupancy &oc);

        //friend std::ifstream &operator>>(std::ifstream &is, Occupancy &oc);

        friend class BGKOctoMap;

    public:
        /*
         * @brief Constructors and destructor.
         */
        Semantics() : 
          ms(std::vector<float>(num_class, prior)),
          state(State::UNKNOWN), 
          tm_A(Semantics::prior_A),
          tm_B(Semantics::prior_B) { 
            classified = false; }

        //Occupancy(float A, float B);

        Semantics(const Semantics &other) : 
          ms(other.ms),
          state(other.state),
          semantics(other.semantics),
          tm_A(other.tm_A), 
          tm_B(other.tm_B) { }

        Semantics &operator=(const Semantics &other) {
            ms = other.ms;
            state = other.state;
            semantics = other.semantics;
            tm_A = other.tm_A;
            tm_B = other.tm_B;
            return *this;
        }

        ~Semantics() { }

        /*
         * @brief Exact updates for nonparametric Bayesian kernel inference
         * @param ybar kernel density estimate of positive class (occupied)
         * @param kbar kernel density of negative class (unoccupied)
         */
        void update(std::vector<float>& ybars);
        
        void update_traversability_with_semantics();

        void update_traversability(float ybar, float kbar);

        /// Get probability of occupancy.
        void get_probs(std::vector<float>& probs) const;

        float get_prob_traversability() const;

        /// Get variance of occupancy (uncertainty)
        //inline float get_var() const { return (m_A * m_B) / ( (m_A + m_B) * (m_A + m_B) * (m_A + m_B + 1.0f)); }

        /*
         * @brief Get occupancy state of the node.
         * @return occupancy state (see State).
         */
        inline State get_state() const { return state; }

        inline int get_semantics() const {return semantics; }

        /// Prune current node; set state to PRUNED.
        inline void prune() { state = State::PRUNED; }

        /// Only FREE and OCCUPIED nodes can be equal.
        /*inline bool operator==(const Occupancy &rhs) const {
            return this->state != State::UNKNOWN && this->state == rhs.state;
        }*/

        bool classified;

    private:
        // For semantics
        std::vector<float> ms;
        State state;
        int semantics;
        static int num_class;

        static float sf2;
        static float ell;   // length-scale

        static float prior;

        static float free_thresh;     // FREE occupancy threshold
        static float occupied_thresh; // OCCUPIED occupancy threshold
        static float var_thresh;

        // For traversability
        float tm_A;
        float tm_B;
    
        static float prior_A; // prior on alpha
        static float prior_B; // prior on beta
    };
}

#endif // LA3DM_SEMANTICS_H
