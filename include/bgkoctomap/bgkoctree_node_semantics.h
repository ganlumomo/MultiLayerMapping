#ifndef LA3DM_BGK_SEMANTICS_H
#define LA3DM_BGK_SEMANTICS_H

#include <iostream>
#include <fstream>
#include <vector>

namespace la3dm {
    
    /// Semantics state: before pruning: FREE, OCCUPIED, UNKNOWN; after pruning: PRUNED
    enum class State : char {
        FREE, OCCUPIED, UNKNOWN, PRUNED
    };
    
    /*
     * @brief Inference ouputs and semantics state.
     *
     * Before using this class, set the static member variables first.
     */
    class Semantics {

        friend class BGKOctoMap;
        
    public:
        /*
         * @brief Constructors and destructor.
         */
        Semantics() : 
          ms(std::vector<float>(num_class, prior)),
          state(State::UNKNOWN), 
          tm_A(Semantics::prior_A),
          tm_B(Semantics::prior_B),
	  stm_A(0),
	  stm_B(0) { 
            classified = false; }

        Semantics(const Semantics &other) : 
          ms(other.ms),
          state(other.state),
          semantics(other.semantics),
          tm_A(other.tm_A), 
          tm_B(other.tm_B),
	  stm_A(0),
	  stm_B(0) { }

        Semantics &operator=(const Semantics &other) {
            ms = other.ms;
            state = other.state;
            semantics = other.semantics;
            tm_A = other.tm_A;
            tm_B = other.tm_B;
	    stm_A = 0;
	    stm_B = 0;
            return *this;
        }

        ~Semantics() { }

        /*
         * @brief Exact updates for nonparametric Bayesian kernel inference
         * @param ybar kernel density estimate of positive class (occupied)
         * @param kbar kernel density of negative class (unoccupied)
         */
        void update(std::vector<float>& ybars);
        
        void update_traversability(float ybar, float kbar);
        
	/// Get probability of occupancy.
        void get_probs(std::vector<float>& probs) const;

        void get_vars(std::vector<float>& vars) const;

        float get_prob_traversability() const;

        float get_var_traversability() const;
        
	void get_semantic_traversability();

	float get_prob_semantic_traversability() const;

	int get_meas_semantic_traversability() const;

        /*
         * @brief Get semantics state of the node.
         * @return semantics state (see State).
         */
        inline State get_state() const { return state; }

        inline int get_semantics() const {return semantics; }

        /// Prune current node; set state to PRUNED.
        inline void prune() { state = State::PRUNED; }

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

	// For semantic-traversability
	float stm_A;
	float stm_B;
    
        static float prior_A; // prior on alpha
        static float prior_B; // prior on beta
    };

    typedef Semantics OcTreeNode;
}

#endif // LA3DM_BGK_SEMANTICS_H
