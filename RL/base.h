#include <iostream>
#include <vector>
#include <array>
#include <set>
#include <cmath>
#include <limits>
#include <iomanip>
#include <cmath>

#include <Eigen/Dense>
#include "matplotlibcpp.h"

#include "z_random.h"
#include "global.h"

using namespace Eigen;
namespace plt = matplotlibcpp;
using std::cout;
using std::endl;

class RL_Base {
public:
    RL_Base() = default;
    RL_Base(const double& epsilon, const std::string& algo, const double& ramda = 1, const double& alpha = 0.1, const double& gamma = 0.6) :
        epsilon(epsilon), ramda(ramda), alpha(alpha), gamma(gamma)
    {
        plot_str = algo + "(e=" + std::to_string(epsilon) + ")";
        int i = 0;
        for (auto& node : nodes) {
            node.node_num = i++;
        }
    }

    virtual void run() {
        std::ios::sync_with_stdio(false);
        init();
        for (int i = 0; i < iterations_target; i++) {
            run_iteration();
            change_seed();
            reset(true);
        }
        calc_average();
        plot();
    }
    virtual ~RL_Base() = default;


    struct Node;
    typedef std::array<int, NumNode> State;
    typedef std::array<int, NumNode> Action;
    typedef std::array<Node, NumNode> NodeArr;


    struct Node {
        friend class RL_Base;
    public:
        Node() = default;
        Node(const int& _node_num) : node_num(_node_num) {}

        RowVectorXd Q = RowVectorXd::Random(NumSlot);
        unsigned int node_num;
        unsigned int remaining_data = 10;
        std::array<int, NumSlot> num_visit = { 0 };
        std::array<int, NumSlot> e_trace = { 0 };
        bool is_success = false;

        void reset(bool iteration_end) {
            remaining_data = 10;
            is_success = false;
            std::fill(e_trace.begin(), e_trace.end(), 0);
            std::fill(num_visit.begin(), num_visit.end(), 0);
            if (iteration_end) {
                Q = RowVectorXd::Random(NumSlot);
            }
        }
    };


    virtual void init() = 0;

    void reset(const bool &episode_end) {
        for (auto& node : nodes) {
            node.reset(true);
        }
        frame_num_data = 0;
    }

    virtual void run_iteration() = 0;
    virtual void update() = 0;
    virtual void final_reward() = 0;
    void plot() {
        plt::subplot(1, 3, 1);
        plt::title("Success Frame");
        plt::named_plot(plot_str, data.steps, data.success_frame);
        plt::xlabel("# Steps");
        plt::subplot(1, 3, 2);
        plt::title("Success Data");
        plt::named_plot(plot_str, data.episodes, data.success_data);
        plt::xlabel("# Episodes");
        plt::subplot(1, 3, 3);
        plt::title("Success Node");
        plt::named_plot(plot_str, data.episodes, data.success_node);
        plt::xlabel("# Episodes");
        plt::legend();
    }

    void calc_average() {
        std::for_each(data.success_frame.begin(), data.success_frame.end(), [](double& val) {val = val / iterations_target; });
        std::for_each(data.success_data.begin(), data.success_data.end(), [](double& val) {val = val / iterations_target; });
        std::for_each(data.success_node.begin(), data.success_node.end(), [](double& val) {val = val / iterations_target; });
    }
    
    std::string plot_str;

    NodeArr nodes;
    Plot_Data data;

    int success_frame = 0;
    int success_data = 0;
    int success_node = 0;
    
    double ramda, alpha, gamma;
    double epsilon;


    unsigned int total_success = 0;
    unsigned int total_failure = 0;
    unsigned int frame_num = 0;
    unsigned int frame_num_data = 0;
    unsigned int episode_num = 0;
};