#pragma once

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


// Every Node has their own Q matrix and node number
// Each cannot observe other Node's Q matrix
class SlottedAlohaRL_Ramda {
public:
    SlottedAlohaRL_Ramda(const double& ramda = 1, const double& alpha = 0.1, const double& gamma = 0.6) :
        ramda(ramda), alpha(alpha), gamma(gamma)
    {
        plot_str = "TD(" + std::to_string(ramda) + ")";
    }
    void run() {
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
    void reset() {
        *this = SlottedAlohaRL_Ramda();
    }



private:
    struct Node {
        friend class SlottedAlohaRL_Ramda;
    public:
        Node() : node_num(counter++) {}

        RowVectorXd Q = RowVectorXd::Random(NumSlot);
        unsigned int node_num;
        unsigned int remaining_data = 10;
        bool is_success = false;
        std::array<int, NumSlot> e_trace = { 0 };
        void reset(bool iteration_end) {
            remaining_data = 10;
            is_success = false;
            std::fill(e_trace.begin(), e_trace.end(), 0);
            if (iteration_end) {
                Q = RowVectorXd::Random(NumSlot);
            }
        }
    private:
        static unsigned int counter;
    };


    typedef std::array<int, NumNode> State;
    typedef std::array<int, NumNode> Action;
    typedef std::array<Node, NumNode> NodeArr;

    void init() {
        for (auto& node : nodes) {
            A_1[node.node_num] = get_rand_int(0, NumSlot - 1);
        }
    }

    // This is a block where it runs target number of episodes and finishes
    void run_iteration() {
        for (episode_num = 0; episode_num < episode_num_target; episode_num++) {
            A_1 = choose_action(A_1);
            returns[0] = A_1;

#ifdef DEBUG
            cout << "Episode #" << episode_num << ":" << endl;
#endif 

            // choose action and update Q matrix every step
            for (frame_num = 0; frame_num < frame_num_target; frame_num++) {
                check_collision(A_1);
                A_2 = choose_action(A_1);
                update();
                render(frame_num);
                A_1 = A_2;
            }


            // figure out if every node has successfully finished their transmissions
            bool is_complete = true;
            for (auto node : nodes) {
                node.is_success ? ++success_node : is_complete = false;
            }
            data.success_data[episode_num] += success_data;
            data.success_node[episode_num] += success_node;
            success_data = 0;
            success_node = 0;
            final_reward();

#ifdef DEBUG
            cout << "Final policy matrix" << endl;
            for (auto node : nodes) {
                cout << std::setprecision(3) << std::fixed << node.Q << endl;
            }

            if (!is_complete) {
                cout << "Could not finish transmitting data in " << frame_num_target << " frames." << endl;
                ++total_failure;
            }
            else {
                cout << "Finished transmitting in " << frame_num << " frames." << endl;
                ++total_success;
            }
            cout << "Total Success: " << total_success << endl;
            cout << "Total Failure: " << total_failure << endl;
#endif

            std::for_each(nodes.begin(), nodes.end(), [](Node& node) { node.reset(false); });
        }
    }

    // Choose an action based on the given state
    Action choose_action(const Action& action) {
        Action action_ret;
        for (auto& node : nodes) {
            if (node.is_success) {
                action_ret[node.node_num] = -1;
                continue;
            }
            if (get_epsilon() >= get_rand_real(0, 1)) {
                //random action
                auto random_num = get_rand_int(0, NumSlot - 1);
                action_ret[node.node_num] = random_num;
                ++node.e_trace[random_num];
            }
            else {
                //optimal action
                int index;
                node.Q.maxCoeff(&index);
                action_ret[node.node_num] = index;
                ++node.e_trace[index];
            }
        }
        return action_ret;
    }

    void check_collision(const Action& action) {
        for (auto& node : nodes) {
            if (std::count(action.begin(), action.end(), action[node.node_num]) == 1) {
                --node.remaining_data;
                ++success_data;
                ++success_frame;
                if (node.remaining_data == 0) {
                    node.is_success = true;
                }
            }
        }
        data.success_frame[frame_num_data++] += success_frame;
        success_frame = 0;
    }

    // Update Q matrix based on TD algorithm
    void update() {
        int reward;

        double target[NumNode] = { 0.0 };
        double predict = 0.0;

        unsigned int node_num;
        double delta;
        for (auto& node : nodes) {
            // getting appropriate rewards
            if (node.remaining_data != 0) {
                node_num = node.node_num;
                // collision X
                if (std::count(A_2.begin(), A_2.end(), A_2[node_num]) == 1) {
                    reward = positive_feedback;
                }
                // collision O
                else {
                    reward = negative_feedback;
                }
                double predict = node.Q(A_1[node_num]);
                double target = reward + gamma * node.Q(A_2[node_num]);
                delta = target - predict;
                node.Q(A_1[node_num]) += alpha * delta * node.e_trace[node_num];
                node.e_trace[node_num] *= gamma * ramda;
            }

        }
    }

    // distribute reward at the end of an episode based on 
    // whether a node has finised transmission or not
    void final_reward() {
        for (auto& node : nodes) {
            int index = 0;
            node.Q.maxCoeff(&index);
            if (node.is_success) {
                node.Q[index] += episode_success;
            }
            else {
                node.Q[index] += episode_failure;
            }
        }
    }


    // rendering which node decided to transmit on which slot
    void render(int step) {
#ifdef DEBUG
        printf("Step #%d ", step);
        for (auto i : A_1) {
            printf("%2d ", i);
        }
        printf("\n");
#endif
    }

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

    void reset(bool episode_end) {
        for (auto& node : nodes) {
            node.reset(true);
        }
        frame_num_data = 0;
    }

    inline double get_epsilon() {
        return epsilon / episode_num;
    }
    NodeArr nodes;
    Action A_1, A_2;
    Action returns[frame_num_target + 1];
    Plot_Data data;

    std::string plot_str;

    int success_frame = 0;
    int success_data = 0;
    int success_node = 0;

    double gamma = 0.6;
    double alpha = 0.1;
    double epsilon = 0.1;



    double ramda = 1;


    unsigned int total_success = 0;
    unsigned int total_failure = 0;
    unsigned int frame_num = 0;
    unsigned int frame_num_data = 0;
    unsigned int episode_num = 0;
};