#pragma once
#include "base.h"



// Every Node has their own Q matrix and node number
// Each cannot observe other Node's Q matrix


class SlottedAlohaRL_TD : public RL_Base {
public:
    SlottedAlohaRL_TD(const double& epsilon, const double& ramda = 1, const double& alpha = 0.1, const double& gamma = 0.6) :
        RL_Base(epsilon, "TD", ramda, alpha, gamma) {}

protected:
    NodeArr nodes;
    Plot_Data data;
    Action A_1, A_2;
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

private:
    void init() {
        for (auto& node : nodes) {
            A_1[node.node_num] = get_rand_int(0, NumSlot - 1);
        }
    }
    
    void run_iteration() {
        for (episode_num = 0; episode_num < episode_num_target; episode_num++) {
            A_1 = choose_action(A_1);
#ifdef DEBUG
            cout << "Episode #" << episode_num << ":" << endl;
#endif 
            int frame_num = 0;

            // choose action and update Q matrix every step
            for (frame_num = 0; frame_num < frame_num_target; frame_num++) {
                A_2 = choose_action(A_1);
                update();
                render(frame_num);
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
            final_reward();

            for (auto& node : nodes) {
                node.reset(false);
            }
        }
    }
    // Choose an action based on the given state
    Action choose_action(const Action &action) {
        Action temp_action;
        for (auto& node : nodes) {
            if (node.is_success) {
                temp_action[node.node_num] = -1;
                continue;
            }
            if (epsilon / episode_num >= get_rand_real(0, 1)) {
                //random action
                auto random_num = get_rand_int(0, NumSlot - 1);
                temp_action[node.node_num] = random_num;
            }
            else {
                //optimal action
                int index;
                node.Q.maxCoeff(&index);
                temp_action[node.node_num] = index;
            }
        }
        return temp_action;
    }

    // Update Q matrix based on TD algorithm
    void update() {
        int reward;
        for (auto& node : nodes) {
            if (node.remaining_data != 0) {
                auto nn = node.node_num;
                // collision X
                if (std::count(A_2.begin(), A_2.end(), A_2[nn]) == 1) {
                    reward = positive_feedback;
                    node.remaining_data -= 1;
                    ++success_data;
                    ++success_frame;
                    if (node.remaining_data == 0) {
                        node.is_success = true;
                    }
                }
                // collision O
                else {
                    reward = negative_feedback;
                }
                double predict = node.Q(A_1[nn]);
                double target = reward + gamma * node.Q(A_2[nn]);
                node.Q(A_1[nn]) += alpha * (target - predict);
            }
        }
        data.success_frame[frame_num_data++] += success_frame;
        success_frame = 0;
        A_1 = A_2;
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



};