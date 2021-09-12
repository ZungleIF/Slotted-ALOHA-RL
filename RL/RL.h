#pragma once
#include "base.h"

struct Node_MC;
class SlottedAlohaRL_MC;

// Every Node has their own Q matrix and node number
// Each cannot observe other Node's Q matrix



class SlottedAlohaRL_MC : public RL_Base {
public:
    SlottedAlohaRL_MC(const double& epsilon, const double& ramda = 1, const double& alpha = 0.1, const double& gamma = 0.6) :
        RL_Base(epsilon, "MC", ramda, alpha, gamma) {}




private:
    void init() {
    }
    void run_iteration() {
        // iterate episodes
        for (episode_num = 0; episode_num < episode_num_target; episode_num++) {
#ifdef DEBUG
            cout << "Episode #" << episode_num << ":" << endl;
#endif 
            // choose action and update Q matrix every step
            for (frame_num = 0; frame_num < frame_num_target; frame_num++) {
                choose_action();
                render(frame_num);
            }

            update();
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
            for (auto& node : nodes) {
                node.reset(false);
            }

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
        }
    }
    // Choose an action based on the given state
    // and save returned actions to a vector according to MC algorithm
    void choose_action() {
        Action temp_action;
        for (auto& node : nodes) {
            //random action
            if (epsilon / episode_num >= get_rand_real(0, 1)) {
                auto random_num = get_rand_int(0, NumSlot - 1);
                temp_action[node.node_num] = random_num;
                ++node.num_visit[random_num];
            }
            //optimal action
            else {
                int index;
                node.Q.maxCoeff(&index);
                temp_action[node.node_num] = index;
                ++node.num_visit[index];
            }
        }
        now = temp_action;
        returns.push_back(temp_action);
    }

    // Update Q matrix based on MC algorithm
    void update() {
        int reward;
        for (auto action : returns) {
            for (auto& node : nodes) {
                if (node.remaining_data != 0) {
                    auto nn = node.node_num;
                    // collision X
                    if (std::count(action.begin(), action.end(), action[nn]) == 1) {
                        reward = positive_feedback;
                        --node.remaining_data;
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
                    node.Q(action[nn]) += reward;
                }
            }
            data.success_frame[frame_num_data++] += success_frame;
            success_frame = 0;
        }
        returns.clear();
        // make an average out of all rewards
        for (auto& node : nodes) {
            int i = 0;
            for (auto& q : node.Q) {
                if (node.num_visit[i] != 0)
                    q = q / node.num_visit[i];
                ++i;
            }
        }
    }

    // distribute rewards at the end of an episode based on 
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
        for (auto i : now) {
            printf("%2d ", i);
        }
        printf("\n");
#endif
    }

    std::vector<Action> returns;
    Action now;
};