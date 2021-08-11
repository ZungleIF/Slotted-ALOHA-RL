#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <set>
#include <cmath>
#include <limits>
#include <iomanip>

#include <Eigen/Dense>
#include "matplotlibcpp.h"

#include "z_random.h"
#include "global.h"

using namespace Eigen;
namespace plt = matplotlibcpp;
using std::cout;
using std::endl;

struct Node_TD;
class SlottedAlohaRL_TD;

// Every Node has their own Q matrix and node number
// Each cannot observe other Node's Q matrix
struct Node_TD {
  friend class SlottedAlohaRL_TD;
public:
  Node_TD() : node_num(counter++), remaining_data(10), is_success(false) {}

  RowVectorXd Q = RowVectorXd::Random(NumSlot);
  unsigned int node_num;
  unsigned int remaining_data;
  bool is_success;

  void reset() {
    remaining_data = 10;
    is_success = false;
  }
private:
  static unsigned int counter;
};


class SlottedAlohaRL_TD {
public:
  void run() {
    std::ios::sync_with_stdio(false);
    init();
    for (int i = 0; i < num_episode; i++) {
      A_1 = choose_action(A_1);
#ifdef DEBUG
      cout << "Episode #" << i << ":" << endl;
#endif 
      int frame_num = 0;

      // choose action and update Q matrix every step
      while (frame_num < frame_num_target) {
        A_2 = choose_action(A_1);
        update();
        render(frame_num);
        frame_num++;
      }


      // figure out if every node has successfully finished their transmissions
      bool is_complete = true;
      for (auto node : nodes) {
        node.is_success ? ++success_node : is_complete = false;
      }
      data.success_data.push_back(success_data);
      data.success_node.push_back(success_node);
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
        node.reset();
      }
    }
    plot();
  }
  void reset() {
    *this = SlottedAlohaRL_TD();
  }

 

private:
  typedef std::array<int, NumNode> State;
  typedef std::array<int, NumNode> Action;
  typedef std::array<Node_TD, NumNode> NodeArr;
  void init() {
    for (auto& node : nodes) {
      A_1[node.node_num] = get_rand_int(0, NumSlot - 1);
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
      if (epsilon >= get_rand_real(0, 1)) {
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
    data.success_frame.push_back(success_frame);
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

  void plot() {
    plt::suptitle("MC vs TD in Slotted ALOHA");
    plt::subplot(1, 3, 1);
    plt::title("Success Frame");
    plt::named_plot("TD", data.steps, data.success_frame);
    plt::xlabel("# Steps");
    plt::subplot(1, 3, 2);
    plt::title("Success Data");
    plt::named_plot("TD", data.episodes, data.success_data);
    plt::xlabel("# Episodes");
    plt::subplot(1, 3, 3);
    plt::title("Success Node");
    plt::named_plot("TD", data.episodes, data.success_node);
    plt::xlabel("# Episodes");
    plt::legend();
  }

  NodeArr nodes;
  Action A_1, A_2;
  Plot_Data data;

  int success_frame = 0;
  int success_data = 0;
  int success_node = 0;

  double gamma = 0.8;
  double alpha = 0.1;
  double epsilon = 0.005;


  unsigned int total_success = 0;
  unsigned int total_failure = 0;
  unsigned int step_num = 0;
};