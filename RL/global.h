#pragma once
#include <numeric>
#include <vector>

constexpr int NumNode = 10;
constexpr int NumSlot = 10;

constexpr double positive_feedback = 1;
constexpr double negative_feedback = -1;
constexpr double neutral_feedback = -0.5;
constexpr double episode_success = 10;
constexpr double episode_failure = -10;

constexpr int frame_num_target = 10;
constexpr int num_episode = 100;

struct MC_Data {
  std::vector<size_t> t_opt;
  std::vector<double> con_vec;
};

struct Plot_Data {
  Plot_Data() : 
                episodes(num_episode), steps(frame_num_target * num_episode) 
  {
    std::iota(episodes.begin(), episodes.end(), 0);
    std::iota(steps.begin(), steps.end(), 0);
  }
  std::vector<int> success_frame; // sucesssful frames that every node was successful per step
  std::vector<int> success_data;  // successful data transmitted per episode
  std::vector<int> success_node;  // successful nodes per episode
  std::vector<int> episodes;      // x axis for plotting
  std::vector<int> steps;         // x axis for plotting
};