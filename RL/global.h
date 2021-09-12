#pragma once
#include <numeric>
#include <vector>

constexpr int NumNode = 10;
constexpr int NumSlot = 10;

// rewards
constexpr double positive_feedback = 1.0;
constexpr double negative_feedback = -1.0;
constexpr double episode_success = 10.0;
constexpr double episode_failure = -10.0;

constexpr int frame_num_target = 10;
constexpr int episode_num_target = 150;

// iterations w/ changing random seeds
constexpr int iterations_target = 100;
constexpr int data_target = 10;

struct Plot_Data {
    Plot_Data() :   success_frame(frame_num_target * episode_num_target, 0), success_data(episode_num_target, 0), success_node(episode_num_target, 0),
                    episodes(episode_num_target), steps(frame_num_target * episode_num_target)
    {
        std::iota(episodes.begin(), episodes.end(), 0);
        std::iota(steps.begin(), steps.end(), 0);
    }
    std::vector<double> success_frame;  // sucesssful frames that every node was successful per step
    std::vector<double> success_data;   // successful data transmitted per episode
    std::vector<double> success_node;   // successful nodes per episode
    std::vector<int> episodes;          // x axis for plotting
    std::vector<int> steps;             // x axis for plotting
};