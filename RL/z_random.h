#pragma once
#include <random>

static std::random_device rd;
static std::default_random_engine e(rd());
/*

*/
inline int get_rand_int(int min, int max) {
  //static std::random_device rd;
//std::default_random_engine e(rd());
  std::uniform_int_distribution<> dist(min, max);
  return dist(e);
}
inline double get_rand_real(double min, double max) {
  //static std::random_device rd;
  //std::default_random_engine e(rd());
  std::uniform_real_distribution<> dist(min, max);
  return dist(e);
}

inline void change_seed() {
  e = std::default_random_engine(rd());
}