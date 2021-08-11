//#define DEBUG


#include <iostream>
#include <string>

#include "TD.h"
#include "RL.h"


int main() {
  SlottedAlohaRL_MC MC;
  SlottedAlohaRL_TD TD;
  MC.run();
  TD.run();
  plt::show();
  return 0;
}