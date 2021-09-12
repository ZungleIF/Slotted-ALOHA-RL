//#define DEBUG


#include <iostream>
#include <string>

#include "TD.h"
#include "RL.h"


int main() {
    SlottedAlohaRL_MC MC_1(0.05);
    SlottedAlohaRL_MC MC_2(0.5);
    SlottedAlohaRL_TD TD_1(0.05);
    SlottedAlohaRL_TD TD_2(0.5);

    MC_1.run();
    TD_1.run();
    MC_2.run();
    TD_2.run();

    plt::suptitle("Comparison of RL Algorithms via Slotted ALOHA");
    plt::show();
    return 0;
}