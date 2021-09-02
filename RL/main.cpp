//#define DEBUG


#include <iostream>
#include <string>

#include "TD.h"
#include "RL.h"
#include "nstep.h"
#include "sarsa_ramda.h"

int main() {
    SlottedAlohaRL_MC MC;
    SlottedAlohaRL_TD TD;
    SlottedAlohaRL_n n(3);
    SlottedAlohaRL_Ramda r(0.7);

    MC.run();
    TD.run();
    n.run();
    r.run();

    plt::suptitle("Comparison of RL Algorithms via Slotted ALOHA");
    plt::show();
    return 0;
}