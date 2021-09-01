//#define DEBUG


#include <iostream>
#include <string>

#include "TD.h"
#include "RL.h"
#include "nstep.h"

int main() {
    SlottedAlohaRL_MC MC;
    SlottedAlohaRL_TD TD;
    SlottedAlohaRL_n n(2, "2-step TD");
    MC.run();
    TD.run();
    n.run();

    plt::suptitle("");
    plt::show();
    return 0;
}