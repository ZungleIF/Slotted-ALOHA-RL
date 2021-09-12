//#define DEBUG



#include "RL.h"
#include "TD.h"

int main() {
    SlottedAlohaRL_MC MC_1(0.1);
    SlottedAlohaRL_MC MC_2(0.5);

    SlottedAlohaRL_TD TD(0.1);

    MC_1.run();
    MC_2.run();
    TD.run();

    plt::suptitle("Comparison of RL Algorithms via Slotted ALOHA");
    plt::show();
    return 0;
}