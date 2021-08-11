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
    std::string title_str;
    title_str = "MC vs TD in Slotted ALOHA averaged w/ " + std::to_string(iterations_target) + " cases";
    plt::suptitle(title_str);
    plt::show();
    return 0;
}