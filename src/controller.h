//
// Created by Benjamin Sylvanus on 3/8/23.
//

#ifndef CUDAPREP_CONTROLLER_H
#define CUDAPREP_CONTROLLER_H


#include "simulation.h"
#include <cstring>
#include <iostream>
#include <vector>
#include "viewer.h"
#include <map>

enum class Controls : int { Invalid,Commands,
Help,
Args,
Show,
StepSize,
PermeationProbability,
IntrinsicDiffusivity,
Distance,
TimeStep,
Scale,
VoxelSize,
NStep,
NPar
};

class controller {
private:

    simulation sim;
    viewer view;
    std::vector<std::string> commands;
    std::vector<std::string> args;
    std::map<std::string, Controls> map;
    std::map<std::string, std::string> targets;


public:
    // explicit controller(simulation& sim);
    explicit controller(std::string path);
    void start();
    void handleinput(std::string input);
    void handlecommand(std::vector<std::string>  command);
    simulation getSim();

};


#endif //CUDAPREP_CONTROLLER_H
