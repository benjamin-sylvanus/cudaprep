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

enum class Controls : int {
Invalid,
Commands,
Help,
Args,
Show,
StepSize,
PermeationProbability,
Init_in,
IntrinsicDiffusivity,
Distance,
TimeStep,
Scale,
VoxelSize,
NStep,
NPar,
Quit,
InPath,
OutPath,
SaveAll
};

class controller {
private:

    simulation sim;
    viewer view;
    std::string pathIn;
    std::string pathOut;
    std::vector<std::string> commands;
    std::vector<std::string> args;
    std::map<std::string, Controls> map;
    std::map<std::string, std::string> targets;


public:
    controller();
    // explicit controller(simulation& sim);
    void Setup(std::string InPath,std::string OutPath, int c);
    // constructor with main functions argc and argv
    void Setup(int argc, char **argv, int c);
    void start(std::string buf, bool   b);
    void start();
    void handleinput(std::string input,bool * b);
    void handlecommand(std::vector<std::string>  command, bool * b);
    simulation getSim();
    void setSim(std::string path);
    void setPathOut(std::string path);
};


#endif //CUDAPREP_CONTROLLER_H
