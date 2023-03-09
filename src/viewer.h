//
// Created by Benjamin Sylvanus on 3/8/23.
//
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "simulation.h"
#include <vector>


#ifndef CUDAPREP_VIEW_H
#define CUDAPREP_VIEW_H


class viewer {
private:
    std::string help;
    std::string command;
    std::string mods;

public:

    viewer();
    void display(int option);
    void show();
    void showargs(std::vector<std::string> args);
    void AlertNoParameter(std::string target);
    void welcome();
};


#endif //CUDAPREP_VIEW_H
