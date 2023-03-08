//
// Created by Benjamin Sylvanus on 3/8/23.
//
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "simulation.h"


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


    void welcome();
};


#endif //CUDAPREP_VIEW_H
