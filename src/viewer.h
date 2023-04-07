//
// Created by Benjamin Sylvanus on 3/8/23.
//
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "simulation.h"
#include <vector>
#include <map>


#ifndef CUDAPREP_VIEW_H
#define CUDAPREP_VIEW_H


class viewer {
private:
    std::string help="-";
    std::string command="-";
    std::string mods="-";
    std::string str="-";
    int useColor = 0;

public:
    viewer();
    viewer(int c);
    void display(int option);
    void show(simulation sim);
    void showHelp();
    void showCommands();
    void showargs(std::map<std::string,std::string> args);
    void AlertNoParameter(std::string target);
    void welcome();
};


#endif //CUDAPREP_VIEW_H
