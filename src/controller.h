//
// Created by Benjamin Sylvanus on 3/8/23.
//

#ifndef CUDAPREP_CONTROLLER_H
#define CUDAPREP_CONTROLLER_H


#include "simulation.h"
#include "viewer.h"

class controller {
private:
    simulation sim;
    viewer view;

public:
    explicit controller(simulation& sim);
    void start();



};


#endif //CUDAPREP_CONTROLLER_H
