//
// Created by Benjamin Sylvanus on 2/20/23.
//

#ifndef CUDAPREP_SIMULATION_H
#define CUDAPREP_SIMULATION_H


#include "simreader.h"

class simulation {
    // Props
private:
    simreader reader;
    double swc=0; // swc array;
    int lut; // lut of simulation
    int index; // index array
    int pairs; // pairs of swc
    int bounds; // bounds of geometry
    int particle_num; // number of particles
    int step_num; // number of steps to simulate
    bool init_in=true; // initialize particles inside?
    /**These variable are dependent on eachother */
    int step_size; // step size of sim (voxel units)
    int perm_prob; // permeation probability
    int D0 = 2; // intrinsic diffusivity
    int d; // 10% min of Radii (um)
    int tstep; // time step of simulation

public:
    // Constructors
    simulation();
    explicit simulation(simreader reader);

    // Mutators

    // Accessors

    // Facilitators

    // Enquiry

    // Destructor
    ~simulation() = delete;


};


#endif //CUDAPREP_SIMULATION_H
