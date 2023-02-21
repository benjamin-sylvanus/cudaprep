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
    std::vector<double> swc; // swc array;
    std::vector<unsigned long long int> lut; // lut of simulation
    std::vector<unsigned long long int> index; // index array
    std::vector<unsigned long long int> pairs; // pairs of swc
    std::vector<unsigned long long int> bounds; // bounds of geometry
    double particle_num; // number of particles
    double step_num; // number of steps to simulate
    double init_in; // initialize particles inside?
    /**These variable are dependent on eachother */
    double step_size; // step size of sim (voxel units)
    double perm_prob; // permeation probability
    double D0; // intrinsic diffusivity
    double d; // 10% min of Radii (um)
    double tstep; // time step of simulation
    std::vector<double> parameterdata;
    std::vector<std::vector<uint64_t>> arraydims;
    double scale;
    double vsize;

public:
    // Constructors
    explicit simulation(simreader reader);

    // Mutators
    void setScale();
    void setVsize();
    void setParticle_num();
    void setStep_num();
    void setInit_in();
    void setStep_size();
    void setPerm_prob();
    void setD0();
    void setD();
    void setTstep();
    void setSwc();
    void setLut();
    void setIndex();
    void setPairs();
    void setbounds();
    void setParameterdata();
    void setArraydims();

    // Accessors
    double getScale();
    double getVsize();
    double getParticle_num();
    double getStep_num();
    double getInit_in();
    double getStep_size();
    double getPerm_prob();
    double getD0();
    double getD();
    double getTstep();
    std::vector<double> getSwc();
    std::vector<unsigned long long int> getLut();
    std::vector<unsigned long long int> getIndex();
    std::vector<unsigned long long int> getPairs();
    std::vector<unsigned long long int> getbounds();
    std::vector<double> getParameterdata();
    std::vector<std::vector<uint64_t>> getArraydims();


    /**Facilitators:
     * Do things
     */


    /**Enquiry:
     * Is Simulation and Is Simulation Valid;
     */

    // Destructor
    ~simulation() = default;

};


#endif //CUDAPREP_SIMULATION_H
