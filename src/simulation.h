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
    std::vector<std::uint64_t> lut; // lut of simulation
    std::vector<std::uint64_t> index; // index array
    std::vector<std::uint64_t> pairs; // pairs of swc
    std::vector<std::uint64_t> bounds; // bounds of geometry
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
    std::string resultPath;
    int SaveAll = 1;

    // Signal Parameters
    // TODO Implement Reads of Signal Data
    std::vector<std::vector<uint64_t>> signalDims;


    // t2 relaxation...
    std::vector<double> T2; // TODO: Figure out what this is
    std::vector<double> bvec; // Gradient Direction
    std::vector<double> bval; // Gradient Strength



public:
    // Constructors
    simulation(simreader reader, std::string outpath);
    simulation();


    // Mutators
    void setScale(double update);
    void setVsize(double vsize);
    void setParticle_num(double pnum);
    void setStep_num(double snum);
    void setInit_in(double initin);
    void setStep_size(double stepsize);
    void setPerm_prob(double permprob);
    void setD0(double d0);
    void setD(double d);
    void setTstep(double tstep);
    void setSwc();
    void setLut();
    void setIndex();
    void setPairs();
    void setbounds();
    void setParameterdata();
    void setArraydims();
    void setResultPath(std::string path);
    void setSaveAll(int value);

    // Accessors
    double getScale() const;
    double getVsize() const;
    double getParticle_num() const;
    double getStep_num() const;
    double getInit_in() const;
    double getStep_size();
    double getPerm_prob() const;
    double getD0() const;
    double getD() const;
    double getTstep() const;
    std::string getResultPath() const;
    std::vector<double> getSwc();
    std::vector<std::uint64_t> getLut();
    std::vector<std::uint64_t> getIndex();
    std::vector<std::uint64_t> getPairs();
    std::vector<std::uint64_t> getbounds();
    std::vector<double> getParameterdata();
    std::vector<std::vector<uint64_t>> getArraydims();
    int getSaveAll() const;



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
