//
// Created by Benjamin Sylvanus on 2/21/23.
//

#ifndef CUDAPREP_PARTICLE_H
#define CUDAPREP_PARTICLE_H


#include "simulation.h"

class particle {
private:



    bool flag = false;
    bool state = false;
    double pos[3] = {0,0,0};
    double * stepsize = nullptr;
public:
    particle();
    particle(bool flag, bool state, const double *pInt,double * pStep);
    explicit particle(double *pStep, bool flag = false, bool state=true);


    double * getPos();
    bool * getFlag();
    bool * getState();
    double * getStep();

    void setState(bool nstate);
    void setFlag(bool nflag);
    void setPos(const double *npos);
    void setStep(double *step);
    void setState();
    void setFlag();
    void display();

};


#endif //CUDAPREP_PARTICLE_H
