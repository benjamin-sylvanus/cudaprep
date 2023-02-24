//
// Created by Benjamin Sylvanus on 2/21/23.
//

#include "particle.h"
#include "iostream"
#include<random>
#include<cmath>
#include<chrono>
#include <iostream>

particle::particle(bool flag, bool state, const double *pos, double * stepsize) {
    this->flag = flag;
    this->state = state;
    this->pos[0] = pos[0];
    this->pos[1] = pos[1];
    this->pos[2] = pos[2];
    this->stepsize = stepsize;
}

double * particle::getPos() {return pos;}

bool *particle::getFlag() {return &flag;}

bool *particle::getState() {return &state;}
double * particle::getStep() {return stepsize;}

void particle::setPos(const double * npos) {for (int i = 0; i<3; i++) pos[i]=npos[i];}

void particle::setFlag(bool nflag) {this->flag=nflag;}

void particle::setState(bool nstate) {this->state = nstate;}

void particle::setStep(double * step) {this->stepsize=step;}

void particle::setState() {this->state = not(this->state);}

void particle::setFlag() {this->flag = not(this->flag);}

void particle::display()
{
    printf("Particle:\npos: [%.3f, %.3f, %.3f]\nstate: %d\nflag: %d\nstepsize: %f\n",pos[0],pos[1],pos[2],state,flag,*stepsize);
}

particle::particle(double *pStep, bool flag, bool state)
{
    this->stepsize=pStep;
    this->flag=flag;
    this->state=state;
}

particle::particle() = default;



