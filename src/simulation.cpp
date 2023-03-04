#pragma clang diagnostic push
#pragma ide diagnostic ignored "readability-convert-member-functions-to-static"
//
// Created by Benjamin Sylvanus on 2/20/23.
//

#include "simulation.h"
#include "iostream"
#include<random>
#include<cmath>
#include<chrono>
#include <iostream>

#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCUnusedGlobalDeclarationInspection"


simulation::simulation(simreader reader) {
    this->swc = reader.read<double>("/swc.bin");
    this->parameterdata = reader.read<double>("/constants.bin");
    this->index = reader.read<std::uint64_t>("/index.bin");
    this->lut = reader.read<uint64_t>("/lut.bin");
    this->pairs = reader.read<uint64_t>("/pairs.bin");
    this->bounds = reader.read<uint64_t>("/bounds.bin");
    this->arraydims = reader.readdims();
    this->particle_num = this->parameterdata[0];
    this->step_num = this->parameterdata[1];
    this->step_size = this->parameterdata[2];
    this->perm_prob = this->parameterdata[3];
    this->init_in = this->parameterdata[4];
    this->D0 = this->parameterdata[5];
    this->d = this->parameterdata[6];
    this->scale = this->parameterdata[7];
    this->tstep = this->parameterdata[8];
    this->vsize = this->parameterdata[9];

}

double simulation::getParticle_num() const { return this->particle_num; }

double simulation::getVsize() const { return this->vsize; }

double simulation::getScale() const { return this->scale; }

double simulation::getStep_num() const { return this->step_num; }

double simulation::getInit_in() const { return this->init_in; }

double *simulation::getStep_size() { return &this->step_size; }

double simulation::getPerm_prob() const { return this->perm_prob; }

double simulation::getD0() const { return this->D0; }

double simulation::getD() const { return this->d; }

double simulation::getTstep() const { return this->tstep; }

std::vector<double> simulation::getSwc() { return this->swc; }

std::vector <std::uint64_t> simulation::getLut() { return this->lut; }

std::vector <std::uint64_t> simulation::getIndex() { return this->index; }

std::vector <std::uint64_t> simulation::getPairs() { return this->pairs; }

std::vector <std::uint64_t> simulation::getbounds() { return this->bounds; }

std::vector<double> simulation::getParameterdata() { return this->parameterdata; }

std::vector <std::vector<uint64_t>> simulation::getArraydims() { return this->arraydims; }

void simulation::setScale(double nscale) {
    printf("Scale: %f -> %f\n", this->scale, nscale);
    this->scale = nscale;
}

void simulation::setVsize(double vs) {
    printf("Vsize: %f -> %f\n", this->vsize, vs);
    this->vsize = vs;

}

void simulation::setParticle_num(double pnum) {
    printf("Particle Num: %.0f -> %.0f\n", this->particle_num, pnum);
    this->particle_num = pnum;
}

void simulation::setStep_num(double snum) {
    printf("Step Num: %.0f -> %.0f\n", this->step_num, snum);
    this->step_num = snum;
}

void simulation::setInit_in(double initin) {
    printf("Initialize Random Walker Inside: %f -> %f\n", this->init_in, initin);
    this->init_in = initin;
}

void simulation::setStep_size(double stepsize) {
    printf("Step Size: %f -> %f\n", this->step_size, stepsize);
    this->step_size = stepsize;
}

void simulation::setPerm_prob(double permprob) {
    printf("Permeation Probability: %f -> %f\n", this->perm_prob, permprob);
    this->perm_prob = permprob;
}

void simulation::setD0(double d0) {
    printf("D0: %f -> %f\n", this->D0, d0);
    this->D0 = d0;
}

void simulation::setD(double d) {
    printf("d: %f -> %f\n", this->d, d);
    this->d = d;
}

void simulation::setTstep(double timestep) {
    printf("Time Step (ms): %f -> %f\n", this->tstep, timestep);
    this->tstep = timestep;
}

void simulation::setSwc() {
    printf("Unsupported\n");
}

void simulation::setLut() {
    printf("Unsupported\n");
}

void simulation::setIndex() {
    printf("Unsupported\n");
}

void simulation::setPairs() {
    printf("Unsupported\n");
}

void simulation::setbounds() {
    printf("Unsupported\n");
}

void simulation::setParameterdata() {
    printf("Unsupported\n");
}

void simulation::setArraydims() {
    printf("Unsupported\n");
}

double *simulation::nextPosition(double *nexts) const {
    auto elapsed = clock();
    clock_t time_req;
    time_req = clock();
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    // generate N random numbers
    int N = int(particle_num);
    for (int i = 0; i < N; i++) {
        double theta = 2 * M_PI * uniform01(generator);
        double phi = acos(1 - 2 * uniform01(generator));
        nexts[3 * i + 0] = sin(phi) * cos(theta);
        nexts[3 * i + 1] = sin(phi) * sin(theta);
        nexts[3 * i + 2] = cos(phi);
    }
//    time_req = clock()-time_req;
//    std::cout << std::endl << (float)time_req/CLOCKS_PER_SEC << " seconds" << std::endl;

    return nexts;
}

// todo: implement a display method

#pragma clang diagnostic pop
#pragma clang diagnostic pop
