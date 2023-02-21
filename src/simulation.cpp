#pragma clang diagnostic push
#pragma ide diagnostic ignored "readability-convert-member-functions-to-static"
//
// Created by Benjamin Sylvanus on 2/20/23.
//

#include "simulation.h"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCUnusedGlobalDeclarationInspection"


simulation::simulation(simreader reader)
{
    this->swc = reader.read<double>("/swc.bin");
    this->parameterdata = reader.read<double>("/constants.bin");
    this->index = reader.read<uint64_t>("/index.bin");
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

double simulation::getParticle_num() {
    return 0;
}

double simulation::getVsize() {
    return 0;
}

double simulation::getScale() {
    return 0;
}

double simulation::getStep_num() {
    return 0;
}

double simulation::getInit_in() {
    return 0;
}

double simulation::getStep_size() {
    return 0;
}

double simulation::getPerm_prob() {
    return 0;
}

double simulation::getD0()
{
    return 0;
}
double simulation::getD()
{
    return 0;
}
double simulation::getTstep()
{
    return 0;
}
std::vector<double> simulation::getSwc()
{
    return{};
}
std::vector<unsigned long long int> simulation::getLut()
{
    return {};
}
std::vector<unsigned long long int> simulation::getIndex()
{
    return {};
}
std::vector<unsigned long long int> simulation::getPairs()
{
    return {};
}
std::vector<unsigned long long int> simulation::getbounds()
{
    return {};
}
std::vector<double> simulation::getParameterdata()
{
    return {};
}
std::vector<std::vector<uint64_t>> simulation::getArraydims()
{
    return {};
}

void simulation::setScale() {

}

void simulation::setVsize() {

}

void simulation::setParticle_num() {

}

void simulation::setStep_num() {

}

void simulation::setInit_in() {

}

void simulation::setStep_size() {

}

void simulation::setPerm_prob() {

}

void simulation::setD0() {

}

void simulation::setD() {

}

void simulation::setTstep() {

}

void simulation::setSwc() {

}

void simulation::setLut() {

}

void simulation::setIndex() {

}

void simulation::setPairs() {

}

void simulation::setbounds() {

}

void simulation::setParameterdata() {

}

void simulation::setArraydims() {

}


#pragma clang diagnostic pop
#pragma clang diagnostic pop