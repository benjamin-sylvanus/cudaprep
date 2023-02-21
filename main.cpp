#include <iostream>
#include "src/simreader.h"
#include "src/simulation.h"
#include "src/particle.h"
#include "vector"

using std::cout, std::endl;
int main()
{
    std::string path = "/Users/benjaminsylvanus/Documents/work/cudaprep/data";
    simreader reader(&path);
    simulation sim(reader);
    sim.setStep_num(1000);
    sim.setParticle_num(10000);
    particle par;
    par.display();
    par.setFlag();
    par.setState();
    double position[3] = {1,2,3};
    par.setPos(position);
    par.setState(false);
    par.display();
    return 0;
}
