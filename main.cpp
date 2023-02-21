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
    double * stepsize =  sim.getStep_size();
    particle par(stepsize);
    par.display();
    par.setFlag();
    par.setState();
    double position[3] = {1,2,3};
    par.setPos(position);
    par.setState(false);
    par.display();

    sim.setParticle_num(1000000);
    double poses[int(3*sim.getParticle_num())];
    double * nextPositions = sim.nextPosition(poses);

    for (int i = 0; i < 3 * int(sim.getParticle_num()); i+=3)
    {
//        printf("particle: %.3f %.3f %.3f\n",poses[i],poses[i+1],poses[i+2]);
    }

    double nextvectors[int(3*sim.getParticle_num())];
    double *coords;
    auto elapsed = clock();
    clock_t time_req;
    time_req = clock();
    for (int i = 0; i < 10; i++)
    {
        time_req = clock();
        coords = sim.nextPosition(nextvectors);
        for (int j = 0; j<int(3*sim.getParticle_num()); j+=3)
        {
//            printf("Sum:\n%.3f + %.3f | %.3f\n%.3f + %.3f | %.3f\n%.3f + %.3f | %.3f\n\n",nextPositions[j+0],coords[j+0],nextPositions[j+0]+coords[j+0],nextPositions[j+1],coords[j+1],nextPositions[j+1]+coords[j+1],nextPositions[j+2],coords[j+2],nextPositions[j+2]+coords[j+2]);
            nextPositions[j+0] = nextPositions[j+0] + coords[j+0];
            nextPositions[j+1] = nextPositions[j+1] + coords[j+1];
            nextPositions[j+2] = nextPositions[j+2] + coords[j+2];
//            printf("Position After:\t[%.3f, %.3f, %.3f]\n",nextPositions[j+0],nextPositions[j+1],nextPositions[j+2]);
        }

        time_req = clock()-time_req;
        std::cout << std::endl << (float)time_req/CLOCKS_PER_SEC << " seconds" << std::endl;
    }
    return 0;
}
