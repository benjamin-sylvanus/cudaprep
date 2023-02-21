#include <iostream>
#include "src/simreader.h"
#include "src/simulation.h"
#include "vector"

using std::cout, std::endl;
int main()
{
    std::string path = "/Users/benjaminsylvanus/Documents/work/cudaprep/data";
    simreader reader(&path);
    simulation sim(reader);
    return 0;
}
