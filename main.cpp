#include <iostream>
#include "src/simreader.h"
#include "vector"

using std::cout, std::endl;
int main()
{
    std::string path = "/Users/benjaminsylvanus/Documents/work/cudaprep/data";
    simreader reader(&path);
    cout << reader.getpath() << endl;
    std::vector<double> swcdata = reader.read<double>("/swc.bin");
    std::vector<uint64_t> indexdata = reader.read<uint64_t>("/index.bin");
    std::vector<uint64_t> lutdata = reader.read<uint64_t>("/lut.bin");
    std::vector<uint64_t> pairsdata = reader.read<uint64_t>("/pairs.bin");
    std::vector<uint64_t> boundsdata = reader.read<uint64_t>("/bounds.bin");
    std::vector<std::vector<uint64_t>> v = reader.readdims();
    return 0;
}
