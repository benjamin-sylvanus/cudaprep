//
// Created by Benjamin Sylvanus on 2/20/23.
//

#ifndef CUDAPREP_SIMREADER_H
#define CUDAPREP_SIMREADER_H

#include <string>
#include "datatemplate.h"

class simreader {

private:

    std::string *genpath;
public:
    explicit simreader(std::string *path);

    void setpath(std::string *pString);

    simreader();

    std::string getpath();

    void read(std::string str);

    template<class T>
    std::vector <T> read(std::string str);


    std::vector <std::vector<uint64_t>> readdims();

    std::vector<double> readconstants();

};

template<class T>
std::vector <T> simreader::read(std::string str) // NOLINT(performance-unnecessary-value-param)
{
    std::string FullPath = this->getpath().append(str);

    datatemplate<T> Data(FullPath);

    return Data.data;
}

#endif //CUDAPREP_SIMREADER_H
