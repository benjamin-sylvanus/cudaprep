//
// Created by Benjamin Sylvanus on 2/20/23.
//

#include <iostream>
#include <fstream>
#include "simreader.h"
#include "vector"
#include "datatemplate.h"


simreader::simreader() = default;

simreader::simreader(std::string *path)
{
    this->setpath(path);
}

void simreader::setpath(std::string *pString)
{
    this->genpath = pString;
}


std::string simreader::getpath()
{
    return *this->genpath;
}

std::vector<std::vector<uint64_t>> simreader::readdims()
{
    std::string path =  this->getpath().append("/dims.txt");
    std::cout<<"Path: "<<path<<std::endl;

    std::fstream newfile;

    newfile.open(path,std::ios::in); //open a file to perform read operation using file object
    std::vector<std::vector<uint64_t>> v;
    if (newfile.is_open())
    {   //checking whether the file is open
        std::string tp;
        while(getline(newfile, tp))
        { //read data from file object and put it into string.
//            std::cout << tp << "\n"; //print the data of the string
            std::vector<uint64_t> temp;
            size_t pos = 0;
            std::string token;
            std::string delimiter = "\t";
            while ((pos = tp.find(delimiter)) != std::string::npos)
            {
                token = tp.substr(0, pos);
                std::cout << token << std::endl;
                temp.push_back(std::stoi(token));
                tp.erase(0, pos + delimiter.length());
            }
            v.push_back(temp);
        }

        newfile.close(); //close the file object.
    }
    std::cout<< "Gen path: " <<this->getpath()<<std::endl;
    return v;
}

std::vector<double> simreader::readconstants()
{
    std::string path =  this->getpath().append("/constants.bin");
    return std::vector<double>();


}











