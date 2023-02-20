//
// Created by Benjamin Sylvanus on 2/20/23.
//
#include "vector"
#include "iostream"
#include "fstream"

#ifndef CUDAPREP_DATATEMPLATE_H
#define CUDAPREP_DATATEMPLATE_H

template<class T>
class datatemplate{
public:
    explicit datatemplate<T>(std::string path);
    void load_data();
    std::vector<T> data;

private:
    std::string path;
};

template<class T>
void datatemplate<T>::load_data()
{
    std::ifstream fin(this->path, std::ios::binary);
    if(!fin)
    {
        std::cout << " Error, Couldn't find the file" << "\n";
        return;
    }

    fin.seekg(0, std::ios::end);

    size_t num_elements = fin.tellg() / sizeof(T);

    fin.seekg(0, std::ios::beg);

    std::vector<T> vector(num_elements);

    fin.read(reinterpret_cast<char*>(&vector[0]), num_elements * sizeof(T)); // NOLINT(cppcoreguidelines-narrowing-conversions)
    this->data = vector;
}

template<class T>
datatemplate<T>::datatemplate(std::string path)
{
    this->path=path;
    this->load_data();
}



#endif //CUDAPREP_DATATEMPLATE_H
