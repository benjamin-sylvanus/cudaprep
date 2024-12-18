#include "funcs.h"
#include "vector"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cstring>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include <sys/stat.h>
#include "overloads.h" 

__device__ bool swc2v(double3 nextpos, double4 child, double4 parent, double dist) {
    bool pos;
    double cr2;
    double pr2;
    if (dist == 0) {
      cr2 = pow(child.w, 2);
      pr2 = pow(parent.w, 2);
      pos = (pow(nextpos.x - child.x, 2) + pow(nextpos.y - child.y, 2) + pow(nextpos.z - child.z, 2)) < cr2 ||
                   (pow(nextpos.x - parent.x, 2) + pow(nextpos.y - parent.y, 2) + pow(nextpos.z - parent.z, 2)) < pr2;
        // dist = 0.000000000000000001;
    }
    else
    {
    double t = ((nextpos.x - child.x) * (parent.x - child.x) + (nextpos.y - child.y) * (parent.y - child.y) +
                (nextpos.z - child.z) * (parent.z - child.z)) / dist;
    // printf("t: %f\n", t);
    double x = child.x + (parent.x - child.x) * t;
    double y = child.y + (parent.y - child.y) * t;
    double z = child.z + (parent.z - child.z) * t;
    cr2 = pow(child.w, 2);
    pr2 = pow(parent.w, 2);
    double L = sqrt(pow(parent.x-child.x,2) + pow(parent.y-child.y,2) + pow(parent.z-child.z,2));
    double sin_theta = fabs(child.w - parent.w)/L;
    double cos_theta = sqrt(1-pow(sin_theta,2));
    bool list1 = (t > ( (child.w * sin_theta) / L)) && (t < (1 + (parent.w * sin_theta) / L));
    if (list1)
    {
      double dist2 = (pow(nextpos.x - x, 2) + pow(nextpos.y - y, 2) + pow(nextpos.z - z, 2));
      double m = (parent.w-child.w) * cos_theta/(L+(parent.w-child.w)*sin_theta);
      double rx = L * t;
      double ry = m * rx + (cos_theta - m * sin_theta) * child.w;
      pos = dist2 < pow(ry,2);
    }
    else
    {
      pos = (pow(nextpos.x - child.x, 2) + pow(nextpos.y - child.y, 2) + pow(nextpos.z - child.z, 2)) < cr2 ||
             (pow(nextpos.x - parent.x, 2) + pow(nextpos.y - parent.y, 2) + pow(nextpos.z - parent.z, 2)) < pr2;
    }
  }
    return pos;
}


__device__ __host__ int s2i(int3 i, int3 b) {
    return 0 + b.x * (b.y * i.z + i.y) + i.x;
}

__device__ double3 initPosition(int gid, double *dx2, int3 &Bounds, curandStatePhilox4_32_10_t *state,
                    double *SimulationParams, double4 *d4swc, int *nlut, int *NewIndex,
                    int3 &IndexSize, int size, int iter, int init_in, bool debug, double3 point) {

    double3 nextpos;
    int3 upper;
    int3 lower;
    int3 floorpos;
    int3 b_int3 = make_int3(Bounds.x, Bounds.y, Bounds.z);
    int3 i_int3 = make_int3(IndexSize.x, IndexSize.y, IndexSize.z);
    double3 b_d3 = make_double3((double) b_int3.x, (double) b_int3.y, (double) b_int3.z);
    curandStatePhilox4_32_10_t localstate = state[gid];
    double4 xr;
    int2 parstate;
    double3 A;
    parstate.y = 0;
    bool cont = true;

    /**
    * Case 1: particles inside.
    * Case 2: Particles outside.
    * Case 3: Particles at point.
    * Case 4: Particles everywhere based on concentration ratio

    Case 1 done
    Case 2:
    // init local state var
    xr = curand_uniform4_double(&localstate);

    // set particle initial position
    A = make_double3(xr.x * b_d3.x, xr.y * b_d3.y, xr.z * b_d3.z);

    Case 3:
    A point must be passed to this function -> point

    A = make double3(point.x,point.y point.z);
    */
    if (init_in == 1)
    {
    while (cont) {

        // init local state var
        xr = curand_uniform4_double(&localstate);

        // set particle initial position
        A = make_double3(xr.x * b_d3.x, xr.y * b_d3.y, xr.z * b_d3.z);

        nextpos = A;

        // floor of position -> check voxels
        floorpos.x = (int) A.x;
        floorpos.y = (int) A.y;
        floorpos.z = (int) A.z;

        // upper bounds of lookup table
        upper.x = floorpos.x < b_int3.x;
        upper.y = floorpos.y < b_int3.y;
        upper.z = floorpos.z < b_int3.z;

        // lower bounds of lookup table
        lower.x = floorpos.x > 0;
        lower.y = floorpos.y > 0;
        lower.z = floorpos.z > 0;

        // position inside the bounds of volume -> state of next position true : false
        parstate.x = (lower.x && lower.y && lower.z && upper.x && upper.y && upper.z) ? 1 : 0;
        double4 parent;
        double4 child;
        int2 vindex;
        int id_test = s2i(floorpos, b_int3);
        int test_lutvalue = nlut[id_test];
        double dist2;
        if (parstate.x) {
            for (int page = 0; page < i_int3.z; page++) {
                int3 c_new = make_int3(test_lutvalue, 0, page);
                int3 p_new = make_int3(test_lutvalue, 1, page);
                vindex.x = NewIndex[s2i(c_new, i_int3)] - 1;
                vindex.y = NewIndex[s2i(p_new, i_int3)] - 1;

                if ((vindex.x) != -1) {
                    child = make_double4(d4swc[vindex.x].x, d4swc[vindex.x].y, d4swc[vindex.x].z, d4swc[vindex.x].w);
                    parent = make_double4(d4swc[vindex.y].x, d4swc[vindex.y].y, d4swc[vindex.y].z, d4swc[vindex.y].w);

                    //distance squared between child parent
                    dist2 = pow(parent.x - child.x, 2) + pow(parent.y - child.y, 2) + pow(parent.z - child.z, 2);

                    // determine whether particle is inside this connection
                    bool inside = swc2v(nextpos, child, parent, dist2);

                    // if it is inside the connection we don't need to check the remaining.
                    if (inside) {

                        // update the particles state
                        parstate.y = 1;
                        // end for p loop
                        page = i_int3.z;
                        cont = false;
                    }
                }
            }
        }
    }
  }
  if (init_in == 2)
  {
    // init local state var
    xr = curand_uniform4_double(&localstate);

    // set particle initial position
    A = make_double3(xr.x * b_d3.x, xr.y * b_d3.y, xr.z * b_d3.z);
  }
  if (init_in == 3)
  {
    A = make_double3(point.x,point.y,point.z);
  }
    return A;
}

__device__ void
diffusionTensor(double3 * A, double3 * xnot, double vsize, double * dx2, double * dx4, double3 * d2, int i, int gid, int iter, int size) {

    d2->x = fabs((A->x - xnot->x) * vsize);
    d2->y = fabs((A->y - xnot->y) * vsize);
    d2->z = fabs((A->z - xnot->z) * vsize);

    // Diffusion tensor
    atomicAdd(&dx2[6 * i + 0], d2->x * d2->x);
    atomicAdd(&dx2[6 * i + 1], d2->x * d2->y);
    atomicAdd(&dx2[6 * i + 2], d2->x * d2->z);
    atomicAdd(&dx2[6 * i + 3], d2->y * d2->y);
    atomicAdd(&dx2[6 * i + 4], d2->y * d2->z);
    atomicAdd(&dx2[6 * i + 5], d2->z * d2->z);

    // Kurtosis Tensor
    atomicAdd(&dx4[15 * i + 0], d2->x * d2->x * d2->x * d2->x);
    atomicAdd(&dx4[15 * i + 1], d2->x * d2->x * d2->x * d2->y);
    atomicAdd(&dx4[15 * i + 2], d2->x * d2->x * d2->x * d2->z);
    atomicAdd(&dx4[15 * i + 3], d2->x * d2->x * d2->y * d2->y);
    atomicAdd(&dx4[15 * i + 4], d2->x * d2->x * d2->y * d2->z);
    atomicAdd(&dx4[15 * i + 5], d2->x * d2->x * d2->z * d2->z);
    atomicAdd(&dx4[15 * i + 6], d2->x * d2->y * d2->y * d2->y);
    atomicAdd(&dx4[15 * i + 7], d2->x * d2->y * d2->y * d2->z);
    atomicAdd(&dx4[15 * i + 8], d2->x * d2->y * d2->z * d2->z);
    atomicAdd(&dx4[15 * i + 9], d2->x * d2->z * d2->z * d2->z);
    atomicAdd(&dx4[15 * i + 10], d2->y * d2->y * d2->y * d2->y);
    atomicAdd(&dx4[15 * i + 11], d2->y * d2->y * d2->y * d2->z);
    atomicAdd(&dx4[15 * i + 12], d2->y * d2->y * d2->z * d2->z);
    atomicAdd(&dx4[15 * i + 13], d2->y * d2->z * d2->z * d2->z);
    atomicAdd(&dx4[15 * i + 14], d2->z * d2->z * d2->z * d2->z);
}


__device__ double3 setNextPos(double3 nextpos, double3 A, double4 xi, double step)
{
    nextpos.x = A.x + ((2.0 * xi.x - 1.0) * step);
    nextpos.y = A.y + ((2.0 * xi.y - 1.0) * step);
    nextpos.z = A.z + ((2.0 * xi.z - 1.0) * step);
    return nextpos;
}

// TODO : check new header
__host__ void writeResults(double * w_swc, double * hostSimP, double * hostdx2, double * mdx2, double * hostdx4,
                                    double * mdx4, double * u_t, double * u_Reflections, double * u_uref, double * u_Sig0,
                                    double * u_SigRe, double * u_AllData,int iter, int size, int nrow,int timepoints, int Nbvec, int sa_size, int SaveAll,
                                    std::string outpath, std::string binaryFilePath, std::string jsonFilePath)
{
  // Check outdir exists
  // isdir()?  mkdir : ...

  /**
  * Check if Path exists

  *
  */
    // Structure which would store the metadata
    struct stat sb;

    // Calls the function with path as argument
    // If the file/directory exists at the path returns 0
    // If block executes if path exists
    printf("Outpath: %s\n",outpath.c_str());
    if (stat(outpath.c_str(), &sb) == 0)
    printf("Valid Path\n");
    else
    if (mkdir(outpath.c_str(), 0777) == -1)
      std::cerr << "Error :  " << std::strerror(errno) << std::endl;
    else
      std::cout << "Directory created\n";

      // Write input paths to JSON file
    std::string inputPathsJson = outpath + "/input_paths.json";
    std::ofstream jsonFile(inputPathsJson);
    if (jsonFile.is_open()) {
        jsonFile << "{\n";
        jsonFile << "  \"binaryFilePath\": \"" << binaryFilePath << "\",\n";
        jsonFile << "  \"jsonFilePath\": \"" << jsonFilePath << "\"\n";
        jsonFile << "}\n";
        jsonFile.close();
    } else {
        std::cerr << "Error: Unable to create input paths JSON file" << std::endl;
    }

    double t[iter];
    double tstep = hostSimP[8];
    for (int i = 0; i < iter; i++) {
      t[i] = tstep * i;
      mdx2[6 * i + 0] = (hostdx2[6 * i + 0] / size) / (2.0 * t[i]);
      mdx2[6 * i + 1] = (hostdx2[6 * i + 1] / size) / (2.0 * t[i]);
      mdx2[6 * i + 2] = (hostdx2[6 * i + 2] / size) / (2.0 * t[i]);
      mdx2[6 * i + 3] = (hostdx2[6 * i + 3] / size) / (2.0 * t[i]);
      mdx2[6 * i + 4] = (hostdx2[6 * i + 4] / size) / (2.0 * t[i]);
      mdx2[6 * i + 5] = (hostdx2[6 * i + 5] / size) / (2.0 * t[i]);
      mdx4[15 * i + 0] = (hostdx4[15 * i + 0] / size) / (2.0 * t[i]);
      mdx4[15 * i + 1] = (hostdx4[15 * i + 1] / size) / (2.0 * t[i]);
      mdx4[15 * i + 2] = (hostdx4[15 * i + 2] / size) / (2.0 * t[i]);
      mdx4[15 * i + 3] = (hostdx4[15 * i + 3] / size) / (2.0 * t[i]);
      mdx4[15 * i + 4] = (hostdx4[15 * i + 4] / size) / (2.0 * t[i]);
      mdx4[15 * i + 5] = (hostdx4[15 * i + 5] / size) / (2.0 * t[i]);
      mdx4[15 * i + 6] = (hostdx4[15 * i + 6] / size) / (2.0 * t[i]);
      mdx4[15 * i + 7] = (hostdx4[15 * i + 7] / size) / (2.0 * t[i]);
      mdx4[15 * i + 8] = (hostdx4[15 * i + 8] / size) / (2.0 * t[i]);
      mdx4[15 * i + 9] = (hostdx4[15 * i + 9] / size) / (2.0 * t[i]);
      mdx4[15 * i + 10] = (hostdx4[15 * i + 10] / size) / (2.0 * t[i]);
      mdx4[15 * i + 11] = (hostdx4[15 * i + 11] / size) / (2.0 * t[i]);
      mdx4[15 * i + 12] = (hostdx4[15 * i + 12] / size) / (2.0 * t[i]);
      mdx4[15 * i + 13] = (hostdx4[15 * i + 13] / size) / (2.0 * t[i]);
      mdx4[15 * i + 14] = (hostdx4[15 * i + 14] / size) / (2.0 * t[i]);
    }
    FILE * outFile;

    // outFile = fopen("./results/filename.bin", "wb");
    // fwrite (hostAllData,sizeof(double),iter*size*3,outFile);
    // fclose (outFile);

    std::string dx2Path = outpath; dx2Path.append("/dx2.bin");
    std::string mdx2Path = outpath; mdx2Path.append("/mdx2.bin");
    std::string dx4Path = outpath; dx4Path.append("/dx4.bin");
    std::string mdx4Path = outpath; mdx4Path.append("/mdx4.bin");
    std::string swcpath = outpath; swcpath.append("/swc.bin");
    std::string cfgpath = outpath; cfgpath.append("/outcfg.bin");
    std::string tpath = outpath; tpath.append("/t.bin");
    std::string reflectionsPath = outpath; reflectionsPath.append("/reflections.bin");
    std::string urefPath = outpath; urefPath.append("/uref.bin");
    std::string sig0Path = outpath; sig0Path.append("/sig0.bin");
    std::string sigRePath = outpath; sigRePath.append("/sigRe.bin");
    std::string allDataPath = outpath; allDataPath.append("/allData.bin");


    outFile = fopen(swcpath.c_str(),"wb");
    fwrite (w_swc,sizeof(double),nrow*4,outFile);
    fclose(outFile);

    outFile = fopen(cfgpath.c_str(),"wb");
    fwrite (hostSimP,sizeof(double),10,outFile);
    fclose(outFile);

    outFile = fopen(dx2Path.c_str(),"wb");
    fwrite (hostdx2,sizeof(double),6 * iter,outFile);
    fclose(outFile);

    outFile = fopen(mdx2Path.c_str(),"wb");
    fwrite (mdx2,sizeof(double),6 * iter,outFile);
    fclose(outFile);

    outFile = fopen(dx4Path.c_str(),"wb");
    fwrite (hostdx4,sizeof(double),15 * iter,outFile);
    fclose(outFile);

    outFile = fopen(mdx4Path.c_str(),"wb");
    fwrite(mdx4,sizeof(double),15 * iter,outFile);
    fclose(outFile);

    outFile = fopen(tpath.c_str(),"wb");
    fwrite(t,sizeof(double),iter,outFile);
    fclose(outFile);

    outFile = fopen(reflectionsPath.c_str(), "wb");
    fwrite(u_Reflections, sizeof(double), iter * size * 3, outFile);
    fclose(outFile);

    outFile = fopen(urefPath.c_str(), "wb");
    fwrite(u_uref, sizeof(double), iter * size * 3, outFile);
    fclose(outFile);

    outFile = fopen(sig0Path.c_str(), "wb");
    fwrite(u_Sig0, sizeof(double), timepoints, outFile);
    fclose(outFile);

    outFile = fopen(sigRePath.c_str(), "wb");
    fwrite(u_SigRe, sizeof(double), Nbvec * iter, outFile);
    fclose(outFile);

    if (SaveAll) {
        outFile = fopen(allDataPath.c_str(), "wb");
        fwrite(u_AllData, sizeof(double),  3 * sa_size, outFile);
        fclose(outFile);
    }

    
}


// TODO : check new header
__host__ void writeResults_new(double * w_swc, double * hostSimP, double * hostdx2, double * mdx2, double * hostdx4,
                         double * mdx4, double * u_t, double * u_Reflections, double * u_uref, double * u_Sig0,
                         double * u_SigRe, double * u_AllData, int iter, int size, int nrow, int timepoints, 
                         int Nbvec, int sa_size, int SaveAll, std::string outpath, 
                         std::string binaryFilePath, std::string jsonFilePath)
{
    // Check/create output directory
    struct stat sb;
    if (stat(outpath.c_str(), &sb) != 0) {
        if (mkdir(outpath.c_str(), 0777) == -1) {
            std::cerr << "Error creating directory: " << std::strerror(errno) << std::endl;
            return;
        }
    }

    // Calculate time array
    double* t = new double[iter];
    double tstep = hostSimP[8];
    for (int i = 0; i < iter; i++) {
        t[i] = tstep * i;
        // Calculate mdx2 and mdx4 as before
        for (int j = 0; j < 6; j++) {
            mdx2[6 * i + j] = (hostdx2[6 * i + j] / size) / (2.0 * t[i]);
        }
        for (int j = 0; j < 15; j++) {
            mdx4[15 * i + j] = (hostdx4[15 * i + j] / size) / (2.0 * t[i]);
        }
    }

    // Write all data to single binary file
    std::string binPath = outpath + "/simulation_data.bin";
    FILE* outFile = fopen(binPath.c_str(), "wb");
    if (!outFile) {
        std::cerr << "Error opening binary file" << std::endl;
        delete[] t;
        return;
    }

    // Write all data sequentially
    fwrite(w_swc, sizeof(double), nrow * 4, outFile);           // order: 1
    fwrite(hostSimP, sizeof(double), 10, outFile);              // order: 2
    fwrite(hostdx2, sizeof(double), 6 * iter, outFile);         // order: 3
    fwrite(mdx2, sizeof(double), 6 * iter, outFile);            // order: 4
    fwrite(hostdx4, sizeof(double), 15 * iter, outFile);        // order: 5
    fwrite(mdx4, sizeof(double), 15 * iter, outFile);           // order: 6
    fwrite(t, sizeof(double), iter, outFile);                   // order: 7
    fwrite(u_Reflections, sizeof(double), iter * size * 3, outFile); // order: 8
    fwrite(u_uref, sizeof(double), iter * size * 3, outFile);   // order: 9
    fwrite(u_Sig0, sizeof(double), timepoints, outFile);        // order: 10
    fwrite(u_SigRe, sizeof(double), Nbvec * iter, outFile);     // order: 11
    if (SaveAll) {
        fwrite(u_AllData, sizeof(double), sa_size * 3, outFile); // order: 12
    }
    fclose(outFile);

    // Create JSON metadata file
    std::string jsonPath = outpath + "/output_metadata.json";
    std::ofstream jsonFile(jsonPath);
    if (jsonFile.is_open()) {
        jsonFile << "{\n";
        
        // Write metadata for each dataset
        jsonFile << "  \"swc_data\": {\n"
                << "    \"dataType\": \"double\",\n"
                << "    \"size\": [4, " << nrow << "],\n"
                << "    \"order\": 1\n"
                << "  },\n";
        
        jsonFile << "  \"simulation_params\": {\n"
                << "    \"dataType\": \"double\",\n"
                << "    \"size\": [10, 1],\n"
                << "    \"order\": 2\n"
                << "  },\n";
        
        jsonFile << "  \"dx2\": {\n"
                << "    \"dataType\": \"double\",\n"
                << "    \"size\": [" << iter << ", 6],\n"
                << "    \"order\": 3\n"
                << "  },\n";
        
        jsonFile << "  \"mdx2\": {\n"
                << "    \"dataType\": \"double\",\n"
                << "    \"size\": [" << iter << ", 6],\n"
                << "    \"order\": 4\n"
                << "  },\n";
        
        jsonFile << "  \"dx4\": {\n"
                << "    \"dataType\": \"double\",\n"
                << "    \"size\": [" << iter << ", 15],\n"
                << "    \"order\": 5\n"
                << "  },\n";
        
        jsonFile << "  \"mdx4\": {\n"
                << "    \"dataType\": \"double\",\n"
                << "    \"size\": [" << iter << ", 15],\n"
                << "    \"order\": 6\n"
                << "  },\n";
        
        jsonFile << "  \"time\": {\n"
                << "    \"dataType\": \"double\",\n"
                << "    \"size\": [" << iter << ", 1],\n"
                << "    \"order\": 7\n"
                << "  },\n";
        
        jsonFile << "  \"reflections\": {\n"
                << "    \"dataType\": \"double\",\n"
                << "    \"size\": [" << size << ", " << iter << ", 3],\n"
                << "    \"order\": 8\n"
                << "  },\n";
        
        jsonFile << "  \"uref\": {\n"
                << "    \"dataType\": \"double\",\n"
                << "    \"size\": [" << size << ", " << iter << ", 3],\n"
                << "    \"order\": 9\n"
                << "  },\n";
        
        jsonFile << "  \"sig0\": {\n"
                << "    \"dataType\": \"double\",\n"
                << "    \"size\": [" << timepoints << ", 1],\n"
                << "    \"order\": 10\n"
                << "  },\n";
        
        jsonFile << "  \"sigRe\": {\n"
                << "    \"dataType\": \"double\",\n"
                << "    \"size\": [" << Nbvec << ", " << iter << "],\n"
                << "    \"order\": 11\n"
                << "  }";

        if (SaveAll) {
            jsonFile << ",\n  \"all_data\": {\n"
                    << "    \"dataType\": \"double\",\n"
                    << "    \"size\": [" << size << ", " << iter << ", 3],\n"
                    << "    \"order\": 12\n"
                    << "  }";
        }
        
        jsonFile << "\n}\n";
        jsonFile.close();
    } else {
        std::cerr << "Error: Unable to create metadata JSON file" << std::endl;
    }

    delete[] t;
}

__device__ void computeNext(double3 &A, double &step, double4 &xi, double3 &nextpos, double &pi) {
    double theta = 2 * pi * xi.x;
    double v = xi.y;
    double cos_phi = 2 * v - 1;
    double sin_phi = sqrt(1 - pow(cos_phi, 2));
    nextpos.x = A.x + (step * sin_phi * cos(theta));
    nextpos.y = A.y + (step * sin_phi * sin(theta));
    nextpos.z = A.z + (step * cos_phi);
}

__device__ bool checkConnections(int3 i_int3, int test_lutvalue, double3 nextpos, int *NewIndex, double4 *d4swc, double &fstep) {
    int3 vindex;
    double4 child, parent;
    double dist2;

    // for each connection check if particle inside
    for (int page = 0; page < i_int3.z; page++) {

        // create a subscript indices
        int3 c_new = make_int3(test_lutvalue, 0, page);
        int3 p_new = make_int3(test_lutvalue, 1, page);

        // convert subscripted index to linear index and get value from Index Array
        vindex.x = NewIndex[s2i(c_new, i_int3)] - 1;
        vindex.y = NewIndex[s2i(p_new, i_int3)] - 1;

        if ((vindex.x) != -1) {
            //extract child parent values from swc
            child = d4swc[vindex.x];
            parent = d4swc[vindex.y];

            // calculate euclidean distance
            dist2 = distance2(parent, child);

            // determine whether particle is inside this connection
            bool inside = swc2v(nextpos, child, parent, dist2);

            // if it is inside the connection we don't need to check the remaining.
            if (inside) {
                return true;
            }
        }
            // if the value of the index array is -1 we have checked all pairs for this particle.
        else {
            return false;
        }
    }
    return false;
}


__device__ void validCoord(double3 &nextpos, double3 &pos, int3 &b_int3, int3 &upper, int3 &lower, int3 &floorpos,
                           double * reflections, double * uref, int gid, int i, int size, int iter, int * flips) {
    double3 High = make_double3((double)b_int3.x, (double)b_int3.y, (double) b_int3.z);
    double3 Low = make_double3(0.0, 0.0, 0.0);

    // determine the index of the reflection storage should match the save data index
    int3 dix = make_int3(size, iter, 3);
    int3 did[4];
    did[0] = make_int3(gid, i, 0);
    did[1] = make_int3(gid, i, 1);
    did[2] = make_int3(gid, i, 2);
    did[3] = make_int3(s2i(did[0], dix), s2i(did[1], dix), s2i(did[2], dix));

    int fidx; // flip index for reflection

    int count = 0;
    while(true) {
        int3 UPPER = nextpos > High;
        int3 LOWER = nextpos < Low;

        // normal vector
        double3 normal;

        // point on plane
        double3 pointOnPlane;

        // par hit boundary, flip sign of that x,y,z. 
        if (LOWER.x) {
            fidx = 6*gid + 0;
            pointOnPlane = make_double3(Low.x, nextpos.y, nextpos.z);
            normal = make_double3(1.0, 0.0, 0.0);
        }
        else if (UPPER.x) {
            fidx = 6*gid + 1;
            pointOnPlane = make_double3(High.x, nextpos.y, nextpos.z);
            normal = make_double3(-1.0, 0.0, 0.0);
        }
        else if (LOWER.y) {
            fidx = 6*gid + 2;
            pointOnPlane = make_double3(nextpos.x, Low.y, nextpos.z);
            normal = make_double3(0.0, 1.0, 0.0);
        }
        else if (UPPER.y) {
            fidx = 6*gid + 3;
            pointOnPlane = make_double3(nextpos.x, High.y, nextpos.z);
            normal = make_double3(0.0, -1.0, 0.0);
        }
        else if (LOWER.z) {
            fidx = 6*gid + 4;
            pointOnPlane = make_double3(nextpos.x, nextpos.y, Low.z);
            normal = make_double3(0.0, 0.0, 1.0);
        }
        else if (UPPER.z) {
            fidx = 6*gid + 5;
            pointOnPlane = make_double3(nextpos.x, nextpos.y, High.z);
            normal = make_double3(0.0, 0.0, -1.0);
        }
        else {
            return; // no reflection needed
        }
        // 
        // Calculate D  (Ax + By + Cz + D = 0) 
        double D = -(dot(normal, pointOnPlane));

        double3 intersectionPoint;
        double3 d = pos - nextpos;

        double t1 = -((dot(normal, nextpos) + D)) / dot(normal, d);
        intersectionPoint = nextpos + d * t1;

        double3 reflectionVector = nextpos - intersectionPoint;
        reflectionVector = reflectionVector - normal * (2 * dot(reflectionVector,normal));

        // record the unreflected position
        double3 unreflected = nextpos;
        // double3 intersection = intersectionPoint;
        nextpos = intersectionPoint + reflectionVector;

        // printf("NextPos: %f %f %f -> %f %f %f\n", nextpos.x, nextpos.y, nextpos.z, intersectionPoint.x+reflectionVector.x, intersectionPoint.y + reflectionVector.y, intersectionPoint.z + reflectionVector.z);
        // printf("Count: %d\n", count);
        count += 1;

        // store the intersection point and unreflected position
        set(reflections, did[3], intersectionPoint);
        set(uref, did[3], unreflected);

        // Update the particle's position
        nextpos = intersectionPoint + reflectionVector;

        // flip the particle's direction
        flips[fidx] += 1; // no need for atomicAdd since gid is what is parallelized
    }
}


__host__ void setup_data(double * u_dx2, double * u_dx4, double * u_SimP, double4 * u_D4Swc, int * u_NewLut,
                                    int * u_NewIndex, int * u_Flip, double * simparam, double4 * swc_trim,
                                    double * mdx2, double * mdx4, double * u_AllData,double * u_Reflections, 
                                    double * u_Uref, double * u_T2, double * u_T, double * u_SigRe, double * u_Sig0,
                                    double * u_bvec, double * u_bval, double * u_TD, std::vector<uint64_t> lut, std::vector<uint64_t> indexarr, 
                                    int3 bounds, int size, int iter, int nrow, int prod, int newindexsize, size_t sa_size, int Nbvec, int timepoints, int NC, int n_vf, double * u_vf, int * u_label) {
    // Set Values for Host
    
        
        for (int i = 0; i < 10; i++) {
            u_SimP[i] = simparam[i];
        }
        for (int i = 0; i < nrow; i++) {
            u_D4Swc[i].x = swc_trim[i].x;
            u_D4Swc[i].y = swc_trim[i].y;
            u_D4Swc[i].z = swc_trim[i].z;
            u_D4Swc[i].w = swc_trim[i].w;
        }
        for (int i = 0; i < prod; i++) {
            u_NewLut[i] = lut[i];
        }
        for (int i = 0; i < newindexsize; i++) {
            u_NewIndex[i] = indexarr[i];
        }
        
        memset(mdx2, 0.0, 6 * iter * sizeof(double));
        memset(mdx4, 0.0, 15 * iter * sizeof(double));
        memset(u_AllData, 0.0, 3 * sa_size * sizeof(double));
        memset(u_Reflections, 0.0, 3 * iter * size * sizeof(double));
        memset(u_Uref, 0.0, 3 * iter * size * sizeof(double));
        memset(u_Flip, 0.0, 3 * iter * sizeof(int));
        memset(u_T2, 0.0, NC * sizeof(double));                            // T2 is read from file?
        memset(u_T, 0.0, NC * sizeof(double));                             // T is set to 0.0
        memset(u_SigRe, 0.0, Nbvec * timepoints * sizeof(double));         // Calculated in kernel
        memset(u_Sig0, 0.0, timepoints * sizeof(double));                  // Calculated in kernel
        memset(u_bvec, 0.0, Nbvec * 3 * sizeof(double));                   // bvec is read from file
        memset(u_bval, 0.0, Nbvec * sizeof(double));                       // bval is read from file
        memset(u_TD, 0.0,   Nbvec * sizeof(double));                       // TD is read from file
        memset(u_vf, 0.0, sizeof(double));                                 // Calc in vf kernel
        memset(u_label, 0.0,  n_vf * sizeof(int));                         // arg in or default
        printf("Set Host Values\n");
    
}

// Set Values for Host
/*    {
    memset(u_dx2, 0.0, 6 * iter * SOD);
    memset(u_dx4, 0.0, 15 * iter * SOD);
    {

        for (int i = 0; i < 10; i++) {
            u_SimP[i] = simparam[i];
        }

        for (int i = 0; i < nrow; i++) {
            u_D4Swc[i].x = swc_trim[i].x;
            u_D4Swc[i].y = swc_trim[i].y;
            u_D4Swc[i].z = swc_trim[i].z;
            u_D4Swc[i].w = swc_trim[i].w;
        }

        for (int i = 0; i < prod; i++) {
            int value = lut[i];
            u_NewLut[i] = value;
        }

        for (int i = 0; i < indexarr.size(); i++) {
            int value = indexarr[i];
            u_NewIndex[i] = value;
        }
    }
    memset(mdx2, 0.0, 6 * iter * SOD);
    memset(mdx4, 0.0, 15 * iter * SOD);
    memset(u_AllData, 0.0, 3 * sa_size * SOD);
    memset(u_Reflections, 0.0, 3 * iter * size * SOD);
    memset(u_uref, 0.0, 3 * iter * size * SOD);
    memset(u_Flip, 0.0, 3 * size * SOI);
    memset(u_T2, 0.0, Nc * SOD); // T2 is read from file?
    memset(u_T, 0.0, Nc * SOD); // T is set to 0.0
    memset(u_SigRe, 0.0, Nbvec * timepoints * SOD); // Calculated in kernel
    memset(u_Sig0, 0.0, timepoints * SOD); // Calculated in kernel
    memset(u_bvec, 0.0, Nbvec * 3 * SOD); // bvec is read from file
    memset(u_bval, 0.0, Nbvec * SOD); // bval is read from file
    memset(u_TD, 0.0, Nbvec * SOD); // TD is read from file
    printf("Set Host Values\n");
}*/
