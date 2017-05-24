/*
 * main.cpp
 *
 *  Created on: Apr 22, 2017
 *      Author: jacey
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "utillib/helper_string.h"
#include "utillib/helper_cuda.h"

#include "EM_gpu.h"

using namespace std;

int main()
{
	/* Get GPU statistics */
	int devID = 0; 			// Assuming using 1 GPU
    cudaDeviceProp deviceProp;

    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    // compute the scaling factor (for GPUs with fewer MPs)
    float scale_factor, total_tiles;
    scale_factor = max((192.0f / (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * (float)deviceProp.multiProcessorCount)), 1.0f);

    printf("-- GPU statistics --\n");
    printf("Device %d: \"%s\"\n", devID, deviceProp.name);
    printf("SM Capability %d.%d detected:\n", deviceProp.major, deviceProp.minor);
    printf("[%s] has %d MP(s) x %d (Cores/MP) = %d (Cores)\n\n",
           deviceProp.name, deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

    /* Read in sample data */
	const char *csv_file = "./data/static_G3_N100000.csv";
	const int dim = 3;
	const int size = 50000;
	const int num_gaus = 3;
	double exit_threshold = 1e-20;
	const int max_iter = 5000;
	int i;

	double *samples = new double[size * dim];
	string line, token;
	ifstream file(csv_file);

	if (! file.good()) cout << "> Not able to open file " << csv_file << endl;
	else cout << "> Start reading input file : " << csv_file << endl;

	i = 0;
	while (file.good() && i < size * dim) {
		getline(file, line);
		stringstream ss(line);

		while (getline(ss, token, ','))
		{
			samples[i] = strtod(token.c_str(), NULL);
			++i;
		}
	}
	cout << "> Finished reading " << i/dim << " samples" << endl;

	cout << "> Start executing EM_GPU...\n" << endl;

	printf("-- Experiment summary --\n");
	run_EM(samples, size, dim, num_gaus, exit_threshold, max_iter, false);

	cout << "> Finished EM_GPU" << endl;

	delete[] samples;
	return 0;
}
