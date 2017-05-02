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

#include "EM_gpu.h"

using namespace std;

int main()
{

	const char *csv_file = "./data/static_750.csv";
	const int dim = 3;
	const int size = 750;
	const int num_gaus = 3;
	double exit_threshold = 1e-20;
	const int max_iter = 10;
	int i;

	// Read in samples
	double *samples = new double[size * dim];
	string line, token;
	ifstream file(csv_file);

	if (! file.good()) cout << "Not able to open file " << csv_file << endl;

	i = 0;
	while (file.good()) {
		getline(file, line);
		stringstream ss(line);

		while (getline(ss, token, ','))
		{
			samples[i] = strtod(token.c_str(), NULL);
			cout << "read in " << i << " sample token: " << token << endl;
			++i;
		}
	}
	cout << "Finished reading file." << endl;

	// GaussianParam *params = run_EM_gpu(samples, size, dim, num_gaus, exit_threshold, 2);
	run_EM(samples, size, dim, num_gaus, exit_threshold, max_iter);

	delete[] samples;

	return 0;
}
