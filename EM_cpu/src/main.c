#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "EM_cpu.h"


void print_params(GaussianParam *params, int num_models, int dim)
{
	printf("-- The Gaussian parameters estimated --\n");
	for (int i = 0; i < num_models; i++)
	{
		printf("The %d Guassian model: \nMU = \n", i + 1);
		for (int d = 0; d < dim; d++) printf("\t%lf", params[i].mu[d]);
		printf("\nSIGMA = \n");
		for (int p = 0; p < dim; p++)
		{
			for (int q = 0; q < dim; q++)
				printf("\t%lf", params[i].sigma[p * dim + q]);
			printf("\n");
		}
	}
}

int main(int argc, char *argv[])
{
	const char *csv_file = "src/data/static_G3_N100000.csv";
	const int dim = 3;
	const int size = 50000;
	const int num_gaus = 3;
	const int num_iter = 5000;
	double exit_threshold = 1e-20;

	// Read samples from csv file
	char buffer[1024];
	char *token;
	int i, j;
	double *sample;
	double **samples = (double **)malloc(sizeof(double *) * size);

	FILE *fstream = fopen(csv_file, "r");
	if (fstream == NULL)
	{
		printf("\n file opening failed ");
		return -1;
	}

	i = 0;
	while (fgets(buffer, sizeof(buffer), fstream) != NULL)
	{
		j = 0;
		sample = (double *)malloc(sizeof(double) * dim);
		token = strtok(buffer, ",");
		while (token != NULL)
		{
			// printf("read in %d sample token: %s\n", i, token);
			sample[j] = atof(token);
			token = strtok(NULL, ",");
			j++;
		}
		samples[i] = sample;
		i++;
	}

	// run Expectation-Maximization algo
	printf("CPU EM starting ... \n");

	clock_t start = clock(), diff;
	GaussianParam *params = run_EM_cpu(samples, size, dim, num_gaus,
			exit_threshold, num_iter);
	diff = clock() - start;
	float msec = diff * 1000 / CLOCKS_PER_SEC;

	printf("CPU EM finished\n\n");

	// Experiment summary
	printf("-- Experiment summary --\n");
	printf("SampleDim = %d X %d (double), NumGaussians = %d, ", size, dim, num_gaus);
	printf("TotTime = %.4f ms; NumIter = %d; TimePerIter = %.4f ms\n\n", msec, num_iter, msec / num_iter);

	// Print calculated params
	print_params(params, num_gaus, dim);

	printf("Note: Randomly initializing predicted params might cause 'nan' in final estimation.\nPlease re-run program to avoid influences of inproper initialization.\n");

	getchar();
	return 0;
}
