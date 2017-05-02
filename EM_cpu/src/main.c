#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "EM_cpu.h"


void print_params(GaussianPara *params, int num_models, int dim)
{
	for (int i = 0; i < num_models; i++)
	{
		printf("The %d Guassian model: \n MU = \n", i);
		for (int d = 0; d < dim; d++) printf("\t%lf", params[i].mu[d]);
		printf("\n SIGMA = \n");
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
	const char *csv_file = "src/data/static_750.csv";
	const int dim = 3;
	const int size = 750;
	const int num_gaus = 3;
	double exit_threshold = 1e-20;

	// Read samples
	char buffer[1024];
	char *token;
	int i, j;
	double *sample;
	double **samples = (double **)malloc(sizeof(double *) * size);

	//double abc[size][dim];

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
			printf("read in %d sample token: %s\n", i, token);
			sample[j] = atof(token);
			token = strtok(NULL, ",");
			j++;
		}
		samples[i] = sample;
		i++;
	}

	// run Expectation-Maximization algo
	GaussianPara *params = run_EM_cpu(samples, size, dim, num_gaus, exit_threshold, 1000);

	print_params(params, num_gaus, dim);

	printf("End of program\n");

	getchar();
	return 0;
}
