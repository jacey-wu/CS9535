/**
    CS9535 Expactation-maximization (EM) algorithm (CPU impl)
    main.c
    Purpose: The main file of the CPU implementation of EM algorithm. Parse cmd
    arguments and complete I/O operations.

    @author Jacey Wu
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#include "EM_cpu.h"

#define DEFAULT_DIM 3

//extern char *optarg;

/**
 * Print mean vector MU and covariance matrix SIGMA for each Gaussian model
 * stored in Gaussian parameter array.
 *
 * @param params: the Gaussian parameter array
 * @param num_models: the number of Gaussian models included in GMM
 * @param dim: the dimension of input feature
 * @return
 */
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


/**
 * Print help message.
 */
void print_help()
{
	printf("\nUsage  : EM_CPU [options] <inputvalue>\n");
	printf("Multiple arguments can be provided separated by wightspace, e.g.\n");
	printf("EM_cpu -f input_file.csv -n 5000 -g 4 -i 1000\n\n");
	printf("Supported options:\n");
	printf("  -f csv_file  Option to add input file in csv format\n");
	printf("  -n size      Option to specify the number of input samples in given csv file\n");
	printf("  -g num_gaus  Option to specify the number of Gaussian models\n");
	printf("  -i max_iter  Option to specify the max number of iterations of EM algo\n");
	printf("  -h           Option to show this message\n\n");
}


/**
 * Main function to launch EM experiments.
 *
 * @param argc: the number of cmd arguments
 * @param argv: the cmd arguments as strings
 * @return: error code
 */
int main(int argc, char *argv[])
{
	/* Default values of cmd arguments */
	const double exit_threshold = 1e-20;
	const int dim = DEFAULT_DIM;

	char *csv_file = "../src/data/static_G3_N1000.csv";
	int size = 1000;
	int num_gaus = 3;
	int num_iter = 500;

	// Process cmd arguments
	if (argc < 2)
	{
		print_help();
		printf("None options are received. Process with default args\n");
	}
	else
	{
		int opt = 0;
		const char *optString = "f:n:g:i:h?";

		opt = getopt( argc, argv, optString);
		while (opt != -1) {
			switch (opt) {
			case 'f':	// input file
				csv_file = optarg;
				printf("File name: %s\n", csv_file);
				break;
			case 'n':	// num of samples
				size = atoi(optarg);
				break;
			case 'g':	// num of Gaussian models
				num_gaus = atoi(optarg);
				break;
			case 'i':	// num of max iterations
				num_iter = atoi(optarg);
				break;
			case 'h':	// help message
			case '?':
				print_help();
				break;
			default:	// should never reach here
				break;
			}
			opt = getopt(argc, argv, optString);
		}
	}

	// Read samples from csv file
	char buffer[1024];
	char *token;
	int i, j;
	double *sample;
	double **samples = (double **)malloc(sizeof(double *) * size);

	FILE *fstream = fopen(csv_file, "r");
	if (fstream == NULL)
	{
		printf("[ERROR] Failed to open file %s\n", csv_file);
		return -1;
	}

	i = 0;
	while (i < size && fgets(buffer, sizeof(buffer), fstream) != NULL)
	{
		j = 0;
		sample = (double *)malloc(sizeof(double) * dim);
		token = strtok(buffer, ",");
		while (token != NULL)
		{
			//printf("read in %d sample token: %s\n", i, token);
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
