#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "EM_cpu.h"
#include "matrix_cpu.h"

/* List of functions */
double mvnpdf(double *x, double *mu, double *sigma, int dim);
double eval_likelihood(double *prev, double *curr, int m, int n);
GaussianParam *run_EM_cpu(double **samples, int s_size, int s_dim, int num_gaus, double threshold, int max_iter);


/*
 *
 *
 */
double mvnpdf(double *x, double *mu, double *sigma, int dim)
{
	// Allocate memory
	double exponent, denom;
	double *res_vec_subtr = (double *) malloc(sizeof(double) * dim * 1);
	double *res_trans = (double *) malloc(sizeof(double) * dim * 1);
	double *res_mat_inv = (double *) malloc(sizeof(double) * dim * dim);
	double *res_mat_mult = (double *) malloc(sizeof(double) * dim * dim);
	double *res_mat_mult2 = (double *)malloc(sizeof(double));

	//
	matrix_subtr(res_vec_subtr, x, mu, dim, 1);
	transpose(res_trans, res_vec_subtr, dim, 1);
	inverse(res_mat_inv, sigma, dim);
	matrix_mult(res_mat_mult, res_trans, res_mat_inv, 1, dim, dim);
	matrix_mult(res_mat_mult2, res_mat_mult, res_vec_subtr, 1, dim, 1);

	exponent = (-1 / (double) 2) * res_mat_mult2[0];
	denom = sqrt(pow(2 * MATH_PI, dim) * determinant(sigma, dim));

	free(res_vec_subtr);
	free(res_trans);
	free(res_mat_inv);
	free(res_mat_mult);
	free(res_mat_mult2);
	return exp(exponent) / denom;
}

double eval_likelihood(double *prev, double *curr, int m, int n)
{
	if (prev[0] == INFINITY) return 10;

	double diff = 0;
	for (int i = 0; i < m * n; i++)
		diff += (prev[i] > curr[i]) ? (prev[i] - curr[i]) : (curr[i] - prev[i]);
	return diff;
}

GaussianParam *run_EM_cpu(double **samples, int s_size, int s_dim, int num_gaus, double threshold, int max_iter)
{
	GaussianParam *output = (GaussianParam *) malloc(sizeof(GaussianParam) * num_gaus);

	// Sizes for memory allocation
	int size_n_gaus = sizeof(double) * num_gaus;
	int size_n_samp = sizeof(double) * s_size;
	int size_mu = sizeof(double) * s_dim;
	int size_sigma = sizeof(double) * s_dim * s_dim;
	int size_likelihood = sizeof(double) * num_gaus * s_size;

	double *mean_diff = (double *)malloc(size_mu);
	double *weights = (double *)malloc(size_n_gaus);
	double *likelihood = (double *) malloc(size_likelihood);
	double *likelihood_prev = (double *)malloc(size_likelihood);

	double normalization, marginals, change_L = INFINITY; likelihood_prev[0] = INFINITY;
	double *inter_sigma, *trans_sigma;
	int i, j, k, d, iter = 0;
	double pdf;

	/* Initialize Gaussian params and weights */
	time_t t;
	GaussianParam gauPara;
	srand((unsigned) time(&t)); // seed random number generator
	inter_sigma = (double *)malloc(size_sigma);
	trans_sigma = (double *)malloc(size_sigma);

	for (i = 0; i < num_gaus; i++)
	{
		// Init mu randomly
		output[i].mu = malloc(sizeof(double) * s_dim);
		for (d = 0; d < s_dim; d++)
			output[i].mu[d] = ((double)rand() / RAND_MAX) * MEAN_PRIOR;

		// Init sigma randomly
		output[i].sigma = malloc(size_sigma);
		for (d = 0; d < s_dim * s_dim; d++)
			inter_sigma[d] = ((double)rand() / RAND_MAX) * VAR_PRIOR;
		// Let sigma = T'* T so that sigma is positive-definite
		transpose(trans_sigma, inter_sigma, s_dim, s_dim);
		matrix_mult(output[i].sigma, trans_sigma, inter_sigma, s_dim, s_dim, s_dim);

		// Init weights uniformly
		weights[i] = 1 / (double) num_gaus;
	}

	while (change_L > threshold && iter < max_iter)
	{
		// E-step: Calculate normalized likelihood
		for (j = 0; j < s_size; j++)
		{
			normalization = 0;
			for (k = 0; k < num_gaus; k++)
			{
				pdf = mvnpdf(samples[j], output[k].mu, output[k].sigma, s_dim);
				normalization += weights[k] * pdf;
				// printf("sample idx = %d, guassian idx = %d, pdf = %e\n", j, k, pdf);
			}
			// printf("sample idx = %d, normalization = %e\n", j, normalization);
			if (normalization <= 0)
			{
				printf("sample idx = %d, INVALID normalization = %e\n", j, normalization);
				normalization = 1;
			}
			for (i = 0; i < num_gaus; i++)
				likelihood[i * s_size + j] = weights[i] * mvnpdf(samples[j], output[i].mu, output[i].sigma, s_dim) / normalization;
		}

		// print_mat(likelihood, num_gaus, s_size);

		// M-step: update weights, means, covarience matrices
		for (i = 0; i < num_gaus; i++)
		{
			// reset the mu and sigma parameters to zero for updates
			for (d = 0; d < s_dim; d++) output[i].mu[d] = 0;            // Rest components of mean
			for (d = 0; d < s_dim * s_dim; d++) output[i].sigma[d] = 0; // Rest components of covariance

			marginals = 0;
			for (j = 0; j < s_size; j++) marginals += likelihood[i * s_size + j];

			// Update weight
			weights[i] = marginals / s_size;

			// Update mean
			for (j = 0; j < s_size; j++)
				for (d = 0; d < s_dim; d++)
					output[i].mu[d] += likelihood[i * s_size + j] * samples[j][d];
			for (d = 0; d < s_dim; d++) output[i].mu[d] /= marginals;

			// Update covariance matrix
			for (j = 0; j < s_size; j++)
			{
				matrix_subtr(mean_diff, samples[j], output[i].mu, s_dim, 1);
				matrix_mult(inter_sigma, mean_diff, mean_diff, s_dim, 1, s_dim);
				matrix_scalar_mult(inter_sigma, inter_sigma, likelihood[i * s_size + j] / marginals, s_dim, s_dim);
				matrix_add(output[i].sigma, output[i].sigma, inter_sigma, s_dim, s_dim);
			}
		}

		// change_L = eval_likelihood(likelihood_prev, likelihood, num_gaus, s_size);
		change_L = 10;
		// Save likelihood matrix
		// for (k = 0; k < s_size * num_gaus; k++)
		// 	  likelihood_prev[k] = likelihood[k];
		//printf("ITER = %d\nchange_L = %e\n", iter, change_L);
		iter++;
	}

	free(mean_diff);
	free(inter_sigma);
	free(trans_sigma);

	return output;
}
