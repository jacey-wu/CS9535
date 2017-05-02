#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "EM_cpu.h"
#include "matrix_cpu.h"

double mvnpdf(double *x, double *mu, double *sigma, int dim);
double eval_likelihood(double *prev, double *curr, int m, int n);
GaussianPara *run_EM_cpu(double **samples, int s_size, int s_dim, int num_gaus, double threshold, int max_iter);


double mvnpdf(double *x, double *mu, double *sigma, int dim)
{
	double exponent, denom;
	double *res_vec_subtr = matrix_subtr(x, mu, dim, 1);
	double *res_trans = transpose(res_vec_subtr, dim, 1);
	double *res_mat_inv = inverse(sigma, dim);
	double *res_mat_mult = matrix_mult(res_trans, res_mat_inv, 1, dim, dim);
	double *res_mat_mult2 = matrix_mult(res_mat_mult, res_vec_subtr, 1, dim, 1);
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

GaussianPara *run_EM_cpu(double **samples, int s_size, int s_dim, int num_gaus, double threshold, int max_iter)
{
	GaussianPara *output = (GaussianPara *) malloc(sizeof(GaussianPara) * num_gaus);
	double *weights = (double *)malloc(sizeof(double) * num_gaus);
	double *likelihood = (double *) malloc(sizeof(double) * s_size * num_gaus);
	double *likelihood_prev = (double *)malloc(sizeof(double) * s_size * num_gaus);

	double change_L = INFINITY; likelihood_prev[0] = INFINITY;
	double normalization, marginal;
	double *mean_diff, *inter_sigma, *trans_sigma;
	int i, j, k, d, iter = 0;
	double pdf;

	/* Initialize Gaussian params and weights */
	time_t t;
	GaussianPara gauPara;
	srand((unsigned) time(&t)); // seed random number generator

	for (i = 0; i < num_gaus; i++)
	{
		output[i].mu = malloc(sizeof(double) * s_dim);
		for (d = 0; d < s_dim; d++)
			output[i].mu[d] = ((double)rand() / RAND_MAX) * MEAN_PRIOR;

		inter_sigma = malloc(sizeof(double) * s_dim * s_dim);
		for (d = 0; d < s_dim * s_dim; d++) inter_sigma[d] = ((double)rand() / RAND_MAX) * VAR_PRIOR;
		trans_sigma = transpose(inter_sigma, s_dim, s_dim);
		output[i].sigma = matrix_mult(trans_sigma, inter_sigma, s_dim, s_dim, s_dim);

		free(inter_sigma);
		free(trans_sigma);

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

		//print_mat(likelihood, num_gaus, s_size);

		// M-step: update weights, means, covarience matrices
		for (i = 0; i < num_gaus; i++)
		{
			// reset the mu and sigma parameters to zero for updates
			for (d = 0; d < s_dim; d++) output[i].mu[d] = 0;            // Rest components of mean
			for (d = 0; d < s_dim * s_dim; d++) output[i].sigma[d] = 0; // Rest components of covariance

			marginal = 0;
			for (j = 0; j < s_size; j++) marginal += likelihood[i * s_size + j];

			// Update weight
			weights[i] = marginal / s_size;

			// Update mean
			for (j = 0; j < s_size; j++)
				for (d = 0; d < s_dim; d++)
					output[i].mu[d] += likelihood[i * s_size + j] * samples[j][d];
			for (d = 0; d < s_dim; d++) output[i].mu[d] /= marginal;

			// Update covariance matrix
			for (j = 0; j < s_size; j++)
			{
				mean_diff = matrix_subtr(samples[j], output[i].mu, s_dim, 1);
				inter_sigma = matrix_mult(mean_diff, mean_diff, s_dim, 1, s_dim);
				inter_sigma = matrix_scalar_mult(inter_sigma, inter_sigma, likelihood[i * s_size + j] / marginal, s_dim, s_dim);
				matrix_add(output[i].sigma, output[i].sigma, inter_sigma, s_dim, s_dim);

				free(mean_diff);
				free(inter_sigma);
			}
		}

		change_L = eval_likelihood(likelihood_prev, likelihood, num_gaus, s_size);

		// Save likelihood matrix
		for (k = 0; k < s_size * num_gaus; k++)
			likelihood_prev[k] = likelihood[k];
		iter++;
		printf("ITER = %d\nchange_L = %e", iter, change_L);
	}
	return output;
}
