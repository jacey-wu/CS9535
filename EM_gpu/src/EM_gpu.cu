/*
 ============================================================================
 Name        : EM_gpu.cu
 Author      : Jiaxiao Wu
 Version     :
 Copyright   : 
 Description : CUDA expectation-maximization algorithm
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cfloat>

#include "utillib/helper_string.h"
#include "utillib/helper_cuda.h"

#include "EM_gpu.h"
extern "C" {
#include "matrix.h"
}

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define WARPSIZE 32
#define THR_PER_BLOCK_SM 128
#define THR_PER_BLOCK 256

#define MATH_PI 3.14159265358979323846   // pi
#define MEAN_PRIOR 5;
#define VAR_PRIOR 2;


/**
 * CUDA kernels
 */

__global__ void marginal_single_reduction(double *g_idata, double *g_odata)
{
	extern __shared__ double sdata[];
	int tid = threadIdx.x;

	// Read data into shared memory
	sdata[tid] = g_idata[blockIdx.x * blockDim.x + tid];
	__syncthreads();

	// Reduction in shared memory
	for (int s = blockDim.x / 2; s > WARPSIZE; s>>=1)    // Reversed loop
	{
		if (tid < s)                 // Non-divergent branch
			sdata[tid] += sdata[tid + s]; // Sequential addressing
		__syncthreads();
	}

	if (tid < WARPSIZE)
	{
		if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
		if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
		if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8];
		if (blockDim.x >= 8) sdata[tid] += sdata[tid + 4];
		if (blockDim.x >= 4) sdata[tid] += sdata[tid + 2];
		if (blockDim.x >= 2) sdata[tid] += sdata[tid + 1];
	}

	// Write the sum of this block to device memory
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void marginal_reduction(double *likelihood, double *tmp_odata, int size, int num_gaus)
{
	int tid = threadIdx.x;
	int lx = blockDim.x * blockIdx.x + threadIdx.x;
	int ly = blockIdx.y;

	if (lx < size && ly < num_gaus)
	{
		// Read data into shared memory
		extern __shared__ double sdata[];
		sdata[tid] = likelihood[ly * size + lx];
		__syncthreads();

		// Reduction in shared memory
		for (int s = blockDim.x / 2; s > WARPSIZE; s>>=1)    // Reversed loop
		{
			if (tid < s)                 // Non-divergent branch
				sdata[tid] += sdata[tid + s]; // Sequential addressing
			__syncthreads();
		}

		if (tid < WARPSIZE)
		{
			if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
			if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
			if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8];
			if (blockDim.x >= 8) sdata[tid] += sdata[tid + 4];
			if (blockDim.x >= 4) sdata[tid] += sdata[tid + 2];
			if (blockDim.x >= 2) sdata[tid] += sdata[tid + 1];
		}

		// Write the sum of this block to device memory
		if (tid == 0) tmp_odata[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];

	}
}

__global__ void weight_update(double *weights, double *marginals, int num_gaus, int size)
{
	int tid = threadIdx.x;
	if (tid < num_gaus)
	{
		double sum = 0;
		for (int i = 0; i < num_gaus; ++i) sum += marginals[i];

		if (marginals[tid] == 0)
			weights[tid] = DBL_MIN;
		else
			weights[tid] = marginals[tid] / (double) sum;
	}
}

__global__ void weight_update_orig(double *weights, double *marginals, int num_gaus, int size)
{
	int tid = threadIdx.x;
	if (tid < num_gaus)
	{
		if (marginals[tid] == 0)
			weights[tid] = DBL_MIN;
		else
			weights[tid] = marginals[tid] / (double) size;
	}
}

__global__ void mu_reduction(double *likelihood, double *samples, double *marg, double *tmp_odata, int size, int num_gaus, int dim)
{
	int tid = threadIdx.x;
	int blockId = (gridDim.x * gridDim.y) * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;

	int samp_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int dim_idx = blockIdx.y;
	int gaus_idx = blockIdx.z;

	if (samp_idx < size && dim_idx < dim && gaus_idx < num_gaus)
	{
		// Read data into shared memory
		extern __shared__ double sdata[];
		sdata[tid] = likelihood[gaus_idx * size + samp_idx] * samples[samp_idx * dim + dim_idx];
		__syncthreads();

		// Reduction in shared memory
		for (int s = blockDim.x / 2; s > WARPSIZE; s>>=1)    // Reversed loop
		{
			if (tid < s)                      // Non-divergent branch
				sdata[tid] += sdata[tid + s]; // Sequential addressing
			__syncthreads();
		}

		if (tid < WARPSIZE)
		{
			if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
			if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
			if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8];
			if (blockDim.x >= 8) sdata[tid] += sdata[tid + 4];
			if (blockDim.x >= 4) sdata[tid] += sdata[tid + 2];
			if (blockDim.x >= 2) sdata[tid] += sdata[tid + 1];
		}

		// Write the sum of this block to device memory
		if (tid == 0) tmp_odata[blockId] = sdata[0] / (double) marg[gaus_idx];
	}
}

__global__ void sigma_reduction(double *likelihood, double *samples, double *marg, double *mu, double *odata,
		int size, int num_gaus, int dim)
{
	int tid = threadIdx.x;
	int blockId = (gridDim.x * gridDim.y) * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;

	int samp_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int gaus_idx = blockIdx.z;

	int sigy = blockIdx.y / dim;
	int sigx = blockIdx.y - dim * sigy;

	if (samp_idx < size && gaus_idx < num_gaus && sigx < dim && sigy < dim)
	{
		// Read data into shared memory
		extern __shared__ double sdata[];

		sdata[tid] = likelihood[gaus_idx * size + samp_idx]
		                     * (samples[samp_idx * dim + sigx] - mu[gaus_idx * dim + sigx]) // non-coalesced access
		                     * (samples[samp_idx * dim + sigy] - mu[gaus_idx * dim + sigy]);
		__syncthreads();

		// Reduction in shared memory
		for (int s = blockDim.x / 2; s > WARPSIZE; s>>=1)    // Reversed loop
		{
			if (tid < s)                 // Non-divergent branch
				sdata[tid] += sdata[tid + s]; // Sequential addressing
			__syncthreads();
		}

		if (tid < WARPSIZE)
		{
			if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
			if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
			if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8];
			if (blockDim.x >= 8)  sdata[tid] += sdata[tid + 4];
			if (blockDim.x >= 4)  sdata[tid] += sdata[tid + 2];
			if (blockDim.x >= 2)  sdata[tid] += sdata[tid + 1];
		}

		// Write the sum of this block to device memory
		if (tid == 0) odata[blockId] = sdata[0] / (double) marg[gaus_idx];
	}
}

__global__ void single_reduction(double *idata, double *omat)
{
	extern __shared__ double sdata[];
	int tid = threadIdx.x;
	int gid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

	// Read data into shared memory
	sdata[tid] = idata[gid];
	__syncthreads();

	// Reduction in shared memory
	for (int s = blockDim.x / 2; s > WARPSIZE; s>>=1)    // Reversed loop
	{
		if (tid < s)                 // Non-divergent branch
			sdata[tid] += sdata[tid + s]; // Sequential addressing
		__syncthreads();
	}

	if (tid < WARPSIZE)
	{
		if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
		if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
		if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8];
		if (blockDim.x >= 8) sdata[tid] += sdata[tid + 4];
		if (blockDim.x >= 4) sdata[tid] += sdata[tid + 2];
		if (blockDim.x >= 2) sdata[tid] += sdata[tid + 1];
	}

	// Write the sum of this block to device memory
	if (tid == 0) omat[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
}

__global__ void mvnpdf_dim3(double *likelihood, double *samples, double *mu_mat, double *sig_mat, int size, int num_gaus)
{
	const int dim = 3;
	int lx = blockDim.x * blockIdx.x + threadIdx.x;
	int ly = blockIdx.y;
	if (lx < size && ly < num_gaus)
	{
		int i;
		__shared__ double x[dim];
		__shared__ double mu[dim];
		__shared__ double sigma[dim * dim];
		__shared__ double inv_sig[dim * dim];
		__shared__ double d[dim];

		// Read data from global to shared mem
		for (i = 0; i < dim; ++i)
		{
			x[i] = samples[lx * dim + i];
			mu[i] = mu_mat[ly * dim + i];
		}
		for (i = 0; i < dim * dim; ++i) sigma[i] = sig_mat[ly * dim * dim + i];


		// determinant and inverse of sigma
		double det = sigma[0] * (sigma[4] * sigma[8] - sigma[5] * sigma[7])
				- sigma[1] * (sigma[3] * sigma[8] - sigma[5] * sigma[6])
				+ sigma[2] * (sigma[3] * sigma[7] - sigma[4] * sigma[6]);

		inv_sig[0] = (sigma[4] * sigma[8] - sigma[5] * sigma[7]) / (double) det;
		inv_sig[1] = (sigma[2] * sigma[7] - sigma[1] * sigma[8]) / (double) det;
		inv_sig[2] = (sigma[1] * sigma[5] - sigma[2] * sigma[4]) / (double) det;
		inv_sig[3] = (sigma[5] * sigma[6] - sigma[3] * sigma[8]) / (double) det;
		inv_sig[4] = (sigma[0] * sigma[8] - sigma[2] * sigma[6]) / (double) det;
		inv_sig[5] = (sigma[2] * sigma[3] - sigma[0] * sigma[5]) / (double) det;
		inv_sig[6] = (sigma[3] * sigma[7] - sigma[4] * sigma[6]) / (double) det;
		inv_sig[7] = (sigma[1] * sigma[6] - sigma[0] * sigma[7]) / (double) det;
		inv_sig[8] = (sigma[0] * sigma[4] - sigma[1] * sigma[3]) / (double) det;

		// diff = x[i] - mu[i]
		for (i = 0; i < dim; ++i) d[i] = x[i] - mu[i];

		// expon = -1/2 * (x - mu)' * inv_sig * (x - mu)
		double expon = ((d[0]*inv_sig[0] + d[1]*inv_sig[3] + d[2]*inv_sig[6]) * d[0]
                      + (d[0]*inv_sig[1] + d[1]*inv_sig[4] + d[2]*inv_sig[7]) * d[1]
                      + (d[0]*inv_sig[2] + d[1]*inv_sig[5] + d[2]*inv_sig[8]) * d[2]) / (double) -2;

		// denom = sqrt((2pi)^dim * det(sigma))
		double denom = sqrt(pow(2 * MATH_PI, dim) * det);

		likelihood[ly * size + lx] = exp(expon) / (double) denom;
	}
	else likelihood[ly * size + lx] = 0;
}

__global__ void w_mvnpdf_dim3(double *likelihood, double *samples, double *mu_mat, double *sig_mat,
		double *weights, int size, int num_gaus)
{
	int lx = blockDim.x * blockIdx.x + threadIdx.x;
	int ly = blockIdx.y;
	if (lx < size && ly < num_gaus)
	{
		int i, dim = 3;
		int os_samp = dim * threadIdx.x;
		int os_sig = dim * dim * threadIdx.x;
		int thr_per_block = blockDim.x * blockDim.y;

		extern __shared__ double sdata[];
		double *x = (double *) &sdata[0];
		double *mu = (double *) &x[dim * thr_per_block];
		double *diff = (double *) &mu[dim * thr_per_block];
		double *sigma = (double *) &diff[dim * thr_per_block];
		double *inv_sig = (double *) &sigma[dim * dim * thr_per_block];

		double weight = weights[ly];

		// Read data from global to shared mem
		for (i = 0; i < dim; ++i)
		{
			x[os_samp + i] = samples[lx * dim + i];
			// x[i] = 12;
			mu[os_samp + i] = mu_mat[ly * dim + i];
		}
		for (i = 0; i < dim * dim; ++i) sigma[os_sig + i] = sig_mat[ly * dim * dim + i];

		// determinant and inverse of sigma
		double det = sigma[os_sig + 0] * (sigma[os_sig + 4] * sigma[os_sig + 8] - sigma[os_sig + 5] * sigma[os_sig + 7])
				   - sigma[os_sig + 1] * (sigma[os_sig + 3] * sigma[os_sig + 8] - sigma[os_sig + 5] * sigma[os_sig + 6])
				   + sigma[os_sig + 2] * (sigma[os_sig + 3] * sigma[os_sig + 7] - sigma[os_sig + 4] * sigma[os_sig + 6]);

		inv_sig[os_sig + 0] = (sigma[os_sig + 4] * sigma[os_sig + 8] - sigma[os_sig + 5] * sigma[os_sig + 7]) / (double) det;
		inv_sig[os_sig + 1] = (sigma[os_sig + 2] * sigma[os_sig + 7] - sigma[os_sig + 1] * sigma[os_sig + 8]) / (double) det;
		inv_sig[os_sig + 2] = (sigma[os_sig + 1] * sigma[os_sig + 5] - sigma[os_sig + 2] * sigma[os_sig + 4]) / (double) det;
		inv_sig[os_sig + 3] = (sigma[os_sig + 5] * sigma[os_sig + 6] - sigma[os_sig + 3] * sigma[os_sig + 8]) / (double) det;
		inv_sig[os_sig + 4] = (sigma[os_sig + 0] * sigma[os_sig + 8] - sigma[os_sig + 2] * sigma[os_sig + 6]) / (double) det;
		inv_sig[os_sig + 5] = (sigma[os_sig + 2] * sigma[os_sig + 3] - sigma[os_sig + 0] * sigma[os_sig + 5]) / (double) det;
		inv_sig[os_sig + 6] = (sigma[os_sig + 3] * sigma[os_sig + 7] - sigma[os_sig + 4] * sigma[os_sig + 6]) / (double) det;
		inv_sig[os_sig + 7] = (sigma[os_sig + 1] * sigma[os_sig + 6] - sigma[os_sig + 0] * sigma[os_sig + 7]) / (double) det;
		inv_sig[os_sig + 8] = (sigma[os_sig + 0] * sigma[os_sig + 4] - sigma[os_sig + 1] * sigma[os_sig + 3]) / (double) det;

		// diff = x[i] - mu[i]
		for (i = 0; i < dim; ++i) diff[os_samp + i] = x[os_samp + i] - mu[os_samp + i];

		// expon = -1/2 * (x - mu)' * inv_sig * (x - mu)
		double expon = (double) (-1) / 2
				  *((diff[os_samp + 0]*inv_sig[os_sig + 0] + diff[os_samp + 1]*inv_sig[os_sig + 3] + diff[os_samp + 2]*inv_sig[os_sig + 6]) * diff[os_samp + 0]
                  + (diff[os_samp + 0]*inv_sig[os_sig + 1] + diff[os_samp + 1]*inv_sig[os_sig + 4] + diff[os_samp + 2]*inv_sig[os_sig + 7]) * diff[os_samp + 1]
                  + (diff[os_samp + 0]*inv_sig[os_sig + 2] + diff[os_samp + 1]*inv_sig[os_sig + 5] + diff[os_samp + 2]*inv_sig[os_sig + 8]) * diff[os_samp + 2]);

		// denom = sqrt((2pi)^dim * det(sigma))
		double denom = sqrt(pow(2 * MATH_PI, dim) * fabs(det));

		double value = weight * exp(expon) / (double) denom;

		if (det == 0) 							// Sig matrix not invertable
			likelihood[ly * size + lx] = 0;
		else if (isinf(value)) 						// Large enough to exceed precision
			likelihood[ly * size + lx] = DBL_MAX;
		else
			likelihood[ly * size + lx] = value;
	}
	else
		likelihood[ly * size + lx] = 0;
}

__global__ void normalization(double *likelihood, int size, int num_gaus)
{
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int lx = blockDim.x * blockIdx.x + threadIdx.x;
	int ly = threadIdx.y;

	if (lx < size && ly < num_gaus)
	{
		int i;
		double sum = 0;
		extern __shared__ double sdata[];

		// read data from global to shared mem
		sdata[tid] = likelihood[ly * size + lx];
		__syncthreads();

		// sum up likelihood values for the same sample
		for (i = 0; i < num_gaus; ++i)
			sum = sum + sdata[i * blockDim.x + threadIdx.x];

		if (sum == 0) sum = 1;
//			likelihood[ly * size + lx] = 1 / (double) num_gaus;
//		else if (likelihood[ly * size + lx] == 0)
//			likelihood[ly * size + lx] = DBL_MIN;
//		else
			likelihood[ly * size + lx] = sdata[tid] / (double) sum;
	}
	else likelihood[ly * size + lx] = 0;
}

/**
 * Host function that copies the data and launches the work on GPU
 */

GaussianParam run_EM(double *samples, int s_size, int s_dim, int num_gaus, double threshold, int max_iter, bool use_timer)
{
	int i, d, dim_squared = s_dim * s_dim;

	// Sizes for memory allocation
	int size_n_gaus = sizeof(double) * num_gaus;
	int size_n_samp = sizeof(double) * s_size;

	int size_sigma = sizeof(double) * dim_squared;
	int size_likelihood = sizeof(double) * num_gaus * s_size;
	int size_mu_mat = sizeof(double) * num_gaus * s_dim;
	int size_sig_mat = sizeof(double) * num_gaus * dim_squared;

	/* Allocate and initialize host (CPU) memory */
	double *likelihood = (double *) malloc(size_likelihood);
	double *likelihood_prev = (double *)malloc(size_likelihood);
	double *weights = (double *)malloc(size_n_gaus);
	double *mu_mat = (double *)malloc(size_mu_mat);
	double *sig_mat = (double *)malloc(size_sig_mat);

	// debug vars
	double *marginals = (double *)malloc(size_n_gaus);
	// double *tmp_data;
	double *d_tmp_data;

	GaussianParam *output = (GaussianParam *)malloc(sizeof(GaussianParam));

	/* Initialize Gaussian params and weights */
	time_t t;
	srand((unsigned)time(&t)); // seed random number generator

	double *inter_sigma, *trans_sigma, *init_sigma;

	for (i = 0; i < num_gaus; i++)
	{
		// Init mu randomly
		for (d = 0; d < s_dim; d++)
			mu_mat[i * s_dim + d] = ((double)rand() / RAND_MAX) * MEAN_PRIOR;

		// Init sigma randomly. To make sigma positive semi-definite, symmetric, sigma = s'*s
		inter_sigma = (double *)malloc(size_sigma);
		for (d = 0; d < dim_squared; d++)
			inter_sigma[d] = ((double)rand() / RAND_MAX) * VAR_PRIOR;
		trans_sigma = transpose(inter_sigma, s_dim, s_dim);
		init_sigma = matrix_mult(trans_sigma, inter_sigma, s_dim, s_dim, s_dim);
		for (d = 0; d < dim_squared; ++d)
			sig_mat[i * dim_squared + d] = init_sigma[d];

		free(inter_sigma);
		free(trans_sigma);
		free(init_sigma);

		// Init weights uniformly
		weights[i] = 1 / (double) num_gaus;
	}

	/* Allocate device memory & copy data from host to device */
	double *d_samples, *d_weights, *d_likelihood, *d_mu_mat, *d_sig_mat, *d_marginals;

	// samples
	CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_samples, size_n_samp * s_dim));
	CUDA_CHECK_RETURN(cudaMemcpy(d_samples, samples, size_n_samp * s_dim, cudaMemcpyHostToDevice));

	// likelihood
	CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_likelihood, size_likelihood));
	CUDA_CHECK_RETURN(cudaMemset(d_likelihood, 0, size_likelihood));

	// weights
	CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_weights, size_n_gaus));
	CUDA_CHECK_RETURN(cudaMemcpy(d_weights, weights, size_n_gaus, cudaMemcpyHostToDevice));

	// mu matrix
	CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_mu_mat, size_mu_mat));
	CUDA_CHECK_RETURN(cudaMemcpy(d_mu_mat, mu_mat, size_mu_mat, cudaMemcpyHostToDevice));

	// sigma matrix
	CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_sig_mat, size_sig_mat));
	CUDA_CHECK_RETURN(cudaMemcpy(d_sig_mat, sig_mat, size_sig_mat, cudaMemcpyHostToDevice));

	// marginals
	CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_marginals, size_n_gaus));
	CUDA_CHECK_RETURN(cudaMemset(d_marginals, 0, size_n_gaus));

	/* Use CUDA event to time kernel */
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msec;

    // Kernel-wise timer
    cudaEvent_t ker_start, ker_stop;
    float ker_msec;
    EmTimer timer;
    if (use_timer)
    {	// Set time accumulators to be 0
        checkCudaErrors(cudaEventCreate(&ker_start));
        checkCudaErrors(cudaEventCreate(&ker_stop));
		timer.mvnpdf = 0;
		timer.normalization = 0;
		timer.marginal_red = 0;
		timer.marginal_red_sig = 0;
		timer.weight_update = 0;
		timer.mu_red = 0;
		timer.mu_red_sig = 0;
		timer.sigma_red = 0;
		timer.sigma_red_sig = 0;
    }

	/* ---------------- Interative update ------------------ */
	double change = 100;
	int iter = 0;
	int size_reduced = (THR_PER_BLOCK + s_size - 1) / THR_PER_BLOCK;
	int size_reduced_sm = (THR_PER_BLOCK_SM + s_size - 1) / THR_PER_BLOCK_SM;

	dim3 grid_dim(1, 1, 1);
	dim3 block_dim(1, 1);
	int size_shared_mem;

	// Start timer
	checkCudaErrors(cudaEventRecord(start, 0));

	while (iter < max_iter)
	{
		//*******************************************//
		//* E-step: Calculate normalized likelihood *//
		//*******************************************//

		/* compute weighted multivariate normal pdf */
		block_dim.x = THR_PER_BLOCK_SM; // Use less threads per block due to the limitation of shared memory size
		block_dim.y = 1;
		grid_dim.x = size_reduced_sm;
		grid_dim.y = num_gaus;
		size_shared_mem = sizeof(double) * THR_PER_BLOCK_SM * (3 * s_dim + 2 * s_dim * s_dim);

		// Start kernel timer
		if (use_timer) checkCudaErrors(cudaEventRecord(ker_start, 0));
		w_mvnpdf_dim3<<<grid_dim, block_dim, size_shared_mem>>>(d_likelihood, d_samples, d_mu_mat, d_sig_mat, d_weights, s_size, num_gaus);

		// Check kernel timer
		if (use_timer)
		{
			checkCudaErrors(cudaEventRecord(ker_stop, 0));
			checkCudaErrors(cudaEventSynchronize(ker_stop));
			checkCudaErrors(cudaEventElapsedTime(&ker_msec, ker_start, ker_stop));
			timer.mvnpdf += ker_msec;
		}
		else cudaDeviceSynchronize();

		/* -- check likelihood
		CUDA_CHECK_RETURN(cudaMemcpy(likelihood, d_likelihood, size_likelihood, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
		std::cout << "Likelihood at iter = " << iter << std::endl; // PRINT
		print_mat(likelihood, num_gaus, s_size);
		std::cout << std::endl;
		*/

		/* compute normalization */
		block_dim.x = (num_gaus + THR_PER_BLOCK - 1) / num_gaus;
		block_dim.y = num_gaus;
		grid_dim.x = (block_dim.x + s_size - 1) / block_dim.x;
		grid_dim.y = 1;
		size_shared_mem = sizeof(double) * block_dim.x * block_dim.y;

		// Start kernel timer
		if (use_timer) checkCudaErrors(cudaEventRecord(ker_start, 0));
		normalization<<<grid_dim, block_dim, size_shared_mem, 0>>>(d_likelihood, s_size, num_gaus);

		// Check kernel timer
		if (use_timer)
		{
			checkCudaErrors(cudaEventRecord(ker_stop, 0));
			checkCudaErrors(cudaEventSynchronize(ker_stop));
			checkCudaErrors(cudaEventElapsedTime(&ker_msec, ker_start, ker_stop));
			timer.normalization += ker_msec;
		}
		else cudaDeviceSynchronize();

		/* -- check normalization
		CUDA_CHECK_RETURN(cudaMemcpy(likelihood, d_likelihood, size_likelihood, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
		std::cout << "Normalized Likelihood at iter = " << iter << std::endl; // PRINT
		print_mat(likelihood, num_gaus, s_size);
		std::cout << std::endl;
		*/

		//*******************************************//
		// *   M-step: Update weights, mus, sigmas  *//
		//*******************************************//
		block_dim.x = THR_PER_BLOCK;
		block_dim.y = 1;
		size_shared_mem = sizeof(double) * THR_PER_BLOCK;

		/* update marginals */
		grid_dim.x = size_reduced;
		grid_dim.y = num_gaus;

		// tmp_data = (double *) malloc(sizeof(double) * grid_dim.x * grid_dim.y);
		CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_tmp_data, sizeof(double) * grid_dim.x * grid_dim.y));

		// Start kernel timer
		if (use_timer) checkCudaErrors(cudaEventRecord(ker_start, 0));
		marginal_reduction<<<grid_dim, block_dim, size_shared_mem>>>(d_likelihood, d_tmp_data, s_size, num_gaus);

		// Check kernel timer
		if (use_timer)
		{
			checkCudaErrors(cudaEventRecord(ker_stop, 0));
			checkCudaErrors(cudaEventSynchronize(ker_stop));
			checkCudaErrors(cudaEventElapsedTime(&ker_msec, ker_start, ker_stop));
			timer.marginal_red += ker_msec;
		}
		else cudaDeviceSynchronize();

		/* -- check tmp data of marginals
		CUDA_CHECK_RETURN(cudaMemcpy(tmp_data, d_tmp_data, sizeof(double) * grid_dim.x * grid_dim.y, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
		std::cout << "tmp marginals at iter = " << iter << std::endl; // PRINT
		print_mat(tmp_data, grid_dim.y, grid_dim.x);
		std::cout << std::endl;
		*/

		// Start kernel timer
		if (use_timer) checkCudaErrors(cudaEventRecord(ker_start, 0));
		marginal_single_reduction<<<num_gaus, size_reduced, sizeof(double) * size_reduced>>>(d_tmp_data, d_marginals);

		// Check kernel timer
		if (use_timer)
		{
			checkCudaErrors(cudaEventRecord(ker_stop, 0));
			checkCudaErrors(cudaEventSynchronize(ker_stop));
			checkCudaErrors(cudaEventElapsedTime(&ker_msec, ker_start, ker_stop));
			timer.marginal_red_sig += ker_msec;
		}
		else cudaDeviceSynchronize();
		CUDA_CHECK_RETURN(cudaFree(d_tmp_data));

		/* -- check marginals
		CUDA_CHECK_RETURN(cudaMemcpy(marginals, d_marginals, size_n_gaus, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
		std::cout << "Marginals at iter = " << iter << std::endl; // PRINT
		print_mat(marginals, 1, num_gaus);
		std::cout << std::endl;
		*/

		/* Update weights */

		// Start kernel timer
		if (use_timer) checkCudaErrors(cudaEventRecord(ker_start, 0));
		weight_update<<<1, num_gaus>>>(d_weights, d_marginals, num_gaus, s_size);

		// Check kernel timer
		if (use_timer)
		{
			checkCudaErrors(cudaEventRecord(ker_stop, 0));
			checkCudaErrors(cudaEventSynchronize(ker_stop));
			checkCudaErrors(cudaEventElapsedTime(&ker_msec, ker_start, ker_stop));
			timer.weight_update += ker_msec;
		}
		// else cudaDeviceSynchronize();

		/* -- check weights
		CUDA_CHECK_RETURN(cudaMemcpy(weights, d_weights, size_n_gaus, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
		std::cout << "Weights at iter = " << iter << std::endl; // PRINT
		print_mat(weights, 1, num_gaus);
		std::cout << std::endl;
		*/

		/* Update mu */
		grid_dim.x = size_reduced;
		grid_dim.y = s_dim;
		grid_dim.z = num_gaus;
		CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_tmp_data, sizeof(double) * grid_dim.x * grid_dim.y * s_dim));

		// Start kernel timer
		if (use_timer) checkCudaErrors(cudaEventRecord(ker_start, 0));
		mu_reduction<<<grid_dim, block_dim, size_shared_mem>>>(d_likelihood, d_samples, d_marginals, d_tmp_data, s_size, num_gaus, s_dim);

		// Check kernel timer
		if (use_timer)
		{
			checkCudaErrors(cudaEventRecord(ker_stop, 0));
			checkCudaErrors(cudaEventSynchronize(ker_stop));
			checkCudaErrors(cudaEventElapsedTime(&ker_msec, ker_start, ker_stop));
			timer.mu_red += ker_msec;
		}
		else cudaDeviceSynchronize();

		grid_dim.x = s_dim;
		grid_dim.y = num_gaus;
		grid_dim.z = 1;

		// Start kernel timer
		if (use_timer) checkCudaErrors(cudaEventRecord(ker_start, 0));
		single_reduction<<<grid_dim, size_reduced, size_shared_mem>>>(d_tmp_data, d_mu_mat);

		// Check kernel timer
		if (use_timer)
		{
			checkCudaErrors(cudaEventRecord(ker_stop, 0));
			checkCudaErrors(cudaEventSynchronize(ker_stop));
			checkCudaErrors(cudaEventElapsedTime(&ker_msec, ker_start, ker_stop));
			timer.mu_red_sig += ker_msec;
		}
		else cudaDeviceSynchronize();
		cudaFree(d_tmp_data);

		/* -- check mu_mat
		CUDA_CHECK_RETURN(cudaMemcpy(mu_mat, d_mu_mat, size_mu_mat, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
		std::cout << "Mu_mat at iter = " << iter << std::endl; // PRINT
		print_mat(mu_mat, num_gaus, s_dim);
		std::cout << std::endl;
		*/

		/* Update sigma */
		grid_dim.x = size_reduced;
		grid_dim.y = s_dim * s_dim;
		grid_dim.z = num_gaus;

		CUDA_CHECK_RETURN(cudaMalloc( (void **) &d_tmp_data, sizeof(double) * grid_dim.x * grid_dim.y * grid_dim.z));

		// Start kernel timer
		if (use_timer) checkCudaErrors(cudaEventRecord(ker_start, 0));
		sigma_reduction<<<grid_dim, block_dim, size_shared_mem>>>(d_likelihood, d_samples, d_marginals, d_mu_mat, d_tmp_data, s_size, num_gaus, s_dim);

		// Check kernel timer
		if (use_timer)
		{
			checkCudaErrors(cudaEventRecord(ker_stop, 0));
			checkCudaErrors(cudaEventSynchronize(ker_stop));
			checkCudaErrors(cudaEventElapsedTime(&ker_msec, ker_start, ker_stop));
			timer.sigma_red += ker_msec;
		}
		else cudaDeviceSynchronize();

		grid_dim.x = s_dim * s_dim;
		grid_dim.y = num_gaus;
		grid_dim.z = 1;

		// Start kernel timer
		if (use_timer) checkCudaErrors(cudaEventRecord(ker_start, 0));
		single_reduction<<<grid_dim, size_reduced, size_shared_mem>>>(d_tmp_data, d_sig_mat);

		// Check kernel timer
		if (use_timer)
		{
			checkCudaErrors(cudaEventRecord(ker_stop, 0));
			checkCudaErrors(cudaEventSynchronize(ker_stop));
			checkCudaErrors(cudaEventElapsedTime(&ker_msec, ker_start, ker_stop));
			timer.sigma_red_sig += ker_msec;
		}
		else cudaDeviceSynchronize();

		cudaFree(d_tmp_data);

		/* -- check sig_mat
		CUDA_CHECK_RETURN(cudaMemcpy(sig_mat, d_sig_mat, size_sig_mat, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
		std::cout << "Sig_mat at iter = " << iter << std::endl; // PRINT
		print_mat(sig_mat, num_gaus, dim_squared);
		std::cout << std::endl;
		*/

		++iter;
	}
	/* --------------------------------------- */

	// Stop timer
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&msec, start, stop));

	/* Print experiment summary */
	printf("All EM kernels: TotTime = %.4f ms; NumIter = %d; TimePerIter = %.4f ms \n", msec, max_iter, msec / max_iter);

	if (use_timer)
	{
		printf("Mvnpdf kernel:                    TotTime = %.4f ms; NumIter = %d; TimePerIter = %.4f ms \n", timer.mvnpdf, max_iter, timer.mvnpdf / max_iter);
		printf("Normalization kernel:             TotTime = %.4f ms; NumIter = %d; TimePerIter = %.4f ms \n", timer.normalization, max_iter, timer.normalization / max_iter);
		printf("Marginal reduction kernel:        TotTime = %.4f ms; NumIter = %d; TimePerIter = %.4f ms \n", timer.marginal_red, max_iter, timer.marginal_red / max_iter);
		printf("Marginal single reduction kernel: TotTime = %.4f ms; NumIter = %d; TimePerIter = %.4f ms \n", timer.marginal_red_sig, max_iter, timer.marginal_red_sig / max_iter);
		printf("Weight update kernel:             TotTime = %.4f ms; NumIter = %d; TimePerIter = %.4f ms \n", timer.weight_update, max_iter, timer.weight_update/ max_iter);
		printf("Mu reduction kernel:              TotTime = %.4f ms; NumIter = %d; TimePerIter = %.4f ms \n", timer.mu_red, max_iter, timer.mu_red_sig/ max_iter);
		printf("Mu single reduction kernel:       TotTime = %.4f ms; NumIter = %d; TimePerIter = %.4f ms \n", timer.mu_red_sig, max_iter, timer.mu_red_sig/ max_iter);
		printf("Sigma reduction kernel:           TotTime = %.4f ms; NumIter = %d; TimePerIter = %.4f ms \n", timer.sigma_red, max_iter, timer.sigma_red / max_iter);
		printf("Sigma single reduction kernel:    TotTime = %.4f ms; NumIter = %d; TimePerIter = %.4f ms \n", timer.sigma_red_sig, max_iter, timer.sigma_red_sig / max_iter);

	}


	/* Copy data from device to host & free device memory */
	// samples
	CUDA_CHECK_RETURN(cudaFree(d_samples));

	// likelihood
	CUDA_CHECK_RETURN(cudaMemcpy(likelihood, d_likelihood, size_likelihood, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(d_likelihood));
	std::cout << "Likelihood:" << std::endl;
	print_mat(likelihood, num_gaus, s_size);

	// weights
	CUDA_CHECK_RETURN(cudaMemcpy(weights, d_weights, size_n_gaus, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(d_weights));
	std::cout << "Weights:" << std::endl;
	print_mat(weights, num_gaus, 1);

	// mu matrix
	CUDA_CHECK_RETURN(cudaMemcpy(mu_mat, d_mu_mat, size_mu_mat, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(d_mu_mat));
	std::cout << "Mu mat:" << std::endl;
	print_mat(mu_mat, num_gaus, s_dim);

	// sigma matrix
	CUDA_CHECK_RETURN(cudaMemcpy(sig_mat, d_sig_mat, size_sig_mat, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaFree(d_sig_mat));
	std::cout << "Sig mat:" << std::endl;
	print_mat(sig_mat, num_gaus, dim_squared);

	// marginals
	CUDA_CHECK_RETURN(cudaFree(d_marginals));

	output->mu = mu_mat;
	output->sigma = sig_mat;

	return output[0];
}

int main_no()
{
	int size = sizeof(double) * 3;
	double h_x[3] = {0.3188, -1.3077, -0.4336};
	double h_mu[3] = {0, 0, 0};
	double h_sig[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
	double pdf;
	double wpdf;
	double weight = 1 / (double) 3;

	double *d_x, *d_mu, *d_sig, *d_pdf, *d_wpdf, *d_weight;
	cudaMalloc((void **)&d_x, size);
	cudaMalloc((void **)&d_mu, size);
	cudaMalloc((void **)&d_sig, size * 3);
	cudaMalloc((void **)&d_pdf, sizeof(double));
	cudaMalloc((void **)&d_wpdf, sizeof(double));
	cudaMalloc((void **)&d_weight, sizeof(double));

	cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mu, h_mu, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sig, h_sig, size * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, &weight, sizeof(double), cudaMemcpyHostToDevice);

	mvnpdf_dim3<<<1, 1>>>(d_pdf, d_x, d_mu, d_sig, 1, 1);

	int shared = sizeof(double) * (3 * 3 + 9 *2);
	w_mvnpdf_dim3<<<1, 1, shared>>>(d_wpdf, d_x, d_mu, d_sig, d_weight, 1, 1);

	cudaMemcpy(&pdf, d_pdf, sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&wpdf, d_wpdf, sizeof(double), cudaMemcpyDeviceToHost);

	std::cout << "pdf = " << pdf << std::endl;
	std::cout << "wpdf = " << wpdf << std::endl;

	return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned :" << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

