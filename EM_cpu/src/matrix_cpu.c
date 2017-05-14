#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>

#include "matrix_cpu.h"

double *matrix_add(double *result, double *A, double *B, int m, int n);
double *matrix_subtr(double *result, double *A, double *B, int m, int n);
double *matrix_scalar_mult(double *result, double *matrix, double scalar, int m, int n);
double *matrix_mult(double *result, double *A, double *B, int m, int p, int n);
double *transpose(double *result, double *matrix, int m, int n);
double *scalar_transpose(double *result, double *matrix, int m, int n, double scalar);
double determinant(double*, int);
double *inverse(double *result, double* matrix, int dim);
void print_mat(double *matrix, int m, int n);

/* Matrix addition */
double *matrix_add(double *result, double *A, double *B, int m, int n)
{
	for (int i = 0; i < m * n; ++i)
		result[i] = A[i] + B[i];
	return result;
}

/* Matrix subtraction */
double *matrix_subtr(double *result, double *A, double *B, int m, int n)
{
	for (int i = 0; i < m * n; ++i)
		result[i] = A[i] - B[i];
	return result;
}

/* Matrix scalar multiplication */
double *matrix_scalar_mult(double *result, double *matrix, double scalar, int m, int n)
{
	int i;
	for (i = 0; i < m * n; i++)
		result[i] = scalar * matrix[i];
	return result;
}

double *matrix_mult(double *result, double *A, double *B, int m, int p, int n)
{
	int i, j, k;
	for (i = 0; i < m; i++)
		for (j = 0; j < n; j++)
		{
			result[i * n + j] = 0;
			for (k = 0; k < p; k++)
				result[i * n + j] += A[i * p + k] * B[k * n + j];
		}
	return result;
}

double *vector_mult(double *u, double *v, int dim)
{
	double * result = (double *)malloc(sizeof(double) * dim);
	for (int i = 0; i < dim; i++)
		result[i] = u[i] * v[i];
	return result;
}

/*For calculating Determinant of the Matrix */
double determinant(double *matrix, int dim)
{
	int s = 1, sub_dim = dim - 1;
	double det = 0;
	int i,j,m,n,c;
	if (dim == 2) return (matrix[0] * matrix[3] - matrix[1] * matrix[2]);

	else
    {
		double *sub_matrix = malloc(sizeof(double) * sub_dim * sub_dim);
		for (c = 0; c < dim; c++)
		{
			m = 0; n = 0;
			for (i = 0; i < dim; i++)
			{
				for (j = 0; j < dim; j++)
				{
					if (i != 0 && j != c)
					{
						sub_matrix[m * sub_dim + n] = matrix[i * dim + j];
						if (n < (dim - 2)) n++;
						else { n = 0; m++; }
                   }
               }
             }
			det = det + s * (matrix[c] * determinant(sub_matrix, sub_dim));
			s = -1 * s;
		}
		free(sub_matrix);
    }
    return det;
}

/* For calculating the inverse of the matrix */
double *inverse(double *inverse_matrix, double *matrix, int dim)
{
	int sub_dim = dim - 1;
	double *sub_matrix = (double *)malloc(sizeof(double) * sub_dim * sub_dim);
	double *inter_matrix = (double *)malloc(sizeof(double) * dim * dim);
	double det = determinant(matrix, dim);

	if (det == 0)
	{
		printf("\nInverse of covariance matrix SIGMA is not possible.\n");
		printf("This may be caused by randomness in parameter initialization. Please restart the program.\n");
		return NULL;
	}

	if (dim == 2)
	{
		inverse_matrix = inter_matrix;
		inverse_matrix[0] =  1 / det * matrix[3];
		inverse_matrix[1] = -1 / det * matrix[1];
		inverse_matrix[2] = -1 / det * matrix[2];
		inverse_matrix[3] =  1 / det * matrix[0];
		free(sub_matrix);
		return inverse_matrix;
	}
	int p,q,m,n,i,j;
	for (p = 0; p < dim; p++)
	{
		for (q = 0; q < dim; q++)
		{
			m=0; n=0;
			for (i = 0; i < dim; i++)
			{
				for (j = 0; j < dim; j++)
				{
					if (i != p && j != q)
					{
						sub_matrix[m * sub_dim + n] = matrix[i * dim + j];
						if (n < (dim - 2)) n++;
						else { n = 0; m++; }
					}
				}
			}
			inter_matrix[p * dim + q] = pow(-1, q + p) * determinant(sub_matrix, sub_dim);
		}
	}
	scalar_transpose(inverse_matrix, inter_matrix, dim, dim, 1 / (double) det);
	free(sub_matrix);
	free(inter_matrix);
	return inverse_matrix;
}

/*Finding transpose of matrix*/
double *transpose(double *result, double *matrix, int m, int n)
{
	return scalar_transpose(result, matrix, m, n, 1);
}

/*Finding transpose of matrix and multiply a scalar to each entry */
double *scalar_transpose(double *result, double *matrix, int m, int n, double scalar)
{
	// double *result = (double *)malloc(sizeof(double) * n * m);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			result[j * m + i] = scalar * matrix[i * n + j];
	return result;
}



/* Print matrix mat to console */
void print_mat(double *matrix, int m, int n)
{
	int i, j;
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < n; j++)
		{
			printf("\t%e", matrix[i * n + j]);
		}
		printf("\n");
	}
}
