#pragma once

double *matrix_add(double *result, double *A, double *B, int m, int n);
double *matrix_subtr(double *A, double *B, int m, int n);
double *matrix_scalar_mult(double *result, double *matrix, double scalar, int m, int n);
double *matrix_mult(double *A, double *B, int m, int p, int n);
double *transpose(double *matrix, int m, int n);
double *scalar_transpose(double *matrix, int m, int n, double scalar);
double determinant(double*, int);
double *inverse(double*, int);
void print_mat(double *matrix, int m, int n);
