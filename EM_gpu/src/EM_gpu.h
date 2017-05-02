typedef struct GaussianParam {
	double *mu;
	double *sigma;
} GaussianParam;

GaussianParam run_EM(double *samples, int s_size, int s_dim, int num_gaus, double threshold, int max_iter);
