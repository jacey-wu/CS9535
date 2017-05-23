typedef struct GaussianParam {
	double *mu;
	double *sigma;
} GaussianParam;

typedef struct EmTimer {
	float mvnpdf;
	float normalization;
	float marginal_red;
	float marginal_red_sig;
	float weight_update;
	float mu_red;
	float mu_red_sig;
	float sigma_red;
	float sigma_red_sig;
} EmTimer;

GaussianParam run_EM(double *samples, int s_size, int s_dim, int num_gaus, double threshold, int max_iter, bool use_timer);
