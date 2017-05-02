#define MATH_PI 3.14159265358979323846   // pi
#define MEAN_PRIOR 5;
#define VAR_PRIOR 2;

typedef struct GaussianPara {
	double *mu;
	double *sigma;
} GaussianPara;

double mvnpdf(double *x, double *mu, double *sigma, int dim);

double eval_likelihood(double *prev, double *curr, int m, int n);

GaussianPara *run_EM_cpu(double **samples, int s_size, int s_dim, int num_gaus, double threshold, int max_iter);
