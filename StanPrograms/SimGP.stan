data {
  int<lower=1> T;   // length of time we are observing over
  real x[T];        // x-axis 1,2,3,...,T-1,T
  real<lower=0> rho;
  real<lower=0> alpha;
}

transformed data {
  matrix[T, T] cov =   cov_exp_quad(x, alpha, rho)
                     + diag_matrix(rep_vector(1e-10, T));
  matrix[T, T] L_cov = cholesky_decompose(cov);
}

parameters {}

transformed parameters {}

model {}

generated quantities {
  
  // gaussian process
  vector[T] f = multi_normal_cholesky_rng(rep_vector(0, T), L_cov);
  
  // gaussian process squared
  vector[T] squared_f = multi_normal_cholesky_rng(rep_vector(0, T), L_cov);
  
}
