data {
  int<lower=1> T;
  int<lower=1> N;
  matrix[N, T] X;
  int y_obs[N];
  real x[T];
}

transformed data {
}

parameters {

  // mean and dispersion we are trying to estimate
  // non-rescaled
  vector[T] mu_tilde;
  vector[T] phi_tilde;
  
  // mean gaussian process pars
  real<lower=0> rho_mu;
  real<lower=0> alpha_mu;
  
  // dispersion parameter gaussian process
  real<lower=0> rho_phi;
  real<lower=0> alpha_phi;
}

transformed parameters {

  // Covariance for the mean GP
  matrix[T, T] cov_mu =   cov_exp_quad(x, alpha_mu, rho_mu) + diag_matrix(rep_vector(1e-10, T));
  matrix[T, T] L_cov_mu = cholesky_decompose(cov_mu);
  
  // Covariance for the dispersion GP
  matrix[T, T] cov_phi =   cov_exp_quad(x, alpha_phi, rho_phi) + diag_matrix(rep_vector(1e-10, T));
  matrix[T, T] L_cov_phi = cholesky_decompose(cov_phi);
  
  // exponentiated mean gaussian process
  vector[T] mu = exp(rep_vector(0, T) + L_cov_mu * mu_tilde);
  
  // exponentiated dispersion gaussian process
  vector[T] phi = exp(rep_vector(0, T) + L_cov_phi * phi_tilde);
}

model {
  // mu prior
  alpha_mu ~ exponential(3);
  rho_mu ~ exponential(3);
  
  // phi prior
  alpha_phi ~ exponential(3);
  rho_phi ~ exponential(3);
  
  // non-rescaled GP's
  mu_tilde ~ normal(0, 1);
  phi_tilde ~ normal(0, 1);
  
  // design matrix multiplied by mean and dispersion parameters
  y_obs ~ neg_binomial_2(X * mu, X * phi);
}

generated quantities {
  vector[T] mu_predict = mu;
  vector[T] phi_predict = phi;
  int y_predict[T] = neg_binomial_rng(mu, phi);
}