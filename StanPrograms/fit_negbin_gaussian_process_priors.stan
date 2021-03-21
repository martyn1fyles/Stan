data {
  int<lower=1> T;   // length of time we are observing over
  int<lower=1> N;   // total number of observations
  matrix[N, T] X;   // Design matrix, each observation occurs at a single point in time
  int y_obs[N];     // observations
  real x[T];        // x-axis 1,2,3,...,T-1,T
}

transformed data {
  // we need to add to the diagonal of the covariance matrix to guarantee positive
  // definiteness
  vector[T] matrix_diagonal = rep_vector(1e-9, T);
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
  
  // gaussian process means
  real<lower=-10> mu_gp_mean;
  real<lower=-10> phi_gp_mean;
}

transformed parameters {
  // We use an alternative formulation of the Gaussian process
  // Compute the covariance matrix and decompose it using cholesky
  // \Sigma = LL^T
  // let z be a vector of independent normal(0,1) r.v's
  // then x = mu + L*z ~ normal(\mu, LL^T)
  // so  x ~ normal(\mu, \Sigma) where sigma is our covariance matrix
  // since we want a gaussian process the covariance matrix is computed using 
  // we do this because it makes drawing proposal gaussian processes much easier

  // Covariance matrix for the mean GP
  matrix[T, T] cov_mu =   gp_exp_quad_cov(x, alpha_mu, rho_mu);
  // cholesky decompose
  matrix[T, T] L_cov_mu = cholesky_decompose(add_diag(cov_mu, matrix_diagonal));
  
  // Covariance for the dispersion GP
  matrix[T, T] cov_phi =   gp_exp_quad_cov(x, alpha_phi, rho_phi);
  // Cholesky decompose
  matrix[T, T] L_cov_phi = cholesky_decompose(add_diag(cov_phi, matrix_diagonal));
  
  // exponentiated mu gaussian process
  // because we only allow positive values
  vector[T] mu = exp(rep_vector(mu_gp_mean, T) + L_cov_mu * mu_tilde);
  
  // exponentiated dispersion gaussian process
  // because we only allow positive values
  vector[T] phi = exp(rep_vector(phi_gp_mean, T) + L_cov_phi * phi_tilde);
}

model {
  
  // ---- prior distributions ----
  // We assume some mean for the gaussian processes
  mu_gp_mean ~ normal(0, 10);
  phi_gp_mean ~ normal(0, 10);
  
  // mean gaussian process pars
  rho_mu ~ exponential(1);
  alpha_mu ~ exponential(1);
  
  // dispersion parameter gaussian process
  rho_phi ~ exponential(1);
  alpha_phi ~ exponential(1);
  
  // ---- gaussian process proposals ----
  // generate some normal(0,1) data that will be turned into a gaussian process
  // by multiplying using the Cholesky decomposed covariance matrix
  mu_tilde ~ normal(0, 1);
  phi_tilde ~ normal(0, 1);
  
  // ---- observed negative binomial process ----
  // design matrix multiplied by mean and dispersion parameters
  y_obs ~ neg_binomial_2( X * mu, X * phi);
  
}

generated quantities {
  // generate samples of the mean
  vector[T] mu_predict = mu;
  
  // generate samples of the overdispersion
  vector[T] phi_predict = phi;
  
  // generate draws from the negative binomial
  int y_predict[T] = neg_binomial_2_rng(mu, phi);
}
