data {
  int<lower=1> T;  // length of time we observe the process over
  int<lower=1> N;  // number of observations
  matrix[N, T] X;  // design matrix
  vector[T] mu;    // mean over time
  vector[T] phi;   // overdispersion over time
}

transformed data {
  vector[N] mu_pred = X * mu;  // predictor vector
  vector[N] phi_pred = X * phi;// predictor vector
}

parameters {
}

transformed parameters {
}

model {
}

generated quantities {
  // generate some negative binomial data 
  // This is the formulation with multiple observations per time period
  int y_obs[N] = neg_binomial_2_rng(mu_pred, phi_pred);
  
  // generate some samples we can use to describe the credible intervals
  // this is the formulation with only a single obs per time point
  int y[T] = neg_binomial_2_rng(mu, phi);
}
