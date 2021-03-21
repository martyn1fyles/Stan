# Stan

Some code developed to fit negative binomial and poisson distributions on count data observed over time, where the mean and the overdispersion of the process vary over time.

We use Gaussian process prior distributions on the mean and overdispersion.

What's in the repo?
* SimulateNegBinData.Rmd - generate some fake example data, where mean and overdipsersion are modelled using exponentiated random walks and a stan model.
* FitNegBin.Rmd - implements a stan model to fit to the fake generated data using stan, and produces credible intervals etc (needs updating to use cmdstanr)

TODO:
* finish off the poisson implementation
* redo plotting code in ggplot2. We switched to cmdstanr which returns samples in a tidyverse df anyway.
* test removing the lower bound on the gaussian process means. I think they make the model a lot more stable, but unsure exactly.

# Installation notes
* I switched to using cmdstanr, rstan was too unstable on my system and crashed after 50% of MCMC runs
* cmdstanr is not on CRAN, see https://mc-stan.org/cmdstanr/articles/cmdstanr.html for intallation instructions
