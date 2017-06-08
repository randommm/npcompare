functions {
  real lkk(real x, real[] beta) {
    real result_;
    result_ = 0;
    for (i in 1:num_elements(beta)) {
      if (i % 2 == 0)
        result_ = result_ + beta[i] * cos(i * pi() * x);
      else
        result_ = result_ + beta[i] * sin((i + 1) * pi() * x);
    }

    return exp(result_ * sqrt2());
  }
  real[] integ(real x, real[] f, real[] beta, real[] x_r, int[] x_i) {
    real dfdx[1];
    dfdx[1] = lkk(x, beta);
    return dfdx;
  }
}
data {
  int<lower=1> nmaxcomp; // number of mixture components
  int<lower=1> nobs0; // number of data points
  int<lower=1> nobs1; // number of data points

  matrix[nobs0, nmaxcomp] phi1;
  matrix[nobs1, nmaxcomp] phi2;
  real<lower=0> hpp;
  real<lower=0> hpgamma;

  real<lower=0> hplindley;
}
transformed data {
  real x_r[0];
  int x_i[0];
  matrix[nobs0 + nobs1, nmaxcomp] phiconcat;
  real minus_hpp_minus_half;
  int<lower=1> nobsconcat;
  real lhplindley[2];
  real minus_hpgamma_times_i[nmaxcomp];

  phiconcat = append_row(phi1, phi2);

  minus_hpp_minus_half = -hpp - 0.5;

  nobsconcat = nobs0 + nobs1;
  lhplindley[1] = log(hplindley);
  lhplindley[2] = log1m(hplindley);

  for (i in 1:nmaxcomp)
    minus_hpgamma_times_i[i] = -hpgamma * i;
}
parameters {
  matrix[nmaxcomp, nmaxcomp] beta1;
  matrix[nmaxcomp, nmaxcomp] beta2;
  matrix[nmaxcomp, nmaxcomp] betaconcat;
}
transformed parameters {
  vector[2] lpfull;
  vector[nmaxcomp] lp1;
  vector[nmaxcomp] lp2;
  vector[nmaxcomp] lpconcat;
  real lognormconst1[nmaxcomp];
  real lognormconst2[nmaxcomp];
  real lognormconstconcat[nmaxcomp];
  for (i in 1:nmaxcomp) {
    lognormconst1[i] =
      log(integrate_ode_rk45(integ, {0.0}, 0, {1.0},
                             to_array_1d(beta1[1:i, i]),
                             x_r, x_i, 1.49e-08, 1.49e-08, 1e7)[1,1]);
    lp1[i] = sum(phi1[, 1:i] * beta1[1:i, i])
             - nobs0 * lognormconst1[i]
             + minus_hpgamma_times_i[i];

    lognormconst2[i] =
      log(integrate_ode_rk45(integ, {0.0}, 0, {1.0},
                             to_array_1d(beta2[1:i, i]),
                             x_r, x_i, 1.49e-08, 1.49e-08, 1e7)[1,1]);
    lp2[i] = sum(phi2[, 1:i] * beta2[1:i, i])
             - nobs1 * lognormconst2[i]
             + minus_hpgamma_times_i[i];

    lognormconstconcat[i] =
      log(integrate_ode_rk45(integ, {0.0}, 0, {1.0},
                             to_array_1d(betaconcat[1:i, i]),
                             x_r, x_i, 1.49e-08, 1.49e-08, 1e7)[1,1]);
    lpconcat[i] = sum(phiconcat[, 1:i] * betaconcat[1:i, i])
                  - nobsconcat * lognormconstconcat[i]
                  + minus_hpgamma_times_i[i];
  }

  lpfull[1] = log_sum_exp(lp1) + log_sum_exp(lp2) + lhplindley[1];
  lpfull[2] = log_sum_exp(lpconcat) + lhplindley[2];
}
model {
  target += log_sum_exp(lpfull[1], lpfull[2]);
  for (i in 1:nmaxcomp) {
    if (i % 2 == 0) {
      beta1[i, ] ~ normal(0, i ^ minus_hpp_minus_half);
      beta2[i, ] ~ normal(0, i ^ minus_hpp_minus_half);
      betaconcat[i, ] ~ normal(0, i ^ minus_hpp_minus_half);
    } else {
      beta1[i, ] ~ normal(0, (i + 1) ^ minus_hpp_minus_half);
      beta2[i, ] ~ normal(0, (i + 1) ^ minus_hpp_minus_half);
      betaconcat[i, ] ~ normal(0, (i + 1) ^ minus_hpp_minus_half);
    }
  }
}
generated quantities {
  int<lower=1,upper=nmaxcomp> model_index1;
  int<lower=1,upper=nmaxcomp> model_index2;
  int<lower=1,upper=nmaxcomp> model_indexconcat;
  int<lower=1,upper=2> model_indexfull;
  vector[nmaxcomp] weights1;
  vector[nmaxcomp] weights2;
  vector[nmaxcomp] weightsconcat;
  vector[2] weightsfull;

  weights1 = softmax(lp1);
  model_index1 = categorical_rng(weights1);
  weights2 = softmax(lp2);
  model_index2 = categorical_rng(weights2);
  weightsconcat = softmax(lpconcat);
  model_indexconcat = categorical_rng(weightsconcat);
  weightsfull = softmax(lpfull);
  model_indexfull = categorical_rng(weightsfull);
}
