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

  real<lower=0> hplindley;
}
transformed data {
  real x_r[0];
  int x_i[0];
  matrix[nobs0 + nobs1, nmaxcomp] phiconcat;
  real minus_hpp_minus_half;
  int<lower=1> nobsconcat;
  real lhplindley[2];

  phiconcat = append_row(phi1, phi2);

  minus_hpp_minus_half = -hpp - 0.5;

  nobsconcat = nobs0 + nobs1;
  lhplindley[1] = log(hplindley);
  lhplindley[2] = log1m(hplindley);
}
parameters {
  vector[nmaxcomp] beta1;
  vector[nmaxcomp] beta2;
  vector[nmaxcomp] betaconcat;
}
transformed parameters {
  vector[2] lpfull;
  real lognormconst1;
  real lognormconst2;
  real lognormconstconcat;

  lognormconst1 =
    log(integrate_ode_rk45(integ, {0.0}, 0, {1.0},
                           to_array_1d(beta1),
                           x_r, x_i, 1.49e-08, 1.49e-08, 1e7)[1,1]);

  lognormconst2 =
    log(integrate_ode_rk45(integ, {0.0}, 0, {1.0},
                           to_array_1d(beta2),
                           x_r, x_i, 1.49e-08, 1.49e-08, 1e7)[1,1]);

  lognormconstconcat =
    log(integrate_ode_rk45(integ, {0.0}, 0, {1.0},
                           to_array_1d(betaconcat),
                           x_r, x_i, 1.49e-08, 1.49e-08, 1e7)[1,1]);

  lpfull[1] = sum(phi1 * beta1) - nobs0 * lognormconst1
              + sum(phi2 * beta2) - nobs1 * lognormconst2
              + lhplindley[1];

  lpfull[2] = sum(phiconcat * betaconcat)
              - nobsconcat * lognormconstconcat
              + lhplindley[2];
}
model {
  target += log_sum_exp(lpfull[1], lpfull[2]);
  for (i in 1:nmaxcomp) {
    if (i % 2 == 0) {
      beta1[i] ~ normal(0, i ^ minus_hpp_minus_half);
      beta2[i] ~ normal(0, i ^ minus_hpp_minus_half);
      betaconcat[i] ~ normal(0, i ^ minus_hpp_minus_half);
    } else {
      beta1[i] ~ normal(0, (i + 1) ^ minus_hpp_minus_half);
      beta2[i] ~ normal(0, (i + 1) ^ minus_hpp_minus_half);
      betaconcat[i] ~ normal(0, (i + 1) ^ minus_hpp_minus_half);
    }
  }
}
generated quantities {
  int<lower=1,upper=2> model_indexfull;
  vector[2] weightsfull;

  weightsfull = softmax(lpfull);
  model_indexfull = categorical_rng(weightsfull);
}
