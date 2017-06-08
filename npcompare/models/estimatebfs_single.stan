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
  int<lower=1> nobs; // number of data points
  matrix[nobs, nmaxcomp] phi;
  real<lower=0> hpp;
}
transformed data {
  real x_r[0];
  int x_i[0];
  real minus_hpp_minus_half;
  minus_hpp_minus_half = -hpp - 0.5;
}
parameters {
  vector[nmaxcomp] beta;
}
transformed parameters {
  real lognormconst;
  lognormconst =
    log(integrate_ode_rk45(integ, {0.0}, 0, {1.0},
                           to_array_1d(beta),
                           x_r, x_i, 1.49e-08, 1.49e-08, 1e7)[1,1]);
}
model {
  target += sum(phi * beta) - nobs * lognormconst;
  for (i in 1:nmaxcomp) {
    if (i % 2 == 0)
      beta[i] ~ normal(0, i ^ minus_hpp_minus_half);
    else
      beta[i] ~ normal(0, (i + 1) ^ minus_hpp_minus_half);
  }
}
