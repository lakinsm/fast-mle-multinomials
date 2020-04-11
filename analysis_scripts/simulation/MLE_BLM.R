set.seed(1154)

blm_precalc = function(X)
{
  D = dim(X)[2] - 1
  D1 = D+1
  N = dim(X)[1]
  Zd = max(rowSums(X[, 1:D]))
  Zd1 = max(rowSums(X))
  elem_max = max(X)
  
  # U would be more efficient as a ragged array, but alas this is R
  U = matrix(0L, nrow=D1, ncol=elem_max)
  vd = rep(0, Zd)
  vd1 = rep(0, Zd1)
  
  # Algorithm 1
  for(n in 1:N) {
    C = 0
    for(d in 1:D) {
      C = C + X[n, d]
      for(i in 1:(X[n, d])) {
        U[d, i] = U[d, i] + 1
      }
    }
    for(i in 1:C) {
      vd[i] = vd[i] + 1
    }
    
    C = C + X[n, D1]
    for(i in 1:X[n, D1]) {
      U[D1, i] = U[D1, i] + 1
    }
    for(i in 1:C) {
      vd1[i] = vd1[i] + 1
    }
  }
  return(list(U=U, vd=vd, vd1=vd1))
}


blm_init_params = function(X)
{
  Xnorm = t(apply(X, 1, function(x) x / sum(x)))
  mean = apply(Xnorm, 2, mean)
  m2 = apply(Xnorm, 2, function(x) mean(x^2))
  nonzeros = mean > 0
  sum_alpha = (mean[nonzeros] - m2[nonzeros]) / (m2[nonzeros] - ((mean[nonzeros])^2))
  sum_alpha[is.infinite(sum_alpha)] = max(sum_alpha[!is.infinite(sum_alpha)])  # fix zero divisions
  sum_alpha[sum_alpha == 0] = 1  # required to prevent zero division
  var_pk = (mean * (1 - mean)) / (1 + sum_alpha)
  alpha = abs(((mean * (1 - mean)) / var_pk) - 1)
  log_sum_alpha = ((ncol(X) - 1)^(-1)) * sum(log(alpha))
  s = exp(log_sum_alpha)
  if(s == 0) {
    s = 1
  }
  d_params = s * mean
  return(c(d_params, sum(d_params[1:(length(d_params)-1)])))
}


blm_hessian_precompute = function(U, vd, vd1, theta, verbose=F)
{
  if(any(theta <= 0)) {
    print('ERROR, Params must be positive:')
    print(theta)
    stop()
  }
  
  D1 = length(theta) - 1
  D2 = D1 + 1
  Zd = length(vd)
  Zd1 = length(vd1)
  
  u_oob = ncol(U) + 1
  
  lprob = 0
  gradient = rep(0, D2)
  h_diag = rep(0, D2)
  constants = rep(0, 2)
  sum_theta = sum(theta[1:(D1-1)])
  
  # Zd terms
  for(z in 0:(Zd-1)) {
    if(verbose) {
      if(z %% 1000 == 0 & z != 0) {
        print(sprintf('Precompute: %d / %d', z, Zd1))
      }
    }
    
    idx = z + 1  # for R's 1-based indexing
    # alpha_d params
    for(d in 1:(D1-1)) {
      if(idx < u_oob) {
        lprob = lprob + (U[d, idx] * log(theta[d] + z))
        gradient[d] = gradient[d] + (U[d, idx] * ((theta[d] + z)^(-1))) 
        h_diag[d] = h_diag[d] - (U[d, idx] * ((theta[d] + z)^(-2)))
      }
      gradient[d] = gradient[d] - (vd[idx] * ((sum_theta + z)^(-1)))
    }
    lprob = lprob - (vd[idx] * log(sum_theta + z))
    
    # beta param
    if(idx < u_oob) {
      lprob = lprob + (U[D1, idx] * log(theta[D1] + z))
      gradient[D1] = gradient[D1] + (U[D1, idx] * ((theta[D1] + z)^(-1)))
      h_diag[D1] = h_diag[D1] - (U[D1, idx] * ((theta[D1] + z)^(-2)))
    }
    gradient[D1] = gradient[D1] - (vd1[idx] * ((theta[D1] + theta[D2] + z)^(-1)))
    
    
    # alpha param
    h_diag[D2] = h_diag[D2] - (vd[idx] * ((theta[D2] + z)^(-2)))
    gradient[D2] = gradient[D2] + (vd[idx] * ((theta[D2] + z)^(-1))) - (vd1[idx] * ((theta[D1] + theta[D2] + z)^(-1)))
    lprob = lprob + (vd[idx] * log(theta[D2] + z)) - (vd1[idx] * log(theta[D1] + theta[D2] + z))
    
    # constants
    constants[1] = constants[1] + (vd[idx] * ((sum_theta + z)^(-2)))
    constants[2] = constants[2] + (vd1[idx] * ((theta[D1] + theta[D2] + z)^(-2)))
  }
  
  # Zd1 terms
  if(Zd < Zd1) {
    for(z in Zd:(Zd1-1)) {
      if(verbose) {
        if(z %% 1000 == 0 & z != 0) {
          print(sprintf('Precompute: %d / %d', z, Zd1))
        }
      }
      
      idx = z + 1
      
      # beta param
      if(idx < u_oob) {
        lprob = lprob + (U[D1, idx] * log(theta[D1] + z))
        gradient[D1] = gradient[D1] + (U[D1, idx] * ((theta[D1] + z)^(-1)))
        h_diag[D1] = h_diag[D1] - (U[D1, idx] * ((theta[D1] + z)^(-2)))
      }
      gradient[D1] = gradient[D1] - (vd1[idx] * ((theta[D1] + theta[D2] + z)^(-1)))
      
      
      # alpha param
      gradient[D2] = gradient[D2] - (vd1[idx] * ((theta[D1] + theta[D2] + z)^(-1)))
      lprob = lprob - (vd1[idx] * log(theta[D1] + theta[D2] + z))
      
      # constants
      constants[2] = constants[2] + (vd1[idx] * ((theta[D1] + theta[D2] + z)^(-2)))
    }
  }
  
  if(verbose) {
    print(sprintf('Precompute: %d / %d', Zd1, Zd1))
  }
  return(list(g=gradient, h=h_diag, c=constants, lprob=lprob))
}


blm_step = function(h, g, c)
{
  D = length(g) - 2
  D1 = D + 1
  D2 = D + 2
  deltas = rep(0, D)
  h_inv = h^(-1)
  for(d in 1:D) {
    deltas[d] = h[d] * (g[d] - ((g[1:D] %*% h[1:D]) / ((c[1]^(-1)) + sum(h[1:D]))))
  }
  delta_beta = h[D1] * (g[D1] - (((g[D1] * h[D1]) + (g[D2] * h[D2])) / ((c[2]^(-1)) + h[D1] + h[D2])))
  delta_alpha = h[D2] * (g[D2] - (((g[D1] * h[D1]) + (g[D2] * h[D2])) / ((c[2]^(-1)) + h[D1] + h[D2])))
  return(c(deltas, delta_beta, delta_alpha))
}


blm_newton_raphson = function(U, vd, vd1, params,
                              max_steps=1000,
                              delta_eps_threshold=1e-10,
                              delta_lprob_threshold=1e-10,
                              verbose=F)
{
  current_lprob = -2e20
  delta_lprob = 2e20
  delta_params = 2e20
  
  local_params = params
  D = length(params) - 2
  D1 = D + 1
  D2 = D + 2
  
  step = 0
  while((delta_params > delta_eps_threshold) & (step < max_steps) & (delta_lprob > delta_lprob_threshold)) {
    step = step + 1
    hes_precomp = blm_hessian_precompute(U, vd, vd1, local_params)
    delta_lprob = abs(hes_precomp$lprob - current_lprob)
    current_lprob = hes_precomp$lprob
    deltas = blm_step(hes_precomp$h, hes_precomp$g, hes_precomp$c)
    delta_params = sum(abs(deltas[1:D])) + (deltas[D1] / (deltas[D1] + deltas[D2]))
    
    if(verbose) {
      cat(sprintf('Step: %d\tLprob: %0.2f\tDelta Lprob: %0.2f\n',
                  step, hes_precomp$lprob, delta_lprob))
      cat(sprintf('Delta Sum Eps: %0.2f\n', delta_params))
      print(local_params)
      print(deltas)
    }
    
    local_params = local_params - deltas
    
    if(any(local_params < 0)) {
      print('NEGATIVE PARAMS DETECTED')
      local_params = local_params + deltas
      break
    }
  }
  if(verbose) {
    cat(sprintf('BL MLE Exiting, Total steps: %d / %d\n', step, max_steps))
  }
  return(local_params)
}


# pre = blm_precalc(sim$accepted_vectors)
# params = blm_init_params(sim$accepted_vectors)
# res = blm_newton_raphson(pre$U, pre$vd, pre$vd1, params)


