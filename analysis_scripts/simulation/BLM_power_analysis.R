set.seed(100)

source('/mnt/datasets/sampling_methods/BLM/MCMC_BLM.R')
source('/mnt/datasets/sampling_methods/BLM/MLE_BLM.R')

n_iterations = 50  # repetitions of each experiment to get parameter distributions
n_observations = round(seq(2, 202, length.out=10))  # number of vectors (independent observations)
n_draws_mean = round(seq(15, 315, length.out=5))  # average number of draws for each observation
n_draws_sd = 10  # fixed standard deviations for n_draws
n_params = round(seq(4, 504, length.out=10))  # number of multinomial parameters (D+1)

n_burn = 500
n_thin = 10
accept_threshold = 0.7

outputs = c('Type,Iteration,ParamCount,Observations,Draws')
# DiffBetaAlphaRatio,DiffAlpha,DiffBeta,DiffAlphaD   for BLM type
# DiffMultinomialParams   for multinomial type

for(p in n_params) {
  true_params = rnbinom(n=p, size=10, mu=80)
  true_params = c(true_params, 3*sum(true_params) / p)
  alpha_d = true_params[1:(length(true_params)-2)]
  alpha = true_params[length(true_params)]
  beta = true_params[length(true_params)-1]
  
  true_multinom_params = blm_multinom_mean(alpha_d, alpha, beta)
  
  for(o in n_observations) {
    for(d in n_draws_mean) {
      for(iter in 1:n_iterations) {
        sim = blm_mcmc_sampling(fixed_alpha_d = alpha_d,
                                fixed_alpha = alpha,
                                fixed_beta = beta,
                                n_observations = o,
                                n_draws_mean = d,
                                n_draws_sd = n_draws_sd,
                                burn=n_burn,
                                thin=n_thin,
                                threshold=accept_threshold)
        # if(iter == 1) print(sim$chain_acf)
        if(any(colSums(sim$accepted_vectors) == 0)) {
          sim$accepted_vectors = rbind(sim$accepted_vectors, rep(1, ncol(sim$accepted_vectors)))
        }
        
        pre = blm_precalc(sim$accepted_vectors)
        params = blm_init_params(sim$accepted_vectors)
        res = blm_newton_raphson(pre$U, pre$vd, pre$vd1, params)
        multinom_res = blm_multinom_mean(res[1:(length(res)-2)],
                                         res[length(res)],
                                         res[length(res)-1])
        
        param_diffs = res - true_params
        multinom_diffs = multinom_res - true_multinom_params
        
        
        
        exp_string1 = paste(c('BLM', iter, p, o, d,
                              (beta / alpha) - (res[length(res)-1] / res[length(res)]),
                           param_diffs[length(param_diffs)],
                           param_diffs[length(param_diffs)-1], 
                           param_diffs[1:(length(param_diffs)-2)]), sep=',', collapse=',')
        exp_string2 = paste(c('Multinomial', iter, p, o, d, multinom_diffs), sep=',', collapse=',')
        
        print(exp_string1)
        print(exp_string2)
        outputs = c(outputs, exp_string1, exp_string2)
      }
    }
  }
}

handle = file('/mnt/datasets/sampling_methods/BLM/results.csv')
writeLines(outputs, handle)
close(handle)


