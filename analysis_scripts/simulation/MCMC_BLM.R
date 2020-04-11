library(ggplot2)
library(data.table)

set.seed(154)

# True parameter values for alpha_d, alpha, and beta for BLM distribution
alpha_d = c(12, 90, 50)
alpha = 500
beta = 10

# Example test data (multinomial)
# example_X = t(rmultinom(10, 4, prob=c(0.1, 0.5, 0.2, 0.2)))


# Utility functions
dual_lgamma = function(a, n)
{
    i_values = seq(0, n-1)
    return(sum(i_values + a))
}


blm_multinom_mean = function(alpha_d, alpha, beta)
{
    ret = c()
    for(d in 1:length(alpha_d)) {
        ret = c(ret, (alpha / (alpha + beta)) * (alpha_d[d] / sum(alpha_d)))
    }
    ret = c(ret, beta / (alpha + beta))
    return(ret)
}


ggacf <- function(series, title) {
    significance_level = apply(series, 2, function(x) {
        qnorm((1 + 0.95) / 2) / sqrt(sum(!is.na(x)))  
    })
    a = apply(series, 2, function(x) acf(x, plot=F))
    a2 = lapply(a, function(x) data.frame(lag=x$lag, acf=x$acf))
    
    alpha_d_params = ncol(series) - 2
    blm_param_names = c(paste('alpha_', 1:alpha_d_params, sep=''), 'alpha', 'beta')
    
    plot_dat = data.table(lag=numeric(),
                          acf=numeric(),
                          ci.lower=numeric(),
                          ci.upper=numeric(),
                          param=character())
    for(i in 1:length(a2)) {
        plot_dat = rbind(plot_dat, data.table(
            lag=a2[[i]]$lag,
            acf=a2[[i]]$acf,
            param=rep(blm_param_names[i], nrow(a2[[i]])),
            ci.lower=rep(significance_level[i], nrow(a2[[i]])),
            ci.upper=rep(-significance_level[i], nrow(a2[[i]]))
        ))
    }
    
    hline_dat = data.table(param=blm_param_names,
                           ci.lower=-significance_level,
                           ci.upper=significance_level)
    
    g = ggplot(plot_dat, aes(x=lag,y=acf)) + 
        geom_segment(aes(y=0, yend=acf, x=lag, xend=lag)) + xlab('Lag') + ylab('ACF') +
        geom_point(size=1.5) +
        geom_hline(data=hline_dat, aes(yintercept=ci.lower[1]), lty=3, colour='red') +
        geom_hline(data=hline_dat, aes(yintercept=ci.upper[1]), lty=3, colour='red') +
        scale_x_discrete(limits=seq(1, max(plot_dat$lag))) +
        facet_wrap(~param, ncol=1) +
        ggtitle(title)
    return(g)
}


# Likelihood function for one observation of data wrt the BLM distribution
# Based on Lakin & Abdo equation 8
blm_vector_likelihood = function(x, alpha_d, alpha, beta)
{
    term1 = dual_lgamma(1, sum(x))
    term2 = dual_lgamma(alpha, sum(x[1:(length(x)-1)]))
    term3 = dual_lgamma(beta, x[length(x)])
    term4 = dual_lgamma(sum(alpha_d), sum(x[1:(length(x)-1)]))
    
    term5 = rep(0, length(alpha_d))
    for(i in 1:length(alpha_d)) {
        term5[i] = dual_lgamma(alpha_d[i], x[i])
    }
    term5 = sum(term5)
    
    term6 = dual_lgamma(alpha + beta, sum(x))
    
    term7 = rep(0, length(x))
    for(i in 1:length(x)) {
        term7[i] = dual_lgamma(1, x[i])
    }
    term7 = sum(term7)
    
    return(term1 + term2 + term3 - term4 + term5 - term6 - term7)
}


# Log-likelihood of the parameters on flat priors
blm_prior = function(alpha_d, alpha, beta)
{
    ll_alpha_d = sum(dunif(alpha_d, min=0, max=50000, log=TRUE))
    ll_alpha = dunif(alpha, min=0, max=50000, log=TRUE)
    ll_beta = dunif(beta, min=0, max=50000, log=TRUE)
    ll_prior_sum = ll_alpha_d + ll_alpha + ll_beta
    return(ifelse(is.infinite(ll_prior_sum), 0, ll_prior_sum))
}


# Posterior log-likelihood
blm_posterior = function(X, alpha_d, alpha, beta)
{
    if(any(c(alpha_d, alpha, beta) <= 0)) {
        return(0)
    }
    ll_posterior_vectors = c()
    if(is.null(dim(X))) {
        ll_posterior_vectors = blm_vector_likelihood(X, alpha_d, alpha, beta)
    }
    else {
        for(n in 1:nrow(X)) {
            ll_posterior_vectors[n] = blm_vector_likelihood(X[n, ], alpha_d, alpha, beta)
        }
    }
    ll_prior = blm_prior(alpha_d, alpha, beta)
    # print(ll_posterior_vectors)
    # print(ll_prior)
    return(sum(ll_posterior_vectors + ll_prior))
}


#### Metropolis algorithm ####
blm_proposal_function = function(alpha_d, alpha, beta)
{
    proposals = rnorm(length(c(alpha_d, alpha, beta)), 
                      mean=c(alpha_d, alpha, beta), 
                      sd=rep(60, length(c(alpha_d, alpha, beta))))
    return(proposals)
}


blm_mcmc_sampling = function(fixed_alpha_d,
                             fixed_alpha,
                             fixed_beta,
                             n_observations,
                             n_draws_mean,
                             n_draws_sd,
                             burn,
                             thin,
                             threshold,
                             verbose=F)
{
    # fixed_alpha_d = vector of floats (0, 1), fixed parameter values for alpha_ds of BLM
    # fixed_alpha = float (0, 1), fixed parameter value for alpha of BLM
    # fixed_beta = float (0, 1), fixed parameter value for beta of BLM
    # n_observations = integer [1, inf), number of independent vectors to draw for BLM dataset
    # n_draws = integer [1, inf), number of objects drawn from the proposal multinomial distribution
    # burn = integer [0, inf], number of iterations to ignore at the start of MCMC chain
    # threshold = (0, 1), threshold to meet for acceptance of new observation
    dataset = list()
    blm_mean_probs = blm_multinom_mean(fixed_alpha_d, fixed_alpha, fixed_beta)
    
    dataset$accepted_vectors = matrix(0L, nrow=(thin*n_observations)+burn, ncol=length(fixed_alpha_d)+1)
    dataset$n_draws_mean = n_draws_mean
    dataset$n_draws_sd = n_draws_sd
    dataset$n_observations = n_observations
    
    total_iterations = 0
    acceptances = 0
    dataset$accepted_vectors[1, ] = t(rmultinom(1, size=max(c(2, round(rnorm(1, mean=n_draws_mean, sd=n_draws_sd)))), 
                                                prob=blm_mean_probs))
    for(o in 2:((thin*n_observations)+burn)) {
        accept = FALSE
        draw_iteration = 0
        old_ll_blm = blm_vector_likelihood(dataset$accepted_vectors[(o-1), ], fixed_alpha_d, fixed_alpha, fixed_beta)
        old_ll_multinom = dmultinom(dataset$accepted_vectors[(o-1), ], prob=blm_mean_probs, log=TRUE)
        old_ll_ratio = old_ll_blm / old_ll_multinom
        while(!accept) {
            draw_iteration = draw_iteration + 1
            total_iterations = total_iterations + 1
            proposal = t(rmultinom(1, size=max(c(2, round(rnorm(1, mean=n_draws_mean, sd=n_draws_sd)))), 
                                   prob=blm_mean_probs))
            ll_proposal = blm_vector_likelihood(proposal, fixed_alpha_d, fixed_alpha, fixed_beta)
            ll_multinom = dmultinom(proposal, prob=blm_mean_probs, log=TRUE)
            ll_ratio = ll_proposal / ll_multinom
            acceptance_prob = min(c(1, ll_ratio / old_ll_ratio))
            
            if(verbose) {
                cat(sprintf('Obs: %d, Iter: %d, LLR_old: %0.6f, LLR_new: %0.6f,
                          Acceptance: %0.6f, Threshold: %0.6f, Accept: %s, Rate: %0.5f\n', 
                            o,
                            draw_iteration,
                            old_ll_ratio,
                            ll_ratio,
                            acceptance_prob,
                            threshold,
                            acceptance_prob >= threshold,
                            acceptances / total_iterations))
            }
            
            if(acceptance_prob >= threshold) {
                dataset$accepted_vectors[o, ] = proposal
                acceptances = acceptances + 1
                accept = TRUE
            }
        }
    }
    # dataset$burn_plot = plot.ts(dataset$accepted_vectors[1:burn, ])
    dataset$burn_acf = ggacf(dataset$accepted_vectors[1:burn, ], 'Burn-in ACF Plots')
    
    
    dataset$accepted_vectors = dataset$accepted_vectors[seq(burn+1, nrow(dataset$accepted_vectors), thin), ]
    
    # dataset$chain_plot = plot.ts(dataset$accepted_vectors)
    dataset$chain_acf = ggacf(dataset$accepted_vectors, 'Stationary Chain ACF Plots')
    return(dataset)
}



## Main ##
# sim = blm_mcmc_sampling(fixed_alpha_d = alpha_d,
#                         fixed_alpha = alpha,
#                         fixed_beta = beta,
#                         n_observations = 20,
#                         n_draws_mean = 35,
#                         n_draws_sd = 5,
#                         burn=500,
#                         thin=10,
#                         threshold=0.7)
# print(sim$burn_acf)
# print(sim$chain_acf)


# res = blm_rejection_sampling(fixed_alpha_d=alpha_d,
#                              fixed_alpha=alpha,
#                              fixed_beta=beta,
#                              n_observations=20,
#                              n_draws=35,
#                              M=-3000)


# res = run_blm_MCMC(X=example_X,
#                    inits=c(alpha_d, alpha, beta),
#                    n_chains=3,
#                    steps=7000,
#                    burn_in=2000)
# 
# lapply(res, function(L) {
#     colnames(L) = c('alpha_1', 'alpha_2', 'alpha_3', 'alpha', 'beta')
#     plot.ts(L)
# })




