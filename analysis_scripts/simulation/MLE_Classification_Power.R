library(data.table)
library(ggplot2)
library(MASS)
library(assertthat)

source('C:/Users/lakin/PycharmProjects/fast-mle-multinomials/analysis_scripts/simulation/MCMC_BLM.R')
TEST_DIR = 'C:/Users/lakin/Documents/1AbdoBioinformatics/Publications/MLE/Simulations/SimulatedClassification/data/'

set.seed(2718)


# Fixed parameters
NUM_CLASSES = 4
NUM_PARAMS = 100
NUM_TEST_OBS = 200
NUM_BOOTSTRAP = 5


# Variable parameters
n_observations = round(seq(2, 202, length.out=5))  # number of vectors (independent observations)
n_observations_imbalanced = round(seq(20, 220, length.out=5))  # number of vectors for class-imbalance
n_draws_mean = round(seq(15, 300, length.out=5))  # average number of draws for each observation
n_param_sd = round(seq(10, 1.5*NUM_PARAMS, length.out=5))  # standard deviation for class parameter Gaussians
n_param_draws = round(seq(50, 1000, length.out=5))  # number of draws for class parameter Gaussians


# MCMC params
n_burn = 200
n_thin = 3
accept_threshold = 0.7


# Lin's similarity measure for categorical simplices.
## Arguments:
## x: vector of frequencies on the unit simplex
## y: vector of frequencies on the unit simplex
## dissimilarity: boolean, return the dissimilarity instead of similarity
lin_sim = function(x, y, dissimilarity=FALSE)
{
  assert_that(length(x) == length(y))
  X = matrix(c(x, y), nrow=2, ncol=length(x), byrow=T)
  X = X[, colSums(X) > 0]
  
  numerator = sum(apply(X, 2, function(col_vec) {
    sim = 0
    if(col_vec[1] == col_vec[2]) {
      sim = 2 * log(col_vec[1])
    }
    else {
      sim = 2 * log(sum(col_vec))
    }
    return(sim)
  }))
  
  denominator = sum(apply(X, 2, function(col_vec) {
    return(sum(log(col_vec)))
  }))
  
  S = numerator / denominator
  return(ifelse(dissimilarity, 1 - S, S))
}


# Compute pairwise vector distance for the rows of a data matrix on the unit simplex
## Arguments:
## X: float matrix, rows are observations on the unit simplex
## FUN: distance function, takes arguments x and y, which are unit simplex vectors
## dissimilarity: boolean, return dissimilarity (distance) instead of similarity
pair_dist = function(X, FUN, dissimilarity=TRUE)
{
  dist_matrix = matrix(0L, nrow=nrow(X), ncol=nrow(X))
  for(i in 1:(nrow(X)-1)) {
    for(j in (i+1):nrow(X)) {
      dist_matrix[i, j] = FUN(X[i, ], X[j, ], dissimilarity=dissimilarity)
    }
  }
  return(dist_matrix)
}


# This function simulates one vector of multinomial parameters for each class.
# The "mean" location of each class will be centered equidistant from one another
# across the parameter vector, and the parameters that "belong" to each class will
# be determined by a Gaussian distribution centered at each class mean, with standard
# deviation param_sd.  The higher param_sd, the more "overlapping" the classes will be,
# and therefore the less power.  N_draws determines how well-characterized each of these
# Gaussian distributions is (density of the vector).  Theoretically, as param_sd decreases
# and n_draws increases, power should increase.  Additive smoothing adds a count to
# each parameter prior to Gaussian draws, such that all parameters are greater than 0.
#
## Arguments:
## n_classes: integer, number of separate classes to simulate [2, n_params]
## n_params: integer, total number of features (multinomial parameters) [4, inf)
## param_sd: integer, standard deviation of parameters for each class,
##           higher values = larger spread for each class (inverse to power) [1, inf)
## n_draws: integer, total number of draws to make from the Gaussian distribution [1, inf)
## additive_smoothing: integer, count to add to every parameter in the vector [1, inf)
multinomial_simulate_class_params = function(n_classes, 
                                             n_params, 
                                             param_sd, 
                                             n_draws, 
                                             additive_smoothing=1)
{
  class_multinomial_simplex = matrix(additive_smoothing, nrow=n_classes, ncol=n_params)
  class_means = floor(seq(1, n_params, length.out = n_classes))
  for(cm in 1:n_classes) {
    rowsum = sum(class_multinomial_simplex[cm, ]) - (n_params * additive_smoothing)
    while(rowsum < n_draws) {
      remainder = n_draws - rowsum
      class_param_counts = table(round(rnorm(remainder, class_means[cm], param_sd)))
      idxs = as.numeric(names(class_param_counts))
      idxs = idxs[idxs > 0 & idxs < n_params]
      class_param_counts = class_param_counts[as.character(idxs)]
      class_multinomial_simplex[cm, idxs] = class_multinomial_simplex[cm, idxs] + as.numeric(class_param_counts)
      rowsum = sum(class_multinomial_simplex[cm, ]) - (n_params * additive_smoothing)
    }
  }
  return(class_multinomial_simplex)
}


# Draw a dataset based on the multinomial distribution from a parameter simplex vector.
# Number of draws per observation is based on a Gaussian distribution.
#
## Arguments:
## params: float vector, multinomial parameters on the unit simplex
## n_observations: integer, number of data vectors to simulate
## n_draws_mean: integer, mean number of draws to make per data vector
## n_draws_sd: float, standard deviation of darws to make per data vector
multinomial_simulate_data = function(params,
                                     n_observations,
                                     n_draws_mean,
                                     n_draws_sd)
{
  dat = matrix(0L, nrow=n_observations, ncol=length(params))
  for(i in 1:n_observations) {
    dat[i, ] = rmultinom(1, size=max(c(2, round(rnorm(1, n_draws_mean, n_draws_sd)))), prob=params)
  }
  return(dat)
}


# Create the datasets used for power analysis of DM and BLM MLE, as well as
# accuracy of multinomial, DM and BLM in classification.  Three parameters will be
# independently assessed:
# 1. Effect size: how overlapping the probability density is for each class, measured by Lin similarity
#                 This is somewhat of a latent variable, since number of draws and param_sd both affect
#                 the Lin similarity score.  So the three truly independent parameters are below.
# 2. Param_SD: standard deviation around the class probability density center (mean).  The param_sd is
#              used to generate more overlap between classes via a Gaussian distribution.
# 3. Num_Draws: number of draws from each multinomial distribution.  This will also affect the probability
#               density and therefore the effect size.  The more draws, the more well-defined the parameter
#               distributions will be, and the less overlap there will be between classes.
# 4. Num_Observations: number of independent multinomial observations per class, adds to sample size and
#                      theoretically should improve the MLE process.
# 5. Num_Data_Draws: number of independent draws from each class
# In order to use the Python MLE code as a black box, the data are output as pseudo-NLP datasets.
# These datasets will then be passed to the Python MLE code for MLE and classification.  The outcomes
# being assessed are classifiation accuracy measures and standard error of the mean around each parameter.
# Parameters used to generate each dataset will be stored in a commented line at the beginning of each
# dataset NLP file.
#
## Arguments:
## X_multinom: float matrix, rows are unit simplices for each class, columns are the multinomial parameters.
## n_obs: vector of integers, number of observations to generate for each class
## n_data_draws: vector of integers, mean number of draws to make for each multinomial data vector
## param_sd: float, standard deviation used to produce params in X_multinom
## param_draws: integer, number of draws used to produce params in X_multinom
## mean_lin: float, mean Lin distance between each class based on X_multinom
## dataset_name: string, file name prefix for this dataset file
## out_dir: string, absolute file path of the output directory for simulation. This directory should include
##          "train" and "test" as subdirectories.
create_multinom_datasets_from_multinom = function(X_multinom, n_obs, n_data_draws, param_sd, 
                                                  param_draws, mean_lin, dataset_name, out_dir)
{
  multinom_train = list()
  multinom_test = list()
  for(i in 1:nrow(X_multinom)) {
      multinom_train[[length(multinom_train)+1]] = multinomial_simulate_data(X_multinom[i, ],
                                                                             n_obs[i],
                                                                             n_data_draws[i],
                                                                             n_data_draws[i] / 3)
      
      multinom_test[[length(multinom_test)+1]] = multinomial_simulate_data(X_multinom[i, ],
                                                                           NUM_TEST_OBS,
                                                                           n_data_draws[i],
                                                                           n_data_draws[i] / 3)
  }
  
  # Train data
  #dataset,param_sd,param_draws,mean_lin_dist,n_obs,n_data_draws,class1_params (comma-separated)
  #...
  #dataset,param_sd,param_draws,mean_lin_dist,n_obs,n_data_draws,classN_params (comma-separated)
  document = c()
  for(class in 1:nrow(X_multinom)) {
    document = c(document, paste('#', paste(dataset_name, param_sd, param_draws, mean_lin, n_obs[class], 
                                 n_data_draws[class], paste(X_multinom[class, ], sep=',', collapse=','), sep=','), sep=''))
  }
  
  for(class in 1:length(multinom_train)) {
    class_label = paste(paste('class', class, sep='_'), '\t', sep='')
    for(obs in 1:nrow(multinom_train[[class]])) {
      class_vec = c()
      for(wc in 1:ncol(multinom_train[[class]])) {
        class_vec = c(class_vec, rep(as.character(wc), multinom_train[[class]][obs, wc]))
      }
      class_vec = paste(class_label, paste(class_vec, collapse=' '), sep='')
      document = c(document, class_vec)
    }
  }
  handle = file(paste(out_dir, 'train', paste(dataset_name, 'train-stemmed.txt', sep='-'), sep='/'))
  writeLines(document, handle)
  close(handle)
  
  
  # Test data
  document = c()
  for(class in 1:length(multinom_test)) {
    class_label = paste(paste('class', class, sep='_'), '\t', sep='')
    for(obs in 1:nrow(multinom_test[[class]])) {
      class_vec = c()
      for(wc in 1:ncol(multinom_test[[class]])) {
        class_vec = c(class_vec, rep(as.character(wc), multinom_test[[class]][obs, wc]))
      }
      class_vec = paste(class_label, paste(class_vec, collapse=' '), sep='')
      document = c(document, class_vec)
    }
  }
  handle = file(paste(out_dir, 'test', paste(dataset_name, 'test-stemmed.txt', sep='-'), sep='/'))
  writeLines(document, handle)
  close(handle)
}


create_blm_datasets_from_multinom = function(X_multinom, n_obs, n_data_draws, param_sd, 
                                             param_draws, mean_lin, dataset_name, out_dir)
{
  blm_train = list()
  blm_test = list()
  blm_param_matrix = matrix(0L, nrow=nrow(X_multinom), ncol=ncol(X_multinom)+1)
  
  for(i in 1:nrow(X_multinom)) {
    blm_params = multinom_mean_from_blm(X_multinom[i, ])
    blm_param_matrix[i, ] = blm_params
    alpha_d = blm_params[1:(length(blm_params)-2)]
    beta = blm_params[length(blm_params)-1]
    alpha = blm_params[length(blm_params)]
    
    blm_train_res = blm_mcmc_sampling(fixed_alpha_d=alpha_d,
                                      fixed_alpha=alpha,
                                      fixed_beta=beta,
                                      n_observations=n_obs[i],
                                      n_draws_mean=n_data_draws[i],
                                      n_draws_sd=n_data_draws[i] / 3,
                                      burn=n_burn,
                                      thin=n_thin,
                                      threshold=accept_threshold)
    
    blm_train[[length(blm_train)+1]] = blm_train_res$accepted_vectors
    
    blm_test_res = blm_mcmc_sampling(fixed_alpha_d=alpha_d,
                                     fixed_alpha=alpha,
                                     fixed_beta=beta,
                                     n_observations=NUM_TEST_OBS,
                                     n_draws_mean=n_data_draws[i],
                                     n_draws_sd=n_data_draws[i] / 3,
                                     burn=n_burn,
                                     thin=n_thin,
                                     threshold=accept_threshold)
    
    blm_test[[length(blm_test)+1]] = blm_test_res$accepted_vectors
  }
  
  #dataset,param_sd,param_draws,mean_lin_dist,n_obs,n_data_draws,class1_params (comma-separated)
  #...
  #dataset,param_sd,param_draws,mean_lin_dist,n_obs,n_data_draws,classN_params (comma-separated)
  document = c()
  for(class in 1:nrow(blm_param_matrix)) {
    document = c(document, paste('#', paste(dataset_name, param_sd, param_draws, mean_lin, n_obs[class], 
                                            n_data_draws[class], paste(blm_param_matrix[class, ], sep=',', collapse=','), sep=','), sep=''))
  }
  
  for(class in 1:length(blm_train)) {
    class_label = paste(paste('class', class, sep='_'), '\t', sep='')
    for(obs in 1:nrow(blm_train[[class]])) {
      class_vec = c()
      for(wc in 1:ncol(blm_train[[class]])) {
        class_vec = c(class_vec, rep(as.character(wc), blm_train[[class]][obs, wc]))
      }
      class_vec = paste(class_label, paste(class_vec, collapse=' '), sep='')
      document = c(document, class_vec)
    }
  }
  handle = file(paste(out_dir, 'train', paste(dataset_name, 'train-stemmed.txt', sep='-'), sep='/'))
  writeLines(document, handle)
  close(handle)
  
  
  # Test data
  document = c()
  for(class in 1:length(blm_test)) {
    class_label = paste(paste('class', class, sep='_'), '\t', sep='')
    for(obs in 1:nrow(blm_test[[class]])) {
      class_vec = c()
      for(wc in 1:ncol(blm_test[[class]])) {
        class_vec = c(class_vec, rep(as.character(wc), blm_test[[class]][obs, wc]))
      }
      class_vec = paste(class_label, paste(class_vec, collapse=' '), sep='')
      document = c(document, class_vec)
    }
  }
  handle = file(paste(out_dir, 'test', paste(dataset_name, 'test-stemmed.txt', sep='-'), sep='/'))
  writeLines(document, handle)
  close(handle)
}


#### Main ####
for(p_sd in n_param_sd) {
  for(p_draw in n_param_draws) {
    for(d_draw in n_draws_mean) {
      for(b in 1:NUM_BOOTSTRAP) {
        vals = multinomial_simulate_class_params(n_classes=NUM_CLASSES, 
                                                 n_params=NUM_PARAMS, 
                                                 param_sd=p_sd, 
                                                 n_draws=p_draw)
        sparse_vals = vals[, colSums(vals) != NUM_CLASSES]
        sparse_vals = t(apply(sparse_vals, 1, function(x) x / sum(x)))
        multi_dist = pair_dist(sparse_vals, lin_sim)
        mean_lin_dist = mean(multi_dist[multi_dist > 0])
        
        # Balanced observations
        for(obs in n_observations) {
          multinom_name = paste('multinom', 'balanced', p_sd, p_draw, d_draw, obs, b, sep='_')
          blm_name = paste('blm', 'balanced', p_sd, p_draw, d_draw, obs, b, sep='_')
          n_obs = rep(obs, NUM_CLASSES)
          
          create_multinom_datasets_from_multinom(X_multinom=vals, 
                                                 n_obs=n_obs, 
                                                 n_data_draws=rep(d_draw, NUM_CLASSES), 
                                                 param_sd=p_sd, 
                                                 param_draws=p_draw, 
                                                 mean_lin=mean_lin_dist, 
                                                 dataset_name=multinom_name, 
                                                 out_dir=TEST_DIR)
          
          create_blm_datasets_from_multinom(X_multinom=vals,
                                            n_obs=n_obs,
                                            n_data_draws=rep(d_draw, NUM_CLASSES),
                                            param_sd=p_sd,
                                            param_draws=p_draw,
                                            mean_lin=mean_lin_dist,
                                            dataset_name=blm_name,
                                            out_dir=TEST_DIR)
        }
        
        # Class-imbalanced observations
        for(obs in n_observations_imbalanced) {
          multinom_name = paste('multinom', 'imbalanced', p_sd, p_draw, d_draw, obs, b, sep='_')
          blm_name = paste('blm', 'imbalanced', p_sd, p_draw, d_draw, obs, b, sep='_')
          n_obs = c()
          for(c in 1:NUM_CLASSES) {
            proposal = round(obs / c)
            n_obs = c(n_obs, ifelse(proposal > 2, proposal, 2))
          }
          n_obs = sample(n_obs)  # shuffle
          
          create_multinom_datasets_from_multinom(X_multinom=vals, 
                                                 n_obs=n_obs, 
                                                 n_data_draws=rep(d_draw, NUM_CLASSES), 
                                                 param_sd=p_sd, 
                                                 param_draws=p_draw, 
                                                 mean_lin=mean_lin_dist, 
                                                 dataset_name=multinom_name, 
                                                 out_dir=TEST_DIR)
          
          create_blm_datasets_from_multinom(X_multinom=vals,
                                            n_obs=n_obs,
                                            n_data_draws=rep(d_draw, NUM_CLASSES),
                                            param_sd=p_sd,
                                            param_draws=p_draw,
                                            mean_lin=mean_lin_dist,
                                            dataset_name=blm_name,
                                            out_dir=TEST_DIR)
          
        }
      }
    }
  }
}

