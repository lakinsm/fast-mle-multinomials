# library(ggplot2)
# library(data.table)
# 
# setwd('/mnt/datasets/sampling_methods/BLM/')
# 
# res = data.table(read.table('results.csv'))

blm = res[grepl('BLM,[0-9]{1,2},4,', res$V1, perl=T), ]
blm[, c('Type', 'Iteration', 'ParamCount', 'Observations', 'Draws',
     'DiffBetaAlphRatio', 'DiffAlpha', 'DiffBeta', 'DiffAlpha1', 'DiffAlpha2',
     'DiffAlpha3') := tstrsplit(V1, ',')]
blm[, Iteration := NULL]
blm[, ParamCount := NULL]
blm[, Type := NULL]
blm$DiffBetaAlphRatio = as.numeric(blm$DiffBetaAlphRatio)
blm$DiffAlpha = as.numeric(blm$DiffAlpha)
blm$DiffBeta = as.numeric(blm$DiffBeta)
blm$DiffAlpha1 = as.numeric(blm$DiffAlpha1)
blm$DiffAlpha2 = as.numeric(blm$DiffAlpha2)
blm$DiffAlpha3 = as.numeric(blm$DiffAlpha3)

mult = res[grepl('Multinomial,[0-9]{1,2},4,', res$V1, perl=T), ]
mult[, c('Type', 'Iteration', 'ParamCount', 'Observations', 'Draws',
        'Diff1', 'Diff2', 'Diff3', 'Diff4') := tstrsplit(V1, ',')]
mult[, Iteration := NULL]
mult[, ParamCount := NULL]
mult[, Type := NULL]
mult$Diff1 = as.numeric(mult$Diff1)
mult$Diff1 = as.numeric(mult$Diff2)
mult$Diff1 = as.numeric(mult$Diff3)
mult$Diff1 = as.numeric(mult$Diff4)

blm_max_draws = blm[Draws == '315',]
obs_lvl = as.numeric(unique(blm_max_draws$Observations))
blm_max_draws$Observations = factor(blm_max_draws$Observations,
                                    levels=as.character(obs_lvl[order(obs_lvl, decreasing=F)]),
                                    ordered=T)
print(tapply(blm_max_draws$DiffAlpha1, blm_max_draws$Observations, function(x) sum(abs(x))))
g1 = ggplot(blm_max_draws, aes(x=DiffAlpha)) +
  geom_histogram() +
  facet_wrap(~Observations) +
  xlim(-5, 600) +
  ggtitle('Draws=315, BLM Alpha1 by Observation')
print(g1)


blm_max_obs = blm[Observations == '202', ]
draw_lvl = as.numeric(unique(blm_max_obs$Draws))
blm_max_obs$Draws = factor(blm_max_obs$Draws,
                           levels=as.character(draw_lvl[order(draw_lvl, decreasing=F)]),
                           ordered=T)
print(tapply(blm_max_obs$DiffAlpha1, blm_max_obs$Draws, function(x) sum(abs(x))))
g2 = ggplot(blm_max_obs, aes(x=DiffAlpha)) +
  geom_histogram() +
  facet_wrap(~Draws) +
  xlim(-5, 600) +
  ggtitle('Obs=202, BLM Alpha1 by Draws')
print(g2)



## Multinomial
mult_max_draws = mult[Draws == '315',]
obs_lvl = as.numeric(unique(mult_max_draws$Observations))
mult_max_draws$Observations = factor(mult_max_draws$Observations,
                                    levels=as.character(obs_lvl[order(obs_lvl, decreasing=F)]),
                                    ordered=T)
print(tapply(mult_max_draws$Diff1, mult_max_draws$Observations, function(x) sum(abs(x))))
g1 = ggplot(mult_max_draws, aes(x=Diff1)) +
  geom_histogram() +
  facet_wrap(~Observations) +
  xlim(-0.1, 0.1) +
  ggtitle('Draws=315, Multinomial Param1 by Observation')
print(g1)


mult_max_obs = mult[Observations == '202', ]
draw_lvl = as.numeric(unique(mult_max_obs$Draws))
mult_max_obs$Draws = factor(mult_max_obs$Draws,
                           levels=as.character(draw_lvl[order(draw_lvl, decreasing=F)]),
                           ordered=T)
print(tapply(mult_max_obs$Diff1, mult_max_obs$Draws, function(x) sum(abs(x))))
g2 = ggplot(mult_max_obs, aes(x=Diff1)) +
  geom_histogram() +
  facet_wrap(~Draws) +
  xlim(-0.1, 0.1) +
  ggtitle('Obs=202,  Multinomial Param1 by Draws')
print(g2)





# print(tapply(blm_max_draws$DiffAlpha1, blm_max_draws$Observations, function(x) sum(abs(x))))
# g1 = ggplot(blm_max_draws, aes(x=DiffAlpha)) +
#   geom_histogram() +
#   facet_wrap(~Observations) +
#   xlim(-5, 2000) +
#   ggtitle('Draws=315, BLM Alpha by Observation')
# print(g1)
# 
# 
# blm_max_obs = blm[Observations == '202', ]
# draw_lvl = as.numeric(unique(blm_max_obs$Draws))
# blm_max_obs$Draws = factor(blm_max_obs$Draws,
#                            levels=as.character(draw_lvl[order(draw_lvl, decreasing=F)]),
#                            ordered=T)
# print(tapply(blm_max_obs$DiffAlpha1, blm_max_obs$Draws, function(x) sum(abs(x))))
# g2 = ggplot(blm_max_obs, aes(x=DiffAlpha)) +
#   geom_histogram() +
#   facet_wrap(~Draws) +
#   xlim(-5, 2500) +
#   ggtitle('Obs=202, BLM Alpha by Draws')
# print(g2)


# print(tapply(blm_max_draws$DiffAlpha1, blm_max_draws$Observations, function(x) sum(abs(x))))
# g1 = ggplot(blm_max_draws, aes(x=DiffAlpha1)) +
#   geom_histogram() +
#   facet_wrap(~Observations) +
#   ggtitle('Draws=315, BLM Alpha1 by Observation')
# print(g1)
