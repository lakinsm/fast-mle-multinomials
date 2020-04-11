library(ggplot2)
library(data.table)
library(gridExtra)
library(RColorBrewer)

setwd('C:/Users/lakin/PycharmProjects/fast-mle-multinomials/analytic_data/')
graph_path = 'C:/Users/lakin/Documents/1AbdoBioinformatics/Publications/MLE/Analysis/graphs/'

dat = data.table(read.csv('2020Apr6_mle_results.csv', header=F))
colnames(dat) = c('DataSet', 'PosteriorMethod', 'MLEMethod', 'SmoothingMethod',
                  'PrecomputeMethod', 'ParamString',
                  'Class', 'TruePositives', 'FalsePositives', 'FalseNegatives',
                  'TrueNegatives')


dat[, SensitivityRecall := (100*TruePositives / (TruePositives + FalseNegatives))]
dat[, Specificity := (100*TrueNegatives / (TrueNegatives + FalsePositives))]
dat[, PPVPrecision := (100*TruePositives / (TruePositives + FalsePositives))]
dat[, NPV := (100*TrueNegatives / (TrueNegatives + FalseNegatives))]
dat[, Accuracy := (100*(TruePositives + TrueNegatives) / 
                     (TruePositives + TrueNegatives + FalsePositives + FalseNegatives))]
dat[, F1 := (100*2*PPVPrecision*SensitivityRecall / (PPVPrecision + SensitivityRecall))]

dat$NPV[is.nan(dat$NPV)] = 0
dat$PPVPrecision[is.nan(dat$PPVPrecision)] = 0
dat$F1[is.nan(dat$F1)] = 0

extract_n_param = function(x, i)
{
  ret = c()
  split = strsplit(x, '\\|')
  for(vals in split) {
    if(length(vals) >= i) {
      ret = c(ret, vals[i])
    }
    else {
      ret = c(ret, NA)
    }
  }
  return(ret)
}

dat$ParamString = as.character(dat$ParamString)
dat[, Param1Value := (as.numeric(gsub('.+=', '', extract_n_param(ParamString, 1), perl=T)))]
dat[, Param2Value := (as.numeric(gsub('.+=', '', extract_n_param(ParamString, 2), perl=T)))]


subdat = dat[, .SD, .SDcols=c('DataSet', 'MLEMethod',
                              'SmoothingMethod', 'PrecomputeMethod', 'PosteriorMethod',
                              'TruePositives', 'FalsePositives', 'FalseNegatives',
                              'TrueNegatives', 'Param1Value', 'Param2Value')]
subdat = subdat[, lapply(.SD, sum), by=c('DataSet', 'MLEMethod','PosteriorMethod',
                                         'SmoothingMethod', 'PrecomputeMethod', 
                                         'Param1Value', 'Param2Value')]

subdat[, Precompute_PosteriorMethod := paste(PrecomputeMethod, PosteriorMethod, sep='_')]

subdat[, SensitivityRecall := (100*TruePositives / (TruePositives + FalseNegatives))]
subdat[, Specificity := (100*TrueNegatives / (TrueNegatives + FalsePositives))]
subdat[, PPVPrecision := (100*TruePositives / (TruePositives + FalsePositives))]
subdat[, NPV := (100*TrueNegatives / (TrueNegatives + FalseNegatives))]
subdat[, Accuracy := (100*(TruePositives + TrueNegatives) / 
                     (TruePositives + TrueNegatives + FalsePositives + FalseNegatives))]
subdat[, F1 := (2*PPVPrecision*SensitivityRecall / (PPVPrecision + SensitivityRecall))]

subdat$PPVPrecision[is.nan(subdat$PPVPrecision)] = 0
subdat$F1[is.nan(subdat$F1)] = 0

plot_subdat = subdat[, .SD, .SDcols=!c('TruePositives', 'FalsePositives',
                                       'FalseNegatives', 'TrueNegatives',
                                       'Specificity', 'NPV', 'Accuracy',
                                       'PPVPrecision', 'SensitivityRecall')]
mdat = plot_subdat

mdat$SmoothingMethod = factor(mdat$SmoothingMethod,
                              levels=c('lidstone', 'dirichlet', 'jm', 'ad', 'ts'),
                              ordered=T)

mdat$PrecomputeMethod = factor(mdat$PrecomputeMethod,
                               levels=c('approximate', 'vectorized', 'sklar', 'None'))

mdat$Precompute_PosteriorMethod = factor(mdat$Precompute_PosteriorMethod,
                              levels=c('None_None', 
                                       'approximate_None', 'vectorized_None', 'sklar_None',
                                       'approximate_empirical', 'vectorized_empirical', 'sklar_empirical',
                                       'approximate_aposteriori', 'vectorized_aposteriori', 'sklar_aposteriori'),
                              ordered=T)

mdat$PosteriorMethod = factor(mdat$PosteriorMethod,
                              levels=c('None', 'empirical', 'aposteriori'))

lidstone_g = ggplot(mdat[SmoothingMethod == 'lidstone'], 
                    aes(x=Precompute_PosteriorMethod, y=F1, fill=MLEMethod)) +
  geom_bar(stat='identity', position='dodge') +
  facet_wrap(~DataSet, labeller = label_both, scales = 'free_x') +
  ggtitle('Lidstone Smoothing') +
  scale_fill_brewer(type='qual', palette = 'Dark2') +
  theme(
    plot.title=element_text(hjust=0.5, size=22),
    axis.text.x=element_text(hjust=1, angle=45),
    axis.text=element_text(size=14),
    strip.text=element_text(size=20),
    axis.title=element_text(size=20),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  )
png(paste(graph_path, '2020Apr6_LidstoneMLEResults.png', sep=''), width = 1200, height = 1000)
print(lidstone_g)
dev.off()

dirichlet_g = ggplot(mdat[SmoothingMethod == 'dirichlet'], 
                     aes(x=Precompute_PosteriorMethod, y=F1, fill=MLEMethod)) +
  geom_bar(stat='identity', position='dodge') +
  facet_wrap(~DataSet, labeller = label_both, scales = 'free_x') +
  ggtitle('Dirichlet Smoothing') +
  scale_fill_brewer(type='qual', palette = 'Dark2') +
  theme(
    plot.title=element_text(hjust=0.5, size=22),
    axis.text.x=element_text(hjust=1, angle=45),
    axis.text=element_text(size=14),
    strip.text=element_text(size=20),
    axis.title=element_text(size=20),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  )
png(paste(graph_path, '2020Apr6_DirichletMLEResults.png', sep=''), width = 1200, height = 1000)
print(dirichlet_g)
dev.off()

jm_g = ggplot(mdat[SmoothingMethod == 'jm'], 
              aes(x=Param1Value, y=F1, color=Precompute_PosteriorMethod)) +
  geom_point(size=3, alpha=0.8) +
  facet_wrap(DataSet~MLEMethod, labeller = label_both, scales = 'free_x', ncol=3) +
  ggtitle('JM Smoothing') +
  scale_colour_brewer(type='qual', palette = 'Set1') +
  theme(
    plot.title=element_text(hjust=0.5, size=22),
    axis.text=element_text(size=14),
    strip.text=element_text(size=20),
    axis.title=element_text(size=20),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  )
png(paste(graph_path, '2020Apr6_jmMLEResults.png', sep=''), width = 1200, height = 1000)
print(jm_g)
dev.off()

ad_g = ggplot(mdat[SmoothingMethod == 'ad'], 
              aes(x=log(Param1Value), y=F1, color=Precompute_PosteriorMethod)) +
  geom_point(size=3, alpha=0.8) +
  facet_wrap(DataSet~MLEMethod, labeller = label_both, scales = 'free_x', ncol=3) +
  ggtitle('AD Smoothing') +
  scale_colour_brewer(type='qual', palette = 'Set1') +
  theme(
    plot.title=element_text(hjust=0.5, size=22),
    axis.text=element_text(size=14),
    strip.text=element_text(size=20),
    axis.title=element_text(size=20),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  )
png(paste(graph_path, '2020Apr6_adMLEResults.png', sep=''), width = 1200, height = 1000)
print(ad_g)
dev.off()



