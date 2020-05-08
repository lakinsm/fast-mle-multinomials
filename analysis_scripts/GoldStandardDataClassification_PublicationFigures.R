library(data.table)
library(ggplot2)
library(gridExtra)


setwd('C:/Users/lakin/PycharmProjects/fast-mle-multinomials/analytic_data/')

dat = data.table(read.csv('2020Apr15_mle_results.csv', header=F))
colnames(dat) = c('DataSet', 'PosteriorMethod', 'MLEMethod', 'SmoothingMethod', 'PrecomputeMethod', 'ParamString',
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

subdat = dat[, .SD, .SDcols=c('DataSet', 'MLEMethod', 'SmoothingMethod', 'PrecomputeMethod',
                              'PosteriorMethod',
                              'TruePositives', 'FalsePositives', 'FalseNegatives',
                              'TrueNegatives', 'Param1Value', 'Param2Value')]
subdat = subdat[, lapply(.SD, sum), by=c('DataSet', 'MLEMethod', 'SmoothingMethod', 'PosteriorMethod',
                                         'PrecomputeMethod', 'Param1Value', 'Param2Value')]
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

lidstone_g = ggplot(mdat[SmoothingMethod == 'lidstone'], 
                    aes(x=MLEMethod, y=F1, fill=PrecomputeMethod)) +
  geom_bar(stat='identity', position='dodge') +
  facet_wrap(~DataSet) +
  ggtitle('Lidstone Smoothing') +
  theme(plot.title=element_text(hjust=0.5))
print(lidstone_g)

dirichlet_g = ggplot(mdat[SmoothingMethod == 'dirichlet'], 
                     aes(x=Param1Value, y=F1, color=PrecomputeMethod)) +
  geom_point(size=3, alpha=0.8) +
  facet_wrap(MLEMethod~DataSet) +
  ggtitle('JM Smoothing') +
  theme(plot.title=element_text(hjust=0.5))
print(dirichlet_g)

jm_g = ggplot(mdat[SmoothingMethod == 'jm'], 
              aes(x=Param1Value, y=F1, color=PrecomputeMethod)) +
  geom_point(size=3, alpha=0.8) +
  facet_wrap(MLEMethod~DataSet) +
  ggtitle('JM Smoothing') +
  theme(plot.title=element_text(hjust=0.5))
print(jm_g)

ad_g = ggplot(mdat[SmoothingMethod == 'ad'], 
              aes(x=log(Param1Value), y=F1, color=PrecomputeMethod)) +
  geom_point(size=3, alpha=0.8) +
  facet_wrap(MLEMethod~DataSet) +
  ggtitle('AD Smoothing') +
  theme(plot.title=element_text(hjust=0.5))
print(ad_g)


# dirichlet_g = ggplot(mdat[!(SmoothingMethod %in% c('ts', 'lidstone'))], 
#                      aes(x=Param1Value, y=F1, color=MLEMethod)) +
#   geom_line() +
#   facet_wrap(DataSet~SmoothingMethod)
# print(dirichlet_g)

