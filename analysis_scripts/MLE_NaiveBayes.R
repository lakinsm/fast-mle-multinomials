library(ggplot2)
library(data.table)
library(gridExtra)

setwd('/mnt/phd_repositories/fast-mle-multinomials/analytic_data/')

dat = data.table(read.csv('2020Jan20_mle.txt', header=F))
colnames(dat) = c('DataSet', 'MLEMethod', 'SmoothingMethod', 'ParamString',
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

subdat = dat[, .SD, .SDcols=c('DataSet', 'MLEMethod', 'SmoothingMethod',
                              'TruePositives', 'FalsePositives', 'FalseNegatives',
                              'TrueNegatives', 'Param1Value', 'Param2Value')]
subdat = subdat[, lapply(.SD, sum), by=c('DataSet', 'MLEMethod', 'SmoothingMethod', 
                                         'Param1Value', 'Param2Value')]
subdat[, SensitivityRecall := (100*TruePositives / (TruePositives + FalseNegatives))]
subdat[, Specificity := (100*TrueNegatives / (TrueNegatives + FalsePositives))]
subdat[, PPVPrecision := (100*TruePositives / (TruePositives + FalsePositives))]
subdat[, NPV := (100*TrueNegatives / (TrueNegatives + FalseNegatives))]
subdat[, Accuracy := (100*(TruePositives + TrueNegatives) / 
                     (TruePositives + TrueNegatives + FalsePositives + FalseNegatives))]
subdat[, F1 := (2*PPVPrecision*SensitivityRecall / (PPVPrecision + SensitivityRecall))]

plot_subdat = subdat[, .SD, .SDcols=!c('TruePositives', 'FalsePositives',
                                       'FalseNegatives', 'TrueNegatives',
                                       'Specificity', 'NPV', 'Accuracy',
                                       'PPVPrecision', 'SensitivityRecall')]
# mdat = melt(plot_subdat, id.vars=c('DataSet', 'MLEMethod', 'SmoothingMethod',
#                            'Param1Value', 'Param2Value'))
mdat = plot_subdat

mdat$SmoothingMethod = factor(mdat$SmoothingMethod,
                              levels=c('lidstone', 'dirichlet', 'jm', 'ad', 'ts'),
                              ordered=T)

lidstone_g = ggplot(mdat[SmoothingMethod == 'lidstone'], 
                    aes(x=DataSet, y=F1, fill=MLEMethod)) +
  geom_bar(stat='identity', position='dodge') +
  ggtitle('Lidstone Smoothing')
print(lidstone_g)


dirichlet_g = ggplot(mdat[!(SmoothingMethod %in% c('ts', 'lidstone'))], 
                    aes(x=Param1Value, y=F1, color=MLEMethod)) +
  geom_line() +
  facet_wrap(DataSet~SmoothingMethod)
print(dirichlet_g)



