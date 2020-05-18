library(data.table)
library(ggplot2)
library(gridExtra)


setwd('C:/Users/lakin/PycharmProjects/fast-mle-multinomials/analytic_data/')

custom_theme_nolegend = function (base_size = 11, base_family = 'sans', base_line_size = base_size/22, 
                                  base_rect_size = base_size/22) 
{
  theme_grey(base_size = base_size, base_family = base_family, 
             base_line_size = base_line_size, base_rect_size = base_rect_size) %+replace% 
    theme(
      panel.background = element_rect(fill = "white", colour = NA), 
      panel.border = element_rect(fill = NA, colour = "grey20"), 
      panel.grid = element_line(colour = "grey92"), 
      panel.grid.minor = element_line(size = rel(0.5)), 
      strip.background = element_rect(fill = "grey85", colour = "grey20"), 
      legend.key = element_rect(fill = "white", colour = NA), 
      plot.title=element_text(hjust=0.5, vjust=2),
      legend.position='None',
      plot.margin = unit(rep(0.5, 4), "cm"),
      complete = TRUE
    )
}

custom_theme_legend = function(base_size=11, base_family='sans')
{
  custom_theme_nolegend(base_size = base_size, base_family=base_family) %+replace%
    theme(
      legend.position='right',
      legend.text=element_text(size=rel(0.8))
    )
}

extract_highest_f1 = function(X)
{
  return(X[F1 == max(F1)][1])
}


dat = data.table(read.csv('2020May11_mle_results.csv', header=F))
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

plot_subdat2 = plot_subdat[, extract_highest_f1(.SD), by=c('DataSet', 'MLEMethod',
                                                           'SmoothingMethod', 'PosteriorMethod',
                                                           'PrecomputeMethod')]
plot_subdat2[, c('Param1Value', 'Param2Value') := NULL]

mdat = plot_subdat2

smooth_transl = c('lidstone'='Lidstone', 
                  'dirichlet'='Dirichlet', 
                  'jm'='Jelinek-Mercer', 
                  'ad'='Absolute Discounting', 
                  'ts'='Two-Stage')
mdat$SmoothingMethod = smooth_transl[as.character(mdat$SmoothingMethod)]
mdat$SmoothingMethod = factor(mdat$SmoothingMethod,
                              levels=c('Lidstone', 
                                       'Dirichlet', 
                                       'Jelinek-Mercer', 
                                       'Absolute Discounting', 
                                       'Two-Stage'),
                              ordered=T)

posterior_transl = c('None'='Standard Naive Bayes',
                     'empirical'='Empirical Naive Bayes',
                     'aposteriori'='Marginal Likelihood Classification')
mdat$PosteriorMethod = posterior_transl[as.character(mdat$PosteriorMethod)]
mdat$PosteriorMethod = factor(mdat$PosteriorMethod,
                              levels=c('Standard Naive Bayes',
                                       'Empirical Naive Bayes',
                                       'Marginal Likelihood Classification'),
                              ordered=T)


precompute_transl = c('None'='Multinomial',
                      'vectorized'='Vectorized',
                      'approximate'='Approximate',
                      'sklar'='Sklar')
mdat$PrecomputeMethod = precompute_transl[as.character(mdat$PrecomputeMethod)]
mdat$PrecomputeMethod = factor(mdat$PrecomputeMethod,
                               levels=c('Sklar',
                                        'Vectorized',
                                        'Approximate',
                                        'Multinomial'),
                               ordered=T)

mle_transl = c('pooledDM'='Multinomial',
               'DM'='DM',
               'BLM'='BLM')
mdat$MLEMethod = mle_transl[as.character(mdat$MLEMethod)]
mdat$MLEMethod = factor(mdat$MLEMethod,
                        levels=c('DM',
                                 'BLM',
                                 'Multinomial'),
                        ordered=T)

colnames(mdat) = c('DataSet', 'MLEMethod', 'Smoothing', 'Classifier',
                   'PrecomputeMethod', 'F1')

mdat = mdat[Smoothing == 'Lidstone']
mdat[, Smoothing := NULL]

plotdir = 'C:/Users/lakin/PycharmProjects/fast-mle-multinomials/analytic_data/graphs/'

g1 = ggplot(mdat, aes(x=MLEMethod, y=F1, fill=PrecomputeMethod)) +
  geom_bar(stat='identity', position='dodge') +
  geom_text(aes(label=round(F1, 1)), size=6.5, position=position_dodge(width=0.9),
            hjust=0.5, vjust=-0.25) +
  ylim(c(0, 100)) +
  facet_wrap(DataSet~Classifier, labeller = label_both, ncol=2, scales='free_x') +
  scale_fill_brewer('Precompute Method', palette='Set2') +
  # ggtitle('Gold Standard Data Classification Performance') +
  xlab('MLE Method (Distribution)') +
  ylab('F1 Score') +
  custom_theme_legend(
    base_size = 23
  )
png(filename=paste0(plotdir, 'Figure4_GoldStandardData.png'),
    width=2000, height=2000)
print(g1)
dev.off()




## The below are for all smoothing methods
# webkb_g = ggplot(mdat[DataSet == 'webkb'], 
#                     aes(x=MLEMethod, y=F1, fill=PrecomputeMethod)) +
#   geom_bar(stat='identity', position='dodge') +
#   geom_text(aes(label=round(F1, 1)), size=3.5, position=position_dodge(width=1),
#             hjust=0.5, vjust=-0.25) +
#   ylim(c(0, 100)) +
#   facet_wrap(`Classifier`~`Smoothing`, labeller = label_both, ncol=5, scales='free_x') +
#   scale_fill_brewer('Precompute Method', palette='Set2') +
#   ggtitle('WebKB Dataset') +
#   xlab('MLE Method (Distribution)') +
#   ylab('F1 Score') +
#   theme(
#     plot.title=element_text(hjust=0.5),
#     legend.position = 'bottom'
#   )
# print(webkb_g)
# 
# r8_g = ggplot(mdat[DataSet == 'r8'], 
#                  aes(x=MLEMethod, y=F1, fill=PrecomputeMethod)) +
#   geom_bar(stat='identity', position='dodge') +
#   geom_text(aes(label=round(F1, 1)), size=3.5, position=position_dodge(width=1),
#             hjust=0.5, vjust=-0.25) +
#   ylim(c(0, 100)) +
#   facet_wrap(`Classifier`~`Smoothing`, labeller=label_both, ncol=5, scales='free_x') +
#   scale_fill_brewer('Precompute Method', palette='Set2') +
#   ggtitle('R8 Dataset') +
#   xlab('MLE Method (Distribution)') +
#   ylab('F1 Score') +
#   theme(
#     plot.title=element_text(hjust=0.5),
#     legend.position='bottom'
#   )
# print(r8_g)
# 
# cade_g = ggplot(mdat[DataSet == 'cade'], 
#               aes(x=MLEMethod, y=F1, fill=PrecomputeMethod)) +
#   geom_bar(stat='identity', position='dodge') +
#   geom_text(aes(label=round(F1, 1)), size=3.5, position=position_dodge(width=1),
#             hjust=0.5, vjust=-0.25) +
#   ylim(c(0, 100)) +
#   facet_wrap(`Classifier`~`Smoothing`, labeller=label_both, ncol=5, scales='free_x') +
#   scale_fill_brewer('Precompute Method', palette='Set2') +
#   ggtitle('CADE12 Dataset') +
#   xlab('MLE Method (Distribution)') +
#   ylab('F1 Score') +
#   theme(
#     plot.title=element_text(hjust=0.5),
#     legend.position='bottom'
#   )
# print(cade_g)
# 
# ng_g = ggplot(mdat[DataSet == '20ng'], 
#                 aes(x=MLEMethod, y=F1, fill=PrecomputeMethod)) +
#   geom_bar(stat='identity', position='dodge') +
#   geom_text(aes(label=round(F1, 1)), size=3.5, position=position_dodge(width=1),
#             hjust=0.5, vjust=-0.25) +
#   ylim(c(0, 100)) +
#   facet_wrap(`Classifier`~`Smoothing`, labeller=label_both, ncol=5, scales='free_x') +
#   scale_fill_brewer('Precompute Method', palette='Set2') +
#   ggtitle('20 Newsgroups Dataset') +
#   xlab('MLE Method (Distribution)') +
#   ylab('F1 Score') +
#   theme(
#     plot.title=element_text(hjust=0.5),
#     legend.position='bottom'
#   )
# print(ng_g)

