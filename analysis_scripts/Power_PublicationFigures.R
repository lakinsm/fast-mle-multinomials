library(data.table)
library(ggplot2)
library(gridExtra)
library(RColorBrewer)

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

classdat = data.table(read.csv(paste('C:/Users/lakin/Documents/1AbdoBioinformatics/Publications/MLE/Simulations',
                                     '/SimulatedClassification/analytic_data/power_analysis_classification_results.csv', sep=''), 
                               header=F))
# powerdat = data.table(read.csv(paste('C:/Users/lakin/PycharmProjects/fast-mle-multinomials/analytic_data/',
#                                       'simulations/power_analysis_mle_results.csv', sep=''), header=F))

if(colnames(classdat)[1] == 'V1') {
  colnames(classdat) = c('Dataset', 'PosteriorMethod', 'Distribution', 'SmoothingMethod',
                         'PrecomputeMethod', 'ParamString', 'ClassLabel', 'TruePositives',
                         'FalsePositives', 'FalseNegatives', 'TrueNegatives')
  # colnames(powerdat)[1:4] = c('Dataset', 'Distribution', 'ParamOrigin', 'ParamType')
}

classdat[, SensitivityRecall := (100*TruePositives / (TruePositives + FalseNegatives))]
classdat[, Specificity := (100*TrueNegatives / (TrueNegatives + FalsePositives))]
classdat[, PPVPrecision := (100*TruePositives / (TruePositives + FalsePositives))]
classdat[, NPV := (100*TrueNegatives / (TrueNegatives + FalseNegatives))]
classdat[, Accuracy := (100*(TruePositives + TrueNegatives) / 
                          (TruePositives + TrueNegatives + FalsePositives + FalseNegatives))]
classdat[, F1 := (2*PPVPrecision*SensitivityRecall / (PPVPrecision + SensitivityRecall))]

classdat$PPVPrecision[is.nan(classdat$PPVPrecision)] = 0
classdat$F1[is.nan(classdat$F1)] = 0

classdat[, c('DataType', 'Balance', 'ParamSD', 'ParamDraws', 'DataDraws', 'DataObs', 'Bootstrap') 
         := tstrsplit(Dataset, '_', fixed=T)]

classdat[, c('PrecomputeMethod', 'SmoothingMethod', 'ParamString',
             'Balance', 'ParamDraws') := NULL]

classdat$DataDraws = as.numeric(classdat$DataDraws)
classdat$DataObs = as.numeric(classdat$DataObs)
classdat$ParamSD = as.numeric(classdat$ParamSD)

classdat$DataDraws = factor(classdat$DataDraws,
                            levels=sort(unique(classdat$DataDraws)),
                            ordered=T)
classdat$DataObs = factor(classdat$DataObs,
                          levels=sort(as.numeric(unique(classdat$DataObs))),
                          ordered=T)
classdat$ParamSD = factor(classdat$ParamSD,
                          levels=sort(as.numeric(unique(classdat$ParamSD)), decreasing=T),
                          ordered=T)

colnames(classdat)[colnames(classdat) == 'DataObs'] = 'Data Observations'
colnames(classdat)[colnames(classdat) == 'DataDraws'] = 'Data Draws'

multinom = classdat[Distribution == 'pooledDM']
dm = classdat[Distribution == 'DM']
blm = classdat[Distribution == 'BLM']


plotdir = 'C:/Users/lakin/PycharmProjects/fast-mle-multinomials/analytic_data/graphs/'


######### DM NB, Multinomial Data, Balanced #########
g1 = ggplot(multinom, aes(x=ParamSD, y=F1, color=ClassLabel)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score') +
  custom_theme_legend(
    base_size = 23
  ) %+replace%
  theme(
    legend.position = 'bottom'
  )
png(filename=paste0(plotdir, 'S_Figure4_MNB_MultinomialData.png'),
    width=2000, height=2000)
print(g1)
dev.off()


######### DM NB, Multinomial Data, Balanced #########
dm_balanced_multidat = dm[DataType == 'multinom' & PosteriorMethod == 'None']

g1 = ggplot(dm_balanced_multidat, aes(x=ParamSD, y=F1, color=ClassLabel)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score') +
  custom_theme_legend(
    base_size = 23
  ) %+replace%
  theme(
    legend.position = 'bottom'
  )
png(filename=paste0(plotdir, 'S_Figure1_DMNB_MultinomialData.png'),
    width=2000, height=2000)
print(g1)
dev.off()


######### DM NB, BLM Data, Balanced #########
dm_balanced_blmdat = dm[DataType == 'blm' & PosteriorMethod == 'None']

g1 = ggplot(dm_balanced_blmdat, aes(x=ParamSD, y=F1, color=ClassLabel)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score') +
  custom_theme_legend(
    base_size = 23
  ) %+replace%
  theme(
    legend.position = 'bottom'
  )
png(filename=paste0(plotdir, 'S_Figure2_DMNB_BLMData.png'),
    width=2000, height=2000)
print(g1)
dev.off()


######### BLM NB, Multinomial Data, Balanced #########
blm_balanced_multidat = blm[DataType == 'multinom' & PosteriorMethod == 'None']

g1 = ggplot(blm_balanced_multidat, aes(x=ParamSD, y=F1, color=ClassLabel)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score') +
  custom_theme_legend(
    base_size = 23
  ) %+replace%
  theme(
    legend.position = 'bottom'
  )
png(filename=paste0(plotdir, 'Figure3_BLMNB_MultinomialData.png'),
    width=2000, height=2000)
print(g1)
dev.off()


######### BLM NB, BLM Data, Balanced #########
blm_balanced_blmdat = blm[DataType == 'blm' & PosteriorMethod == 'None']

g1 = ggplot(blm_balanced_blmdat, aes(x=ParamSD, y=F1, color=ClassLabel)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score') +
  custom_theme_legend(
    base_size = 23
  ) %+replace%
  theme(
    legend.position = 'bottom'
  )
png(filename=paste0(plotdir, 'S_Figure3_BLMNB_BLMData.png'),
    width=2000, height=2000)
print(g1)
dev.off()


######### DM a posteriori NB, Multinomial Data, Balanced #########
adm_balanced_multidat = dm[DataType == 'multinom' & PosteriorMethod == 'aposteriori']

g1 = ggplot(adm_balanced_multidat, aes(x=ParamSD, y=F1, color=ClassLabel)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score') +
  custom_theme_legend(
    base_size = 23
  ) %+replace%
  theme(
    legend.position = 'bottom'
  )
png(filename=paste0(plotdir, 'S_Figure5_DMANB_MultinomialData.png'),
    width=2000, height=2000)
print(g1)
dev.off()


######### BLM NB, Multinomial Data, Balanced #########
ablm_balanced_multidat = blm[DataType == 'multinom' & PosteriorMethod == 'aposteriori']

g1 = ggplot(ablm_balanced_multidat, aes(x=ParamSD, y=F1, color=ClassLabel)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score') +
  custom_theme_legend(
    base_size = 23
  ) %+replace%
  theme(
    legend.position = 'bottom'
  )
png(filename=paste0(plotdir, 'S_Figure6_BLMANB_MultinomialData.png'),
    width=2000, height=2000)
print(g1)
dev.off()
