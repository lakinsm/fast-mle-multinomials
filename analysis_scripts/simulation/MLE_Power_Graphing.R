library(data.table)
library(ggplot2)
library(gridExtra)
library(RColorBrewer)
library(scatterplot3d)

classdat = data.table(read.csv(paste('C:/Users/lakin/Documents/1AbdoBioinformatics/Publications/MLE/Simulations',
                               '/SimulatedClassification/analytic_data/power_analysis_classification_results.csv',
                               sep=''), header=F))
powerdat = data.table(read.csv(paste('C:/Users/lakin/Documents/1AbdoBioinformatics/Publications/MLE/Simulations',
                                     '/SimulatedClassification/analytic_data/power_analysis_mle_results.csv',
                                     sep=''), header=F))
lin_metadata = data.table(read.csv(paste('C:/Users/lakin/PycharmProjects/fast-mle-multinomials/analytic_data/',
                                         'simulated_data_lin_scores.csv',
                                         sep=''), header=F))

if(colnames(classdat)[1] == 'V1') {
  colnames(classdat) = c('Dataset', 'PosteriorMethod', 'Distribution', 'SmoothingMethod',
                         'PrecomputeMethod', 'ParamString', 'ClassLabel', 'TruePositives',
                         'FalsePositives', 'FalseNegatives', 'TrueNegatives')
  colnames(powerdat)[1:4] = c('Dataset', 'Distribution', 'ParamOrigin', 'ParamType')
  colnames(lin_metadata) = c('Dataset', 'LinDistance')
}


# p_sd, p_draw, d_draw, obs, b
lin_metadata[, c('DataType', 'Balance', 'ParamSD', 'ParamDraws', 'DataDraws',
                 'DataObs', 'Bootstrap') := tstrsplit(Dataset, '_', fixed=T)]

setkey(lin_metadata, Dataset)
setkey(classdat, Dataset)
setkey(powerdat, Dataset)

classdat = lin_metadata[classdat]
powerdat = lin_metadata[powerdat]

classdat[, SensitivityRecall := (100*TruePositives / (TruePositives + FalseNegatives))]
classdat[, Specificity := (100*TrueNegatives / (TrueNegatives + FalsePositives))]
classdat[, PPVPrecision := (100*TruePositives / (TruePositives + FalsePositives))]
classdat[, NPV := (100*TrueNegatives / (TrueNegatives + FalseNegatives))]
classdat[, Accuracy := (100*(TruePositives + TrueNegatives) / 
                        (TruePositives + TrueNegatives + FalsePositives + FalseNegatives))]
classdat[, F1 := (2*PPVPrecision*SensitivityRecall / (PPVPrecision + SensitivityRecall))]

classdat$PPVPrecision[is.nan(classdat$PPVPrecision)] = 0
classdat$F1[is.nan(classdat$F1)] = 0

classdat[, c('PrecomputeMethod', 'SmoothingMethod', 'ParamString') := NULL]

nclr = 11
plotclr = brewer.pal(nclr, 'RdBu')
colornum = cut(rank(classdat$LinDistance), nclr, labels=F)

classdat[, LinDistancePlotColor := plotclr[colornum]]

classdat$DataDraws = as.numeric(classdat$DataDraws)
classdat$DataObs = as.numeric(classdat$DataObs)
classdat$ParamDraws = as.numeric(classdat$ParamDraws)
classdat$ParamSD = as.numeric(classdat$ParamSD)

classdat$DataDraws = factor(classdat$DataDraws,
                            levels=sort(unique(classdat$DataDraws)),
                            ordered=T)
classdat$DataObs = factor(classdat$DataObs,
                            levels=sort(as.numeric(unique(classdat$DataObs))),
                            ordered=T)
classdat$ParamDraws = factor(classdat$ParamDraws,
                          levels=sort(as.numeric(unique(classdat$ParamDraws))),
                          ordered=T)
classdat$ParamSD = factor(classdat$ParamSD,
                             levels=sort(as.numeric(unique(classdat$ParamSD)), decreasing=T),
                             ordered=T)

colnames(classdat)[colnames(classdat) == 'DataObs'] = 'Data Observations'
colnames(classdat)[colnames(classdat) == 'DataDraws'] = 'Data Draws'

multinom = classdat[Distribution == 'pooledDM']
dm = classdat[Distribution == 'DM']
blm = classdat[Distribution == 'BLM']


plotdir = paste('C:/Users/lakin/Documents/1AbdoBioinformatics/Publications/MLE/Simulations',
                '/SimulatedClassification/graphs/',
                sep='')




######### Multinomial NB, Multinomial Data, Balanced #########
multinom_balanced_multidat = multinom[Balance == 'balanced' & DataType == 'multinom']

g1 = ggplot(multinom_balanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'MultinomialNB_MultinomialData_Balanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(multinom_balanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Multinomial Naive Bayes on Multinomial Data with a Balanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'MultinomialNB_MultinomialData_Balanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()







######### Multinomial NB, Multinomial Data, Imbalanced #########
multinom_imbalanced_multidat = multinom[Balance == 'imbalanced' & DataType == 'multinom']

g1 = ggplot(multinom_imbalanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'MultinomialNB_MultinomialData_Imbalanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(multinom_imbalanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Multinomial Naive Bayes on Multinomial Data with an Imbalanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'MultinomialNB_MultinomialData_Imbalanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()






######### Multinomial NB, BLM Data, Balanced #########
multinom_balanced_blmdat = multinom[Balance == 'balanced' & DataType == 'blm']

g1 = ggplot(multinom_balanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'MultinomialNB_BLMData_Balanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(multinom_balanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Multinomial Naive Bayes on BLM Data with a Balanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'MultinomialNB_BLMData_Balanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()









######### Multinomial NB, BLM Data, Imbalanced #########
multinom_imbalanced_blmdat = multinom[Balance == 'imbalanced' & DataType == 'blm']

g1 = ggplot(multinom_imbalanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'MultinomialNB_BLMData_Imbalanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(multinom_imbalanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Multinomial Naive Bayes on BLM Data with an Imbalanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'MultinomialNB_BLMData_Imbalanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()



####


######### DM NB, Multinomial Data, Balanced #########
dm_balanced_multidat = dm[Balance == 'balanced' & DataType == 'multinom' & PosteriorMethod == 'None']

g1 = ggplot(dm_balanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMNB_MultinomialData_Balanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(dm_balanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Dirichlet Multinomial Naive Bayes on Multinomial Data with a Balanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMNB_MultinomialData_Balanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()







######### DM NB, Multinomial Data, Imbalanced #########
dm_imbalanced_multidat = dm[Balance == 'imbalanced' & DataType == 'multinom' & PosteriorMethod == 'None']

g1 = ggplot(dm_imbalanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMNB_MultinomialData_Imbalanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(dm_imbalanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Dirichlet Multinomial Naive Bayes on Multinomial Data with an Imbalanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMNB_MultinomialData_Imbalanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()






######### DM NB, BLM Data, Balanced #########
dm_balanced_blmdat = dm[Balance == 'balanced' & DataType == 'blm' & PosteriorMethod == 'None']

g1 = ggplot(dm_balanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMNB_BLMData_Balanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(dm_balanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Dirichlet Multinomial Naive Bayes on BLM Data with a Balanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMNB_BLMData_Balanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()




######### DM NB, BLM Data, Imbalanced #########
dm_imbalanced_blmdat = dm[Balance == 'imbalanced' & DataType == 'blm' & PosteriorMethod == 'None']

g1 = ggplot(dm_imbalanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMNB_BLMData_Imbalanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(dm_imbalanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Dirichlet Multinomial Naive Bayes on BLM Data with an Imbalanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMNB_BLMData_Imbalanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()








##

######### DM Empirical NB, Multinomial Data, Balanced #########
dme_balanced_multidat = dm[Balance == 'balanced' & DataType == 'multinom' & PosteriorMethod == 'empirical']

g1 = ggplot(dme_balanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMEmpiricalNB_MultinomialData_Balanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(dme_balanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Dirichlet Multinomial Empirical Naive Bayes on Multinomial Data with a Balanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMEmpiricalNB_MultinomialData_Balanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()







######### DM Empirical NB, Multinomial Data, Imbalanced #########
dme_imbalanced_multidat = dm[Balance == 'imbalanced' & DataType == 'multinom' & PosteriorMethod == 'empirical']

g1 = ggplot(dme_imbalanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMEmpiricalNB_MultinomialData_Imbalanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(dme_imbalanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Dirichlet Multinomial Empirical Naive Bayes on Multinomial Data with an Imbalanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMEmpiricalNB_MultinomialData_Imbalanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()






######### DM Empirical NB, BLM Data, Balanced #########
dme_balanced_blmdat = dm[Balance == 'balanced' & DataType == 'blm' & PosteriorMethod == 'empirical']

g1 = ggplot(dme_balanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMEmpiricalNB_BLMData_Balanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(dme_balanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Dirichlet Multinomial Empirical Naive Bayes on BLM Data with a Balanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMEmpiricalNB_BLMData_Balanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()




######### DM Empirical NB, BLM Data, Imbalanced #########
dme_imbalanced_blmdat = dm[Balance == 'imbalanced' & DataType == 'blm' & PosteriorMethod == 'empirical']

g1 = ggplot(dme_imbalanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMEmpiricalNB_BLMData_Imbalanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(dme_imbalanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Dirichlet Multinomial Empirical Naive Bayes on BLM Data with an Imbalanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMEmpiricalNB_BLMData_Imbalanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()








##

######### DM Aposteriori NB, Multinomial Data, Balanced #########
dma_balanced_multidat = dm[Balance == 'balanced' & DataType == 'multinom' & PosteriorMethod == 'aposteriori']

g1 = ggplot(dma_balanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMAposterioriNB_MultinomialData_Balanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(dma_balanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Dirichlet Multinomial A Posteriori Naive Bayes on Multinomial Data with a Balanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMAposterioriNB_MultinomialData_Balanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()







######### DM Aposteriori NB, Multinomial Data, Imbalanced #########
dma_imbalanced_multidat = dm[Balance == 'imbalanced' & DataType == 'multinom' & PosteriorMethod == 'aposteriori']

g1 = ggplot(dma_imbalanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMAposterioriNB_MultinomialData_Imbalanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(dma_imbalanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Dirichlet Multinomial A Posteriori Naive Bayes on Multinomial Data with an Imbalanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMAposterioriNB_MultinomialData_Imbalanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()






######### DM Aposteriori NB, BLM Data, Balanced #########
dma_balanced_blmdat = dm[Balance == 'balanced' & DataType == 'blm' & PosteriorMethod == 'aposteriori']

g1 = ggplot(dma_balanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMAposterioriNB_BLMData_Balanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(dma_balanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Dirichlet Multinomial A Posteriori Naive Bayes on BLM Data with a Balanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMAposterioriNB_BLMData_Balanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()




######### DM Aposteriori NB, BLM Data, Imbalanced #########
dma_imbalanced_blmdat = dm[Balance == 'imbalanced' & DataType == 'blm' & PosteriorMethod == 'aposteriori']

g1 = ggplot(dma_imbalanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMAposterioriNB_BLMData_Imbalanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(dma_imbalanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Dirichlet Multinomial A Posteriori Naive Bayes on BLM Data with an Imbalanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'DMAposterioriNB_BLMData_Imbalanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()


















####


######### BLM NB, Multinomial Data, Balanced #########
blm_balanced_multidat = blm[Balance == 'balanced' & DataType == 'multinom' & PosteriorMethod == 'None']

g1 = ggplot(blm_balanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMNB_MultinomialData_Balanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(blm_balanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Beta-Liouville Multinomial Naive Bayes on Multinomial Data with a Balanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMNB_MultinomialData_Balanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()







######### BLM NB, Multinomial Data, Imbalanced #########
blm_imbalanced_multidat = blm[Balance == 'imbalanced' & DataType == 'multinom' & PosteriorMethod == 'None']

g1 = ggplot(blm_imbalanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMNB_MultinomialData_Imbalanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(blm_imbalanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Beta-Liouville Multinomial Naive Bayes on Multinomial Data with an Imbalanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMNB_MultinomialData_Imbalanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()






######### BLM NB, BLM Data, Balanced #########
blm_balanced_blmdat = blm[Balance == 'balanced' & DataType == 'blm' & PosteriorMethod == 'None']

g1 = ggplot(blm_balanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMNB_BLMData_Balanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(blm_balanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Beta-Liouville Multinomial Naive Bayes on BLM Data with a Balanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMNB_BLMData_Balanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()




######### BLM NB, BLM Data, Imbalanced #########
blm_imbalanced_blmdat = blm[Balance == 'imbalanced' & DataType == 'blm' & PosteriorMethod == 'None']

g1 = ggplot(blm_imbalanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMNB_BLMData_Imbalanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(blm_imbalanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Beta-Liouville Multinomial Naive Bayes on BLM Data with an Imbalanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMNB_BLMData_Imbalanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()








##

######### BLM Empirical NB, Multinomial Data, Balanced #########
blme_balanced_multidat = blm[Balance == 'balanced' & DataType == 'multinom' & PosteriorMethod == 'empirical']

g1 = ggplot(blme_balanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMEmpiricalNB_MultinomialData_Balanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(blme_balanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Beta-Liouville Multinomial Empirical Naive Bayes on Multinomial Data with a Balanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMEmpiricalNB_MultinomialData_Balanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()







######### BLM Empirical NB, Multinomial Data, Imbalanced #########
blme_imbalanced_multidat = blm[Balance == 'imbalanced' & DataType == 'multinom' & PosteriorMethod == 'empirical']

g1 = ggplot(blme_imbalanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMEmpiricalNB_MultinomialData_Imbalanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(blme_imbalanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Beta-Liouville Multinomial Empirical Naive Bayes on Multinomial Data with an Imbalanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMEmpiricalNB_MultinomialData_Imbalanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()






######### BLM Empirical NB, BLM Data, Balanced #########
blme_balanced_blmdat = blm[Balance == 'balanced' & DataType == 'blm' & PosteriorMethod == 'empirical']

g1 = ggplot(blme_balanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMEmpiricalNB_BLMData_Balanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(blme_balanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Beta-Liouville Multinomial Empirical Naive Bayes on BLM Data with a Balanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMEmpiricalNB_BLMData_Balanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()




######### BLM Empirical NB, BLM Data, Imbalanced #########
blme_imbalanced_blmdat = blm[Balance == 'imbalanced' & DataType == 'blm' & PosteriorMethod == 'empirical']

g1 = ggplot(blme_imbalanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMEmpiricalNB_BLMData_Imbalanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(blme_imbalanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Beta-Liouville Multinomial Empirical Naive Bayes on BLM Data with an Imbalanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMEmpiricalNB_BLMData_Imbalanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()








##

######### BLM Aposteriori NB, Multinomial Data, Balanced #########
blma_balanced_multidat = blm[Balance == 'balanced' & DataType == 'multinom' & PosteriorMethod == 'aposteriori']

g1 = ggplot(blma_balanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMAposterioriNB_MultinomialData_Balanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(blma_balanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Beta-Liouville Multinomial A Posteriori Naive Bayes on Multinomial Data with a Balanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMAposterioriNB_MultinomialData_Balanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()







######### BLM Aposteriori NB, Multinomial Data, Imbalanced #########
blma_imbalanced_multidat = blm[Balance == 'imbalanced' & DataType == 'multinom' & PosteriorMethod == 'aposteriori']

g1 = ggplot(blma_imbalanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMAposterioriNB_MultinomialData_Imbalanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(blma_imbalanced_multidat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Beta-Liouville Multinomial A Posteriori Naive Bayes on Multinomial Data with an Imbalanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMAposterioriNB_MultinomialData_Imbalanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()






######### BLM Aposteriori NB, BLM Data, Balanced #########
blma_balanced_blmdat = blm[Balance == 'balanced' & DataType == 'blm' & PosteriorMethod == 'aposteriori']

g1 = ggplot(blma_balanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMAposterioriNB_BLMData_Balanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(blma_balanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Beta-Liouville Multinomial A Posteriori Naive Bayes on BLM Data with a Balanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMAposterioriNB_BLMData_Balanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()




######### BLM Aposteriori NB, BLM Data, Imbalanced #########
blma_imbalanced_blmdat = blm[Balance == 'imbalanced' & DataType == 'blm' & PosteriorMethod == 'aposteriori']

g1 = ggplot(blma_imbalanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMAposterioriNB_BLMData_Imbalanced_NoTitle.png'),
    width=2000, height=2000)
print(g1)
dev.off()

g2 = ggplot(blma_imbalanced_blmdat, aes(x=ParamSD, y=F1, color=ParamDraws)) +
  geom_boxplot(position='dodge') +
  facet_wrap(`Data Observations`~`Data Draws`, labeller = label_both) +
  scale_color_brewer('Parameter Draws', palette='Set2') +
  theme(
    plot.title=element_text(size=32, hjust=0.5),
    strip.text=element_text(size=18),
    axis.text=element_text(size=16),
    axis.title=element_text(size=18),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  ) +
  ggtitle('F1 Score for Beta-Liouville Multinomial A Posteriori Naive Bayes on BLM Data with an Imbalanced Design\n') +
  xlab('Parameter Standard Deviation') +
  ylab('F1 Score')
png(filename=paste0(plotdir, 'BLMAposterioriNB_BLMData_Imbalanced.png'),
    width=2000, height=2000)
print(g2)
dev.off()







