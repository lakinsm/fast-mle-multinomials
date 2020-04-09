library(ggplot2)
library(data.table)
library(gridExtra)
library(RColorBrewer)

setwd('C:/Users/lakin/PycharmProjects/fast-mle-multinomials/analytic_data/')
graph_path = 'C:/Users/lakin/Documents/1AbdoBioinformatics/Publications/MLE/Analysis/graphs/'

dat = data.table(read.csv('2020Apr6_runtime_results.csv', header=F))
colnames(dat) = c('Experiment', 'MLEMethod', 'PrecomputeMethod', 
                  'NumParameters', 'NumObservations', 'NumDraws',
                  'Seconds')

dat[, MeanSeconds := lapply(.SD, mean), by=c('Experiment', 'MLEMethod', 'PrecomputeMethod',
                                             'NumParameters', 'NumObservations', 'NumDraws')]

dat[, SdSeconds := lapply(.SD, sd), by=c('Experiment', 'MLEMethod', 'PrecomputeMethod',
                                             'NumParameters', 'NumObservations', 'NumDraws')]
dat[, Seconds := NULL]

dat = unique(dat)

obs_dat = dat[Experiment == 'VaryObservations', .SD, .SDcols=!c('NumParameters', 'NumDraws')]
g_obs = ggplot(obs_dat, aes(x=NumObservations, y=MeanSeconds, color=PrecomputeMethod)) +
  geom_line(size=1) +
  geom_point(size=2) +
  geom_errorbar(aes(ymin=MeanSeconds - SdSeconds, ymax=MeanSeconds + SdSeconds), size=1, width=1) +
  facet_wrap(~MLEMethod, ncol=1) +
  ggtitle('Runtime Varying # of Observations') +
  scale_colour_brewer(type='qual', palette = 'Set1') +
  theme(
    plot.title=element_text(hjust=0.5, size=22),
    axis.text=element_text(size=14),
    strip.text=element_text(size=20),
    axis.title=element_text(size=20),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16),
    legend.position='None'
  )

param_dat = dat[Experiment == 'VaryParameters', .SD, .SDcols=!c('NumObservations', 'NumDraws')]
g_param = ggplot(param_dat, aes(x=NumParameters, y=MeanSeconds, color=PrecomputeMethod)) +
  geom_line(size=1) +
  geom_point(size=2) +
  geom_errorbar(aes(ymin=MeanSeconds - SdSeconds, ymax=MeanSeconds + SdSeconds), size=1, width=1) +
  facet_wrap(~MLEMethod, ncol=1) +
  ggtitle('Runtime Varying # of Parameters') +
  scale_colour_brewer(type='qual', palette = 'Set1') +
  theme(
    plot.title=element_text(hjust=0.5, size=22),
    axis.text=element_text(size=14),
    strip.text=element_text(size=20),
    axis.title=element_text(size=20),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16),
    legend.position='None'
  )

draw_dat = dat[Experiment == 'VaryMultinomDraws', .SD, .SDcols=!c('NumObservations', 'NumParameters')]
g_draw = ggplot(draw_dat, aes(x=NumDraws, y=MeanSeconds, color=PrecomputeMethod)) +
  geom_line(size=1) +
  geom_point(size=2) +
  geom_errorbar(aes(ymin=MeanSeconds - SdSeconds, ymax=MeanSeconds + SdSeconds), size=1, width=1) +
  facet_wrap(~MLEMethod, ncol=1) +
  ggtitle('Runtime Varying # of Draws') +
  scale_colour_brewer(type='qual', palette = 'Set1') +
  theme(
    plot.title=element_text(hjust=0.5, size=22),
    axis.text=element_text(size=14),
    strip.text=element_text(size=20),
    axis.title=element_text(size=20),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  )

png(paste(graph_path, '2020Apr6_Runtime.png', sep=''), width = 1400, height = 1000)
grid.arrange(g_obs, g_param, g_draw, layout_matrix=matrix(c(1, 1, 2, 2, 3, 3, 3), nrow=1))
dev.off()

