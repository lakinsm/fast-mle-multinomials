library(ggplot2)
library(data.table)
library(gridExtra)
library(RColorBrewer)

setwd('C:/Users/lakin/PycharmProjects/fast-mle-multinomials/analytic_data/')
graph_path = 'C:/Users/lakin/Documents/1AbdoBioinformatics/Publications/MLE/Analysis/graphs/'

dat = data.table(read.csv('2020Apr6_mle_timings.csv', header=F))
colnames(dat) = c('DataSet', 'MLEMethod', 'PrecomputeMethod', 'PosteriorMethod',
                  'ClassLabel', 'NumObservations', 'NumParameters', 'Seconds')

draw_dat = data.table(read.csv('dataset_label_max_rowcounts.csv', header=F))
colnames(draw_dat) = c('ClassLabel', 'NumMultinomDraws')

setkey(dat, ClassLabel)
setkey(draw_dat, ClassLabel)

dat = draw_dat[dat]
dat[, Precompute_PosteriorMethod := paste(PrecomputeMethod, PosteriorMethod, sep='_')]
dat$Precompute_PosteriorMethod = factor(dat$Precompute_PosteriorMethod,
                                         levels=c('None_None', 
                                                  'approximate_None', 'vectorized_None', 'sklar_None',
                                                  'approximate_empirical', 'vectorized_empirical', 'sklar_empirical',
                                                  'approximate_aposteriori', 'vectorized_aposteriori', 'sklar_aposteriori'),
                                         ordered=T)

cade_dat = dat[DataSet == 'cade']
g_cade_param = ggplot(cade_dat, aes(x=NumParameters, y=Seconds, color=NumObservations)) +
  geom_point(size=3) +
  facet_wrap(PrecomputeMethod~MLEMethod, ncol=2, labeller = label_both) +
  ggtitle('Cade MLE Runtime by # of Parameters & Observations') +
  scale_color_continuous() +
  theme(
    plot.title=element_text(hjust=0.5, size=22),
    axis.text=element_text(size=14),
    strip.text=element_text(size=20),
    axis.title=element_text(size=20),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  )
png(paste(graph_path, '2020Apr6_CadeMLETiming.png', sep=''), width = 1200, height = 1000)
print(g_cade_param)
dev.off()

r8_dat = dat[DataSet == 'r8']
g_r8 = ggplot(r8_dat, aes(x=NumParameters, y=Seconds, color=NumObservations)) +
  geom_point(size=3) +
  facet_wrap(PrecomputeMethod~MLEMethod, ncol=2, labeller = label_both) +
  ggtitle('R8 MLE Runtime by # of Parameters & Observations') +
  scale_color_continuous() +
  theme(
    plot.title=element_text(hjust=0.5, size=22),
    axis.text=element_text(size=14),
    strip.text=element_text(size=20),
    axis.title=element_text(size=20),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  )
png(paste(graph_path, '2020Apr6_r8MLETiming.png', sep=''), width = 1200, height = 1000)
print(g_r8)
dev.off()


webkb_dat = dat[DataSet == 'webkb']
g_webkb = ggplot(webkb_dat, aes(x=NumParameters, y=Seconds, color=NumObservations)) +
  geom_point(size=3) +
  facet_wrap(PrecomputeMethod~MLEMethod, ncol=2, labeller = label_both) +
  ggtitle('Webkb MLE Runtime by # of Parameters & Observations') +
  scale_color_continuous() +
  theme(
    plot.title=element_text(hjust=0.5, size=22),
    axis.text=element_text(size=14),
    strip.text=element_text(size=20),
    axis.title=element_text(size=20),
    legend.title=element_text(size=18),
    legend.text=element_text(size=16)
  )
png(paste(graph_path, '2020Apr6_webkbMLETiming.png', sep=''), width = 1200, height = 1000)
print(g_webkb)
dev.off()


