library(ggplot2)
library(data.table)
library(gridExtra)
library(RColorBrewer)
library(ggpubr)

setwd('C:/Users/lakin/PycharmProjects/fast-mle-multinomials/analytic_data/')
graph_path = 'C:/Users/lakin/PycharmProjects/fast-mle-multinomials/analytic_data/graphs/'


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

precompute_transl = c('sklar'='Sklar',
                      'approximate'='Approximate',
                      'vectorized'='Vectorized')

dat$PrecomputeMethod = precompute_transl[as.character(dat$PrecomputeMethod)]

dat$PrecomputeMethod = factor(dat$PrecomputeMethod,
                              levels=c('Sklar',
                                       'Vectorized',
                                       'Approximate'),
                              ordered=T)

obs_dat = dat[Experiment == 'VaryObservations', .SD, .SDcols=!c('NumParameters', 'NumDraws')]
g_obs = ggplot(obs_dat, aes(x=NumObservations, y=MeanSeconds, color=PrecomputeMethod)) +
  # geom_line(size=1) +
  geom_point(size=2) +
  # geom_errorbar(aes(ymin=MeanSeconds - SdSeconds, ymax=MeanSeconds + SdSeconds), size=1, width=1) +
  geom_smooth(method='loess', alpha=0.4) +
  facet_wrap(~MLEMethod, ncol=1) +
  ggtitle('Runtime Varying # of Observations') +
  ylab('Mean Elapsed Real Time (seconds)') +
  xlab('Number of Observations') +
  scale_colour_brewer(type='qual', palette = 'Dark2') +
  custom_theme_nolegend(
    base_size= 20
  )

param_dat = dat[Experiment == 'VaryParameters', .SD, .SDcols=!c('NumObservations', 'NumDraws')]
g_param = ggplot(param_dat, aes(x=NumParameters, y=MeanSeconds, color=PrecomputeMethod)) +
  # geom_line(size=1) +
  geom_point(size=2) +
  # geom_errorbar(aes(ymin=MeanSeconds - SdSeconds, ymax=MeanSeconds + SdSeconds), size=1, width=1) +
  geom_smooth(method='loess', alpha=0.4) +
  facet_wrap(~MLEMethod, ncol=1) +
  ggtitle('Runtime Varying # of Parameters') +
  ylab('Mean Elapsed Real Time (seconds)') +
  xlab('Number of Parameters') +
  scale_colour_brewer(type='qual', palette = 'Dark2') +
  custom_theme_nolegend(
    base_size= 20
  )

draw_dat = dat[Experiment == 'VaryMultinomDraws', .SD, .SDcols=!c('NumObservations', 'NumParameters')]
g_draw = ggplot(draw_dat, aes(x=NumDraws, y=MeanSeconds, color=PrecomputeMethod)) +
  # geom_line(size=1) +
  geom_point(size=2) +
  # geom_errorbar(aes(ymin=MeanSeconds - SdSeconds, ymax=MeanSeconds + SdSeconds), size=1, width=1) +
  geom_smooth(method='loess', alpha=0.4) +
  facet_wrap(~MLEMethod, ncol=1) +
  ggtitle('Runtime Varying # of Draws') +
  ylab('Mean Elapsed Real Time (seconds)') +
  xlab('Number of Draws') +
  scale_colour_brewer('Precompute Method', type='qual', palette = 'Dark2') +
  custom_theme_legend(
    base_size = 20
  ) %+replace%
  theme(
    plot.margin = unit(c(0.5, 0, 0.5, 0.5), 'cm')
  )

png(paste(graph_path, '2020Apr6_Runtime.png', sep=''), width = 1400, height = 1000)
  grid.arrange(g_obs, g_param, g_draw, layout_matrix=matrix(c(1, 1, 2, 2, 3, 3, 3), nrow=1))
dev.off()

