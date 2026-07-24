#!/usr/bin/env Rscript
# OPTIONAL reference fit. Runs EpiNow2 (Stan) on the same example_confirmed
# series with the packaged default distributions and a weekly random walk on
# Rt, then writes output/reference_rt.csv (date, median, lower, upper) for
# plot_results.R to overlay on the Birch posterior.
#
# This compiles and samples a Stan model and can take several minutes.
#   Rscript R/epinow2_reference.R

suppressMessages({
  library(EpiNow2)
  library(data.table)
})

get_root <- function() {
  f <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(f)) {
    normalizePath(file.path(dirname(sub("^--file=", "", f[1])), ".."))
  } else {
    getwd()
  }
}
ROOT <- get_root()
here <- function(...) file.path(ROOT, ...)

data("example_confirmed", package = "EpiNow2")

# Match the Birch model as closely as EpiNow2 allows: example generation time
# and (incubation + reporting) delay, initial R ~ LogNormal(1, 1), a weekly RW
# on Rt (rw = 7) in place of the default GP, default weekly + negbin obs model.
gt <- generation_time_opts(example_generation_time)
dly <- delay_opts(example_incubation_period + example_reporting_delay)
rto <- rt_opts(prior = LogNormal(mean = 1, sd = 1), rw = 7)

est <- estimate_infections(
  data = example_confirmed,
  generation_time = gt,
  delays = dly,
  rt = rto,
  gp = NULL, # rw replaces the GP
  obs = obs_opts(week_effect = TRUE),
  stan = stan_opts(samples = 1000, chains = 2, cores = 2),
  verbose = TRUE
)

s <- as.data.table(est$summarised)

# Robustly pick the 90% interval columns across EpiNow2 versions.
pick <- function(dt, cand) cand[cand %in% names(dt)][1]
Rraw <- s[variable == "R"]
lo <- pick(Rraw, c("lower_90", "lower_0.9", "lower_50"))
hi <- pick(Rraw, c("upper_90", "upper_0.9", "upper_50"))
ref <- data.frame(
  date = as.Date(Rraw$date),
  median = Rraw$median,
  lower = Rraw[[lo]],
  upper = Rraw[[hi]]
)

dir.create(here("output"), showWarnings = FALSE)
write.csv(ref, here("output", "reference_rt.csv"), row.names = FALSE)
cat(sprintf(
  "wrote: %s (%d rows, interval %s/%s)\n",
  here("output", "reference_rt.csv"),
  nrow(ref),
  lo,
  hi
))
