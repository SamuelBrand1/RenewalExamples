#!/usr/bin/env Rscript
# Prepare the Birch input file from EpiNow2's example case series and default
# distributions.
#
# Produces:
#   input/epinow2_example.json   - the real example_confirmed series to fit
#   input/prior_predictive.json  - same length, observations blanked (forward sim)
#
# The Birch input convention is a top-level JSON array:
#   element[0]        -> read(buffer)      (globals: PMFs + scalars)
#   element[1..uot]   -> seeding steps     ({}, no observation)
#   element[uot+1..T] -> reported steps    ({y, dow})
#
# Distribution parameters are EpiNow2's packaged example estimates (Ganyani
# generation time; Lauer incubation; a short reporting delay). We discretise at
# the central parameters with the standard difference-of-CDF scheme and
# renormalise to the truncation, matching EpiNow2's default PMF construction.

suppressMessages({
  library(EpiNow2)
  library(jsonlite)
})

# package root = parent of this R/ script's directory (fallback: working dir)
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

# ---- fixed model dimensions ----
UOT <- 14L # seeding window == generation-interval length
N_DELAY <- 21L # length of the infection->report delay PMF
SEED_SD <- 1.0 # log-scale sd of the seed prior
# initial R0 prior: EpiNow2 LogNormal(mean, sd) -> Normal(meanlog, sdlog^2) on log Rt[1]
R0_MEAN <- 1
R0_SD <- 1
R0_VAR <- log(1 + (R0_SD / R0_MEAN)^2)
R0_MEANLOG <- log(R0_MEAN) - 0.5 * R0_VAR
# static-parameter priors (log scale): initial spread (RW) / outer prior (SMC^2)
PRIOR_LSR_MEAN <- log(0.05)
PRIOR_LSR_VAR <- 0.2^2
PRIOR_LPHI_MEAN <- log(10)
PRIOR_LPHI_VAR <- 0.7^2
PRIOR_DOW_VAR <- 0.2^2

# ---- discretisation helpers (difference-of-CDF, renormalised) ----
discretise_gamma <- function(mean, sd, max_day) {
  shape <- (mean / sd)^2
  rate <- mean / sd^2
  k <- seq_len(max_day) # infection age 1..max_day
  pmf <- pgamma(k, shape, rate) - pgamma(k - 1, shape, rate)
  pmf / sum(pmf)
}

discretise_lognormal <- function(meanlog, sdlog, max_day) {
  k <- 0:(max_day - 1) # delay 0..max_day-1
  pmf <- plnorm(k + 1, meanlog, sdlog) - plnorm(k, meanlog, sdlog)
  pmf / sum(pmf)
}

convolve_pmf <- function(a, b) {
  # delays 0..(La-1) (x) 0..(Lb-1)
  La <- length(a)
  Lb <- length(b)
  out <- numeric(La + Lb - 1)
  for (i in seq_len(La)) {
    for (j in seq_len(Lb)) {
      out[i + j - 1] <- out[i + j - 1] + a[i] * b[j]
    }
  }
  out
}

# ---- EpiNow2 example default distributions (central parameter values) ----
gi_pmf <- discretise_gamma(mean = 3.64, sd = 3.08, max_day = UOT) # Ganyani GI

incubation <- discretise_lognormal(meanlog = 1.621, sdlog = 0.418, max_day = 14) # Lauer
reporting <- discretise_lognormal(meanlog = 0.58, sdlog = 0.47, max_day = 10) # reporting
delay_full <- convolve_pmf(incubation, reporting)
delay_pmf <- delay_full[seq_len(N_DELAY)]
delay_pmf <- delay_pmf / sum(delay_pmf)

# ---- example case series ----
data("example_confirmed", package = "EpiNow2")
cases <- example_confirmed
stopifnot(all(c("date") %in% names(cases)))
count_col <- if ("confirm" %in% names(cases)) "confirm" else "cases"
y <- as.integer(cases[[count_col]])
dates <- as.Date(cases$date)
# day-of-week index 1..7 (Mon=1 .. Sun=7)
dow <- as.integer(strftime(dates, "%u"))

n_obs <- length(y)
I0_guess <- max(1, mean(head(y, 7)))
Tsteps <- UOT + n_obs

# ---- assemble the Birch input array ----
empty_obj <- setNames(list(), character(0)) # renders as {}

globals <- list(
  gi_pmf = gi_pmf,
  delay_pmf = delay_pmf,
  uot = UOT,
  n_delay = N_DELAY,
  I0_guess = I0_guess,
  seed_sd = SEED_SD,
  logR0_mean = R0_MEANLOG,
  logR0_var = R0_VAR,
  prior_lsr_mean = PRIOR_LSR_MEAN,
  prior_lsr_var = PRIOR_LSR_VAR,
  prior_lphi_mean = PRIOR_LPHI_MEAN,
  prior_lphi_var = PRIOR_LPHI_VAR,
  prior_dow_var = PRIOR_DOW_VAR
)

seeding_steps <- replicate(UOT, empty_obj, simplify = FALSE)

fit_obs <- lapply(seq_len(n_obs), function(i) list(y = y[i], dow = dow[i]))
pp_obs <- lapply(seq_len(n_obs), function(i) list(dow = dow[i])) # no y -> forward sim

fit_array <- c(list(globals), seeding_steps, fit_obs)
pp_array <- c(list(globals), seeding_steps, pp_obs)

dir.create(here("input"), showWarnings = FALSE, recursive = TRUE)
write_json(
  fit_array,
  here("input", "epinow2_example.json"),
  auto_unbox = TRUE,
  digits = 10,
  pretty = TRUE
)
write_json(
  pp_array,
  here("input", "prior_predictive.json"),
  auto_unbox = TRUE,
  digits = 10,
  pretty = TRUE
)

# also stash the observed series + dates for the plotting script
write_json(
  list(date = as.character(dates), y = y, dow = dow, uot = UOT),
  here("input", "observed.json"),
  auto_unbox = TRUE,
  pretty = TRUE
)

# ---- sanity summary ----
cat(sprintf(
  "generation interval: length %d, sum %.6f, mean %.2f d\n",
  length(gi_pmf),
  sum(gi_pmf),
  sum(gi_pmf * seq_along(gi_pmf))
))
cat(sprintf(
  "delay PMF:           length %d, sum %.6f, mean %.2f d\n",
  length(delay_pmf),
  sum(delay_pmf),
  sum(delay_pmf * (seq_along(delay_pmf) - 1))
))
cat(sprintf(
  "observations:        n_obs %d, T %d, I0_guess %.1f\n",
  n_obs,
  Tsteps,
  I0_guess
))
cat(sprintf("wrote: %s\n", here("input", "epinow2_example.json")))
cat(sprintf("wrote: %s\n", here("input", "prior_predictive.json")))
