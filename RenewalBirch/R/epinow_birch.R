#!/usr/bin/env Rscript
# epinow_birch(): an EpiNow2-style front-end that swaps Stan/NUTS for Birch-SMC
# in the compute slot, and reuses EpiNow2's OWN machinery on both ends:
#   - INPUT:  EpiNow2 distribution specs -> discretise()/get_pmf() -> Birch PMFs
#   - OUTPUT: Birch JSON -> calc_summary_measures() -> plot_estimates()
# so the epi user gets the familiar Rt / infections / reported-cases plots, with
# Birch's particle-filter SMC doing the inference instead of Stan.
#
# Mirrors the core of epinow(): `data` positional; `generation_time`, `delays`,
# `obs` kwargs. Adds a Birch-specific `method`:
#   "rw"   - self-organizing-RW static params + bootstrap/alive PF (fast, one sweep)
#   "smc2" - nested SMC^2 (proper static-parameter posterior; returns theta only)
# Other epinow() options are out of scope for this toy.
#
#   source("R/epinow_birch.R")
#   fit <- epinow_birch(EpiNow2::example_confirmed)     # run RW + plot
#   plot(fit)

suppressMessages({
  library(jsonlite)
  library(data.table)
  library(EpiNow2)
  library(ggplot2)
})

# ---- EpiNow2 dist spec -> discretised PMF (reuses EpiNow2's discretisation) ----
spec_to_pmf <- function(dist) {
  d <- discretise(fix_parameters(dist))
  pmf <- tryCatch(get_pmf(d), error = function(e) get_pmf(collapse(d))) # composite -> convolve
  as.numeric(pmf)
}

# ---- build the Birch input array from EpiNow2 specs + case data ----
build_birch_input <- function(
  data,
  generation_time,
  delays,
  rt_prior = c(mean = 1, sd = 1), # initial R0 prior (natural scale, as EpiNow2 rt_opts)
  # priors on the (log-scale) static parameters -- initial spread for RW, outer prior for SMC^2:
  sigma_rw_prior = c(meanlog = log(0.05), sd = 0.2), # log sigma_rw (RW step sd of log Rt)
  phi_prior = c(meanlog = log(10), sd = 0.7), # log phi (NB2 overdispersion)
  dow_prior_sd = 0.2, # day-of-week logit sd (mean 0)
  seed_sd = 1.0,
  rw_sd = 0.03,
  dir = "."
) {
  gi_pmf <- spec_to_pmf(generation_time)
  delay_pmf <- spec_to_pmf(delays)
  uot <- length(gi_pmf) # seeding window == generation-interval length
  n_delay <- length(delay_pmf)

  # LogNormal(mean, sd) on R0 -> Normal(meanlog, sdlog^2) on log Rt[1]
  r0_v <- log(1 + (rt_prior[["sd"]] / rt_prior[["mean"]])^2)
  r0_meanlog <- log(rt_prior[["mean"]]) - 0.5 * r0_v

  y <- as.integer(data$confirm)
  dates <- as.Date(data$date)
  dow <- as.integer(strftime(dates, "%u"))
  n_obs <- length(y)

  globals <- list(
    gi_pmf = gi_pmf,
    delay_pmf = delay_pmf,
    uot = uot,
    n_delay = n_delay,
    I0_guess = max(1, mean(head(y, 7))),
    seed_sd = seed_sd,
    logR0_mean = r0_meanlog,
    logR0_var = r0_v,
    prior_lsr_mean = sigma_rw_prior[["meanlog"]],
    prior_lsr_var = sigma_rw_prior[["sd"]]^2,
    prior_lphi_mean = phi_prior[["meanlog"]],
    prior_lphi_var = phi_prior[["sd"]]^2,
    prior_dow_var = dow_prior_sd^2,
    tau_sr = rw_sd,
    tau_phi = rw_sd,
    tau_dow = rw_sd
  )
  empty <- setNames(list(), character(0))
  fit_obs <- lapply(seq_len(n_obs), function(i) list(y = y[i], dow = dow[i]))
  arr <- c(list(globals), replicate(uot, empty, simplify = FALSE), fit_obs)

  dir.create(file.path(dir, "input"), showWarnings = FALSE, recursive = TRUE)
  write_json(
    arr,
    file.path(dir, "input", "epinow2_example.json"),
    auto_unbox = TRUE,
    digits = 10
  )
  write_json(
    list(date = as.character(dates), y = y, dow = dow, uot = uot),
    file.path(dir, "input", "observed.json"),
    auto_unbox = TRUE
  )
  list(uot = uot, n_delay = n_delay, n_obs = n_obs)
}

# ---- ingest a trajectory-style Birch output into EpiNow2 long-format samples ----
birch_to_samples <- function(output_json, observed_json) {
  s <- fromJSON(output_json, simplifyVector = FALSE)
  obs <- fromJSON(observed_json, simplifyVector = TRUE)
  uot <- obs$uot
  first_date <- as.Date(obs$date[1])
  K <- length(s)
  Tt <- length(s[[1]]$sample[[2]]$sample)
  dates <- first_date + (seq_len(Tt) - uot - 1L)

  Rt <- Inf_ <- Mu <- matrix(NA_real_, K, Tt)
  for (k in seq_len(K)) {
    steps <- s[[k]]$sample[[2]]$sample
    for (t in seq_len(Tt)) {
      st <- steps[[t]]
      if (!is.null(st$Rt)) {
        Rt[k, t] <- st$Rt
      }
      if (!is.null(st$I)) {
        Inf_[k, t] <- st$I
      }
      if (!is.null(st$mu)) Mu[k, t] <- st$mu
    }
  }
  mk_long <- function(M, var) {
    dt <- as.data.table(M)
    setnames(dt, as.character(dates))
    dt[, sample := .I]
    m <- melt(
      dt,
      id.vars = "sample",
      variable.name = "date",
      value.name = "value"
    )
    m[, `:=`(variable = var, date = as.Date(as.character(date)))]
    m[is.finite(value)]
  }
  rbindlist(list(
    mk_long(Rt, "R"),
    mk_long(Inf_, "infections"),
    mk_long(Mu, "reported_cases")
  ))
}

# ---- the wrapper ----
epinow_birch <- function(
  data,
  generation_time = EpiNow2::example_generation_time,
  delays = EpiNow2::example_incubation_period +
    EpiNow2::example_reporting_delay,
  obs = EpiNow2::obs_opts(),
  rt_prior = c(mean = 1, sd = 1), # initial R0 prior (EpiNow2 rt_opts default)
  sigma_rw_prior = c(meanlog = log(0.05), sd = 0.2), # static-param priors (log scale):
  phi_prior = c(meanlog = log(10), sd = 0.7), #   initial spread (RW) / outer prior (SMC^2)
  dow_prior_sd = 0.2,
  method = c("rw", "smc2"),
  nparticles = 512,
  nsamples = 200, # RW / bootstrap PF
  ntheta = 100,
  nx = 100,
  nmoves = 5, # SMC^2 PMMH steps per rejuvenation (fixed chain; 1 impoverishes)
  move_sd = 0.02, # SMC^2 (nx scales with T)
  seed_sd = 1.0,
  rw_sd = 0.03,
  dir = ".",
  CrIs = c(0.2, 0.5, 0.9),
  verbose = TRUE
) {
  method <- match.arg(method)
  old <- setwd(dir)
  on.exit(setwd(old))
  say <- function(...) if (verbose) cat(...)
  # the run config and outputs are generated, not committed -- ensure the dirs exist
  dir.create("config", showWarnings = FALSE)
  dir.create("output", showWarnings = FALSE)

  dims <- build_birch_input(
    data,
    generation_time,
    delays,
    rt_prior = rt_prior,
    sigma_rw_prior = sigma_rw_prior,
    phi_prior = phi_prior,
    dow_prior_sd = dow_prior_sd,
    seed_sd = seed_sd,
    rw_sd = rw_sd,
    dir = "."
  )
  say(sprintf(
    "[epinow_birch] gi length %d, delay length %d, %d obs; method=%s\n",
    dims$uot,
    dims$n_delay,
    dims$n_obs,
    method
  ))

  if (method == "rw") {
    cfg <- list(
      model = list(class = "RenewalModel"),
      filter = list(class = "AliveParticleFilter", nparticles = nparticles),
      sampler = list(nsamples = nsamples),
      input = "input/epinow2_example.json",
      output = "output/renewal.json"
    )
    write_json(cfg, "config/_epinow_birch.json", auto_unbox = TRUE)
    unlink("output/renewal.json") # Birch's writer does not truncate; start from a clean file
    say(
      "[epinow_birch] running Birch alive-PF (this can take a few minutes)...\n"
    )
    system2(
      "birch",
      c("sample", "--config", "config/_epinow_birch.json"),
      stdout = if (verbose) "" else FALSE,
      stderr = FALSE
    )

    samples <- birch_to_samples("output/renewal.json", "input/observed.json")
    summarised <- calc_summary_measures(
      samples,
      summarise_by = c("variable", "date"),
      CrIs = CrIs
    )[, type := "estimate"]
    reported <- {
      o <- fromJSON("input/observed.json", simplifyVector = TRUE)
      data.table(date = as.Date(o$date), confirm = o$y)
    }
    plots <- list(
      R = plot_estimates(summarised[variable == "R"], ylab = "Rt", hline = 1),
      infections = plot_estimates(
        summarised[variable == "infections"],
        ylab = "Infections"
      ),
      reported_cases = plot_estimates(
        summarised[variable == "reported_cases"],
        ylab = "Reported cases",
        reported = reported
      )
    )
    out <- list(
      summarised = summarised,
      samples = samples,
      observations = reported,
      plots = plots,
      engine = "birch-rw"
    )
  } else {
    # smc2 -> theta posterior only (trajectory output is a documented next step)
    say(sprintf(
      "[epinow_birch] running SMC^2 (ntheta=%d, nx=%d, nmoves=%d)...\n",
      ntheta,
      nx,
      nmoves
    ))
    unlink("output/smc2.json") # Birch's writer does not truncate; start from a clean file
    system2(
      "birch",
      c(
        "smc2",
        "--ntheta",
        ntheta,
        "--nx",
        nx,
        "--nmoves", # fixed MH chain per rejuvenation; 1 impoverishes the theta-particles
        nmoves,
        "--move-sd", # Birch converts underscores to hyphens in program-arg flags
        move_sd,
        "--output",
        "output/smc2.json"
      ),
      stdout = if (verbose) "" else FALSE,
      stderr = FALSE
    )
    th <- fromJSON("output/smc2.json", simplifyVector = FALSE)
    theta <- data.table(
      sigma_rw = vapply(th, function(p) p$sigma_rw, numeric(1)),
      phi = vapply(th, function(p) p$phi, numeric(1)),
      weight = vapply(th, function(p) p$weight, numeric(1))
    )
    out <- list(
      theta = theta,
      engine = "birch-smc2",
      plots = list(
        sigma_rw = ggplot(theta, aes(sigma_rw, weight = weight / sum(weight))) +
          geom_density(fill = "#2c7fb8", alpha = .35) +
          labs(title = "sigma_rw posterior"),
        phi = ggplot(theta, aes(phi, weight = weight / sum(weight))) +
          geom_density(fill = "#41ab5d", alpha = .35) +
          labs(title = "phi posterior")
      )
    )
  }
  structure(out, class = "epinow_birch")
}

plot.epinow_birch <- function(x, ...) {
  if (requireNamespace("patchwork", quietly = TRUE)) {
    patchwork::wrap_plots(x$plots, ncol = 1) +
      patchwork::plot_annotation(
        title = sprintf("EpiNow2 front-end, %s compute", x$engine)
      )
  } else {
    x$plots
  }
}
print.epinow_birch <- function(x, ...) {
  cat(sprintf("<epinow_birch> engine=%s\n", x$engine))
  invisible(x)
}

# run as a script: ingest the existing RW output (no re-fit) and save the figure
if (sys.nframe() == 0) {
  samples <- birch_to_samples("output/renewal.json", "input/observed.json")
  summarised <- calc_summary_measures(
    samples,
    summarise_by = c("variable", "date"),
    CrIs = c(0.2, 0.5, 0.9)
  )[, type := "estimate"]
  o <- fromJSON("input/observed.json", simplifyVector = TRUE)
  reported <- data.table(date = as.Date(o$date), confirm = o$y)
  fig <- patchwork::wrap_plots(
    list(
      plot_estimates(summarised[variable == "R"], ylab = "Rt", hline = 1),
      plot_estimates(summarised[variable == "infections"], ylab = "Infections"),
      plot_estimates(
        summarised[variable == "reported_cases"],
        ylab = "Reported cases",
        reported = reported
      )
    ),
    ncol = 1
  ) +
    patchwork::plot_annotation(title = "EpiNow2 front-end, Birch-SMC compute")
  dir.create("figures", showWarnings = FALSE)
  ggsave("figures/epinow_birch_rw.png", fig, width = 8, height = 9, dpi = 120)
  cat("wrote figures/epinow_birch_rw.png\n")
}
