#!/usr/bin/env Rscript
# Demo: EpiNow2 front-end with Birch-SMC in the compute slot.
#
# Drives the whole pipeline with a single epinow_birch() call that looks just
# like epinow() — EpiNow2 distribution specs in, EpiNow2-style Rt / infections /
# reported-cases plots out — but a Birch particle filter does the inference
# instead of Stan/NUTS.
#
# Run from the package root (after `birch build`):
#   Rscript demo.R

suppressMessages({
  library(EpiNow2)
  library(ggplot2)
})

# resolve the package root whether run via Rscript, sourced, or from the repo root
locate_root <- function() {
  f <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  path <- if (length(f)) {
    sub("^--file=", "", f[1])
  } else {
    of <- NULL
    for (i in seq_len(sys.nframe())) {
      if (!is.null(sys.frame(i)$ofile)) of <- sys.frame(i)$ofile
    }
    of
  }
  root <- if (!is.null(path)) dirname(normalizePath(path)) else getwd() # demo.R sits in the package root
  if (!file.exists(file.path(root, "R", "epinow_birch.R"))) {
    # graceful fallbacks
    for (cand in c(
      root,
      file.path(root, "RenewalBirch"),
      getwd(),
      file.path(getwd(), "RenewalBirch")
    )) {
      if (file.exists(file.path(cand, "R", "epinow_birch.R"))) {
        root <- cand
        break
      }
    }
  }
  root
}
ROOT <- locate_root()
if (!file.exists(file.path(ROOT, "R", "epinow_birch.R"))) {
  stop(
    "Cannot find R/epinow_birch.R — run this from the RenewalBirch package directory."
  )
}
setwd(ROOT)

if (!length(Sys.glob(".libs/librenewalbirch*.*"))) {
  stop(
    "Birch package not built. Run `birch build` in the RenewalBirch directory first."
  )
}

source(file.path(ROOT, "R", "epinow_birch.R"))

cat("\n==============================================================\n")
cat(" EpiNow2 front-end  x  Birch-SMC compute  —  demo\n")
cat("==============================================================\n\n")
cat(
  "Fitting EpiNow2::example_confirmed (",
  nrow(EpiNow2::example_confirmed),
  " days) with the SAME distribution specs epinow() would use,\n",
  sep = ""
)
cat("but with a Birch particle-filter SMC engine instead of Stan.\n\n")

t0 <- proc.time()[["elapsed"]]
fit <- epinow_birch(
  EpiNow2::example_confirmed, # data (positional)
  generation_time = EpiNow2::example_generation_time, # discretised via EpiNow2
  delays = EpiNow2::example_incubation_period + #   discretise()/get_pmf()
    EpiNow2::example_reporting_delay,
  method = "smc2",
  ntheta = 100,
  nx = 100,
  nparticles = 256,
  nsamples = 80,
  verbose = TRUE
)
cat(sprintf(
  "\nBirch-SMC inference finished in %.0f s.\n",
  proc.time()[["elapsed"]] - t0
))

# report + plot, branching on which engine ran
if (!is.null(fit$summarised)) {
  # RW engine: EpiNow2-format summarised table + plot_estimates panel
  cat(
    "\nEpiNow2-format summary (calc_summary_measures output), R around the peak:\n"
  )
  sm <- fit$summarised
  print(head(
    sm[variable == "R", c("date", "median", "lower_90", "upper_90")],
    5
  ))
  fig <- plot(fit)
  ggsave(
    file.path(ROOT, "figures", "epinow_birch_demo.png"),
    fig,
    width = 8,
    height = 9,
    dpi = 120
  )
  cat("\nWrote figures/epinow_birch_demo.png — the familiar EpiNow2 panel,\n")
  cat("computed by Birch's particle filter.\n\n")
} else {
  # SMC^2 engine: static-parameter (theta) posterior -- no Rt trajectories
  th <- fit$theta
  w <- th$weight / sum(th$weight)
  cat("\nSMC^2 static-parameter posterior (theta):\n")
  cat(sprintf(
    "  sigma_rw:  mean %.3f  range [%.3f, %.3f]\n",
    sum(w * th$sigma_rw),
    min(th$sigma_rw),
    max(th$sigma_rw)
  ))
  cat(sprintf(
    "  phi:       mean %.2f  range [%.2f, %.2f]\n",
    sum(w * th$phi),
    min(th$phi),
    max(th$phi)
  ))
  fig <- plot(fit)
  ggsave(
    file.path(ROOT, "figures", "epinow_birch_smc2_demo.png"),
    fig,
    width = 8,
    height = 5,
    dpi = 120
  )
  cat(
    "\nWrote figures/epinow_birch_smc2_demo.png — the theta posterior from SMC^2.\n"
  )
  cat(
    "(SMC^2 returns the static-parameter posterior; routing its state trajectories\n"
  )
  cat(
    " into the EpiNow2 Rt/infections/reported-cases plots is a documented next step.)\n\n"
  )
}
