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
cat("but with a self-organizing-RW alive particle filter instead of Stan.\n\n")

t0 <- proc.time()[["elapsed"]]
fit <- epinow_birch(
  EpiNow2::example_confirmed, # data (positional)
  generation_time = EpiNow2::example_generation_time, # discretised via EpiNow2
  delays = EpiNow2::example_incubation_period + #   discretise()/get_pmf()
    EpiNow2::example_reporting_delay,
  method = "rw",
  nparticles = 256,
  nsamples = 80,
  verbose = TRUE
)
cat(sprintf(
  "\nBirch-SMC inference finished in %.0f s.\n",
  proc.time()[["elapsed"]] - t0
))

# peek at EpiNow2's summarised table (the exact structure estimate_infections() returns)
cat(
  "\nEpiNow2-format summary (calc_summary_measures output), R around the peak:\n"
)
sm <- fit$summarised
print(head(sm[variable == "R", c("date", "median", "lower_90", "upper_90")], 5))

# render EpiNow2's own plots (plot_estimates) of the Birch results
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
cat("Try method = \"smc2\" for the nested-SMC static-parameter posterior:\n")
cat(
  "  epinow_birch(EpiNow2::example_confirmed, method = \"smc2\", ntheta = 100, nx = 100)\n\n"
)
