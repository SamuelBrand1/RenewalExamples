#!/usr/bin/env Rscript
# Parse a Birch SMC output file and plot posterior Rt, infections, and
# posterior-predictive cases against the observed series (and, if present, an
# EpiNow2 reference Rt from R/epinow2_reference.R).
#
# Usage:
#   Rscript R/plot_results.R [output_json] [figure_suffix]
# defaults: output/renewal.json  rw
#
# RenewalModel (RW) writes sigma_rw/phi per-step and the day-of-week simplex in
# globals. The nsamples runs are EQUAL-weighted posterior trajectory draws
# (lweight is the marginal-likelihood estimate, not a cross-run weight).

suppressMessages({
  library(jsonlite)
  library(ggplot2)
})

args <- commandArgs(TRUE)
out_json <- if (length(args) >= 1) args[1] else "output/renewal.json"
suffix <- if (length(args) >= 2) args[2] else "rw"

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
if (!grepl("^/", out_json)) {
  out_json <- here(out_json)
}

wquant <- function(x, w, probs) {
  ok <- is.finite(x) & is.finite(w)
  if (!any(ok)) {
    return(rep(NA_real_, length(probs)))
  }
  x <- x[ok]
  w <- w[ok]
  o <- order(x)
  x <- x[o]
  w <- w[o]
  cw <- cumsum(w) / sum(w)
  approx(cw, x, probs, rule = 2, ties = "ordered")$y
}

samples <- fromJSON(out_json, simplifyVector = FALSE)
K <- length(samples)
Tt <- length(samples[[1]]$sample[[2]]$sample)

# Each run's draw() already samples one ancestral trajectory in proportion to
# the particle weights, so the nsamples runs are EQUALLY-weighted posterior
# draws. lweight is the per-run marginal-likelihood estimate (for PMMH/evidence),
# NOT a cross-run importance weight -- do not reweight the trajectories by it.
lw <- vapply(samples, function(s) s$lweight, numeric(1))
w <- rep(1 / K, K)
ml_ess <- {
  u <- exp(lw - max(lw))
  1 / sum((u / sum(u))^2)
} # ML-estimator spread (diagnostic only)

Rt <- matrix(NA_real_, K, Tt)
Inf_ <- matrix(NA_real_, K, Tt)
Mu <- matrix(NA_real_, K, Tt)
Sig <- matrix(NA_real_, K, Tt)
Phi <- matrix(NA_real_, K, Tt)
sig_g <- rep(NA_real_, K)
phi_g <- rep(NA_real_, K)
for (k in seq_len(K)) {
  g <- samples[[k]]$sample[[1]]
  if (!is.null(g$sigma_rw)) {
    sig_g[k] <- g$sigma_rw
  }
  if (!is.null(g$phi)) {
    phi_g[k] <- g$phi
  }
  steps <- samples[[k]]$sample[[2]]$sample
  for (t in seq_len(Tt)) {
    st <- steps[[t]]
    if (!is.null(st$I)) {
      Inf_[k, t] <- st$I
    }
    if (!is.null(st$Rt)) {
      Rt[k, t] <- st$Rt
    }
    if (!is.null(st$mu)) {
      Mu[k, t] <- st$mu
    }
    if (!is.null(st$sigma_rw)) {
      Sig[k, t] <- st$sigma_rw
    }
    if (!is.null(st$phi)) Phi[k, t] <- st$phi
  }
}
# fall back to global statics (MALA schema) if not written per-step
if (all(is.na(Sig)) && any(is.finite(sig_g))) {
  Sig <- matrix(sig_g, K, Tt)
}
if (all(is.na(Phi)) && any(is.finite(phi_g))) {
  Phi <- matrix(phi_g, K, Tt)
}

obs <- fromJSON(here("input", "observed.json"), simplifyVector = TRUE)
uot <- obs$uot
first_date <- as.Date(obs$date[1])
date_t <- first_date + (seq_len(Tt) - uot - 1L)
y_t <- rep(NA_real_, Tt)
y_t[(uot + 1):Tt] <- obs$y

summ <- function(M) {
  q <- t(vapply(
    seq_len(Tt),
    function(t) {
      wquant(M[, t], w, c(0.05, 0.25, 0.5, 0.75, 0.95))
    },
    numeric(5)
  ))
  data.frame(
    date = date_t,
    lo90 = q[, 1],
    lo50 = q[, 2],
    med = q[, 3],
    hi50 = q[, 4],
    hi90 = q[, 5]
  )
}
rt_df <- summ(Rt)
inf_df <- summ(Inf_)
mu_df <- summ(Mu)

wmean <- function(M) sum(w * colMeans(M, na.rm = TRUE), na.rm = TRUE) / sum(w)
cat(sprintf(
  "[%s] trajectory draws=%d (equal-weighted)  steps=%d  ML-estimator ESS=%.1f  sigma_rw~%.3f  phi~%.2f\n",
  suffix,
  K,
  Tt,
  ml_ess,
  wquant(rowMeans(Sig, na.rm = TRUE), w, 0.5),
  wquant(rowMeans(Phi, na.rm = TRUE), w, 0.5)
))

ref_path <- here("output", "reference_rt.csv")
ref <- if (file.exists(ref_path)) {
  r <- read.csv(ref_path)
  r$date <- as.Date(r$date)
  r
} else {
  NULL
}

dir.create(here("figures"), showWarnings = FALSE)
theme_set(theme_minimal(base_size = 12))

p_rt <- ggplot(rt_df, aes(date, med)) +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "grey50") +
  geom_ribbon(aes(ymin = lo90, ymax = hi90), fill = "#2c7fb8", alpha = 0.20) +
  geom_ribbon(aes(ymin = lo50, ymax = hi50), fill = "#2c7fb8", alpha = 0.35) +
  geom_line(colour = "#08306b", linewidth = 0.7)
if (!is.null(ref)) {
  p_rt <- p_rt +
    geom_ribbon(
      data = ref,
      aes(date, ymin = lower, ymax = upper),
      inherit.aes = FALSE,
      fill = "#d95f02",
      alpha = 0.12
    ) +
    geom_line(
      data = ref,
      aes(date, median),
      colour = "#d95f02",
      linewidth = 0.6,
      linetype = "twodash"
    )
}
p_rt <- p_rt +
  labs(
    title = sprintf("Posterior Rt (Birch SMC, %s)", suffix),
    subtitle = if (!is.null(ref)) {
      "blue = Birch; orange dashed = EpiNow2 reference"
    } else {
      "median with 50% and 90% credible bands"
    },
    x = NULL,
    y = expression(R[t])
  )
ggsave(
  here("figures", sprintf("rt_%s.png", suffix)),
  p_rt,
  width = 9,
  height = 4.5,
  dpi = 130
)

p_inf <- ggplot(inf_df, aes(date, med)) +
  geom_ribbon(aes(ymin = lo90, ymax = hi90), fill = "#41ab5d", alpha = 0.20) +
  geom_ribbon(aes(ymin = lo50, ymax = hi50), fill = "#41ab5d", alpha = 0.35) +
  geom_line(colour = "#006d2c", linewidth = 0.7) +
  labs(
    title = sprintf("Latent infections (%s)", suffix),
    x = NULL,
    y = "infections / day"
  )
ggsave(
  here("figures", sprintf("infections_%s.png", suffix)),
  p_inf,
  width = 9,
  height = 4.5,
  dpi = 130
)

obs_df <- data.frame(date = date_t, y = y_t)
p_pp <- ggplot(mu_df, aes(date, med)) +
  geom_ribbon(aes(ymin = lo90, ymax = hi90), fill = "#756bb1", alpha = 0.20) +
  geom_ribbon(aes(ymin = lo50, ymax = hi50), fill = "#756bb1", alpha = 0.35) +
  geom_line(colour = "#54278f", linewidth = 0.7) +
  geom_point(
    data = obs_df,
    aes(date, y),
    size = 0.9,
    colour = "grey20",
    alpha = 0.7
  ) +
  labs(
    title = sprintf("Expected reported cases vs observed (%s)", suffix),
    x = NULL,
    y = "cases / day"
  )
ggsave(
  here("figures", sprintf("pp_%s.png", suffix)),
  p_pp,
  width = 9,
  height = 4.5,
  dpi = 130
)

cat(sprintf("wrote figures/{rt,infections,pp}_%s.png\n", suffix))
