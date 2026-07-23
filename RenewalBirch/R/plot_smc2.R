#!/usr/bin/env Rscript
# Plot the SMC^2 static-parameter posterior (sigma_rw, phi, day-of-week effect)
# and overlay the self-organizing-RW variant's estimates for comparison.
#
# Usage: Rscript R/plot_smc2.R [smc2_json] [rw_json]
# defaults: output/smc2.json  output/renewal.json

suppressMessages({
  library(jsonlite)
  library(ggplot2)
})

args <- commandArgs(TRUE)
smc2_json <- if (length(args) >= 1) args[1] else "output/smc2.json"
rw_json <- if (length(args) >= 2) args[2] else "output/renewal.json"

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
if (!grepl("^/", smc2_json)) {
  smc2_json <- here(smc2_json)
}
if (!grepl("^/", rw_json)) {
  rw_json <- here(rw_json)
}

# ---- SMC^2 theta-particles ----
s <- fromJSON(smc2_json, simplifyVector = FALSE)
sigma_rw <- vapply(s, function(p) p$sigma_rw, numeric(1))
phi <- vapply(s, function(p) p$phi, numeric(1))
w <- vapply(s, function(p) p$weight, numeric(1))
w <- w / sum(w)
dow_mat <- t(vapply(s, function(p) as.numeric(p$dow), numeric(7))) # ntheta x 7 (effect, avg 1)
ess <- 1 / sum(w^2)
cat(sprintf(
  "[smc2] ntheta=%d  theta-ESS=%.1f  sigma_rw=%.3f [%.3f,%.3f]  phi=%.2f [%.2f,%.2f]\n",
  length(s),
  ess,
  weighted.mean(sigma_rw, w),
  min(sigma_rw),
  max(sigma_rw),
  weighted.mean(phi, w),
  min(phi),
  max(phi)
))

# ---- RW variant: time-averaged per-trajectory statics (for overlay) ----
rw_sig <- rw_phi <- numeric(0)
if (file.exists(rw_json)) {
  rw <- fromJSON(rw_json, simplifyVector = FALSE)
  grab <- function(traj, key) {
    v <- vapply(
      traj$sample[[2]]$sample,
      function(st) if (!is.null(st[[key]])) st[[key]] else NA_real_,
      numeric(1)
    )
    mean(v, na.rm = TRUE)
  }
  rw_sig <- vapply(rw, grab, numeric(1), key = "sigma_rw")
  rw_phi <- vapply(rw, grab, numeric(1), key = "phi")
}

dir.create(here("figures"), showWarnings = FALSE)
theme_set(theme_minimal(base_size = 11))

wdens <- function(x, w, n = 512) {
  d <- density(x, weights = w, n = n)
  data.frame(x = d$x, y = d$y)
}

# sigma_rw
d1 <- wdens(sigma_rw, w)
p1 <- ggplot(d1, aes(x, y)) +
  geom_area(fill = "#2c7fb8", alpha = 0.35) +
  geom_line(colour = "#08306b")
if (length(rw_sig)) {
  p1 <- p1 +
    geom_density(
      data = data.frame(x = rw_sig),
      aes(x, after_stat(density)),
      inherit.aes = FALSE,
      colour = "#d95f02",
      linetype = "twodash"
    )
}
p1 <- p1 +
  labs(
    title = expression(sigma[rw]),
    subtitle = "blue = SMC²; orange dashed = RW variant",
    x = NULL,
    y = NULL
  )

# phi
d2 <- wdens(phi, w)
p2 <- ggplot(d2, aes(x, y)) +
  geom_area(fill = "#41ab5d", alpha = 0.35) +
  geom_line(colour = "#006d2c")
if (length(rw_phi)) {
  p2 <- p2 +
    geom_density(
      data = data.frame(x = rw_phi),
      aes(x, after_stat(density)),
      inherit.aes = FALSE,
      colour = "#d95f02",
      linetype = "twodash"
    )
}
p2 <- p2 +
  labs(title = expression(phi ~ "(overdispersion)"), x = NULL, y = NULL)

# day-of-week effect: weighted mean + 90% band per day
dw_summ <- data.frame(
  day = factor(
    c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"),
    levels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
  ),
  mean = sapply(1:7, function(k) weighted.mean(dow_mat[, k], w)),
  lo = sapply(1:7, function(k) as.numeric(quantile(dow_mat[, k], 0.05))),
  hi = sapply(1:7, function(k) as.numeric(quantile(dow_mat[, k], 0.95)))
)
p3 <- ggplot(dw_summ, aes(day, mean)) +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "grey60") +
  geom_pointrange(aes(ymin = lo, ymax = hi), colour = "#54278f") +
  labs(
    title = "Day-of-week reporting effect",
    subtitle = "posterior mean with 90% band (averages to 1)",
    x = NULL,
    y = "multiplier"
  )

# stack into one figure
suppressMessages({
  ok <- requireNamespace("patchwork", quietly = TRUE)
})
if (ok) {
  library(patchwork)
  fig <- (p1 | p2) /
    p3 +
    plot_annotation(title = "SMC² static-parameter posterior (renewal model)")
  ggsave(
    here("figures", "smc2_posterior.png"),
    fig,
    width = 9,
    height = 6.5,
    dpi = 130
  )
} else {
  ggsave(
    here("figures", "smc2_sigma_rw.png"),
    p1,
    width = 5,
    height = 3.5,
    dpi = 130
  )
  ggsave(
    here("figures", "smc2_phi.png"),
    p2,
    width = 5,
    height = 3.5,
    dpi = 130
  )
  ggsave(
    here("figures", "smc2_dow.png"),
    p3,
    width = 7,
    height = 3.5,
    dpi = 130
  )
}
cat("wrote SMC² posterior figure(s) to figures/\n")
