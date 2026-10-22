# Appraisal smoothing of monthly market returns, its detection and its inversion.
f <- read.csv("data/ff_factors_monthly.csv"); r <- (f$mkt_rf + f$rf) / 100
n <- length(r); phi <- 0.6
# r*_t = (1 - phi) r_t + phi r*_{t-1}, started at r*_1 = r_1
rs <- as.numeric(stats::filter((1 - phi) * r, phi, method = "recursive", init = r[1]))
ar1 <- function(z) { z <- z - mean(z); sum(z[-1] * z[-length(z)]) / sum(z^2) }
ph <- ar1(rs)                                         # estimate of phi from the record
ru <- (rs[-1] - ph * rs[-n]) / (1 - ph)              # Geltner's inversion
lagm <- function(z, j) z[(4 - j):(n - j)]             # market lagged j months, aligned
dfit <- lm(rs[4:n] ~ lagm(r, 0) + lagm(r, 1) + lagm(r, 2) + lagm(r, 3))
cat(sprintf("phi hat %.3f, correlation of inverted with true %.4f\n",
            ph, cor(ru, r[-1])))
cat(sprintf("volatility %% a year: true %.1f, smoothed %.1f, inverted %.1f\n",
            100 * sqrt(12) * sd(r), 100 * sqrt(12) * sd(rs), 100 * sqrt(12) * sd(ru)))
cat(sprintf("beta %.2f, Dimson beta over four months %.2f\n",
            coef(lm(rs ~ r))[2], sum(coef(dfit)[-1])))
