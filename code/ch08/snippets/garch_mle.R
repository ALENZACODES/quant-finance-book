# Normal GARCH(1,1) by maximum likelihood on daily market returns in percent.
d <- read.csv("data/ff_daily.csv"); x <- d$mkt; n <- length(x)
nll <- function(p) {                      # p maps to mu, omega > 0, a, b, a + b < 1
  mu <- p[1]; om <- exp(p[2]); a <- plogis(p[3]); b <- (1 - a) * plogis(p[4])
  e <- x - mu; h <- numeric(n)
  h[1] <- mean((x - mean(x))^2)           # started at the sample variance
  for (t in 2:n) h[t] <- om + a * e[t - 1]^2 + b * h[t - 1]
  0.5 * sum(log(2 * pi * h) + e^2 / h)
}
p0  <- c(mean(x), log(0.05 * var(x)), qlogis(0.08), qlogis(0.90 / 0.92))
nm  <- optim(p0, nll, control = list(maxit = 4000, reltol = 1e-12))
fit <- optim(nm$par, nll, method = "BFGS", control = list(reltol = 1e-14))
cat(sprintf("convergence codes (0 = converged): Nelder-Mead %d, BFGS %d\n",
            nm$convergence, fit$convergence))
if (fit$convergence != 0) warning("BFGS did not converge; the fit is not reliable")
a <- plogis(fit$par[3]); b <- (1 - a) * plogis(fit$par[4]); om <- exp(fit$par[2])
cat(sprintf("a %.3f  b %.3f  persistence %.4f  half life %.0f days\n",
            a, b, a + b, log(0.5) / log(a + b)))
cat(sprintf("long run volatility %.1f%% a year, log likelihood %.1f\n",
            sqrt(252 * om / (1 - a - b)), -fit$value))
