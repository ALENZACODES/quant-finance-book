# Independent R implementation of the Chapter 3 closed forms and the yield case study (base R only).
# Run from the repository root:  Rscript code/ch03/crosscheck.R
# Writes code/ch03/out/r_check.csv with (name, value) pairs compared with Python in test_crosschecks.py.

out <- list()
add <- function(name, value) out[[name]] <<- value

# reflection principle for the simple random walk, n = 20, a = 4
n <- 20; a <- 4; k <- 0:n; s <- 2 * k - n; p <- dbinom(k, n, 0.5)
add("reflExact", 100 * (sum(p[s >= a]) + sum(p[s > a])))
add("reflNaive", 100 * sum(p[s >= a]))

# Brownian maximum and the Broadie-Glasserman-Kou correction; zeta(1/2) from the alternating series
# accelerated by the Cohen-Rodriguez Villegas-Zagier algorithm
eta <- function(sv, N = 60) {
  d <- (3 + sqrt(8))^N; d <- (d + 1 / d) / 2; b <- -1; cc <- -d; tot <- 0
  for (j in 0:(N - 1)) { cc <- b - cc; tot <- tot + cc / (j + 1)^sv; b <- (j + N) * (j - N) * b / ((j + 0.5) * (j + 1)) }
  tot / d
}
zeta_half <- eta(0.5) / (1 - 2^(1 - 0.5))
beta <- -zeta_half / sqrt(2 * pi)
add("bgkBeta", beta)
add("bmMaxCont", 100 * 2 * pnorm(-1))
add("bmMaxBGKSixteen", 100 * 2 * pnorm(-(1 + beta * sqrt(1 / 16))))
add("bmMaxBGKBig", 100 * 2 * pnorm(-(1 + beta * sqrt(1 / 1024))))

# geometric Brownian motion
S0 <- 100; mu <- 0.08; TT <- 30
add("gbmMeanLow", S0 * exp(mu * TT))
add("gbmMedianLow", S0 * exp((mu - 0.25^2 / 2) * TT))
add("gbmMedianHigh", S0 * exp((mu - 0.5^2 / 2) * TT))
add("gbmProbLossLow", 100 * pnorm(-(mu - 0.25^2 / 2) * sqrt(TT) / 0.25))
add("gbmProbLossHigh", 100 * pnorm(-(mu - 0.5^2 / 2) * sqrt(TT) / 0.5))
add("weakMeanTwelve", 100 * (1 + 0.08 / 12)^12)
add("exactMeanOne", 100 * exp(0.08))

# Ornstein-Uhlenbeck
kap <- 1.2; th <- 0.03; sg <- 0.02
add("ouHalfLife", log(2) / kap)
add("ouSdInf", 100 * sg / sqrt(2 * kap))
add("ouProbNeg", 100 * pnorm(-th / (sg / sqrt(2 * kap))))
add("ouMeanOne", 100 * (th + (0.10 - th) * exp(-kap)))
add("ouPhiMonthly", exp(-kap / 12))
add("ouKendall", -(1 + 3 * exp(-kap / 12)) / 600)

# yield case: month-end values, OLS of x_{k+1} on x_k
d <- read.csv("data/daily_policy_bond.csv", check.names = FALSE)
d <- d[!is.na(d[["BD.CDN.5YR.DQ.YLD"]]), ]
d <- d[d$date <= "2026-09-30", ]   # last complete month-end
ym <- substr(d$date, 1, 7)
last <- tapply(seq_len(nrow(d)), ym, max)
y <- d[["BD.CDN.5YR.DQ.YLD"]][last[order(names(last))]]
fit <- lm(y[-1] ~ y[-length(y)]); phi <- unname(coef(fit)[2]); cst <- unname(coef(fit)[1])
se <- summary(fit)$coefficients[2, 2]; rs <- summary(fit)$sigma
kh <- -log(phi) * 12
add("caseN", length(y)); add("casePhi", phi); add("caseKappa", kh); add("caseTheta", cst / (1 - phi))
add("caseSigma", rs * sqrt(2 * kh / (1 - phi^2))); add("caseHalf", log(2) / kh)
add("caseADF", (phi - 1) / se)

# exit problems for mu t + sigma W
exit_p <- function(m, s, a, b) { k <- 2 * m / s^2; (1 - exp(k * b)) / (exp(-k * a) - exp(k * b)) }
exit_t <- function(m, s, a, b) { p <- exit_p(m, s, a, b); (a * p - b * (1 - p)) / m }
add("exitPup", 100 * exit_p(0.1, 0.2, 0.1, 0.1)); add("exitTau", exit_t(0.1, 0.2, 0.1, 0.1))
add("exitTauZero", 0.1 * 0.1 / 0.04)
add("exSixP", 100 * exit_p(0.05, 0.2, 0.1, 0.2)); add("exSixTau", exit_t(0.05, 0.2, 0.1, 0.2))

# Girsanov example and importance sampling
add("girExact", integrate(function(x) pmax(x - 0.5, 0) * dnorm(x, 0.4, 1), 0.5, Inf, rel.tol = 1e-12)$value)
pc <- pnorm(-3); add("isP", 1e4 * pc)
vis <- integrate(function(y) exp(-6 * y + 9) * dnorm(y, 3, 1), 3, Inf, rel.tol = 1e-12)$value - pc^2
add("isRatio", pc * (1 - pc) / vis)

# Feynman-Kac targets
d1 <- (log(1) + (0.03 + 0.02)) / 0.2; add("fkBS", 100 * pnorm(d1) - 100 * exp(-0.03) * pnorm(d1 - 0.2))
m <- 0.2 + 0.8 * exp(-3); v <- 0.25 * (1 - exp(-6)) / 3; add("fkOUExact", m^2 + v)

# exercises
add("exThreeMean", 100 * exp(0.16)); add("exThreeMedian", 100 * exp((0.08 - 0.03125) * 2))
add("exThreeLoss", 100 * pnorm(-(0.08 - 0.03125) * sqrt(2) / 0.25)); add("exFourHalf", log(2) / 0.5)

dir.create("code/ch03/out", showWarnings = FALSE)
write.csv(data.frame(name = names(out), value = sprintf("%.12f", unlist(out))), "code/ch03/out/r_check.csv",
          row.names = FALSE, quote = FALSE)
cat(length(out), "R checks written\n")
