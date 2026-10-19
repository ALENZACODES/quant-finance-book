# Independent cross-check of Chapter 1 numbers in base R (no packages needed).
# Run from the repository root:  Rscript code/ch01/crosscheck.R
npv <- function(r, cf) sum(cf / (1 + r)^(seq_along(cf) - 1))
irr <- function(cf, lo = -0.5, hi = 1) uniroot(npv, c(lo, hi), cf = cf, tol = 1e-12)$root

g <- 1.15; inv <- c(20, 22, 20, 15, 10)
contrib <- c(inv + 2, rep(0, 5))
distrib <- c(rep(0, 5), inv * g^5) - c(rep(0, 5), 2, 2, 2, 2, 0)
fund <- distrib - contrib

mirr <- function(cf, f, re) {
  n <- length(cf) - 1; t <- 0:n
  pvn <- -sum(pmin(cf, 0) / (1 + f)^t); fvp <- sum(pmax(cf, 0) * (1 + re)^(n - t))
  (fvp / pvn)^(1 / n) - 1
}
jb <- function(x) {
  d <- x - mean(x); m2 <- mean(d^2)
  S <- mean(d^3) / m2^1.5; K <- mean(d^4) / m2^2 - 3
  c(S = S, K = K, JB = length(x) / 6 * (S^2 + K^2 / 4))
}
y <- read.csv("data/daily_policy_bond.csv", check.names = FALSE)
y <- y[!is.na(y[["BD.CDN.5YR.DQ.YLD"]]), ]
dy <- diff(y[["BD.CDN.5YR.DQ.YLD"]]) * 100
jbr <- jb(dy)
mkt <- c(2.0, -1.5, 3.0, 0.5, -2.5, 1.5); fnd <- c(1.2, -0.4, 2.1, 0.8, -1.0, 1.1)

out <- data.frame(
  name = c("fundIRR", "irrXY", "mirrMulti", "dwrPer", "crossRate",
           "gocDsd", "gocDskew", "gocDexk", "gocJB", "exBeta", "exRho", "exSpear"),
  value = c(100 * irr(fund), 100 * irr(c(-200, 150, 121)), 100 * mirr(c(-100, 230, -132), 0.08, 0.12),
            100 * irr(c(-100, -200, 288)), 100 * (sqrt(200 / 130) - 1),
            sd(dy), jbr[["S"]], jbr[["K"]], jbr[["JB"]],
            cov(fnd, mkt) / var(mkt), cor(fnd, mkt), cor(fnd, mkt, method = "spearman")))
dir.create("code/ch01/out", showWarnings = FALSE)
write.csv(out, "code/ch01/out/r_check.csv", row.names = FALSE)
print(out, digits = 8)
