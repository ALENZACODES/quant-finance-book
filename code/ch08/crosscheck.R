# Independent R implementation of the Chapter 8 results (base R only).
# Run from the repository root:  Rscript code/ch08/crosscheck.R
# Writes code/ch08/out/r_check.csv with (name, value, tol) rows compared with Python in test_crosschecks.py.
# Methods differ from Python where possible: quantile(type = 1) for historical VaR, a hand-written
# GARCH recursion maximised with optim(BFGS) on transformed parameters, the generalised Pareto fit by
# optim() on its own likelihood, binomial tail sums by pbinom(), the Kupiec and Christoffersen tests from
# table(), drawdowns by cummax(), and smoothing by stats::filter(). The case study is rebuilt from the
# saved VaR series (out/case_var.csv) only for the counts and tests, not for the forecasts.

out <- list(); tols <- list()
add <- function(name, value, tol = 1e-8) { out[[name]] <<- value; tols[[name]] <<- tol }

d <- read.csv("data/ff_daily.csv"); r <- d$mkt / 100; n <- length(r); loss <- -r
f <- read.csv("data/ff_factors_monthly.csv"); x <- f$mkt_rf / 100; rfm <- f$rf / 100
mom <- function(z) { m <- mean(z); s <- sd(z); c(m, s, mean((z - m)^3) / mean((z - m)^2)^1.5,
                                                  mean((z - m)^4) / mean((z - m)^2)^2 - 3) }
M <- mom(r); mu <- M[1]; s <- M[2]
add("tdSd", 100 * s, 1e-10); add("tdSkew", M[3], 1e-8); add("tdKurt", M[4], 1e-8)
add("tdBeyondFive", sum(abs((r - mu) / s) > 5), 0)

# ---------------------------------------------------------------- VaR and ES
hv <- function(z, a) { L <- sort(-z, decreasing = TRUE); L[max(1, ceiling(length(L) * (1 - a)))] }
he <- function(z, a) { L <- sort(-z, decreasing = TRUE); mean(L[1:max(1, ceiling(length(L) * (1 - a)))]) }
add("vrNorN", 100 * (-(mu + s * qnorm(0.01))), 1e-10)
add("esNorS", 100 * (-mu + s * dnorm(qnorm(0.975)) / 0.025), 1e-10)
add("esHsN", 100 * he(r, 0.99), 1e-12)
add("vrHsS", 100 * hv(r, 0.975), 1e-12); add("esHsS", 100 * he(r, 0.975), 1e-12)
# historical VaR by R's own quantile rule (type 1 is the inverse of the empirical distribution)
add("vrHsN", 100 * quantile(loss, 0.99, type = 1, names = FALSE), 1e-12)
z <- qnorm(0.01); sk <- M[3]; ku <- M[4]
zc <- z + (z^2 - 1) * sk / 6 + (z^3 - 3 * z) * ku / 24 - (2 * z^3 - 5 * z) * sk^2 / 36
add("vrCfN", 100 * (-(mu + s * zc)), 1e-8)
ten <- exp(stats::filter(log1p(r), rep(1, 10), sides = 1)) - 1; ten <- ten[!is.na(ten)]
add("srTen", 100 * hv(ten, 0.99), 1e-9)

# ---------------------------------------------------------------- GARCH(1,1), normal
rp <- 100 * r
nll <- function(p) {
  m <- p[1]; om <- exp(p[2]); a <- plogis(p[3]); b <- (1 - a) * plogis(p[4])   # a + b < 1
  e <- rp - m; h <- numeric(n); h[1] <- var(e)
  for (t in 2:n) h[t] <- om + a * e[t - 1]^2 + b * h[t - 1]
  0.5 * sum(log(2 * pi * h) + e^2 / h)
}
p0 <- c(mean(rp), log(0.012), qlogis(0.1), qlogis(0.89 / 0.9))
fit <- optim(p0, nll, control = list(maxit = 4000, reltol = 1e-12))
fit <- optim(fit$par, nll, method = "BFGS", control = list(maxit = 500, reltol = 1e-14))
a <- plogis(fit$par[3]); b <- (1 - a) * plogis(fit$par[4])
add("gaAlphaN", a, 5e-3); add("gaBetaN", b, 5e-3); add("gaPersN", a + b, 5e-4)
add("gaLogLikN", -fit$value, 1e-6)

# ---------------------------------------------------------------- extreme values
u <- quantile(loss, 0.95, type = 7, names = FALSE); ex <- loss[loss > u] - u
gnll <- function(p) { xi <- p[1]; be <- exp(p[2]); if (any(1 + xi * ex / be <= 0)) return(1e10)
  length(ex) * log(be) + (1 + 1 / xi) * sum(log1p(xi * ex / be)) }
g <- optim(c(0.2, log(mean(ex))), gnll, control = list(reltol = 1e-12, maxit = 5000))
xi <- g$par[1]; be <- exp(g$par[2])
add("evXi", xi, 2e-3); add("evU", 100 * u, 1e-10); add("evNu", length(ex), 0)
vq <- u + be / xi * ((n / length(ex) * 0.001)^(-xi) - 1)
add("evVarT", 100 * vq, 2e-3); add("evEsT", 100 * (vq + be - xi * u) / (1 - xi), 3e-3)
Lp <- sort(loss[loss > 0], decreasing = TRUE)
add("evHillTwoHundred", 1 / mean(log(Lp[1:200]) - log(Lp[201])), 1e-10)

# ---------------------------------------------------------------- backtests and zones
add("znGreen", 100 * pbinom(4, 250, 0.01), 1e-10); add("znRed", 100 * (1 - pbinom(9, 250, 0.01)), 1e-8)
kup <- function(k, N, p) { ph <- k / N
  -2 * (((N - k) * log(1 - p) + k * log(p)) - ((N - k) * log(1 - ph) + ifelse(k > 0, k * log(ph), 0))) }
add("kpSixLR", kup(6, 250, 0.01), 1e-10); add("xdLR", kup(8, 250, 0.01), 1e-10)
add("xdP", pchisq(kup(8, 250, 0.01), 1, lower.tail = FALSE), 1e-8)
cv <- read.csv("code/ch08/out/case_var.csv")
for (m in c("hs", "gt", "fhs", "nor")) {
  hit <- as.integer(cv$loss > cv[[m]]); k <- sum(hit)
  add(paste0("ca", toupper(substr(m, 1, 1)), substr(m, 2, 9), "X"), k, 0)
  tb <- table(factor(head(hit, -1), 0:1), factor(tail(hit, -1), 0:1))
  p01 <- tb[1, 2] / sum(tb[1, ]); p11 <- tb[2, 2] / sum(tb[2, ]); p1 <- sum(tb[, 2]) / sum(tb)
  xl <- function(k, q) ifelse(k > 0, k * log(q), 0)
  l1 <- xl(tb[1, 1], 1 - p01) + xl(tb[1, 2], p01) + xl(tb[2, 1], 1 - p11) + xl(tb[2, 2], p11)
  l0 <- xl(tb[1, 1] + tb[2, 1], 1 - p1) + xl(tb[1, 2] + tb[2, 2], p1)
  pc <- pchisq(-2 * (l0 - l1), 1, lower.tail = FALSE)
  if (pc >= 0.001) add(paste0("ca", toupper(substr(m, 1, 1)), substr(m, 2, 9), "Pc"), pc, 1e-6)
}

# ---------------------------------------------------------------- ratios
th <- mean(x) / sd(x); TT <- length(x)
add("prSR", sqrt(12) * th, 1e-10)
sx <- mom(x)
add("prSeNn", sqrt(12) * sqrt((1 + th^2 / 2 - sx[3] * th + sx[4] / 4 * th^2) / TT), 1e-9)
xc <- x - mean(x); rho <- sapply(1:11, function(k) sum(xc[(k + 1):TT] * xc[1:(TT - k)]) / sum(xc^2))
add("prEta", 12 / sqrt(12 + 2 * sum((12 - 1:11) * rho)), 1e-10)
ind <- read.csv("data/industry10_monthly.csv"); p <- ind$HiTec / 100 - rfm
fl <- lm(p ~ x); add("prHtBeta", coef(fl)[[2]], 1e-10); add("prHtAlpha", 1200 * coef(fl)[[1]], 1e-9)
add("prHtIR", sqrt(12) * coef(fl)[[1]] / (sd(resid(fl)) * sqrt((TT - 1) / (TT - 2))), 1e-8)
add("prHtMsq", 1200 * mean(p) * sd(x) / sd(p), 1e-9)
dn <- pmin(p, 0); add("prHtSortino", sqrt(12) * mean(p) / sqrt(mean(dn^2)), 1e-10)
add("prHtOmega", mean(pmax(p, 0)) / mean(pmax(-p, 0)), 1e-10)

# ---------------------------------------------------------------- drawdowns
W <- cumprod(1 + r); P <- cummax(c(1, W))[-1]; dd <- 1 - W / P
add("ddMax", 100 * max(dd), 1e-10); add("ddShareTwenty", 100 * mean(dd > 0.2), 1e-10)
lx <- log1p(r); add("ddLawMean", 100 * var(lx) / (2 * mean(lx)), 1e-10)
add("ddEmpMean", 100 * mean(-log1p(-dd)), 1e-10)
yrs <- n / 252; add("ddCalmar", (W[n]^(1 / yrs) - 1) / max(dd), 1e-10)

# ---------------------------------------------------------------- smoothing
rm_ <- f$mkt_rf / 100 + rfm; phi <- 0.6
rs <- as.numeric(stats::filter((1 - phi) * rm_, phi, method = "recursive", init = rm_[1]))
add("smVolS", 100 * sqrt(12) * sd(rs), 1e-9)
ar <- function(z) { zc <- z - mean(z); sum(zc[-1] * zc[-length(zc)]) / sum(zc^2) }
ph <- ar(rs); add("smPhiHat", ph, 1e-10)
ru <- (rs[-1] - ph * rs[-length(rs)]) / (1 - ph); add("smVolU", 100 * sqrt(12) * sd(ru), 1e-9)
add("smSRS", sqrt(12) * mean(rs - rfm) / sd(rs - rfm), 1e-9)

# ---------------------------------------------------------------- exercises
add("xaVar", 10 * 0.012 * qnorm(0.99), 1e-10); add("xeVar", 100 * (0.02 + 0.007 / 0.25 * ((10000 / 500 * 0.001)^(-0.25) - 1)), 1e-10)
add("xbEsTwo", (0.04^2 * 200 + (0.05 - 0.04^2) * 100) / 0.05, 1e-10)
add("xdTail", 100 * (1 - pbinom(7, 250, 0.01)), 1e-8)

dir.create("code/ch08/out", showWarnings = FALSE)
res <- data.frame(name = names(out), value = sprintf("%.15g", unlist(out)), tol = unlist(tols))
write.csv(res, "code/ch08/out/r_check.csv", row.names = FALSE, quote = FALSE)
cat(nrow(res), "R checks written\n")
