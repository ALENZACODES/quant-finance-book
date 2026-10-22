# Independent R implementation of the Chapter 12 results (base R only).
# Run from the repository root:  Rscript code/ch12/crosscheck.R
# Writes code/ch12/out/r_check.csv with (name, value, tol) rows compared with Python in test_crosschecks.py.
# The equity premium predictors are rebuilt from the data file and every expanding-window least squares
# forecast is recomputed with qr.solve(); the Clark and West statistics, the certainty equivalents, the
# deflated Sharpe ratio, the expected maximum of normals by integrate(), Holm and Benjamini-Hochberg by
# p.adjust(), and the closed forms of the exercises.

out <- list(); tols <- list()
add <- function(name, value, tol = 1e-8) { out[[name]] <<- value; tols[[name]] <<- tol }

# ---- equity premium (skipped if the data file has not been built)
f <- "data/goyal_welch_monthly.csv"
if (file.exists(f)) {
  d <- read.csv(f)
  n0 <- nrow(d)
  x <- data.frame(month = d$month,
    dp = log(d$d12) - log(d$index), dy = log(d$d12) - log(c(NA, d$index[-n0])),
    ep = log(d$e12) - log(d$index), de = log(d$d12) - log(d$e12),
    svar = d$svar, bm = d$bm, ntis = d$ntis, tbl = d$tbl, lty = d$lty, ltr = d$ltr,
    tms = d$lty - d$tbl, dfy = d$baa - d$aaa, dfr = d$corpr - d$ltr, infl = c(NA, d$infl[-n0]),
    r = log(1 + d$ret) - log(1 + d$rf), rx = d$ret - d$rf, rf = d$rf)
  x <- x[complete.cases(x), ]
  nm <- c("dp", "dy", "ep", "de", "svar", "bm", "ntis", "tbl", "lty", "ltr", "tms", "dfy", "dfr", "infl")
  X <- as.matrix(x[, nm]); r <- x$r
  oos <- which(x$month >= 196501)
  Fc <- matrix(NA, length(oos), length(nm) + 2, dimnames = list(NULL, c("hm", nm, "ols")))
  for (k in seq_along(oos)) {
    j <- oos[k]; Xi <- X[1:(j - 2), , drop = FALSE]; yi <- r[2:(j - 1)]; xj <- X[j - 1, ]
    Fc[k, "hm"] <- mean(yi)
    for (c in nm) { b <- qr.solve(cbind(1, Xi[, c]), yi); Fc[k, c] <- b[1] + b[2] * xj[c] }
    b <- qr.coef(qr(cbind(1, Xi), tol = 1e-10), yi); b[is.na(b)] <- 0   # de = dp - ep and tms = lty - tbl exactly
    Fc[k, "ols"] <- sum(c(1, xj) * b)
  }
  ro <- r[oos]; hm <- Fc[, "hm"]
  comb <- rowMeans(Fc[, nm])
  r2 <- function(f) 1 - sum((ro - f)^2) / sum((ro - hm)^2)
  cw <- function(f) { a <- (ro - hm)^2 - ((ro - f)^2 - (hm - f)^2); mean(a) / (sd(a) / sqrt(length(a))) }
  tag <- c(dp = "Dp", dy = "Dy", ep = "Ep", de = "De", svar = "Svar", bm = "Bm", ntis = "Ntis", tbl = "Tbl",
           lty = "Lty", ltr = "Ltr", tms = "Tms", dfy = "Dfy", dfr = "Dfr", infl = "Infl", ols = "Ols")
  for (c in c(nm, "ols")) { add(paste0("eqRtwo", tag[[c]]), 100 * r2(Fc[, c]), 1e-7); add(paste0("eqCw", tag[[c]]), cw(Fc[, c]), 1e-7) }
  add("eqRtwoComb", 100 * r2(comb), 1e-7); add("eqCwComb", cw(comb), 1e-7)
  add("eqOosN", length(oos), 0)
  v <- sapply(oos, function(j) var(r[(j - 60):(j - 1)]))
  ce <- function(f) { w <- pmin(pmax(f / (3 * v), 0), 1.5); rp <- w * x$rx[oos] + x$rf[oos]; 12 * (mean(rp) - 1.5 * var(rp)) }
  add("eqCeHm", 100 * ce(hm), 1e-7); add("eqCeComb", 100 * (ce(comb) - ce(hm)), 1e-7)
  pv <- 1 - pnorm(sapply(nm, function(c) cw(Fc[, c])))
  for (a in list(c(0.10, "Ten"), c(0.05, "Five"))) {
    al <- as.numeric(a[1])
    add(paste0("ovHolm", a[2]), sum(p.adjust(pv, "holm") <= al), 0)
    add(paste0("ovBh", a[2]), sum(p.adjust(pv, "BH") <= al), 0)
    add(paste0("ovRaw", a[2]), sum(pv <= al), 0)
  }
}

# ---- deflated Sharpe ratio of the best equity premium model, from the saved forecasts
g <- "code/ch12/out/eq_forecasts.csv"
if (file.exists(g)) {
  e <- read.csv(g)
  wf <- function(f) pmin(pmax(f / (3 * e$var), 0), 1.5)
  ms <- sub("^f_", "", grep("^f_", names(e), value = TRUE)); ms <- ms[ms != "hm"]
  act <- sapply(ms, function(m) (wf(e[[paste0("f_", m)]]) - wf(e$f_hm)) * e$rx)
  srs <- apply(act, 2, function(a) mean(a) / sd(a))
  b <- which.max(srs); a <- act[, b]; Tn <- length(a)
  m2 <- mean((a - mean(a))^2); sk <- mean((a - mean(a))^3) / m2^1.5 * sqrt(Tn * (Tn - 1)) / (Tn - 2)
  ku <- {m4 <- mean((a - mean(a))^4); n <- Tn; g2 <- m4 / m2^2 - 3
         ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3)) + 3}
  se <- sqrt((1 - sk * srs[b] + (ku - 1) / 4 * srs[b]^2) / (Tn - 1))
  gam <- 0.5772156649015329; N <- length(ms)
  bench <- sd(srs) * ((1 - gam) * qnorm(1 - 1 / N) + gam * qnorm(1 - 1 / (N * exp(1))))
  add("ovEqSr", srs[b]); add("ovEqPsr", pnorm(srs[b] / se)); add("ovEqDsr", pnorm((srs[b] - bench) / se)); add("ovEqBench", bench)
}

# ---- expected maximum of N standard normals
emax <- function(N) integrate(function(z) z * N * dnorm(z) * pnorm(z)^(N - 1), -12, 12, rel.tol = 1e-12, subdivisions = 2000)$value
add("ovMaxTen", emax(10), 1e-8); add("ovMaxHundred", emax(100), 1e-8); add("ovMaxThousand", emax(1000), 1e-7)
ap <- function(N) (1 - 0.5772156649015329) * qnorm(1 - 1 / N) + 0.5772156649015329 * qnorm(1 - 1 / (N * exp(1)))
add("ovApproxHundred", ap(100)); add("ziMax", ap(100)); add("ziExact", emax(100), 1e-8)

# ---- overlapping labels, Black-Scholes, the no-trade band and the exercises
add("kfCorrThOne", 11 / 12); add("kfCorrThSix", 6 / 12)
d1 <- 0.5 * 0.2 * sqrt(0.25); add("hdBs", 100 * (pnorm(d1) - pnorm(d1 - 0.2 * sqrt(0.25))))
gm <- dnorm(d1) / (100 * 0.2 * sqrt(0.25)); add("hdWwAtm", (1.5 * 0.002 * 100 * gm^2 / 1)^(1 / 3))
add("zaRidge", 0.8 / 1.25); add("zbC", 2 / 3); add("zcVar", 0.307)
a <- tanh(0.5); add("zeWtwo", 2 * a * a); add("zeWone", 2 * a * 2 * (1 - a^2))
add("zgGain", 100 * 0.005 * (1 + 0.12^2) / (0.12^2 * 0.995))
add("zkPsr", pnorm(0.25 / sqrt((1 + 0.5 * 0.25 + 0.75 * 0.25^2) / 119)))
p <- c(0.001, 0.008, 0.012, 0.04, 0.2)
add("zjHolm", sum(p.adjust(p, "holm") <= 0.05), 0); add("zjBh", sum(p.adjust(p, "BH") <= 0.05), 0)
add("zjBonf", sum(p.adjust(p, "bonferroni") <= 0.05), 0)
add("zlH", 0.9 / 1.6)

res <- data.frame(name = names(out), value = unlist(out), tol = unlist(tols))
write.csv(res, "code/ch12/out/r_check.csv", row.names = FALSE, quote = FALSE)
cat(nrow(res), "R checks written\n")
