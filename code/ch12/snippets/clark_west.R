# Expanding-window forecasts of the log equity premium from the Treasury bill rate,
# January 1965 on, against the historical mean: R2_OS and the Clark and West test.
# Reads data/goyal_welch_monthly.csv, built by scripts/pull_ch12.py (not shipped).
d <- read.csv("data/goyal_welch_monthly.csv")
lag1 <- function(v) c(NA, v[-length(v)])
x <- data.frame(month = d$month,
  dp = log(d$d12) - log(d$index), dy = log(d$d12) - log(lag1(d$index)),
  ep = log(d$e12) - log(d$index), de = log(d$d12) - log(d$e12),
  d[c("svar", "bm", "ntis", "tbl", "lty", "ltr")], tms = d$lty - d$tbl,
  dfy = d$baa - d$aaa, dfr = d$corpr - d$ltr, infl = lag1(d$infl),
  r = log1p(d$ret) - log1p(d$rf))
x <- x[complete.cases(x), ]          # the Python sample: every predictor present
f <- hm <- c(); oos <- which(x$month >= 196501)
for (J in oos) {
  xi <- x$tbl[1:(J - 2)]; yi <- x$r[2:(J - 1)]   # pairs (x_i, r_{i+1}) known before J
  b <- coef(lm(yi ~ xi))
  f <- c(f, b[1] + b[2] * x$tbl[J - 1]); hm <- c(hm, mean(yi))
}
r <- x$r[oos]
r2 <- 1 - sum((r - f)^2) / sum((r - hm)^2)
dd <- (r - hm)^2 - ((r - f)^2 - (hm - f)^2)
t <- mean(dd) / (sd(dd) / sqrt(length(dd)))
cat(sprintf("%d forecasts: R2_OS %.2f%%, Clark-West t %.2f, one-sided p %.3f\n",
            length(r), 100 * r2, t, 1 - pnorm(t)))
