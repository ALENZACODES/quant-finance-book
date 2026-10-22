# Independent R implementation of the Chapter 9 results (base R only).
# Run from the repository root:  Rscript code/ch09/crosscheck.R
# Writes code/ch09/out/r_check.csv with (name, value, tol) rows compared with Python in test_crosschecks.py.
# The market series are rebuilt from the raw daily files with R's own month-end sampling, the trend
# program by explicit loops, the fee by its own high-water mark loop, the smile by lm(), the database
# formulas by pnorm() and dnorm(), the risk share by uniroot(), and drawdowns by cummax(); the closure and
# hurdle formulas by integrate(), the carry splits from the raw rate files, the attribution by lm(), and
# the white paper's calculation with dpois().

out <- list(); tols <- list()
add <- function(name, value, tol = 1e-8) { out[[name]] <<- value; tols[[name]] <<- tol }

f <- read.csv("data/ff_factors_monthly.csv"); f[, -1] <- f[, -1] / 100
d <- read.csv("data/fred_markets_daily.csv"); g <- read.csv("data/goc_bills_bonds.csv")
me <- function(dates, x) {            # last non-missing value in each month, keyed by yyyymm
  ok <- !is.na(x); k <- as.integer(substr(dates[ok], 1, 4)) * 100 + as.integer(substr(dates[ok], 6, 7))
  v <- tapply(x[ok], k, function(z) z[length(z)]); setNames(as.numeric(v), names(v))
}
price <- function(c, y, m) { n <- 2 * m; v <- (1 + y / 2)^(-n); c / y * (1 - v) + v }
bret <- function(y0, y1) price(y0, y1, 10 - 1 / 12) - 1 + y0 / 12
months <- f$month[f$month >= 199902 & f$month <= 202608]
lag1 <- function(s, k) s[as.character(prev(k))]
prev <- function(k) ifelse(k %% 100 == 1, k - 89, k - 1)
at <- function(s, k) as.numeric(s[as.character(k)])
y <- me(d$date, d$DGS10) / 100; yc <- me(g$date, g$BD.CDN.10YR.DQ.YLD) / 100; bc <- me(g$date, g$TB.CDN.90D.MID) / 100
oil <- me(d$date, d$DCOILWTICO); eur <- me(d$date, d$DEXUSEU); jpy <- me(d$date, d$DEXJPUS); cad <- me(d$date, d$DEXCAUS)
rf <- f$rf[match(months, f$month)]
R <- cbind(EQ = f$mkt_rf[match(months, f$month)],
           UST = bret(at(y, prev(months)), at(y, months)) - rf,
           GOC = bret(at(yc, prev(months)), at(yc, months)) - at(bc, prev(months)) / 12,
           OIL = at(oil, months) / at(oil, prev(months)) - 1,
           EUR = at(eur, months) / at(eur, prev(months)) - 1,
           JPY = at(jpy, prev(months)) / at(jpy, months) - 1,
           CAD = at(cad, prev(months)) / at(cad, months) - 1)
nm <- c("Eq", "Ust", "Goc", "Oil", "Eur", "Jpy", "Cad")
for (j in 1:7) add(paste0("tfVol", nm[j]), 100 * sd(R[, j], na.rm = TRUE) * sqrt(12), 1e-9)

program <- function(R, look, target = 0.20, cost = 0.0005) {
  Tn <- nrow(R); k <- ncol(R); W <- matrix(0, Tn, k)
  for (t in 1:Tn) {
    sg <- rep(NA, k); vo <- rep(NA, k)
    for (j in 1:k) {
      if (t >= look) { z <- R[(t - look + 1):t, j]; if (all(!is.na(z))) sg[j] <- sign(prod(1 + z) - 1) }
      z <- R[max(1, t - 23):t, j]; z <- z[!is.na(z)]
      if (length(z) >= 12) vo[j] <- sd(z) * sqrt(12)
    }
    ok <- !is.na(sg) & !is.na(vo) & vo > 0
    if (any(ok)) W[t, ok] <- sg[ok] * target / vo[ok] / sum(ok)
  }
  P <- numeric(Tn)
  for (t in 2:Tn) {
    wprev2 <- if (t >= 3) W[t - 2, ] else rep(0, k)
    x <- R[t, ]; x[is.na(x)] <- 0
    P[t] <- sum(W[t - 1, ] * x) - cost * sum(abs(W[t - 1, ] - wprev2))
  }
  P
}
keep <- months >= 200101
p <- program(R, 12)[keep]; mk <- months[keep]; rfk <- rf[keep]; eq <- R[keep, "EQ"]
sr <- function(x) sqrt(12) * mean(x) / sd(x)
mdd <- function(x) { w <- c(1, cumprod(1 + x)); max(1 - w / cummax(w)) }
add("tfSR", sr(p), 1e-9); add("tfMean", 100 * 12 * mean(p), 1e-9); add("tfVolP", 100 * sqrt(12) * sd(p), 1e-9)
add("tfCorrEq", cor(p, eq), 1e-9); add("tfMdd", 100 * mdd(p), 1e-9)
add("lbSRThree", sr(program(R, 3)[keep]), 1e-9); add("lbSRTwentyFour", sr(program(R, 24)[keep]), 1e-9)
# fees with a high-water mark crystallised each December
nav <- 1; hwm <- 1; net <- numeric(length(p))
for (t in seq_along(p)) {
  s0 <- nav; nav <- nav * (1 + rfk[t] + p[t]) * (1 - 0.02 / 12)
  if (mk[t] %% 100 == 12 || t == length(p)) { if (nav > hwm) nav <- nav - 0.2 * (nav - hwm); hwm <- max(hwm, nav) }
  net[t] <- nav / s0 - 1 - rfk[t]
}
add("tfSRnet", sr(net), 1e-9); add("tfMeanNet", 100 * 12 * mean(net), 1e-9)
# quarterly smile
q <- (mk %/% 100) * 10 + ((mk %% 100) - 1) %/% 3
cq <- table(q); qs <- as.numeric(names(cq)[cq == 3])
pq <- sapply(qs, function(z) prod(1 + p[q == z]) - 1); eqq <- sapply(qs, function(z) prod(1 + eq[q == z]) - 1)
fit <- summary(lm(pq ~ eqq + I(eqq^2)))$coefficients
add("slQuad", fit[3, 1], 1e-9); add("slQuadT", fit[3, 3], 1e-9); add("slLin", fit[2, 1], 1e-9)
# case study
A <- 0.6 * eq + 0.4 * R[keep, "UST"]; B <- 0.9 * A + 0.1 * net
add("csASR", sr(A), 1e-9); add("csBSR", sr(B), 1e-9); add("csAMdd", 100 * mdd(A), 1e-9); add("csBMdd", 100 * mdd(B), 1e-9)
# database formulas
sm <- function(mu, s, Tn, c) { se <- s / sqrt(Tn); a <- (c - mu) / se; mu + se * dnorm(a) / pnorm(a, lower.tail = FALSE) }
add("svBiasZeroTh", 100 * 12 * (sm(0.006, 0.03, 36, 0) - 0.006), 1e-9)
add("bfIncTheory", 100 * 12 * sm(0.006, 0.03, 24, 0), 1e-9)
add("exDBias", 100 * 12 * (sm(0.005, 0.04, 24, 0) - 0.005), 1e-9)
add("exEBias", 100 * 12 * 12 / 60 * (sm(0.005, 0.03, 12, 0) - 0.005), 1e-9)
add("pvTheory", 1 + 7 / (24 - 7 - 1), 1e-12); add("exFInf", 1 + 10 / 25, 1e-12)
# funds of funds
cm <- function(mu, s) mu * pnorm(mu / s) + s * dnorm(mu / s)
add("ffSepOne", 100 * 0.2 * cm(0.08, 0.15), 1e-10)
add("ffPoolTen", 100 * 0.2 * cm(0.08, 0.15 * sqrt(0.3 + 0.7 / 10)), 1e-10)
add("ffPoolLimit", 100 * 0.2 * cm(0.08, 0.15 * sqrt(0.3)), 1e-10)
add("exCSR", 0.2 * sqrt(20 / (1 + 19 * 0.1)), 1e-12)
# straddle identity and exercise numbers
r5 <- c(0.02, -0.01, 0.03, -0.02, 0.04); S5 <- cumsum(r5)
add("exAPnl", 1e4 * sum(c(0, S5[-5]) * r5), 1e-9)
add("exJMax", integrate(function(z) z * 20 * dnorm(z) * pnorm(z)^19, -Inf, Inf)$value, 1e-7)
share <- function(w, a, b, c) (w^2 * b + w * (1 - w) * c) / ((1 - w)^2 * a + 2 * w * (1 - w) * c + w^2 * b)
a <- 0.01; b <- 0.5625; cc <- 0.3 * 0.1 * 0.75
add("exHW", 100 * uniroot(function(w) share(w, a, b, cc) - 0.25, c(1e-6, 0.5), tol = 1e-14)$root, 1e-7)
add("exHShare", 100 * share(0.03, a, b, cc), 1e-10)
# bitcoin
cr <- read.csv("data/crypto_daily.csv"); cr <- cr[cr$date <= "2026-09-30", ]
bt <- me(cr$date, cr$CBBTCUSD); rb <- bt[-1] / bt[-length(bt)] - 1
add("btVol", 100 * sqrt(12) * sd(rb), 1e-9)
add("btGeo", 100 * (prod(1 + rb)^(12 / length(rb)) - 1), 1e-9)
px <- cr$CBBTCUSD[!is.na(cr$CBBTCUSD)]; add("btMddDaily", 100 * max(1 - px / cummax(px)), 1e-9)
kb <- as.integer(names(rb)); late <- kb >= 202001 & kb <= 202608
add("btCorrLate", cor(rb[late], f$mkt_rf[match(kb[late], f$month)]), 1e-9)
m15 <- kb[kb <= 202608]
e15 <- f$mkt_rf[match(m15, f$month)]; u15 <- R[match(m15, months), "UST"]; x15 <- rb[as.character(m15)] - f$rf[match(m15, f$month)]
P <- 0.6 * e15 + 0.4 * u15
add("btWQuarter", 100 * uniroot(function(w) share(w, var(P), var(x15), cov(P, x15)) - 0.25, c(1e-6, 0.9), tol = 1e-14)$root, 1e-7)
add("btShareTwo", 100 * share(0.02, var(P), var(x15), cov(P, x15)), 1e-9)
C2 <- 0.98 * P + 0.02 * x15; add("ccTwoSR", sr(C2), 1e-9)


# ---- closure at a loss limit: survival and survivors' mean by numerical integration of the density
cl_b <- -log(1 - 0.40); cl_s <- 0.15
dens <- function(x, T) { v <- cl_s * sqrt(T); (dnorm(x / v) - dnorm((x + 2 * cl_b) / v)) / v }   # zero drift
for (T in c(5, 10)) {
  P <- integrate(function(x) dens(x, T), -cl_b, Inf, rel.tol = 1e-12)$value
  M <- integrate(function(x) x * dens(x, T), -cl_b, Inf, rel.tol = 1e-12)$value / P
  tag <- if (T == 5) "Five" else "Ten"
  add(paste0("clSurv", tag), 100 * P, 1e-7); add(paste0("clMean", tag), 100 * M / T, 1e-6)
}
# ---- fees: hurdles by integration, the option value, the reset formula and a high-water mark loop
fe_f <- 0.2; fe_h <- 0.05
softfee <- function(x) pmin(pmax(x - fe_h, 0), fe_f * pmax(x, 0))
add("feSoft", 100 * integrate(function(x) softfee(x) * dnorm(x, 0.08, 0.12), -2, 3, rel.tol = 1e-12)$value, 1e-7)
add("feHard", 100 * integrate(function(x) fe_f * pmax(x - fe_h, 0) * dnorm(x, 0.08, 0.12), -2, 3, rel.tol = 1e-12)$value, 1e-7)
add("feOptFwd", 100 * fe_f * (2 * pnorm(0.05) - 1), 1e-9)
bs <- function(s, r, K) { d1 <- (log(1 / K) + r) / s + s / 2; pnorm(d1) - K * exp(-r) * pnorm(d1 - s) }
add("feOptAtm", 100 * fe_f * bs(0.10, 0.03, 1), 1e-9)
cc <- fe_f * bs(0.15, 0.03, 1); add("feTenClosed", 100 * (1 - (1 - cc)^10), 1e-9)
nav <- 1; hw <- 1; tot <- 0
for (gr0 in c(0.20, -0.25, 0.30)) { pre <- nav * (1 + gr0); fee <- fe_f * max(pre - hw, 0); nav <- pre - fee; hw <- max(hw, nav); tot <- tot + fee }
add("feExFeeH", 100 * tot, 1e-9); add("feExMark", 100 * (hw - 1), 1e-9)
# ---- carry: the Canadian dollar from the Chapter 5 file and the Canada 10 year bond
u <- read.csv("data/usdcad_rates.csv"); u <- u[u$date <= "2026-09-30", ]
S <- me(u$date, u$usdcad); ic <- me(u$date, u$cad_tbill_1m) / 100; iu <- me(u$date, u$us_tbill_1m) / 100
k <- names(S); n <- length(k)
spot <- S[-n] / S[-1] - 1; carry <- (ic[k[-n]] - iu[k[-n]]) / 12
ex <- (1 + ic[k[-n]] / 12) * S[-n] / S[-1] - (1 + iu[k[-n]] / 12)
add("cadEx", 100 * 12 * mean(ex), 1e-9); add("cadSpot", 100 * 12 * mean(spot), 1e-9); add("cadCarry", 100 * 12 * mean(carry), 1e-9)
add("cadSlope", unname(coef(lm(ex ~ carry))[2]), 1e-8)
y10 <- me(g$date, g$BD.CDN.10YR.DQ.YLD) / 100; y7 <- me(g$date, g$BD.CDN.7YR.DQ.YLD) / 100; b90 <- me(g$date, g$TB.CDN.90D.MID) / 100
gk <- intersect(intersect(names(y10), names(y7)), names(b90)); gk <- gk[as.integer(gk) <= 202608]
yr <- y10[gk] - (y10[gk] - y7[gk]) / 36; m <- length(gk)
expd <- (y10[gk] - b90[gk]) / 12 + price(y10[gk], yr, 10 - 1 / 12) - 1
real <- price(y10[gk][-m], yr[-1], 10 - 1 / 12) - 1 + y10[gk][-m] / 12 - b90[gk][-m] / 12
add("gcExp", 100 * 12 * mean(expd[-m]), 1e-9); add("gcReal", 100 * 12 * mean(real), 1e-9)
# ---- attribution regression of the programme on markets and absolute moves
A <- R[keep, ]; full <- rowSums(is.na(A)) == 0        # the months with all seven returns
A <- A[full, ]; pf <- p[full]
fit1 <- lm(pf ~ A); fit2 <- lm(pf ~ A + abs(A))
add("dcAlphaA", 100 * 12 * unname(coef(fit1)[1]), 1e-7); add("dcRtwoB", summary(fit2)$r.squared, 1e-8)
# ---- the protocol: issuance, the white paper's calculation and the Fisher test
add("ocSupply", sum(210000 * floor(5e9 / 2^(0:63))) / 1e8, 1e-12)
nak <- function(q, z) { p <- 1 - q; lam <- z * q / p; kk <- 0:z; 1 - sum(dpois(kk, lam) * (1 - (q / p)^(z - kk))) }
add("ocNakSix", nak(0.1, 6), 1e-9); add("exNNak", nak(0.2, 6), 1e-9); add("exNRuin", (0.2 / 0.8)^6, 1e-12)
kb2 <- kb[kb <= 202608]; e2 <- f$mkt_rf[match(kb2, f$month)]; r2 <- rb[as.character(kb2)]
ea <- kb2 < 202001; r1c <- cor(r2[ea], e2[ea]); r2c <- cor(r2[!ea], e2[!ea])
add("ocFisherZ", (atanh(r2c) - atanh(r1c)) / sqrt(1 / (sum(ea) - 3) + 1 / (sum(!ea) - 3)), 1e-9)

res <- data.frame(name = names(out), value = unlist(out), tol = unlist(tols))
write.csv(res, "code/ch09/out/r_check.csv", row.names = FALSE)
cat(nrow(res), "checks written\n")
