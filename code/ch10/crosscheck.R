# Independent R implementation of the Chapter 10 results (base R only).
# Run from the repository root:  Rscript code/ch10/crosscheck.R
# Writes code/ch10/out/r_check.csv with (name, value, tol) rows compared with Python in test_crosschecks.py.
# The pro forma by explicit loops and uniroot(), the loan by stepping through its monthly payments, the
# loan vintages marked to the property index with their own loops, autocorrelations and betas with lm(), the rotation with optimize(), the concession by direct sums, the convenience yields and
# the inventory norms and gas spreads with their own loops, and the inflation regressions, also over
# overlapping long horizons, with lm() and a hand-written Newey-West variance.

out <- list(); tols <- list()
add <- function(name, value, tol = 1e-8) { out[[name]] <<- value; tols[[name]] <<- tol }

# ---- pro forma
noi <- 2.40e6 * (1 - 0.07) + 0.05e6 - 0.82e6; r <- 0.075; g <- 0.025; kap <- 0.10; T <- 10
cap <- (r - g) / (1 - kap); V <- noi * (1 - kap) / (r - g)
add("pfNoi", noi / 1e6); add("pfCap", 100 * cap); add("pfValue", V / 1e6)
dcf <- function(ec, sc = 0, TT = T) { s <- 0; for (t in 1:TT) s <- s + noi * (1 + g)^(t - 1) * (1 - kap) / (1 + r)^t
  s + noi * (1 + g)^TT / ec * (1 - sc) / (1 + r)^TT }
add("pfValueDcf", dcf(cap + 0.0025, 0.02) / 1e6); add("pfUpHundred", 100 * (dcf(cap + 0.01) / V - 1))
irr <- function(cf) uniroot(function(x) sum(cf / (1 + x)^(0:(length(cf) - 1))), c(-0.5, 1), tol = 1e-14)$root
cf <- c(-V, noi * (1 + g)^(0:(T - 1)) * (1 - kap)); cf[T + 1] <- cf[T + 1] + noi * (1 + g)^T / (cap + 0.0025) * 0.98
add("pfIrr", 100 * irr(cf), 1e-7)
h <- 1e-6; add("pfElast", (log(dcf(cap * (1 + h))) - log(dcf(cap * (1 - h)))) / (2 * h), 1e-7)

# ---- the loan, by stepping through the payments
const <- function(rate, yrs) { i <- rate / 12; n <- 12 * yrs; 12 * i / (1 - (1 + i)^(-n)) }
bal <- function(rate, yrs, t) { b <- 1; p <- const(rate, yrs) / 12; for (m in seq_len(12 * t)) b <- b * (1 + rate / 12) - p; b }
k <- const(0.0625, 25); add("lvK", 100 * k); add("lvBal", 100 * bal(0.0625, 25, 10), 1e-9)
loan <- min(noi / (1.25 * k), 0.65 * V); add("lvLoan", loan / 1e6); add("lvLtv", 100 * loan / V)
add("lvDscrAtMax", (noi / V) / (0.65 * k))
add("lvRateSwitch", 100 * uniroot(function(x) noi / (1.25 * const(x, 25)) - 0.65 * V, c(0.001, 0.2), tol = 1e-12)$root, 1e-7)
lam <- loan / V; add("lvLevRet", 100 * (0.075 - lam * 0.0625) / (1 - lam)); add("lvVolMult", 1 / (1 - lam))
add("lvCashYield", 100 * (noi * 0.9 - k * loan) / (V - loan))

# ---- smoothing
q <- read.csv("data/real_assets_quarterly.csv"); q <- q[q$date <= "2026-09-30", ]
lc <- diff(log(q$cre)); qd <- q$date[-1]
ar <- function(x) { x <- x[!is.na(x)]; unname(coef(lm(x[-1] ~ x[-length(x)]))[2]) }
pre <- lc[qd < "1996-01-01"]; post <- lc[qd > "1996-01-01"]
rr <- log1p(q$reit_reval[-1] / q$reit_mv[-nrow(q)]); reit <- rr[qd >= "1997-01-01"]
hp <- diff(log(q$hpi)); hp <- hp[!is.na(hp)]
add("smArPre", ar(pre)); add("smArPost", ar(post)); add("smArReit", ar(reit)); add("smArHpi", ar(hp))
add("smVolPre", 200 * sd(pre)); add("smVolPost", 200 * sd(post)); add("smVolReit", 200 * sd(reit)); add("smVolHpi", 200 * sd(hp))
uns <- function(x, a) (x[-1] - a * x[-length(x)]) / (1 - a)
add("smUvolPre", 200 * sd(uns(pre, ar(pre)))); add("smUvolPost", 200 * sd(uns(post, ar(post))))
add("smUvolHpi", 200 * sd(uns(hp, ar(hp))))
cc <- q$cre[q$date >= "2006-01-01" & q$date <= "2012-12-31"]; ccp <- q$cre[q$date >= "2006-01-01" & q$date <= "2008-12-31"]
add("smCreFall", 100 * (min(cc) / max(ccp) - 1))

# ---- loan-to-value through the cycle, vintage by vintage
m <- read.csv("data/real_assets_monthly.csv")
cycle <- function(from, to = "2100-01-01") {
  qq <- q[!is.na(q$cre) & q$date > from, ]; res <- NULL
  for (i in seq_len(nrow(qq))) {
    if (i + 20 > nrow(qq)) break
    if (qq$date[i] >= to) break
    rate <- m$gs10[m$date == qq$date[i]] / 100 + 0.018
    lt <- sapply(0:20, function(h) 0.65 * bal(rate, 25, h / 4) / (qq$cre[i + h] / qq$cre[i]))
    res <- rbind(res, data.frame(start = qq$date[i], peak = max(lt), end = lt[21]))
  }
  res
}
bal <- function(rate, yrs, t) { b <- 1; p <- const(rate, yrs) / 12; for (mm in seq_len(round(12 * t))) b <- b * (1 + rate / 12) - p; b }
cy <- cycle("1996-01-01")
add("lcN", nrow(cy)); add("lcShareEighty", 100 * mean(cy$peak > 0.8)); add("lcShareHundred", 100 * mean(cy$peak > 1))
add("lcWorstPeak", 100 * max(cy$peak), 1e-9); add("lcShareRefi", 100 * mean(cy$end > 0.65)); add("lcMedPeak", 100 * median(cy$peak), 1e-9)
ca <- cycle("1970-01-01"); ca <- ca[ca$start >= "1985-01-01" & ca$start < "1991-01-01", ]
add("lcApprPeak", 100 * max(ca$peak), 1e-9)

# ---- REITs and the market (the French library)
f <- read.csv("data/ff_factors_monthly.csv")
lm1 <- log1p((f$mkt_rf + f$rf) / 100); qk <- (f$month %/% 100) * 10 + ((f$month %% 100) - 1) %/% 3
nq <- tapply(lm1, qk, length); mq <- tapply(lm1, qk, sum)[nq == 3]
qkey <- as.integer(substr(qd, 1, 4)) * 10 + (as.integer(substr(qd, 6, 7)) - 1) %/% 3
reitq <- setNames(ifelse(qd >= "1997-01-01", rr, NA), qkey); creq <- setNames(lc, qkey)  # the break quarter is left out
keys <- intersect(as.character(qkey[qd >= "1997-01-01"]), names(mq))
x <- reitq[keys]; yv <- creq[keys]; mm <- mq[keys]
add("rtN", length(keys)); add("rtCorrReitMkt", cor(x, mm)); add("rtCorrCreMkt", cor(yv, mm)); add("rtCorrCreReit", cor(yv, x))
lagk <- function(s, k) { i <- match(keys, names(s)) - k; i[i < 1] <- NA; unname(s[i]) }
cl <- function(k) { z <- lagk(reitq, k); o <- !is.na(z); cor(yv[o], z[o]) }
add("rtLeadOne", cl(1)); add("rtLeadFour", cl(4))
X <- sapply(0:4, function(j) lagk(mm, j)); ok <- complete.cases(X)  # market lags within the sample
add("rtBetaCreDim", sum(coef(lm(yv[ok] ~ X[ok, ]))[-1]), 1e-8)
X2 <- sapply(0:4, function(j) lagk(reitq, j)); ok2 <- complete.cases(X2); add("rtSumReit", sum(coef(lm(yv[ok2] ~ X2[ok2, ]))[-1]), 1e-8)
yr <- substr(keys, 1, 4); full <- tapply(yr, yr, length) == 4
ya <- tapply(yv, yr, sum)[full]; xa <- tapply(x, yr, sum)[full]
add("rtCorrY", cor(ya, xa)); add("rtCorrYLag", cor(ya[-1], xa[-length(xa)]))

# ---- rotation and concession
vol <- function(t) 400 * (1 - exp(-0.06 * t))^3
lev <- function(T, r) (40 * vol(T) * exp(-r * T) - 1500) / (1 - exp(-r * T))
for (z in list(c(0.03, "Three"), c(0.04, "Four"), c(0.06, "Six"))) {
  rr0 <- as.numeric(z[1]); o <- optimize(function(T) -lev(T, rr0), c(10, 80), tol = 1e-10)
  add(paste0("tbF", z[2]), o$minimum, 1e-5); add(paste0("tbLev", z[2]), -o$objective, 1e-8)
  add(paste0("tbS", z[2]), optimize(function(T) -vol(T) * exp(-rr0 * T), c(5, 120), tol = 1e-10)$minimum, 1e-5)
}
add("tbMai", optimize(function(t) -vol(t) / t, c(5, 150), tol = 1e-10)$minimum, 1e-5)
for (z in list(c(30, "Thirty"), c(50, "Fifty"), c(99, "NinetyNine"))) {
  n <- as.numeric(z[1]); t <- 1:n; w <- 100 * 1.025^(t - 1) / 1.07^t
  add(paste0("tbCon", z[2]), sum(w)); add(paste0("tbDur", z[2]), sum(t * w) / sum(w))
}

# ---- storage
e <- read.csv("data/eia_curves_monthly.csv")
e$tb <- m$tbill[match(e$date, m$date)]
cyo <- log1p(e$tb / 100) - 4 * log(e$cl4 / e$cl1); cyg <- log1p(e$tb / 100) - 4 * log(e$ng4 / e$ng1)
norm_dev <- function(s) { mo <- substr(e$date, 6, 7); yy <- as.integer(substr(e$date, 1, 4)); out <- rep(NA, length(s))
  for (i in seq_along(s)) { if (is.na(s[i])) next
    past <- which(mo == mo[i] & yy < yy[i] & yy >= yy[i] - 5 & !is.na(s)); past <- past[order(-yy[past])]
    past <- past[yy[past] >= yy[i] - 5]
    # the previous five calendar years with data in that month, at least three
    if (length(past) >= 3) out[i] <- log(s[i]) - mean(log(s[past[1:min(5, length(past))]])) }
  out }
devo <- norm_dev(e$crude_stocks); devg <- norm_dev(e$gas_storage)
ok <- !is.na(cyo) & !is.na(devo); add("sgNOil", sum(ok)); add("sgCorrOil", cor(cyo[ok], devo[ok]))
add("sgSlopeOil", unname(coef(lm(cyo[ok] ~ devo[ok]))[2]))
okg <- !is.na(cyg) & !is.na(devg); add("sgCorrGas", cor(cyg[okg], devg[okg]))
bb <- !is.na(e$cl1) & !is.na(e$cl4) & e$date >= "1990-01-01"; add("sgBackShare", 100 * mean(e$cl4[bb] < e$cl1[bb]))
a <- e$date == "2020-04-01"; add("sgAprCy", 100 * cyo[a])
sl <- 12 * log(e$ng2 / e$ng1); mo <- substr(e$date, 6, 7)
add("sgSlopeSep", 100 * mean(sl[mo == "09"], na.rm = TRUE)); add("sgSlopeJan", 100 * mean(sl[mo == "01"], na.rm = TRUE))

# ---- inflation regressions with a Newey-West variance, two lags
dec <- m[substr(m$date, 6, 7) == "12" & m$date <= "2026-09-30", ]; yrs <- as.integer(substr(dec$date, 1, 4))
infl <- c(NA, diff(log(dec$cpi))); names(infl) <- yrs
q4 <- q[substr(q$date, 6, 7) == "10", ]; cre <- c(NA, diff(log(q4$cre))); names(cre) <- substr(q4$date, 1, 4)
farm <- c(NA, diff(log(dec$ppi_farm))); names(farm) <- yrs
mb <- log1p(m$tbill / 1200); by <- substr(m$date, 1, 4); bill <- tapply(mb, by, sum); bill[tapply(mb, by, length) < 12] <- NA
nw <- function(yv, X, L = 2) {
  X <- cbind(1, as.numeric(X)); fit <- lm.fit(X, as.numeric(yv)); u <- as.numeric(fit$residuals); n <- nrow(X)
  S <- crossprod(X * u); for (j in 1:L) { G <- crossprod(X[(j + 1):n, , drop = FALSE] * u[(j + 1):n], X[1:(n - j), , drop = FALSE] * u[1:(n - j)])
    S <- S + (1 - j / (L + 1)) * (G + t(G)) }
  B <- solve(crossprod(X)); V <- B %*% S %*% B; list(b = fit$coefficients, se = sqrt(diag(V)))
}
yy <- as.character(1971:2025)
for (z in list(list(cre, "Cre"), list(farm, "Farm"), list(bill, "Bill"))) {
  v <- z[[1]][yy]; res <- nw(v, infl[yy]); add(paste0("ihB", z[[2]]), res$b[2], 1e-8); add(paste0("ihSe", z[[2]]), res$se[2], 1e-8)
  add(paste0("ihReal", z[[2]]), 100 * mean(v - infl[yy]), 1e-8)
}
add("ihInflMean", 100 * mean(infl[yy]))

# ---- slopes of overlapping k year sums, Newey-West with k lags
roll <- function(x, k) { cs <- cumsum(x); out <- cs[k:length(x)] - c(0, cs)[1:(length(x) - k + 1)]; out }
for (z in list(list(cre, "Cre"), list(farm, "Farm"), list(bill, "Bill"))) {
  v <- z[[1]][yy]; ok <- !is.na(v) & !is.na(infl[yy]); vv <- as.numeric(v[ok]); ii <- as.numeric(infl[yy][ok])
  for (h in list(c(3, "Three"), c(5, "Five"), c(10, "Ten"))) { k <- as.integer(h[1])
    res <- nw(roll(vv, k), roll(ii, k), L = k)
    add(paste0("ihH", z[[2]], h[2]), res$b[2], 1e-8); add(paste0("ihHs", z[[2]], h[2]), res$se[2], 1e-8) }
}

# ---- gas: the August spread fixed by the curve against the spot spread that followed
sp <- log(m$gas); names(sp) <- m$date
lk <- c(); rl <- c()
for (i in which(substr(e$date, 6, 7) == "08")) {
  y <- substr(e$date[i], 1, 4); d1 <- paste0(y, "-09-01"); d4 <- paste0(y, "-12-01")
  if (is.na(e$ng1[i]) || is.na(e$ng4[i]) || !(d1 %in% names(sp)) || !(d4 %in% names(sp)) || is.na(sp[d1]) || is.na(sp[d4])) next
  lk <- c(lk, log(e$ng4[i] / e$ng1[i])); rl <- c(rl, sp[d4] - sp[d1])
}
add("sgLockN", length(lk)); add("sgLockMean", 100 * mean(lk)); add("sgRealMean", 100 * mean(rl)); add("sgRealSd", 100 * sd(rl))
add("sgGapSe", 100 * sd(lk - rl) / sqrt(length(lk)))

res <- data.frame(name = names(out), value = unlist(out), tol = unlist(tols))
write.csv(res, "code/ch10/out/r_check.csv", row.names = FALSE)
cat(nrow(res), "checks written\n")
