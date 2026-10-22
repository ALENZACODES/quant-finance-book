# Independent R implementation of the Chapter 11 results (base R only).
# Run from the repository root:  Rscript code/ch11/crosscheck.R
# Writes code/ch11/out/r_check.csv with (name, value, tol) rows compared with Python in test_crosschecks.py.
# The waterfall by a tier-by-tier loop written afresh, IRRs by uniroot(), the PME and direct alpha of the
# listing, the CalPERS multiples and implied durations with aggregate(), the buyout schedule by its own
# loop, the cost of capital and the DCF from the data, and the expected PME and the lookback bound by
# numerical integration with integrate().

out <- list(); tols <- list()
add <- function(name, value, tol = 1e-8) { out[[name]] <<- value; tols[[name]] <<- tol }
irr <- function(t, cf, lo = -0.9, hi = 5) uniroot(function(x) sum(cf * (1 + x)^(-t)), c(lo, hi), tol = 1e-14)$root
irrc <- function(t, cf) uniroot(function(a) sum(cf * exp(-a * t)), c(-3, 3), tol = 1e-14)$root

# ---- the stylised fund and its whole fund waterfall
deals <- rbind(c(0, 3, 2.8), c(1, 5, 0.0), c(1, 5, 2.2), c(2, 6, 1.6), c(2, 6, 3.2), c(3, 7, 0.8), c(3, 8, 2.6), c(4, 9, 2.0))
yrs <- 0:10; C <- numeric(11); X <- numeric(11); inv <- numeric(11); Cg <- numeric(11)
for (j in 1:nrow(deals)) { b <- deals[j, 1]; s <- deals[j, 2]
  C[b + 1] <- C[b + 1] + 10; Cg[b + 1] <- Cg[b + 1] + 10; X[s + 1] <- X[s + 1] + 10 * deals[j, 3]
  if (s > b) inv[(b + 1):s] <- inv[(b + 1):s] + 10 }
for (y in 0:9) C[y + 1] <- C[y + 1] + if (y < 5) 2 else 0.015 * inv[y + 1]
waterfall <- function(C, X, h = 0.08, cy = 0.2) {
  hb <- 0; owed <- 0; lp <- 0; gp <- 0; called <- 0; pref <- 0; catch <- 0; lpf <- numeric(length(C))
  for (j in seq_along(C)) {
    if (j > 1) hb <- hb * (1 + h)
    hb <- hb + C[j]; owed <- owed + C[j]; called <- called + C[j]; x <- X[j]
    a <- min(x, hb); r1 <- min(a, owed); pref <- pref + a - r1; hb <- hb - a; owed <- owed - r1; lp <- lp + a; x <- x - a
    lpf[j] <- a
    if (x > 0 && hb <= 1e-12) {
      need <- max(cy * (lp - called) / (1 - cy) - gp, 0)   # GP's share of profit so far, with GP at 100%
      cu <- min(need, x); gp <- gp + cu; catch <- catch + cu; x <- x - cu
      lp <- lp + (1 - cy) * x; gp <- gp + cy * x; lpf[j] <- lpf[j] + (1 - cy) * x
    }
  }
  list(gp = gp, lp = lp, pref = pref, catch = catch, lpf = lpf)
}
w <- waterfall(C, X)
add("wfPaidIn", sum(C)); add("wfFees", sum(C) - 80); add("wfGross", sum(X)); add("wfGrossMoic", sum(X) / 80)
add("wfPref", w$pref); add("wfCatch", w$catch); add("wfGpTotal", w$gp); add("wfLpTotal", w$lp)
add("wfGpShare", 100 * w$gp / (sum(X) - sum(C)))
add("wfGrossIrr", 100 * irr(yrs, X - Cg), 1e-7); add("wfNetIrrFee", 100 * irr(yrs, X - C), 1e-7)
add("wfNetIrr", 100 * irr(yrs, w$lpf - C), 1e-7); add("wfNetTvpi", w$lp / sum(C))

# ---- the toy fund of the listing
t <- c(0, 1, 2, 4, 6); calls <- c(40, 35, 25, 0, 0); dists <- c(0, 0, 20, 70, 90); nav <- 30; I <- c(100, 108, 101, 125, 140)
gr <- I[5] / I
add("pkSnipPme", (sum(dists * gr) + nav) / sum(calls * gr))
add("pkSnipAlpha", 100 * irrc(c(t, 6), c((dists - calls) * gr, nav)), 1e-7)
add("pkSnipIrr", 100 * irr(c(t, 6), c(dists - calls, nav)), 1e-7); add("pkSnipTvpi", (sum(dists) + nav) / sum(calls))

# ---- expected PME of a fund with beta b and no alpha, by integration over the index's log return
# the distribution is a position of b in the index, rebalanced continuously and financed at r
epme <- function(b, r, mu = 0.07, s = 0.16, T = 5) { m <- (mu - s^2 / 2) * T; v <- s * sqrt(T)
  integrate(function(x) exp(b * x + (1 - b) * r * T - b * (b - 1) * s^2 * T / 2 - x) * dnorm(x, m, v),
            m - 15 * v, m + 15 * v, rel.tol = 1e-12)$value }
gs10 <- read.csv("data/real_assets_monthly.csv")
rf10 <- gs10$gs10[as.character(gs10$date) == "2026-09-01"] / 100
add("pkBetaHalf", epme(0.5, rf10), 1e-9); add("pkBetaOneHalf", epme(1.5, rf10), 1e-9); add("pkBetaTwo", epme(2, rf10), 1e-9)
add("yePme", epme(1.5, 0.04), 1e-9); add("yePmeLow", epme(0.5, 0.04), 1e-9)

# ---- CalPERS
d <- read.csv("data/calpers_pe_funds.csv"); d$nm <- d$not_meaningful %in% c("True", "TRUE", TRUE)
add("cvFunds", nrow(d)); add("cvCommitted", sum(d$committed) / 1e9); add("cvNav", sum(d$total_value - d$cash_out) / 1e9)
add("cvNm", sum(d$nm)); add("cvNmFrom", min(d$vintage[d$nm]))
p <- d[d$cash_in > 0, ]
pool <- aggregate(cbind(total_value, cash_out, cash_in) ~ vintage, data = p, FUN = sum)
pool$n <- aggregate(cash_in ~ vintage, data = p, FUN = length)$cash_in
for (v in list(c(2007, "Seven"), c(2015, "Fifteen"), c(2019, "Nineteen"), c(2023, "TwentyThree"))) {
  r <- pool[pool$vintage == as.numeric(v[1]), ]
  add(paste0("cvTvpi", v[2]), r$total_value / r$cash_in); add(paste0("cvDpi", v[2]), r$cash_out / r$cash_in) }
old <- d[d$vintage <= 2012, ]; add("cvOldN", nrow(old)); add("cvOldTvpi", sum(old$total_value) / sum(old$cash_in))
e <- d[!d$nm & !is.na(d$net_irr) & d$net_irr > 0.005 & d$total_value / d$cash_in > 1, ]
dur <- log(e$total_value / e$cash_in) / log1p(e$net_irr); bound <- 2026.25 - e$vintage
add("cvDurN", nrow(e)); add("cvDurMed", median(dur)); add("cvDurRatioMax", 100 * max(dur / bound)); add("cvDurShareMax", 100 * median(dur / bound))
add("cvDurPre", median(dur[e$vintage >= 2005 & e$vintage <= 2008])); add("cvDurMid", median(dur[e$vintage >= 2015 & e$vintage <= 2018]))

# ---- the buyout, by its own loop
lbo <- function(m0 = 10, mT = 10.5, lev = 5) {
  e0 <- 50; ev0 <- m0 * e0; debt <- lev * e0; eq0 <- ev0 * 1.02 - debt; cash <- 0; rev <- 250
  g <- c(0.06, 0.06, 0.05, 0.05, 0.04); mg <- c(0.205, 0.21, 0.215, 0.22, 0.22)
  for (y in 1:5) { cash <- cash * (1 + 0.08 * 0.75); r0 <- rev; rev <- rev * (1 + g[y]); eb <- rev * mg[y]
    int <- 0.08 * debt; f <- eb - 0.25 * max(eb - 0.03 * rev - int, 0) - 0.035 * rev - 0.10 * (rev - r0) - int
    rp <- min(max(f, 0), debt); debt <- debt - rp; cash <- cash + f - rp }
  eqT <- max(mT * eb - (debt - cash), 0)
  list(eT = eb, evT = mT * eb, ndT = debt - cash, eqT = eqT, eq0 = eq0, moic = eqT / eq0, irr = (eqT / eq0)^(1 / 5) - 1)
}
b <- lbo(); u <- lbo(lev = 0)
add("lxEt", b$eT); add("lxEvt", b$evT); add("lxNdt", b$ndT); add("lxEqt", b$eqT); add("lxMoic", b$moic); add("lxIrr", 100 * b$irr)
add("lxMoicU", u$moic); add("lxIrrU", 100 * u$irr); add("lxK", (1 + 0.08 * 0.75)^5)
add("lxBrG", 10.25 * (b$eT - 50)); add("lxBrM", 0.5 * (50 + b$eT) * 0.5); add("lxBrD", 250 - b$ndT); add("lxGain", b$eqT - b$eq0)
add("lxGridLoLo", 100 * lbo(8, 8)$irr); add("lxGridHiHi", 100 * lbo(12, 12)$irr); add("lxGridHiLo", 100 * lbo(12, 8)$irr); add("lxGridLoHi", 100 * lbo(8, 12)$irr)
add("lxMHi", (2.0 - 0.6 * 1.06^5) / 0.4); add("lxMLo", (1.1 - 0.6 * 1.06^5) / 0.4)

# ---- cost of capital and the DCF
m <- read.csv("data/real_assets_monthly.csv"); rf <- m$gs10[m$date == "2026-09-01"] / 100
erp <- read.csv("data/damodaran_erp.csv"); ERP <- erp$erp_fcfe[erp$year == 2025]
ind <- read.csv("data/damodaran_industry.csv"); row <- ind[ind$industry == "Machinery", ]
de <- row$debt_weight / row$equity_weight; det <- 0.3 / 0.7; tax <- 0.25; rd <- rf + 0.0175
buH <- row$beta / (1 + 0.75 * de); buP <- row$beta / (1 + de)
add("wkBuHam", buH); add("wkBuHp", buP); add("wkBrHam", buH * (1 + 0.75 * det)); add("wkBrHp", buP * (1 + det))
add("wkWHam", 100 * (0.7 * (rf + buH * (1 + 0.75 * det) * ERP) + 0.3 * rd * 0.75))
add("wkWHp", 100 * (0.7 * (rf + buP * (1 + det) * ERP) + 0.3 * rd * 0.75))
rU <- rf + buP * ERP; add("wkRu", 100 * rU)
reME <- rU + (rU - rd) * det * (1 - tax * rd / (1 + rd)); add("wkReMe", 100 * reME)
W <- 0.7 * reME + 0.3 * rd * 0.75; add("wkWme", 100 * W)
mg <- (18.4 + 1.2 + 0.9 - 0.5 + 0.6 - 0.4) / 120; rev <- 120; fcf <- numeric(5)
for (y in 1:5) { r0 <- rev; rev <- rev * (1 + c(0.05, 0.05, 0.04, 0.04, 0.03)[y])
  fcf[y] <- rev * (mg - 0.03) * 0.75 + 0.03 * rev - 0.035 * rev - 0.12 * (rev - r0) }
n6 <- rev * 1.025 * (mg - 0.03) * 0.75
dcf <- function(W) { tv <- n6 * (1 - 0.025 / 0.12) / (W - 0.025); sum(fcf / (1 + W)^(1:5)) + tv / (1 + W)^5 }
add("vlEv", dcf(W)); add("vlTv", n6 * (1 - 0.025 / 0.12) / (W - 0.025)); add("vlEq", dcf(W) - 25)
add("vlVu", dcf(rU)); add("vlShield", dcf(W) - dcf(rU)); add("vlCapEarn", fcf[1] / (W - 0.025)); add("vlEvEbitda", dcf(W) / (mg * 120))
add("vlNopatSix", n6)

# ---- discounts: the put by its formula, the lookback bound by integrating the law of the maximum
put <- function(T, s, r) { d1 <- (r + s^2 / 2) * sqrt(T) / s; d2 <- d1 - s * sqrt(T); exp(-r * T) * pnorm(-d2) - pnorm(-d1) }
look <- function(T, s) { v <- s * sqrt(T); mu <- -s^2 / 2
  tail <- function(x) pnorm((-x + mu * T) / v) + exp(2 * mu * x / s^2) * pnorm((-x - mu * T) / v)
  integrate(function(x) exp(x) * tail(x), 0, 20 * v + 5, rel.tol = 1e-12)$value }
add("dmChaffe", 100 * put(2, 0.45, rf)); add("dmChaffeZero", 100 * put(2, 0.45, 0)); add("dmLong", 100 * look(2, 0.45), 1e-8)
add("dmLsThirtyOne", 100 * look(1, 0.3), 1e-8); add("dmLsSixtyFour", 100 * look(4, 0.6), 1e-8)
comb <- 1 - 0.8 * (1 - put(2, 0.45, rf)); add("dmComb", 100 * comb); add("dmStake", 0.1 * (dcf(W) - 25) * (1 - comb))
add("ymPut", 100 * put(2, 0.4, 0.04)); add("ymLs", 100 * look(2, 0.4), 1e-8)

# ---- exercises
w1 <- waterfall(c(100, 0, 0, 0, 0, 0, 0), c(0, 0, 0, 60, 0, 0, 120))
add("yaGp", w1$gp); add("yaLp", w1$lp); add("yaPref", w1$pref); add("yaCatch", w1$catch)
add("yaLpIrr", 100 * irr(0:6, w1$lpf - c(100, 0, 0, 0, 0, 0, 0)), 1e-7)
add("ydAlpha", 100 * irrc(c(0, 4), c(-130, 160)), 1e-7); add("ydIrr", 100 * irr(c(0, 4), c(-100, 160)), 1e-7)
add("yfMoic", (1.8 - 0.6 * 1.0525^5) / 0.4); add("yfMoicLow", (1.2 - 0.6 * 1.0525^5) / 0.4)
add("yhV", 10 * 0.8 / 0.05); add("yhVhigh", 10 * (1 - 0.05 / 0.15) / 0.03)
add("yjHam", 1.3 / 1.375); add("yjHp", 1.3 / 1.5); add("ylComb", 100 * (1 - 0.75 / 1.3)); add("ylStake", 9 * 0.75 / 1.3)

res <- data.frame(name = names(out), value = unlist(out), tol = unlist(tols))
write.csv(res, "code/ch11/out/r_check.csv", row.names = FALSE)
cat(nrow(res), "checks written\n")
