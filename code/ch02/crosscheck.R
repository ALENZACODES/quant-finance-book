# Independent R implementation of the Chapter 2 calculations (base R only).
# Run from the repository root:  Rscript code/ch02/crosscheck.R
# Writes code/ch02/out/r_check.csv with (name, value) pairs that test_ch02.py compares with Python.

whole_fund <- function(ct, cc, dt, dd, h = 0.08, carry = 0.20, k = 1.0, hard = FALSE) {
  o <- order(dt); gp <- numeric(length(dd)); lp <- numeric(length(dd))
  lp_t <- c(); lp_a <- c(); cum_tot <- 0; cum_gp <- 0
  for (j in o) {
    t <- dt[j]; x <- dd[j]; m <- ct <= t + 1e-12
    bal <- sum(cc[m] * (1 + h)^(t - ct[m])) - sum(lp_a * (1 + h)^(t - lp_t))
    a <- min(x, max(bal, 0)); x <- x - a; l <- a; g <- 0
    if (!hard && x > 0) {
      need <- (carry * (cum_tot + a - sum(cc[m])) - cum_gp) / (k - carry)
      y <- min(x, max(need, 0)); g <- g + k * y; l <- l + (1 - k) * y; x <- x - y
    }
    if (x > 0) { l <- l + (1 - carry) * x; g <- g + carry * x }
    lp[j] <- l; gp[j] <- g; cum_tot <- cum_tot + dd[j]; cum_gp <- cum_gp + g
    lp_t <- c(lp_t, t); lp_a <- c(lp_a, l)
  }
  list(lp = lp, gp = gp)
}

hf_fees <- function(gross, mgmt = 0.02, inc = 0.20, hurdle = 0, type = "hard", hwm = TRUE) {
  nav <- 100; mark <- 100; out <- matrix(0, length(gross), 2)
  for (t in seq_along(gross)) {
    pre <- nav * (1 + gross[t]) - mgmt * nav
    base <- if (hwm) mark else nav
    thr <- base * (1 + hurdle)
    fee <- if (pre <= thr) 0 else inc * (if (type == "hard") pre - thr else pre - base)
    out[t, ] <- c(fee, pre - fee); nav <- pre - fee; if (hwm) mark <- max(mark, nav)
  }
  out
}

bs_call <- function(S, K, T, r, s) {
  d1 <- (log(S / K) + (r + s^2 / 2) * T) / (s * sqrt(T)); d2 <- d1 - s * sqrt(T)
  S * pnorm(d1) - K * exp(-r * T) * pnorm(d2)
}

ta <- function(years, rc, G, B, L) {
  C <- D <- NAV <- numeric(years + 1); U <- 1
  for (t in 1:years) {
    r <- rc[min(t, length(rc))]; C[t + 1] <- r * U; U <- U - C[t + 1]
    D[t + 1] <- min(1, (t / L)^B) * NAV[t] * (1 + G)
    NAV[t + 1] <- NAV[t] * (1 + G) + C[t + 1] - D[t + 1]
  }
  list(C = C, D = D, NAV = NAV)
}

res <- list()
# single period waterfall, V = 180
w <- whole_fund(0, 100, 5, 180); res$spGP <- w$gp
res$spGPPart <- whole_fund(0, 100, 5, 180, k = 0.8)$gp
res$spGPHard <- whole_fund(0, 100, 5, 180, hard = TRUE)$gp
# Fund B
fees <- c(rep(2, 5), 0.015 * 30, 0.015 * 10, 0.015 * 10, 0)
ct <- c(0, 1, 1, 2, 0:8); cc <- c(30, 25, 20, 10, fees)
dt <- c(3, 5, 6, 8); dd <- c(75, 40, 0, 20)
wb <- whole_fund(ct, cc, dt, dd); res$bWfCarry <- sum(wb$gp); res$bWfGPD <- wb$gp[4]
res$bFeeTotal <- sum(fees)
npv <- function(r, t, cf) sum(cf / (1 + r)^t)
res$bLPirrWf <- 100 * uniroot(npv, c(-0.5, 1), t = c(ct, dt), cf = c(-cc, dd - wb$gp), tol = 1e-12)$root
# deal by deal, one deal at a time
dbd <- 0
for (i in 1:4) dbd <- dbd + whole_fund(c(0, 1, 1, 2)[i], c(30, 25, 20, 10)[i], dt[i], dd[i])$gp
res$bDbdCarry <- dbd
# hedge fund worked example
hf <- hf_fees(c(0.15, -0.10, 0.08, 0.20)); res$hfIncOne <- hf[1, 1]; res$hfIncFour <- hf[4, 1]
res$hfCloseFour <- hf[4, 2]
res$hfHardIncOne <- hf_fees(c(0.15, -0.10, 0.08, 0.20), hurdle = 0.05)[1, 1]
res$hfSoftIncOne <- hf_fees(c(0.15, -0.10, 0.08, 0.20), hurdle = 0.05, type = "soft")[1, 1]
res$hfNoHwmIncThree <- hf_fees(c(0.15, -0.10, 0.08, 0.20), hwm = FALSE)[3, 1]
# carry as options, sigma = 30%
H <- 100 * 1.08^5; vstar <- (H - 20) / 0.8
res$carryFullThirty <- bs_call(100, H, 5, 0.03, 0.3) - 0.8 * bs_call(100, vstar, 5, 0.03, 0.3)
res$carryHardThirty <- 0.2 * bs_call(100, H, 5, 0.03, 0.3)
res$optAtSigTwenty <- 0.2 * bs_call(100, 100, 1, 0.03, 0.2)
# Takahashi-Alexander
m <- ta(15, c(0.25, 1 / 3, 0.5), 0.12, 2.5, 12)
res$taPeakNav <- max(m$NAV); res$taDistTotal <- sum(m$D)
res$progNavSteady <- sum(ta(30, c(0.25, 1 / 3, 0.5), 0.12, 2.5, 12)$NAV)

dir.create("code/ch02/out", showWarnings = FALSE)
write.csv(data.frame(name = names(res), value = sprintf("%.12f", unlist(res))),
          "code/ch02/out/r_check.csv", row.names = FALSE)
cat(length(res), "R checks written\n")
