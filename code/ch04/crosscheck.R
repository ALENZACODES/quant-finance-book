# Independent R implementation of the Chapter 4 results and the forward rate case study (base R only).
# Run from the repository root:  Rscript code/ch04/crosscheck.R
# Writes code/ch04/out/r_check.csv with (name, value) pairs compared with Python in test_crosschecks.py.

out <- list()
add <- function(name, value) out[[name]] <<- value

# ---------------------------------------------------------------- discount curves
curveD <- function(kt, kD, x) {                  # log-linear in D, knot (0, 1) added
  t <- c(0, kt); l <- c(0, log(kD)); o <- order(t); t <- t[o]; l <- l[o]
  exp(approx(t, l, xout = x, rule = 2)$y)
}
boot <- function(bills, bonds, m = 2) {
  kt <- as.numeric(names(bills)); kD <- 1 / (1 + unlist(bills) * kt)
  for (Tm in sort(as.numeric(names(bonds)))) {
    cpn <- bonds[[as.character(Tm)]]; times <- (1:round(Tm * m)) / m
    f <- function(DT) cpn / m * sum(curveD(c(kt, Tm), c(kD, DT), times)) + DT - 1
    DT <- uniroot(f, c(1e-6, 1.5), tol = 1e-15)$root
    kt <- c(kt, Tm); kD <- c(kD, DT)
  }
  list(t = kt, D = kD)
}
D <- function(cv, x) curveD(cv$t, cv$D, x)
cv <- boot(list("0.5" = 0.04, "1" = 0.042), list("2" = 0.045, "3" = 0.047))
add("hxDTwo", D(cv, 2)); add("hxDThree", D(cv, 3)); add("hxDOneHalf", D(cv, 1.5))
add("hxZTwo", -100 * log(D(cv, 2)) / 2)
add("hxFwdOneTwo", 100 * (D(cv, 1) / D(cv, 2) - 1)); add("hxFwdTwoThree", 100 * (D(cv, 2) / D(cv, 3) - 1))
add("hxBondFour", 100 * (0.04 * sum(D(cv, 1:3)) + D(cv, 3)))
add("fraRepay", 1e6 * D(cv, 1) / D(cv, 2))

# ---------------------------------------------------------------- data
raw <- read.csv("data/goc_bills_bonds.csv", check.names = FALSE, na.strings = "")
raw$date <- as.Date(raw$date)
month_end <- function(df) {
  key <- format(df$date, "%Y-%m"); me <- df[!duplicated(key, fromLast = TRUE), ]
  last <- max(df$date); eom <- as.Date(format(seq(as.Date(format(last, "%Y-%m-01")), by = "month", length.out = 2)[2] - 1))
  if (last < eom - 5) me <- me[format(me$date, "%Y-%m") != format(last, "%Y-%m"), ]
  me
}
allcols <- c("TB.CDN.90D.MID", "TB.CDN.180D.MID", "TB.CDN.1Y.MID", "BD.CDN.2YR.DQ.YLD", "BD.CDN.3YR.DQ.YLD",
             "BD.CDN.5YR.DQ.YLD", "BD.CDN.7YR.DQ.YLD", "BD.CDN.10YR.DQ.YLD")
full <- raw[complete.cases(raw[, allcols]), c("date", allcols)]
me <- month_end(full); r <- me[nrow(me), ]
g <- boot(list("0.25" = r[[2]] / 100, "0.5" = r[[3]] / 100, "1" = r[[4]] / 100),
          list("2" = r[[5]] / 100, "3" = r[[6]] / 100, "5" = r[[7]] / 100, "7" = r[[8]] / 100, "10" = r[[9]] / 100))
add("gocZeroFive", -100 * log(D(g, 5)) / 5); add("gocZeroTen", -100 * log(D(g, 10)) / 10)
add("gocFwdOneTwo", 100 * (D(g, 1) / D(g, 2) - 1)); add("gocFwdSevenTen", 100 * (D(g, 7) / D(g, 10) - 1) / 3)

# ---------------------------------------------------------------- one-period market
A <- cbind(rep(1.02, 3), c(120, 100, 80), c(20, 0, 0)); p <- c(1, 100, 7)
psi <- solve(t(A), p)
add("psiUp", psi[1]); add("psiMid", psi[2]); add("psiDown", psi[3]); add("qrnMid", 100 * 1.02 * psi[2])
X <- pmax(110 - c(120, 100, 80), 0)
add("putFair", sum(psi * X)); th <- solve(A, X); add("repBond", th[1]); add("repCall", th[3])
# bounds for the K = 100 call when only bond and stock trade: q_up ranges over [0.1, 0.55]
add("boundLo", 0.1 * 20 / 1.02); add("boundHi", 0.55 * 20 / 1.02)

# ---------------------------------------------------------------- forwards and futures
add("fwdIndex", 100 * exp(0.015 * 0.5))
F0 <- 100 * exp(0.015 * 0.5); Ft <- 103 * exp(0.015 * 0.25)
add("fwdValueLater", (Ft - F0) * exp(-0.04 * 0.25))
add("convYield", 100 * (0.05 - log(78 / 80) / 0.5))
add("bandLo", 99.95 * exp(0.038)); add("bandHi", 100.05 * exp(0.044))
add("cipFwd", 1.37 * 1.0275 / 1.04); add("cipProfit", 1040000 * 1.36 - 1370000 * 1.0275)
ff <- function(k, sr, ss, rho, Tm) {
  B <- function(u) (1 - exp(-k * (Tm - u))) / k
  sr^2 * integrate(function(u) B(u)^2, 0, Tm, rel.tol = 1e-13)$value +
    rho * ss * sr * integrate(B, 0, Tm, rel.tol = 1e-13)$value
}
add("ffFivePos", 1e4 * ff(0.3, 0.01, 0.2, 0.5, 5)); add("ffTenNeg", 1e4 * ff(0.3, 0.01, 0.2, -0.5, 10))
add("ffOneZero", 1e4 * ff(0.3, 0.01, 0.2, 0, 1))
add("ffRhoStar", uniroot(function(rho) ff(0.3, 0.01, 0.2, rho, 5), c(-1, 1), tol = 1e-14)$root)

# ---------------------------------------------------------------- commodity term structure
sch <- function(S, tau, k = 1.2, a = log(70), s = 0.35) {
  e <- exp(-k * tau); exp(e * log(S) + (1 - e) * a + s^2 / (4 * k) * (1 - e^2))
}
add("schOneLow", sch(50, 1)); add("schOneHigh", sch(100, 1)); add("schFinf", exp(log(70) + 0.35^2 / 4.8))
add("schSFlat", exp(log(70) + 0.35^2 / 2.4))
k <- 1.2; h <- 1 / 12; tau <- 2 / 12
add("rollTheory", 100 * 12 * ((exp(-k * (tau - h)) - exp(-k * tau)) * 0.05 -
                               0.35^2 / (4 * k) * (exp(-2 * k * (tau - h)) - exp(-2 * k * tau))))

# ---------------------------------------------------------------- binomial trees
bs <- function(S, K, r, s, Tm, call = TRUE) {
  d1 <- (log(S / K) + (r + s^2 / 2) * Tm) / (s * sqrt(Tm)); d2 <- d1 - s * sqrt(Tm)
  if (call) S * pnorm(d1) - K * exp(-r * Tm) * pnorm(d2) else K * exp(-r * Tm) * pnorm(-d2) - S * pnorm(-d1)
}
crr_eu <- function(S, K, r, s, Tm, n, call = TRUE) {
  hh <- Tm / n; u <- exp(s * sqrt(hh)); d <- 1 / u; q <- (exp(r * hh) - d) / (u - d); j <- 0:n
  ST <- S * u^j * d^(n - j); pay <- if (call) pmax(ST - K, 0) else pmax(K - ST, 0)
  exp(-r * Tm) * sum(dbinom(j, n, q) * pay)
}
add("bsCall", bs(100, 110, 0.05, 0.2, 1)); add("bsPut", bs(100, 100, 0.05, 0.2, 1, FALSE))
add("crrFifty", crr_eu(100, 110, 0.05, 0.2, 1, 50)); add("crrHundredOne", crr_eu(100, 110, 0.05, 0.2, 1, 101))
add("crrFiveHundred", crr_eu(100, 110, 0.05, 0.2, 1, 500))
pp <- function(z, n) 0.5 + sign(z) * 0.5 * sqrt(1 - exp(-((z / (n + 1 / 3 + 0.1 / (n + 1)))^2) * (n + 1 / 6)))
lr <- function(S, K, r, s, Tm, n) {
  d1 <- (log(S / K) + (r + s^2 / 2) * Tm) / (s * sqrt(Tm)); d2 <- d1 - s * sqrt(Tm)
  q <- pp(d2, n); qs <- pp(d1, n); gg <- exp(r * Tm / n); u <- gg * qs / q; d <- (gg - q * u) / (1 - q)
  j <- 0:n; exp(-r * Tm) * sum(dbinom(j, n, q) * pmax(S * u^j * d^(n - j) - K, 0))
}
add("lrHundredOne", lr(100, 110, 0.05, 0.2, 1, 101))
am_put <- function(S, K, r, s, Tm, n) {
  hh <- Tm / n; u <- exp(s * sqrt(hh)); d <- 1 / u; q <- (exp(r * hh) - d) / (u - d); disc <- exp(-r * hh)
  V <- pmax(K - S * u^(0:n) * d^(n:0), 0)
  for (i in (n - 1):0) {
    V <- pmax(disc * (q * V[2:(i + 2)] + (1 - q) * V[1:(i + 1)]), K - S * u^(0:i) * d^(i:0))
  }
  V[1]
}
add("amPutThousand", am_put(100, 100, 0.05, 0.2, 1, 1000))
add("tpAmerican", max(5, 0.5 * 15 / 1.05)); add("perpBoundary", 100 * 2.5 / 3.5)
hh <- 1 / 1000; u <- exp(0.2 * sqrt(hh))
add("treeDelta", (crr_eu(100 * u, 110, 0.05, 0.2, 1 - hh, 999) - crr_eu(100 / u, 110, 0.05, 0.2, 1 - hh, 999)) / (100 * (u - 1 / u)))

# ---------------------------------------------------------------- case study
cc <- c("TB.CDN.180D.MID", "TB.CDN.1Y.MID", "BD.CDN.2YR.DQ.YLD", "BD.CDN.3YR.DQ.YLD")
cs <- raw[complete.cases(raw[, cc]), c("date", cc)]; mc <- month_end(cs)
f12 <- f23 <- numeric(nrow(mc))
for (i in seq_len(nrow(mc))) {
  b <- boot(list("0.5" = mc[i, 2] / 100, "1" = mc[i, 3] / 100), list("2" = mc[i, 4] / 100, "3" = mc[i, 5] / 100))
  f12[i] <- D(b, 1) / D(b, 2) - 1; f23[i] <- D(b, 2) / D(b, 3) - 1
}
y1 <- mc[, 3] / 100; n <- length(y1)
nw <- function(y, x, L) {
  fit <- lm(y ~ x); Xm <- cbind(1, x); e <- resid(fit); S <- crossprod(Xm * e)
  for (l in 1:L) { w <- 1 - l / (L + 1); G <- t(Xm[(l + 1):length(e), ] * e[(l + 1):length(e)]) %*% (Xm[1:(length(e) - l), ] * e[1:(length(e) - l)]); S <- S + w * (G + t(G)) }
  Bi <- solve(crossprod(Xm)); V <- Bi %*% S %*% Bi; list(b = coef(fit)[2], se = sqrt(V[2, 2]), r2 = summary(fit)$r.squared)
}
r1 <- nw(y1[13:n] - y1[1:(n - 12)], f12[1:(n - 12)] - y1[1:(n - 12)], 12)
r2 <- nw(y1[25:n] - y1[1:(n - 24)], f23[1:(n - 24)] - y1[1:(n - 24)], 24)
add("caseMonths", n); add("fbOneSlope", r1$b); add("fbOneSE", r1$se); add("fbOneRSq", r1$r2)
add("fbTwoSlope", r2$b); add("fbTwoSE", r2$se)
add("fwdErrMean", 1e4 * mean(f12[1:(n - 12)] - y1[13:n]))
add("naiveMAE", 1e4 * mean(abs(y1[1:(n - 12)] - y1[13:n])))

# ---------------------------------------------------------------- exercises
add("exOneDTwo", (1 - 0.035 / 1.03) / 1.035); add("exTwoF", 50 * exp(0.0225))
add("exThreeV", 55 * exp(-0.02 / 3) - 52 * exp(-0.05 / 3)); add("exFourF", 1.35 * 1.015 / 1.025)
add("exFourProfit", 1025000 * 1.34 - 1370250); add("exSixY", 100 * (0.06 + log(70 / 66)))
q <- 0.6; add("exSevenP", 0.4 * 5 / 1.02)
Vu <- max(0 - 0, (q * 0 + (1 - q) * 2.5) / 1.02); Vd <- max(7, (q * 2.5 + (1 - q) * 11.5) / 1.02)
add("exEightAm", max(2, (q * Vu + (1 - q) * Vd) / 1.02))
add("exNineRoll", 100 * (log(74 / 72) - log(73 / 70)))

df <- data.frame(name = names(out), value = sprintf("%.15g", unlist(out)))
write.csv(df, "code/ch04/out/r_check.csv", row.names = FALSE, quote = FALSE)
cat(nrow(df), "R checks written\n")
