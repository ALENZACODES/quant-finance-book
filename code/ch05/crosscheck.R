# Independent R implementation of the Chapter 5 results and the USD/CAD hedging case study (base R).
# Run from the repository root:  Rscript code/ch05/crosscheck.R
# Writes code/ch05/out/r_check.csv with (name, value, tol) rows compared with Python in test_crosschecks.py.
# Methods differ from Python where possible: Greeks from the closed forms written out again, the gamma
# peak by optimize(), implied volatilities by uniroot(), Crank-Nicolson with a hand-written Thomas
# solver, quadrature by integrate(), and the whole case study recomputed from the CSV file.

out <- list(); tols <- list()
add <- function(name, value, tol = 1e-8) { out[[name]] <<- value; tols[[name]] <<- tol }

bs <- function(S, K, r, s, T, call = TRUE, q = 0) {
  d1 <- (log(S / K) + (r - q + s^2 / 2) * T) / (s * sqrt(T)); d2 <- d1 - s * sqrt(T)
  if (call) S * exp(-q * T) * pnorm(d1) - K * exp(-r * T) * pnorm(d2)
  else K * exp(-r * T) * pnorm(-d2) - S * exp(-q * T) * pnorm(-d1)
}
dd <- function(S, K, r, s, T, q = 0) { d1 <- (log(S / K) + (r - q + s^2 / 2) * T) / (s * sqrt(T)); c(d1, d1 - s * sqrt(T)) }

# ---------------------------------------------------------------- closed form and Greeks
S0 <- 100; K <- 100; r <- 0.05; s <- 0.2; T <- 1
d <- dd(S0, K, r, s, T); c0 <- bs(S0, K, r, s, T); p0 <- bs(S0, K, r, s, T, FALSE)
add("callBase", c0); add("putBase", p0); add("dOne", d[1]); add("dTwo", d[2])
add("deltaC", pnorm(d[1])); add("deltaP", pnorm(d[1]) - 1)
add("gammaB", dnorm(d[1]) / (S0 * s * sqrt(T))); add("vegaB", S0 * dnorm(d[1]) * sqrt(T))
add("thetaC", -S0 * dnorm(d[1]) * s / (2 * sqrt(T)) - r * K * exp(-r * T) * pnorm(d[2]))
add("thetaP", -S0 * dnorm(d[1]) * s / (2 * sqrt(T)) + r * K * exp(-r * T) * pnorm(-d[2]))
add("rhoC", K * T * exp(-r * T) * pnorm(d[2])); add("rhoP", -K * T * exp(-r * T) * pnorm(-d[2]))
add("vannaB", -dnorm(d[1]) * d[2] / s); add("volgaB", S0 * dnorm(d[1]) * sqrt(T) * d[1] * d[2] / s)
gam <- function(x) { e <- dd(x, K, r, s, T); dnorm(e[1]) / (x * s * sqrt(T)) }
add("gammaPeakS", optimize(gam, c(60, 140), maximum = TRUE, tol = 1e-12)$maximum, 1e-7)
add("callDeep", bs(150, K, r, s, T))

# ---------------------------------------------------------------- dividends, Black-76, Garman-Kohlhagen
add("divYieldCall", bs(100, 100, r, s, 1, TRUE, 0.03)); add("divYieldPut", bs(100, 100, r, s, 1, FALSE, 0.03))
add("divYieldDelta", exp(-0.03) * pnorm(dd(100, 100, r, s, 1, 0.03)[1]))
b76 <- function(F, K, r, s, T, call = TRUE) exp(-r * T) * bs(F, K, 0, s, T, call)
add("blackCall", b76(70, 72, 0.04, 0.35, 0.5)); add("blackPut", b76(70, 72, 0.04, 0.35, 0.5, FALSE))
add("gkCall", 1e6 * bs(1.38, 1.40, 0.025, 0.07, 0.25, TRUE, 0.04))
add("gkPut", 1e6 * bs(1.38, 1.40, 0.025, 0.07, 0.25, FALSE, 0.04))
add("gkFwd", 1.38 * exp((0.025 - 0.04) * 0.25))
add("gkDelta", exp(-0.04 * 0.25) * pnorm(dd(1.38, 1.40, 0.025, 0.07, 0.25, 0.04)[1]))
pv <- 2 * exp(-r * 0.25) + 2 * exp(-r * 0.75); add("divPV", pv)
add("escCall", bs(100 - pv, 100, r, s, 1)); add("sigAdj", 100 * s * 100 / (100 - pv))
add("adjCall", bs(100 - pv, 100, r, s * 100 / (100 - pv), 1))

# ---------------------------------------------------------------- Monte Carlo theory
v <- s * sqrt(T); g <- exp(-2 * r * T)
EY2 <- g * (S0^2 * exp((2 * r + s^2) * T) * pnorm(d[1] + v) - 2 * K * S0 * exp(r * T) * pnorm(d[1]) + K^2 * pnorm(d[2]))
EYX <- g * (S0^2 * exp((2 * r + s^2) * T) * pnorm(d[1] + v) - K * S0 * exp(r * T) * pnorm(d[1]))
varY <- EY2 - c0^2; varX <- S0^2 * (exp(s^2 * T) - 1); rho <- (EYX - c0 * S0) / sqrt(varY * varX)
m <- (r - s^2 / 2) * T
f <- function(z) exp(-r * T) * pmax(S0 * exp(m + v * z) - K, 0)
zm <- m / v
EA <- integrate(function(z) f(z) * f(-z) * dnorm(z), -zm, zm, rel.tol = 1e-13)$value
corrA <- (EA - c0^2) / varY
add("mcSdY", sqrt(varY)); add("mcRho", rho); add("mcCorrAnti", corrA, 1e-7)
add("vrAntiTheory", 1 / (1 + corrA), 1e-7); add("vrCVTheory", 1 / (1 - rho^2))

# ---------------------------------------------------------------- finite differences (theta scheme, Thomas solver)
thomas <- function(a, b, c, d) {      # sub, main, super diagonals and right side
  n <- length(b); cp <- numeric(n); dp <- numeric(n)
  cp[1] <- c[1] / b[1]; dp[1] <- d[1] / b[1]
  for (i in 2:n) { den <- b[i] - a[i] * cp[i - 1]; cp[i] <- c[i] / den; dp[i] <- (d[i] - a[i] * dp[i - 1]) / den }
  x <- numeric(n); x[n] <- dp[n]; for (i in (n - 1):1) x[i] <- dp[i] - cp[i] * x[i + 1]; x
}
fd <- function(M, Nt, th, S0 = 100, K = 100, r = 0.05, s = 0.2, T = 1, width = 6) {
  L <- width * s * sqrt(T); x <- log(S0) + seq(-L, L, length.out = M + 1); dx <- x[2] - x[1]; S <- exp(x)
  a <- s^2 / 2; b <- r - s^2 / 2
  lo <- a / dx^2 - b / (2 * dx); di <- -2 * a / dx^2 - r; up <- a / dx^2 + b / (2 * dx)
  V <- pmax(S - K, 0); dt <- T / Nt; n <- M - 1
  for (k in 1:Nt) {
    tau1 <- k * dt; Vi <- V[2:M]
    rhs <- Vi + (1 - th) * dt * (lo * V[1:(M - 1)] + di * Vi + up * V[3:(M + 1)])
    b1 <- S[M + 1] - K * exp(-r * tau1)
    rhs[n] <- rhs[n] + th * dt * up * b1
    new <- if (th == 0) rhs else thomas(rep(-th * dt * lo, n), rep(1 - th * dt * di, n), rep(-th * dt * up, n), rhs)
    V <- c(0, new, b1)
  }
  V[M / 2 + 1]
}
add("fdCNValue", fd(400, 200, 0.5)); add("fdErrImpl", abs(fd(400, 200, 1) - c0))
dx <- 2 * 6 * 0.2 / 400; add("fdErrExpl", abs(fd(400, ceiling(0.5 * 0.04 / (0.4 * dx^2)), 0) - c0), 1e-6)

# ---------------------------------------------------------------- implied volatility
ivu <- function(p, S, K, r, T, call = TRUE) uniroot(function(x) bs(S, K, r, x, T, call) - p, c(1e-4, 5), tol = 1e-15)$root
add("ivExPrice", bs(100, 120, r, 0.3, 0.5))
F <- 100 * exp(r * 0.5); x <- log(F / 120); vs <- sqrt(2 * abs(x)); add("ivExStart", vs / sqrt(0.5))
target <- bs(100, 120, r, 0.3, 0.5) / (exp(-r * 0.5) * F)
ncall <- function(v) pnorm(x / v + v / 2) - exp(-x) * pnorm(x / v - v / 2)
vk <- vs; steps <- c()
for (i in 1:3) { vk <- vk - (ncall(vk) - target) / dnorm(x / vk + vk / 2); steps <- c(steps, vk / sqrt(0.5)) }
add("ivExStepA", steps[1], 1e-7); add("ivExStepB", steps[2], 1e-7)

# Merton jump diffusion smiles
mjd <- function(S, K, r, s, T, lam, mu, dl, call) {
  k <- exp(mu + dl^2 / 2) - 1; lp <- lam * (1 + k) * T; tot <- 0
  for (n in 0:79) tot <- tot + dpois(n, lp) * bs(S, K, r - lam * k + n * log(1 + k) / T, sqrt(s^2 + n * dl^2 / T), T, call)
  tot
}
ms <- seq(-0.45, 0.35, length.out = 33); idx <- c(which.min(abs(ms + 0.1)), which.min(abs(ms)), which.min(abs(ms - 0.1)))
nm <- c("Short", "Mid", "Long"); Ts <- c(0.1, 0.5, 2)
for (j in 1:3) {
  T_ <- Ts[j]; F_ <- 100 * exp(r * T_); iv <- numeric(3)
  for (i in 1:3) { Kk <- F_ * exp(ms[idx[i]]); cl <- ms[idx[i]] >= 0
    iv[i] <- ivu(mjd(100, Kk, r, 0.15, T_, 0.5, -0.1, 0.15, cl), 100, Kk, r, T_, cl) }
  add(paste0("atmMer", nm[j]), 100 * iv[2], 1e-7); add(paste0("skewMer", nm[j]), 100 * (iv[1] - iv[3]), 1e-6)
}

# Heston by the Lewis single integral, written with R complex arithmetic
hcf <- function(u, S, r, T, v0, ka, th, xi, rho) {
  iu <- 1i * u; dq <- sqrt((rho * xi * iu - ka)^2 + xi^2 * (iu + u^2))
  gq <- (ka - rho * xi * iu - dq) / (ka - rho * xi * iu + dq); e <- exp(-dq * T)
  C <- (ka * th / xi^2) * ((ka - rho * xi * iu - dq) * T - 2 * log((1 - gq * e) / (1 - gq)))
  D <- (v0 / xi^2) * (ka - rho * xi * iu - dq) * (1 - e) / (1 - gq * e)
  exp(iu * (log(S) + r * T) + C + D)
}
heston <- function(S, K, r, T, call = TRUE, v0 = 0.04, ka = 2, th = 0.04, xi = 0.5, rho = -0.7) {
  F <- S * exp(r * T); k <- log(F / K)
  f <- function(u) Re(exp(1i * u * k) * hcf(u - 0.5i, S, r, T, v0, ka, th, xi, rho) * exp(-1i * (u - 0.5i) * log(F))) / (u^2 + 0.25)
  val <- integrate(f, 0, Inf, rel.tol = 1e-13, abs.tol = 1e-15, subdivisions = 5000)$value
  c <- exp(-r * T) * (F - sqrt(F * K) / pi * val)
  if (call) c else c - exp(-r * T) * (F - K)
}
add("hesFourier", heston(100, 100 * exp(r * 0.5), r, 0.5), 1e-8)
for (j in 1:3) {
  T_ <- Ts[j]; F_ <- 100 * exp(r * T_); iv <- numeric(3)
  for (i in 1:3) { Kk <- F_ * exp(ms[idx[i]]); cl <- ms[idx[i]] >= 0
    iv[i] <- ivu(heston(100, Kk, r, T_, cl), 100, Kk, r, T_, cl) }
  add(paste0("atmHes", nm[j]), 100 * iv[2], 1e-6); add(paste0("skewHes", nm[j]), 100 * (iv[1] - iv[3]), 1e-6)
}

# ---------------------------------------------------------------- hedging: leading-order error, gaps, Leland
g2 <- function(t, S0, K, r, s, T, mu) {
  tau <- T - t; s2 <- s^2 * t; vv <- s^2 * tau / 2; M <- log(S0) + (mu - s^2 / 2) * t
  cc <- log(K) - (r + s^2 / 2) * tau; a <- M + 2 * s2
  1 / (2 * pi * s^2 * tau) * sqrt(2 * pi * vv) * exp(2 * M + 2 * s2) * dnorm(a, cc, sqrt(s2 + vv))
}
hsd <- function(n, mu = 0.10) {
  f <- function(u) sapply(u, function(w) 2 * w * exp(-2 * r * (T - w^2)) * s^4 * g2(T - w^2, S0, K, r, s, T, mu))
  sqrt(T / n / 2 * integrate(f, 0, sqrt(T), rel.tol = 1e-11, subdivisions = 2000)$value)
}
add("hedgeThThirteen", hsd(13), 1e-7); add("hedgeThFiftyTwo", hsd(52), 1e-7)
add("hedgeThDaily", hsd(252), 1e-7); add("hedgeThQuarter", hsd(1008), 1e-7)
add("volGap", bs(100, 100, r, 0.2, 1) - bs(100, 100, r, 0.15, 1)); add("underGap", bs(100, 100, r, 0.2, 1) - bs(100, 100, r, 0.25, 1))
sL <- 0.2 * sqrt(1 + sqrt(2 / pi) * 0.004 / (0.2 * sqrt(1 / 52)))
add("lelandVol", 100 * sL); add("lelandPrice", bs(100, 100, r, sL, 1)); add("costInit", 0.002 * pnorm(d[1]) * 100)

# ---------------------------------------------------------------- case study recomputed from the data file
df <- read.csv("data/usdcad_rates.csv", stringsAsFactors = FALSE)
ff <- function(x) { for (i in seq_along(x)) if (is.na(x[i]) && i > 1) x[i] <- x[i - 1]; x }
df$cad_tbill_1m <- ff(df$cad_tbill_1m); df$us_tbill_1m <- ff(df$us_tbill_1m)
df <- df[!is.na(df$usdcad) & !is.na(df$cad_tbill_1m) & !is.na(df$us_tbill_1m), ]
dt <- as.Date(df$date); S <- df$usdcad; ym <- format(dt, "%Y-%m")
me <- which(c(ym[-1] != ym[-length(ym)], TRUE))
# a month counts only once it is over; the file is cut at the end of September 2026, so its last
# month is complete, and a file running into a later month has that unfinished month dropped
lastday <- seq(as.Date(format(dt[length(dt)], "%Y-%m-01")), by = "month", length.out = 2)[2] - 1
if (lastday > as.Date("2026-09-30")) me <- me[-length(me)]
cont <- function(y) log(1 + y / 100 * 30 / 365) / (30 / 365)
gkd <- function(S, K, rd, rf, s, tau) { e <- dd(S, K, rd, s, tau, rf); c(exp(-rf * tau) * pnorm(e[1]), exp(-rf * tau) * dnorm(e[1]) / (S * s * sqrt(tau)), S * exp(-rf * tau) * dnorm(e[1]) * sqrt(tau)) }
res <- NULL
for (j in 1:(length(me) - 1)) {
  i0 <- me[j]; i1 <- me[j + 1]; if (i0 <= 63) next
  tau0 <- as.numeric(dt[i1] - dt[i0]) / 365
  rr <- log(S[(i0 - 62):i0] / S[(i0 - 63):(i0 - 1)]); span <- as.numeric(dt[i0] - dt[i0 - 63]) / 365
  sh <- sqrt(sum(rr^2) / span); rd <- cont(df$cad_tbill_1m[i0]); rf <- cont(df$us_tbill_1m[i0])
  Kx <- S[i0] * exp((rd - rf) * tau0); V0 <- bs(S[i0], Kx, rd, sh, tau0, TRUE, rf)
  gg <- gkd(S[i0], Kx, rd, rf, sh, tau0); D <- gg[1]; cash <- V0 - D * S[i0]; gts <- 0
  for (k in i0:(i1 - 1)) {
    h <- as.numeric(dt[k + 1] - dt[k]) / 365; tau <- as.numeric(dt[i1] - dt[k]) / 365
    G <- gkd(S[k], Kx, rd, rf, sh, tau)[2]; R_ <- S[k + 1] / S[k] - 1
    gts <- gts + exp(-rd * as.numeric(dt[k] - dt[i0]) / 365) * 0.5 * G * S[k]^2 * (sh^2 * h - R_^2)
    cash <- cash * exp(rd * h) + D * S[k + 1] * (exp(rf * h) - 1)
    if (k + 1 < i1) { Dn <- gkd(S[k + 1], Kx, rd, rf, sh, as.numeric(dt[i1] - dt[k + 1]) / 365)[1]; cash <- cash - (Dn - D) * S[k + 1]; D <- Dn }
  }
  pnl <- (cash + D * S[i1] - max(S[i1] - Kx, 0)) * exp(-rd * tau0)
  life <- log(S[(i0 + 1):i1] / S[i0:(i1 - 1)]); sr <- sqrt(sum(life^2) / tau0); bp <- 1e4 / S[i0]
  res <- rbind(res, c(pnl * bp, gts * bp, sh, sr, gg[3] * bp))
}
colnames(res) <- c("pnl", "gts", "sh", "sr", "vega"); res <- as.data.frame(res)
add("fxN", nrow(res)); add("fxPnlMean", mean(res$pnl), 1e-7); add("fxPnlSd", sd(res$pnl), 1e-7)
add("fxShMean", 100 * mean(res$sh)); add("fxSrMean", 100 * mean(res$sr)); add("fxCorrGT", cor(res$pnl, res$gts))
fit <- lm(res$pnl ~ I(res$vega * (res$sh - res$sr)))
add("fxSlope", coef(fit)[2], 1e-7); add("fxRsq", summary(fit)$r.squared, 1e-7); add("fxWorst", min(res$pnl), 1e-7)
rl <- diff(log(S)); src <- df$usdcad_source[-1]; srcp <- df$usdcad_source[-length(S)]
ac <- function(x) cor(x[-1], x[-length(x)])
add("acRetNoon", ac(rl[src == "noon" & srcp == "noon"]), 1e-3); add("acRetAvg", ac(rl[src == "average" & srcp == "average"]), 1e-3)

# ---------------------------------------------------------------- exercises
add("xvOneCall", bs(50, 55, 0.04, 0.25, 0.5)); add("xvOnePut", bs(50, 55, 0.04, 0.25, 0.5, FALSE))
e1 <- dd(50, 55, 0.04, 0.25, 0.5); add("xvTwoDelta", pnorm(e1[1])); add("xvTwoGamma", dnorm(e1[1]) / (50 * 0.25 * sqrt(0.5)))
add("xvTwoExact", bs(51, 55, 0.04, 0.25, 0.5)); add("xvThreeS", 55 * exp(-(0.04 + 1.5 * 0.0625) * 0.5))
add("xvFourCall", b76(80, 80, 0.03, 0.3, 0.25)); add("xvFivePut", 1e6 * bs(1.36, 1.36, 0.03, 0.08, 0.5, FALSE, 0.045))
add("xvEightSig", 100 * ivu(8, 100, 105, 0.03, 0.75)); add("xvTenGap", bs(100, 100, r, 0.25, 1) - bs(100, 100, r, 0.2, 1))

dfo <- data.frame(name = names(out), value = sprintf("%.15g", unlist(out)), tol = sprintf("%.1e", unlist(tols)))
write.csv(dfo, "code/ch05/out/r_check.csv", row.names = FALSE, quote = FALSE)
cat(nrow(dfo), "R checks written\n")
