# Independent R implementation of the Chapter 6 results and of the refinancing case study (base R).
# Run from the repository root:  Rscript code/ch06/crosscheck.R
# Writes code/ch06/out/r_check.csv with (name, value, tol) rows compared with Python in test_crosschecks.py.
# Methods differ from Python where possible: duration and convexity by numerical differentiation,
# the curve refitted with optim(), bond options by integrate() against the Gaussian density, the Merton
# calibration by optim(), CDS legs by integrate(), the copula by integrate() over the factor, and the
# case study rebuilt from the CSV file with its own month-end selection.

out <- list(); tols <- list()
add <- function(name, value, tol = 1e-8) { out[[name]] <<- value; tols[[name]] <<- tol }

# ---------------------------------------------------------------- bonds
price <- function(y, c, T) { k <- 1:(2 * T); cf <- rep(100 * c / 2, 2 * T); cf[2 * T] <- cf[2 * T] + 100; sum(cf / (1 + y / 2)^k) }
g <- read.csv("data/goc_bills_bonds.csv", stringsAsFactors = FALSE)
y10 <- g$BD.CDN.10YR.DQ.YLD; y <- tail(y10[!is.na(y10)], 1) / 100
h <- 1e-4
D <- -(price(y + h, y, 10) - price(y - h, y, 10)) / (2 * h) / 100
C <- (price(y + h, y, 10) - 2 * price(y, y, 10) + price(y - h, y, 10)) / h^2 / 100
add("bdD", D, 1e-6); add("bdC", C, 1e-5); add("bdDpar", (1 - (1 + y / 2)^-20) / y, 1e-10)
add("bdExUp", price(y + 0.01, y, 10), 1e-10)
for (TT in c(5, 10)) {
  col <- if (TT == 5) g$BD.CDN.5YR.DQ.YLD else g$BD.CDN.10YR.DQ.YLD
  s <- col[!is.na(col)] / 100; y0 <- s[-length(s)]; y1 <- s[-1]; dy <- y1 - y0
  e1 <- e2 <- numeric(length(dy))
  for (i in seq_along(dy)) {
    p0 <- price(y0[i], y0[i], TT); ex <- price(y1[i], y0[i], TT) / p0 - 1
    k <- 1:(2 * TT); cf <- rep(100 * y0[i] / 2, 2 * TT); cf[2 * TT] <- cf[2 * TT] + 100; v <- 1 + y0[i] / 2
    Dm <- sum(cf * k / 2 * v^(-k - 1)) / p0; Cv <- sum(cf * k * (k + 1) / 4 * v^(-k - 2)) / p0
    e1[i] <- 1e4 * (ex + Dm * dy[i]); e2[i] <- 1e4 * (ex + Dm * dy[i] - 0.5 * Cv * dy[i]^2)
  }
  nm <- if (TT == 5) "Five" else "Ten"
  add(paste0("bdRmsDur", nm), sqrt(mean(e1^2)), 1e-8); add(paste0("bdMaxDur", nm), max(abs(e1)), 1e-8)
  add(paste0("bdRmsQd", nm), sqrt(mean(e2^2)), 1e-6)
}
z <- 0.035; pv <- 100 * exp(-7 * z)
a <- solve(matrix(c(exp(-2 * z), exp(-15 * z), 2 * exp(-2 * z), 15 * exp(-15 * z)), 2, byrow = TRUE), c(pv, 7 * pv))
add("imAone", a[1]); add("imAtwo", a[2])
add("imBarUp", a[1] * exp(-2 * (z + 0.01)) + a[2] * exp(-15 * (z + 0.01)) - 100 * exp(-7 * (z + 0.01)))

# ---------------------------------------------------------------- short-rate models
fp <- read.csv("code/ch06/out/fit_params.csv", stringsAsFactors = FALSE)
vasP <- function(r, t, k, rb, s) { B <- (1 - exp(-k * t)) / k; exp((B - t) * (k^2 * rb - s^2 / 2) / k^2 - s^2 * B^2 / (4 * k) - B * r) }
cirP <- function(r, t, k, rb, s) {
  hh <- sqrt(k^2 + 2 * s^2); e <- exp(hh * t) - 1; den <- 2 * hh + (k + hh) * e
  (2 * hh * exp((k + hh) * t / 2) / den)^(2 * k * rb / s^2) * exp(-2 * e / den * r)
}
quotes <- function(P) {
  b <- sapply(c(0.25, 0.5, 1), function(t) (1 / P(t) - 1) / t)
  p <- sapply(c(2, 3, 5, 7, 10), function(T) { t <- (1:(2 * T)) / 2; 2 * (1 - P(T)) / sum(P(t)) })
  c(b, p)
}
cols <- c("TB.CDN.90D.MID", "TB.CDN.180D.MID", "TB.CDN.1Y.MID", "BD.CDN.2YR.DQ.YLD", "BD.CDN.3YR.DQ.YLD",
          "BD.CDN.5YR.DQ.YLD", "BD.CDN.7YR.DQ.YLD", "BD.CDN.10YR.DQ.YLD")
qnow <- unlist(g[g$date == "2026-09-29", cols]) / 100
vn <- fp[fp$date == "2026-09-29" & fp$model == "vasicek", ]
cn <- fp[fp$date == "2026-09-29" & fp$model == "cir", ]
obj <- function(x) sum((1e4 * (quotes(function(t) vasP(x[1], t, x[2], x[3], vn$sigma)) - qnow))^2)
fit <- optim(c(0.02, 0.5, 0.04), obj, method = "Nelder-Mead", control = list(reltol = 1e-16, maxit = 20000))
fit <- optim(fit$par, obj, method = "BFGS", control = list(reltol = 1e-16, maxit = 2000))
add("srVasNowRms", sqrt(fit$value / 8), 1e-4)
m <- g[, c("date", "TB.CDN.90D.MID")]; m <- m[!is.na(m[, 2]), ]
last <- tapply(seq_len(nrow(m)), substr(m$date, 1, 7), max)
add("srSigV", 100 * sd(diff(m[last, 2] / 100)) * sqrt(12), 1e-10)
add("srMcExact", cirP(cn$r0, 10, cn$kappa, cn$rbar, cn$sigma), 1e-10)
# Vasicek caplet as a put on a zero, integrated against the Gaussian law of the zero price under the T-forward measure
r0 <- vn$r0; k <- vn$kappa; rb <- vn$rbar; s <- vn$sigma
capl <- function(T, dl, K) {
  PT <- vasP(r0, T, k, rb, s); PS <- vasP(r0, T + dl, k, rb, s)
  sp <- s * sqrt((1 - exp(-2 * k * T)) / (2 * k)) * (1 - exp(-k * dl)) / k
  X <- 1 / (1 + dl * K); mu <- log(PS / PT) - sp^2 / 2
  f <- function(x) pmax(X - exp(x), 0) * dnorm(x, mu, sp)
  (1 + dl * K) * PT * integrate(f, mu - 12 * sp, log(X), rel.tol = 1e-12, abs.tol = 0)$value
}
fw <- function(T) (vasP(r0, T, k, rb, s) / vasP(r0, T + 0.25, k, rb, s) - 1) / 0.25
add("cpAtmTwo", 1e4 * capl(2, 0.25, fw(2)), 1e-7)
blk <- function(F, K, v, T, P) { d1 <- (log(F / K) + v^2 * T / 2) / (v * sqrt(T)); P * 0.25 * (F * pnorm(d1) - K * pnorm(d1 - v * sqrt(T))) }
add("cpVolOne", 100 * uniroot(function(v) blk(fw(1), fw(1), v, 1, vasP(r0, 1.25, k, rb, s)) - capl(1, 0.25, fw(1)),
                               c(1e-4, 3), tol = 1e-14)$root, 1e-7)
add("swFive", 100 * (1 - vasP(r0, 5, k, rb, s)) / (sum(vasP(r0, (1:10) / 2, k, rb, s)) / 2), 1e-10)
add("swFiveFive", 100 * (vasP(r0, 5, k, rb, s) - vasP(r0, 10, k, rb, s)) / (sum(vasP(r0, (11:20) / 2, k, rb, s)) / 2), 1e-10)

# ---------------------------------------------------------------- Merton
mert <- function(V, D, r, sv, T) { d1 <- (log(V / D) + (r + sv^2 / 2) * T) / (sv * sqrt(T)); d2 <- d1 - sv * sqrt(T)
  E <- V * pnorm(d1) - D * exp(-r * T) * pnorm(d2); c(E = E, s = -log((V - E) / D) / T - r, d1 = d1, pd = pnorm(-d2)) }
sl <- function(l, sv, T) mert(1, l, 0, sv, T)["s"]
add("mtSSevenFive", 1e4 * sl(0.7, 0.25, 5), 1e-9); add("mtSNineOne", 1e4 * sl(0.9, 0.25, 1), 1e-9)
cal <- optim(c(110, 0.3), function(x) { m <- mert(x[1], 80, 0.03, x[2], 5); (m["E"] - 40)^2 + (pnorm(m["d1"]) * x[2] * x[1] - 0.6 * 40)^2 },
             method = "BFGS", control = list(reltol = 1e-20, maxit = 5000))
add("mcV", cal$par[1], 1e-5); add("mcSigV", 100 * cal$par[2], 1e-5)

# ---------------------------------------------------------------- hazard rates and CDS
Pd <- function(t) vasP(r0, t, k, rb, s)
mats <- c(1, 3, 5, 7, 10); qs <- c(40, 55, 70, 80, 90) / 1e4; R <- 0.4
haz <- function(t, lam) { i <- findInterval(t, c(0, mats[-5]), left.open = TRUE); i[i < 1] <- 1; lam[i] }
Hc <- function(t, lam) sapply(t, function(u) integrate(function(x) haz(x, lam), 0, max(u, 1e-12), subdivisions = 200, rel.tol = 1e-12)$value)
surv <- function(t, lam) { kn <- c(0, mats[-5], Inf); H <- 0; for (j in 1:5) H <- H + lam[j] * pmax(pmin(t, kn[j + 1]) - kn[j], 0); exp(-H) }
legs <- function(T, lam) {
  ann <- 0; prot <- 0
  for (i in 1:(4 * T)) { a <- (i - 1) / 4; b <- i / 4
    ann <- ann + 0.25 * Pd(b) * surv(b, lam)
    f <- function(u) Pd(u) * haz(u, lam) * surv(u, lam)
    prot <- prot + integrate(f, a, b, rel.tol = 1e-12)$value
    ann <- ann + integrate(function(u) (u - a) * f(u), a, b, rel.tol = 1e-12)$value }
  c(ann, (1 - R) * prot) }
lam <- numeric(0)
for (j in 1:5) {
  fj <- function(l) { L <- c(lam, rep(l, 5 - length(lam))); v <- legs(mats[j], L); v[2] / v[1] - qs[j] }
  lam <- c(lam, uniroot(fj, c(1e-6, 1), tol = 1e-14)$root)
}
add("hzLOne", 100 * lam[1], 1e-6); add("hzLFive", 100 * lam[3], 1e-6); add("hzLTen", 100 * lam[5], 1e-6)
add("hzSFive", 100 * surv(5, lam), 1e-6)

# ---------------------------------------------------------------- Gaussian copula
p5 <- 1 - surv(5, lam)
cpd <- function(m, rho) pnorm((qnorm(p5) - sqrt(rho) * m) / sqrt(1 - rho))
trEL <- function(rho, a, d, n = 100) {
  Lk <- (1 - R) * (0:n) / n; tl <- pmin(pmax(Lk - a, 0), d - a)
  f <- function(m) sapply(m, function(x) sum(dbinom(0:n, n, cpd(x, rho)) * tl)) * dnorm(m)
  integrate(f, -12, 12, rel.tol = 1e-12, subdivisions = 1000)$value / (d - a) }
add("trEqThree", 100 * trEL(0.3, 0, 0.03), 1e-6); add("trSnBThree", 100 * trEL(0.3, 0.15, 0.3), 1e-6)
add("trMzASix", 100 * trEL(0.6, 0.03, 0.07), 1e-6)
lhEL <- function(rho, a, d) integrate(function(m) pmin(pmax((1 - R) * cpd(m, rho) - a, 0), d - a) * dnorm(m), -12, 12,
                                       rel.tol = 1e-12, subdivisions = 2000)$value / (d - a)
add("lhEqThree", 100 * lhEL(0.3, 0, 0.03), 1e-6); add("lhSnBThree", 100 * lhEL(0.3, 0.15, 0.3), 1e-5)
# far-tail tranches, split at the kinks so that the thin region where they are hit is not missed
lhKink <- function(rho, x) (qnorm(p5) - sqrt(1 - rho) * qnorm(x / (1 - R))) / sqrt(rho)
lhELk <- function(rho, a, d) {
  e <- sort(c(-12, 12, sapply(Filter(function(x) x > 0 && x < 1 - R, c(a, d)), function(x) lhKink(rho, x))))
  f <- function(m) pmin(pmax((1 - R) * cpd(m, rho) - a, 0), d - a) * dnorm(m)
  sum(sapply(seq_len(length(e) - 1), function(i) integrate(f, e[i], e[i + 1], rel.tol = 1e-12)$value)) / (d - a) }
add("lhSupThree", 100 * lhELk(0.3, 0.3, 1), 1e-6); add("lhSnBOne", 100 * lhELk(0.1, 0.15, 0.3), 1e-6)
add("lhSupSix", 100 * lhELk(0.6, 0.3, 1), 1e-6)

# ---------------------------------------------------------------- private credit
ads <- function(L, r, am) if (am == 0) L * r / 100 else { i <- r / 1200; 12 * L * i / (1 - (1 + i)^(-12 * am)) }
bal <- function(L, r, am, n) if (am == 0) L else { i <- r / 1200; p <- ads(L, r, am) / 12; L * (1 + i)^n - p * ((1 + i)^n - 1) / i }
score <- function(d) {
  ltv <- d$loan / d$value * 100; dscr <- d$noi / ads(d$loan, d$rate, d$amort); dy <- d$noi / d$loan * 100
  ev <- d$noi * (1 + d$noiGrowth / 100)^d$termYrs / ((d$capRate + d$exitCapShift) / 100)
  el <- bal(d$loan, d$rate, d$amort, 12 * d$termYrs) / ev * 100
  cl <- function(x) min(1, max(0, x))
  t <- 0.25 * cl((85 - ltv) / 35) + 0.25 * cl((dscr - 1) / 0.6) + 0.15 * cl((dy - 6) / 8) + 0.25 * cl((90 - el) / 40) +
    0.1 * cl((d$sponsorScore - 1) / 4)
  floor(100 * t + 0.5) }
add("pcCalScore", score(list(loan = 6e6, value = 10e6, noi = 7e5, rate = 9.5, amort = 0, termYrs = 2, capRate = 7, exitCapShift = 0.25, noiGrowth = 2, sponsorScore = 4)), 0)
add("pcTorScore", score(list(loan = 14e6, value = 18.5e6, noi = 9e5, rate = 11.5, amort = 0, termYrs = 1, capRate = 5, exitCapShift = 0.75, noiGrowth = 0, sponsorScore = 3)), 0)
add("pcOttScore", score(list(loan = 4.2e6, value = 6e6, noi = 4.3e5, rate = 10.25, amort = 25, termYrs = 3, capRate = 6.75, exitCapShift = 0.5, noiGrowth = 1, sponsorScore = 3)), 0)
lnput <- function(K, m, s) { d1 <- (log(m / K) + s^2 / 2) / s; K * pnorm(-d1 + s) - m * pnorm(-d1) }
Bs <- bal(60, 6, 25, 60); Bm <- bal(75, 6, 25, 60); mm <- 0.9 * 100 * 1.02^5
add("slSenFifteen", 100 * lnput(Bs, mm, 0.15 * sqrt(5)) / Bs, 1e-9)
add("slMezFifteen", 100 * (lnput(Bm, mm, 0.15 * sqrt(5)) - lnput(Bs, mm, 0.15 * sqrt(5))) / (Bm - Bs), 1e-9)

# ---------------------------------------------------------------- case study
f5 <- g[!is.na(g$BD.CDN.5YR.DQ.YLD), c("date", "BD.CDN.5YR.DQ.YLD")]
ix <- tapply(seq_len(nrow(f5)), substr(f5$date, 1, 7), max); y5 <- f5[ix, 2]
nv <- length(y5) - 60; cover <- numeric(nv)
# half-yearly compounding, both tests at origination and at maturity, cap rate 6% + 0.5 (y - 4)
mi <- function(j) (1 + j / 200)^(1 / 6) - 1
kh <- function(j, am) { i <- mi(j); 12 * i / (1 - (1 + i)^(-12 * am)) }
bh <- function(L, j, am, m) { i <- mi(j); p <- L * kh(j, am) / 12; L * (1 + i)^m - p * ((1 + i)^m - 1) / i }
for (i in 1:nv) {
  j0 <- y5[i] + 2.25; j1 <- y5[i + 60] + 2.25; q0 <- 0.06 + 0.5 * (y5[i] - 4) / 100
  L0 <- min(65, 100 * q0 / (1.25 * kh(j0, 25))); B <- bh(L0, j0, 25, 60)
  noi1 <- 100 * q0 * 1.02^5; q1 <- q0 + 0.5 * (y5[i + 60] - y5[i]) / 100
  cover[i] <- min(0.65 * noi1 / q1, noi1 / (1.25 * kh(j1, 25))) / B }
sg <- sqrt(0.1^2 + 0.15^2); pdv <- pnorm((log(1 / cover) + sg^2 / 2) / sg)
add("csN", nv, 0); add("csCoverMin", min(cover), 1e-10); add("csCoverMed", median(cover), 1e-10)
add("csFails", sum(cover < 1), 0)
add("csPdLast", 100 * pdv[nv], 1e-10); add("csPdMed", 100 * median(pdv), 1e-10)
q0 <- 0.06 + 0.5 * (y5[nv] - 4) / 100; L0 <- min(65, 100 * q0 / (1.25 * kh(y5[nv] + 2.25, 25)))
BL <- bh(L0, y5[nv] + 2.25, 25, 60); V1 <- 100 * q0 * 1.02^5 / (q0 + 0.5 * (y5[nv + 60] - y5[nv]) / 100)
elf <- integrate(function(x) pmax(BL - 0.9 * V1 * x, 0) * (x < 1 / cover[nv]) * dlnorm(x, -sg^2 / 2, sg), 0, 1 / cover[nv],
                 rel.tol = 1e-12)$value
add("csEl", 1e4 * elf / BL, 1e-7); add("csLastB", BL, 1e-9)

# ---------------------------------------------------------------- exercises
add("xsOneEx", price(0.055, 0.04, 5), 1e-12); add("xsFiveS", 1e4 * mert(120, 100, 0.03, 0.2, 3)["s"], 1e-9)
add("xsSeven", 100 * pnorm((qnorm(0.02) + sqrt(0.2) * qnorm(0.99)) / sqrt(0.8)), 1e-10)
add("xsNineCover", 9e4 / (1.25 * ads(1, 6, 25)) / bal(1e6, 4, 25, 60), 1e-10)

dfo <- data.frame(name = names(out), value = sprintf("%.15g", unlist(out)), tol = sprintf("%.1e", unlist(tols)))
write.csv(dfo, "code/ch06/out/r_check.csv", row.names = FALSE, quote = FALSE)
cat(nrow(dfo), "R checks written\n")
