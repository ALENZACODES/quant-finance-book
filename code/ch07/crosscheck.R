# Independent R implementation of the Chapter 7 results (base R only).
# Run from the repository root:  Rscript code/ch07/crosscheck.R
# Writes code/ch07/out/r_check.csv with (name, value, tol) rows compared with Python in test_crosschecks.py.
# Methods differ from Python where possible: lm() for every regression, a hand-written Newey-West
# sandwich, the long-only tangency by its own search over supports with combn(), the AR(1) smoothing by
# stats::filter(), the equal-risk-contribution weights by optim() on the log-barrier objective, the
# Benjamini-Hochberg rule by p.adjust(), and the case study rebuilt from the CSV files.

out <- list(); tols <- list()
add <- function(name, value, tol = 1e-8) { out[[name]] <<- value; tols[[name]] <<- tol }

f <- read.csv("data/ff_factors_monthly.csv") ; ind <- read.csv("data/industry10_monthly.csv")
nm <- c("NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other")
fac <- c("mkt_rf", "smb", "hml", "rmw", "cma", "mom")
rf <- f$rf / 100; mkt <- f$mkt_rf / 100; FF <- as.matrix(f[, fac]) / 100
X <- as.matrix(ind[, nm]) / 100 - rf; TT <- nrow(X); n <- ncol(X)
mx <- colMeans(X); S <- cov(X)

# ---------------------------------------------------------------- diversification and frontier
v <- diag(S); cc <- S[upper.tri(S)]
add("dvOne", 100 * sqrt(12 * mean(v)), 1e-10); add("dvLimit", 100 * sqrt(12 * mean(cc)), 1e-10)
add("dvTen", 100 * sqrt(12 * sum(S) / n^2), 1e-10)
y <- solve(S, mx); add("frTanSR", sqrt(12 * sum(mx * y)), 1e-10)
g <- solve(S, rep(1, n)); add("frGmvVol", 100 * sqrt(12 / sum(g)), 1e-10)
best <- Inf; wlo <- NULL
for (k in 1:n) for (s in combn(n, k, simplify = FALSE)) {
  ys <- solve(S[s, s, drop = FALSE], mx[s]); ys <- ys / sum(mx[s] * ys)       # min y'Sy with mx'y = 1 on s
  if (any(ys < -1e-12)) next
  yy <- rep(0, n); yy[s] <- ys
  val <- sum(yy * (S %*% yy)); if (val < best) { best <- val; wlo <- yy / sum(yy) }
}
add("frLoSR", sqrt(12) * sum(wlo * mx) / sqrt(sum(wlo * (S %*% wlo))), 1e-9)
th2 <- sum(mx * y); add("esInTheorySixty", sqrt(12 * (59 / (60 - n - 2)) * (th2 + n / 60)), 1e-10)

# ---------------------------------------------------------------- CAPM and the security market line
betas <- sapply(1:n, function(i) coef(lm(X[, i] ~ mkt))[2])
cs <- lm(mx ~ betas); add("smSlope", 1200 * coef(cs)[2], 1e-9); add("smIcpt", 1200 * coef(cs)[1], 1e-9)

# ---------------------------------------------------------------- alphas with Newey-West errors
nw_se <- function(fit, L = 6) {
  Xm <- model.matrix(fit); e <- resid(fit); Tn <- nrow(Xm); k <- ncol(Xm)
  u <- Xm * e; Sg <- crossprod(u)
  for (l in 1:L) { G <- t(u[(l + 1):Tn, , drop = FALSE]) %*% u[1:(Tn - l), , drop = FALSE]; Sg <- Sg + (1 - l / (L + 1)) * (G + t(G)) }
  B <- solve(crossprod(Xm)); sqrt(diag(B %*% Sg %*% B) * Tn / (Tn - k))
}
for (k in c("HiTec", "Other")) {
  fit <- lm(X[, k] ~ FF); add(paste0("al", k, "A"), 1200 * coef(fit)[1], 1e-9)
  add(paste0("al", k, "T"), coef(fit)[1] / nw_se(fit)[1], 1e-8)
}
fit <- lm(X[, "Hlth"] ~ mkt); add("alHlthA", 1200 * coef(fit)[1], 1e-9); add("alHlthT", coef(fit)[1] / nw_se(fit)[1], 1e-8)
add("alHlthSe", 1200 * summary(fit)$coefficients[1, 2], 1e-9)

# ---------------------------------------------------------------- GRS
grs <- function(R, F) {
  F <- as.matrix(F); Tn <- nrow(R); N <- ncol(R); K <- ncol(F)
  fit <- lm(R ~ F); a <- coef(fit)[1, ]; E <- resid(fit)
  Sig <- crossprod(E) / Tn; fb <- colMeans(F); Fc <- sweep(F, 2, fb); Om <- crossprod(Fc) / Tn
  J <- (Tn - N - K) / N * sum(a * solve(Sig, a)) / (1 + sum(fb * solve(Om, fb)))
  c(J, pf(J, N, Tn - N - K, lower.tail = FALSE))
}
gc <- grs(X, mkt); add("grsCapm", gc[1], 1e-9); add("grsPCapm", gc[2], 1e-8)
add("grsSix", grs(X, FF)[1], 1e-9); add("grsFive", grs(X, FF[, 1:5])[1], 1e-9)

# ---------------------------------------------------------------- rolling betas
W <- 60
rb <- function(i) sapply(W:TT, function(t) coef(lm(X[(t - W + 1):t, i] ~ mkt[(t - W + 1):t]))[2])
add("rbMaxHiTec", max(rb(5)), 1e-9); add("rbMinUtils", min(rb(9)), 1e-9)

# ---------------------------------------------------------------- multiple testing
mp <- read.csv("code/ch07/out/managers_p.csv")
add("mgBh", sum(p.adjust(mp$p120, "BH") <= 0.10), 0); add("mgLongBh", sum(p.adjust(mp$p240, "BH") <= 0.10), 0)
add("mgLongBhFalse", sum(p.adjust(mp$p240, "BH") <= 0.10 & mp$skilled == 0), 0)
add("mgPiZero", min(1, mean(mp$p120 > 0.5) / 0.5), 1e-12)

# ---------------------------------------------------------------- smoothing
r <- X[, "Other"]
for (ph in c(0.5, 0.8)) {
  ro <- numeric(TT); ro[1] <- r[1]; ro[-1] <- as.numeric(stats::filter((1 - ph) * r[-1], ph, method = "recursive", init = r[1]))
  tag <- if (ph == 0.5) "Half" else "Eight"
  add(paste0("smoBeta", tag), coef(lm(ro ~ mkt))[2], 1e-9); add(paste0("smoAlpha", tag), 1200 * coef(lm(ro ~ mkt))[1], 1e-9)
  q <- 12; lagm <- sapply(0:q, function(j) mkt[(q + 1 - j):(TT - j)])
  add(paste0("smoDimson", tag), sum(coef(lm(ro[(q + 1):TT] ~ lagm))[-1]), 1e-9)
}

# ---------------------------------------------------------------- put writing
bsput <- function(K, r, s, T = 1 / 12) { d1 <- (log(1 / K) + (r + s^2 / 2) * T) / (s * sqrt(T)); d2 <- d1 - s * sqrt(T)
  K * exp(-r * T) * pnorm(-d2) - pnorm(-d1) }
G <- 1 + mkt + rf
for (mk in c(0, 0.2)) {
  yv <- c(); pa <- c()
  for (t in 13:TT) { s0 <- sd(log(G[(t - 12):(t - 1)])) * sqrt(12); rr <- 12 * log1p(rf[t])
    p <- bsput(0.95, rr, s0 * (1 + mk)); yv <- c(yv, (p * (1 + rf[t]) - max(0.95 - G[t], 0)) / 0.95)
    pa <- c(pa, bsput(1 + rf[t], rr, s0 * (1 + mk)) * (1 + rf[t])) }
  xm <- mkt[13:TT]; tag <- if (mk == 0) "Zero" else "Twenty"
  add(paste0("po", tag, "A"), 1200 * coef(lm(yv ~ xm))[1], 1e-8)
  hm <- coef(lm(yv ~ xm + pmax(-xm, 0))); add(paste0("po", tag, "HmA"), 1200 * hm[1], 1e-8)
  add(paste0("po", tag, "Adj"), 1200 * (hm[1] + hm[3] * mean(pa)), 1e-8)
}

# ---------------------------------------------------------------- attribution
capm <- as.matrix(ind[, paste0("cap_", nm)]); ew <- as.matrix(ind[, paste0("ew_", nm)]) / 100; vw <- as.matrix(ind[, nm]) / 100
rows <- which(ind$month >= 202601); b0 <- capm[rows[1], ] / sum(capm[rows[1], ])
w <- b0; w["cap_HiTec"] <- w["cap_HiTec"] - 0.10; w["cap_Hlth"] <- w["cap_Hlth"] + 0.04
w["cap_Enrgy"] <- w["cap_Enrgy"] + 0.03; w["cap_Utils"] <- w["cap_Utils"] + 0.03; w <- as.numeric(w)
Rp <- Rb <- A <- Se <- c()
for (t in rows) { b <- capm[t, ] / sum(capm[t, ]); rb_ <- vw[t, ]; rp_ <- ew[t, ]; RB <- sum(b * rb_)
  Rp <- c(Rp, sum(w * rp_)); Rb <- c(Rb, RB); A <- c(A, sum((w - b) * (rb_ - RB))); Se <- c(Se, sum(b * (rp_ - rb_))) }
P <- prod(1 + Rp) - 1; B <- prod(1 + Rb) - 1; kt <- (log1p(Rp) - log1p(Rb)) / (Rp - Rb); k <- (log1p(P) - log1p(B)) / (P - B)
add("atActive", 100 * (P - B), 1e-10); add("atAlloc", 100 * sum(kt / k * A), 1e-10); add("atSel", 100 * sum(kt / k * Se), 1e-10)

# ---------------------------------------------------------------- Black-Litterman
S60 <- cov(X[(TT - 59):TT, ]); wm <- capm[TT, ] / sum(capm[TT, ]); delta <- mean(mkt) / var(mkt); tau <- 1 / 60
pi <- delta * S60 %*% wm; Pm <- matrix(0, 2, n); Pm[1, 8] <- 1; Pm[1, 5] <- -1; Pm[2, 4] <- 1
qv <- c(0.02, 0.05) / 12; Om <- diag(diag(Pm %*% (tau * S60) %*% t(Pm)))
mu <- solve(solve(tau * S60) + t(Pm) %*% solve(Om) %*% Pm, solve(tau * S60) %*% pi + t(Pm) %*% solve(Om) %*% qv)
add("blMuHlth", 1200 * mu[8], 1e-9); add("blWblHlth", 100 * solve(delta * S60, mu)[8], 1e-8); add("blDelta", delta, 1e-12)

# ---------------------------------------------------------------- risk parity and rebalancing
SF <- cov(FF); obj <- function(z) { yy <- exp(z); 0.5 * sum(yy * (SF %*% yy)) - sum(z) }
o <- optim(rep(0, 6), obj, method = "BFGS", control = list(reltol = 1e-16, maxit = 5000))
we <- exp(o$par) / sum(exp(o$par)); add("rpErcMkt", 100 * we[1], 1e-5)
pr <- FF %*% we; add("rpSRerc", sqrt(12) * mean(pr) / sd(pr), 1e-5)
Gi <- 1 + vw; reb <- Gi %*% rep(1 / n, n); add("rbGrowReb", 100 * 12 * mean(log(reb)), 1e-10)
Sb <- cov(vw) * (TT - 1) / TT; add("rbDivApprox", 100 * 6 * (mean(diag(Sb)) - sum(Sb) / n^2), 1e-10)

# ---------------------------------------------------------------- case study
cr <- read.csv("code/ch07/out/case_returns.csv"); m60 <- mkt[61:TT]
memmel <- function(x, y) { Tn <- length(x); mi <- mean(x); mj <- mean(y); si <- sd(x); sj <- sd(y); sij <- cov(x, y)
  th <- (2 * si^2 * sj^2 - 2 * si * sj * sij + 0.5 * mi^2 * sj^2 + 0.5 * mj^2 * si^2 - mi * mj / (2 * si * sj) * (sij^2 + si^2 * sj^2)) / Tn
  (sj * mi - si * mj) / sqrt(th) }
add("csLwZ", memmel(cr$gmvlw, cr$ew), 1e-9); add("csTanZ", memmel(cr$tan, cr$ew), 1e-9)
add("csLotanSR", sqrt(12) * mean(cr$lotan) / sd(cr$lotan), 1e-10)
ewr <- sapply(61:TT, function(t) mean(X[t, ])); add("csEwSR", sqrt(12) * mean(ewr) / sd(ewr), 1e-10)
gm <- sapply(61:TT, function(t) { Sw <- cov(X[(t - 60):(t - 1), ]); g <- solve(Sw, rep(1, n)); sum(g / sum(g) * X[t, ]) })
add("csGmvSR", sqrt(12) * mean(gm) / sd(gm), 1e-10)
lw <- function(Z) { Tn <- nrow(Z); N <- ncol(Z); Zc <- sweep(Z, 2, colMeans(Z)); Sm <- crossprod(Zc) / Tn
  mm <- sum(diag(Sm)) / N; d2 <- sum((Sm - mm * diag(N))^2) / N
  b2 <- sum(sapply(1:Tn, function(i) sum((tcrossprod(Zc[i, ]) - Sm)^2))) / N / Tn^2; b2 <- min(b2, d2)
  (b2 / d2) * mm * diag(N) + (1 - b2 / d2) * Sm }
lwr <- sapply(61:TT, function(t) { g <- solve(lw(X[(t - 60):(t - 1), ]), rep(1, n)); sum(g / sum(g) * X[t, ]) })
add("csLwSR", sqrt(12) * mean(lwr) / sd(lwr), 1e-9)
add("csLwA", 1200 * coef(lm(lwr ~ m60))[1], 1e-8)

# ---------------------------------------------------------------- exercises
add("xrOneW", 100 * (0.01 - 0.3 * 0.02) / (0.05 - 2 * 0.3 * 0.02), 1e-12)
add("xrTwoSR", sqrt(sum((c(4, 6, 8) / c(10, 15, 20))^2)), 1e-12)
add("xrFour", 4 * (1 + 0.45^2 / 12) / (0.4^2 / 12) / 12, 1e-12)
add("xrNineW", 100 * 0.05 / 0.25, 1e-9)

df <- data.frame(name = names(out), value = sprintf("%.15g", unlist(out)), tol = unlist(tols))
write.csv(df, "code/ch07/out/r_check.csv", row.names = FALSE, quote = FALSE)
cat(nrow(df), "R checks written\n")
