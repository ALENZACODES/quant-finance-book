# Gibbons, Ross and Shanken test that ten industry alphas are jointly zero.
f <- read.csv("data/ff_factors_monthly.csv")
i <- read.csv("data/industry10_monthly.csv")
nm <- c("NoDur", "Durbl", "Manuf", "Enrgy", "HiTec",
        "Telcm", "Shops", "Hlth", "Utils", "Other")
Y <- as.matrix(i[, nm] - f$rf) / 100
grs <- function(Y, Fm) {
  Fm <- as.matrix(Fm); T <- nrow(Y); N <- ncol(Y); K <- ncol(Fm)
  fit <- lm(Y ~ Fm)                                  # one regression per column of Y
  a <- coef(fit)[1, ]; E <- resid(fit)
  Sig <- crossprod(E) / T                            # divisor T, as in the test
  fb <- colMeans(Fm); Om <- cov(Fm) * (T - 1) / T
  q <- drop(t(a) %*% solve(Sig, a)); h <- drop(t(fb) %*% solve(Om, fb))
  J <- (T - N - K) / N * q / (1 + h)
  c(J = J, p = pf(J, N, T - N - K, lower.tail = FALSE))
}
six <- c("mkt_rf", "smb", "hml", "rmw", "cma", "mom")
print(round(rbind(CAPM = grs(Y, f$mkt_rf / 100), six = grs(Y, f[, six] / 100)), 4))
