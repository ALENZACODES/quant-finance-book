# Perpetual growth implied by a trailing multiple M = (1 + g)(1 - g/ROC)/(W - g), by
# uniroot and by the root of the quadratic
# g^2 + (1 - ROC - M ROC) g + ROC (M W - 1) = 0.
implied_g <- function(M, W, ROC) {
  f <- function(g) (1 + g) * (1 - g / ROC) / (W - g) - M
  root <- uniroot(f, c(-0.05, W - 1e-4), tol = 1e-12)$root
  b <- 1 - ROC - M * ROC; c0 <- ROC * (M * W - 1)
  closed <- (-b - sqrt(b^2 - 4 * c0)) / 2             # the root below W
  c(uniroot = root, closed = closed)
}
# medians across industries, rounded to the digits shown, from code/ch11/valuation.py
m <- list(multiple = 25.03, wacc = 0.07134, roc = 0.15596)
g <- implied_g(m$multiple, m$wacc, m$roc)
cat(sprintf("W %.2f%%, ROC %.1f%%, multiple %.1f: g %.2f%% (uniroot), %.2f%% (root)\n",
            100 * m$wacc, 100 * m$roc, m$multiple, 100 * g[1], 100 * g[2]))
