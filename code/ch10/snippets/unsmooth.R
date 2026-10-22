# Unsmoothing the appraisal-era Z.1 commercial property index (Proposition 10.7).
q <- read.csv("data/real_assets_quarterly.csv")
q <- q[as.Date(q$date) <= as.Date("2026-09-30"), ]
r <- diff(log(q$cre))
pre <- na.omit(r[as.Date(q$date[-1]) < as.Date("1996-01-01")])
ar1 <- function(x) unname(coef(lm(x[-1] ~ x[-length(x)]))[2])
a <- ar1(pre)                                           # smoothing parameter alpha
u <- (pre[-1] - a * pre[-length(pre)]) / (1 - a)   # r_t = (r*_t - a r*_{t-1})/(1-a)
cat(sprintf("alpha %.3f, volatility %.1f%% reported and %.1f%% unsmoothed\n",
            a, 200 * sd(pre), 200 * sd(u)))
cat(sprintf("ratio %.2f against sqrt((1+a)/(1-a)) = %.2f, residual AR(1) %.2f\n",
            sd(u) / sd(pre), sqrt((1 + a) / (1 - a)), ar1(u)))
