# The property row of Table 10.3: calendar year log price changes of the Z.1 index
# on December to December CPI inflation, 1971 to 2025, Newey-West over two lags.
m <- read.csv("data/real_assets_monthly.csv")
q <- read.csv("data/real_assets_quarterly.csv")
dec <- m[format(as.Date(m$date), "%m") == "12", ]
q4 <- q[format(as.Date(q$date), "%m") == "10", ]       # the fourth quarter
year <- function(x) as.integer(substr(x$date, 1, 4))[-1]
infl <- data.frame(year = year(dec), infl = diff(log(dec$cpi)))
cre <- data.frame(year = year(q4), cre = diff(log(q4$cre)))
d <- merge(infl, cre); d <- d[d$year >= 1971 & d$year <= 2025, ]
X <- cbind(1, d$infl); b <- solve(crossprod(X), crossprod(X, d$cre))
g <- X * as.vector(d$cre - X %*% b)                     # scores
S <- crossprod(g)
for (j in 1:2) {                        # Bartlett weights, no df factor
  G <- crossprod(g[-(1:j), ], g[1:(nrow(g) - j), ])
  S <- S + (1 - j / 3) * (G + t(G))
}
V <- solve(crossprod(X)) %*% S %*% solve(crossprod(X))
cat(sprintf("%d years: slope %.2f, Newey-West standard error %.2f\n",
            nrow(d), b[2], sqrt(V[2, 2])))
