# A 2% management fee charged monthly and a 20% incentive fee above a
# high-water mark, crystallised each December, on the trend programme.
p <- read.csv("code/ch09/out/program.csv")   # month, gross excess return, rf
net_of_fees <- function(x, rf, month, mgmt = 0.02, perf = 0.20) {
  nav <- 1; hwm <- 1; net <- numeric(length(x)); paid <- 0
  for (t in seq_along(x)) {
    start <- nav
    nav <- nav * (1 + rf[t] + x[t]) * (1 - mgmt / 12)   # management fee first
    if (month[t] %% 100 == 12 || t == length(x)) {      # December and the end
      if (nav > hwm) {
        fee <- perf * (nav - hwm); nav <- nav - fee; paid <- paid + fee
      }
      hwm <- max(hwm, nav)
    }
    net[t] <- nav / start - 1 - rf[t]       # the bill rate is inside the gain
  }
  list(net = net, paid = paid, hwm = hwm)
}
f <- net_of_fees(p$gross, p$rf, p$month)
m <- net_of_fees(p$gross, p$rf, p$month, perf = 0)   # no incentive fee
sr <- function(x) sqrt(12) * mean(x) / sd(x)
cat(sprintf("management fee %.2f, incentive fee %.2f points a year; ",
            1200 * (mean(p$gross) - mean(m$net)), 1200 * (mean(m$net) - mean(f$net))))
cat(sprintf("Sharpe %.2f gross, %.2f net\n", sr(p$gross), sr(f$net)))
cat(sprintf("incentive fees paid %.6f = f / (1 - f) times the rise in the mark %.6f\n",
            f$paid, 0.2 / 0.8 * (f$hwm - 1)))
