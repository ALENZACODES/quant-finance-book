# Screening 1,000 simulated managers: Benjamini-Hochberg at q = 10% and Storey's share.
d <- read.csv("code/ch07/out/managers_p.csv")      # p values for 10 and 20 years
for (col in c("p120", "p240")) {
  p <- d[[col]]
  flag <- p.adjust(p, method = "BH") <= 0.10         # same as the step-up rule
  cat(sprintf("%s: %d flagged, %d without skill; share without skill %.2f\n", col,
              sum(flag), sum(flag & d$skilled == 0), min(1, 2 * mean(p > 0.5))))
}
