1;
% Value of the 10% stake in millions: a DLOC of 20% and a DLOM from the protective
% put (Chaffe) or from the lookback value (Longstaff), for volatilities and
% restriction periods.
function p = put_atm(T, s, r)
  N = @(x) 0.5 * erfc(-x / sqrt(2));
  d1 = (r + 0.5 * s ^ 2) * T / (s * sqrt(T)); d2 = d1 - s * sqrt(T);
  p = exp(-r * T) * N(-d2) - N(-d1);
end
function v = lookback(T, s)
  N = @(x) 0.5 * erfc(-x / sqrt(2)); w = s ^ 2 * T; q = sqrt(w);
  v = (2 + w / 2) * N(q / 2) + q / sqrt(2 * pi) * exp(-w / 8) - 1;
end
x = dlmread("code/ch11/out/minority_inputs.csv", ",", 1, 0);  % control value, DLOC, rf
ctl = x(1); dloc = x(2); rf = x(3);
for s = [0.30 0.45 0.60]
  printf("volatility %.0f%%:", 100 * s);
  for T = [1 2 4]
    d = lookback(T, s); v = ctl * (1 - dloc) * (1 - d);
    p = ctl * (1 - dloc) * (1 - put_atm(T, s, rf));
    if d >= 1, lb = "n.a."; else, lb = sprintf("%.2f", v); end
    printf("  %dy put %.2f lookback %s", T, p, lb);
  end
  printf("\n");
end
