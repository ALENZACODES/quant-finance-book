% Survival to T of a fund closed when its log value first falls to -b: the
% reflection formula, a check by quadrature, and the barrier moved for
% monthly monitoring.
1;
function p = survival(mu, s, b, T)
  v = s * sqrt(T); Phi = @(z) 0.5 * erfc(-z / sqrt(2));
  p = Phi((b + mu * T) / v) - exp(-2 * mu * b / s^2) * Phi((-b + mu * T) / v);
end
function p = survival_quad(mu, s, b, T)
  v = s * sqrt(T); phi = @(x, m) exp(-(x - m).^2 / (2 * v^2)) / (v * sqrt(2 * pi));
  k = exp(-2 * mu * b / s^2);
  dens = @(x) phi(x, mu * T) - k * phi(x, mu * T - 2 * b);  % image at -2b
  p = quadgk(dens, -b, Inf);
end
mu = 0; s = 0.15; b = -log(1 - 0.40); dt = 1 / 12;
bc = b + 0.5826 * s * sqrt(dt);      % move the barrier, not the volatility
for T = [5 10]
  printf('%2d years: continuous %.1f%% (quadrature %.1f%%), corrected %.1f%%\n', ...
         T, 100 * survival(mu, s, b, T), 100 * survival_quad(mu, s, b, T), ...
         100 * survival(mu, s, bc, T));
end
