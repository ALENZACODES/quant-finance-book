function [m, sd, sk, ek] = ln_moments_quad(mu, s)
  % Moments of exp(Y), Y ~ N(mu, s^2), by numerical integration of the raw moments
  % E[exp(kY)] = int exp(k y) phi(y) dy, k = 1..4, then central moments from raw moments.
  phi = @(y) exp(-(y - mu) .^ 2 / (2 * s ^ 2)) / (s * sqrt(2 * pi));
  a = mu - 14 * s; b = mu + 14 * s + 4 * s ^ 2 * 4;
  e = zeros(1, 4);
  for k = 1:4
    e(k) = quadgk(@(y) exp(k * y) .* phi(y), a, b, 'AbsTol', 0, 'RelTol', 1e-13);
  end
  m = e(1);
  v = e(2) - m ^ 2;
  m3 = e(3) - 3 * m * e(2) + 2 * m ^ 3;
  m4 = e(4) - 4 * m * e(3) + 6 * m ^ 2 * e(2) - 3 * m ^ 4;
  sd = sqrt(v); sk = m3 / v ^ 1.5; ek = m4 / v ^ 2 - 3;
end
