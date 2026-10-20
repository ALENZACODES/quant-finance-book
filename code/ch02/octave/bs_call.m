function c = bs_call(S, K, T, r, s)
  % Black-Scholes call price with the normal CDF written through erfc (no toolboxes needed).
  d1 = (log(S ./ K) + (r + s.^2 / 2) .* T) ./ (s .* sqrt(T));
  d2 = d1 - s .* sqrt(T);
  N = @(x) 0.5 * erfc(-x / sqrt(2));
  c = S .* N(d1) - K .* exp(-r .* T) .* N(d2);
end
