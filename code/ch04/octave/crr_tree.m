function V0 = crr_tree(S, K, r, sigma, T, n, isput, american, q)
  % Cox-Ross-Rubinstein tree priced by backward induction. isput: 1 for a put, 0 for a call.
  if nargin < 9, q = 0; end
  h = T / n; u = exp(sigma * sqrt(h)); d = 1 / u; p = (exp((r - q) * h) - d) / (u - d); disc = exp(-r * h);
  j = (0:n)'; ST = S * u .^ j .* d .^ (n - j);
  if isput, V = max(K - ST, 0); else, V = max(ST - K, 0); end
  for i = n - 1:-1:0
    V = disc * (p * V(2:i + 2) + (1 - p) * V(1:i + 1));
    if american
      Si = S * u .^ (0:i)' .* d .^ (i - (0:i)');
      if isput, V = max(V, K - Si); else, V = max(V, Si - K); end
    end
  end
  V0 = V(1);
end
