function V0 = crr_tree_uqd(S, K, u, d, R, n)
  % American put on a tree with given up and down factors and gross riskless return per step.
  p = (R - d) / (u - d); j = (0:n)'; V = max(K - S * u .^ j .* d .^ (n - j), 0);
  for i = n - 1:-1:0
    Si = S * u .^ (0:i)' .* d .^ (i - (0:i)');
    V = max((p * V(2:i + 2) + (1 - p) * V(1:i + 1)) / R, K - Si);
  end
  V0 = V(1);
end
