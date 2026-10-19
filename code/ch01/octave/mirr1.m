function m = mirr1(cf, f, re)
  % Modified IRR: negatives discounted to date 0 at f, positives compounded to date n at re.
  n = numel(cf) - 1; t = 0:n;
  pvn = -sum(min(cf, 0) ./ (1 + f) .^ t);
  fvp = sum(max(cf, 0) .* (1 + re) .^ (n - t));
  m = (fvp / pvn) ^ (1 / n) - 1;
end
