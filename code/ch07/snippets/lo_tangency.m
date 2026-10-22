% Long-only tangency portfolio of the ten industries with Octave's qp.
F = dlmread('data/ff_factors_monthly.csv', ',', 1, 0);
I = dlmread('data/industry10_monthly.csv', ',', 1, 0);
X = (I(:, 2:11) - F(:, 8)) / 100;                  % excess returns, decimals
mu = mean(X)'; S = cov(X); n = numel(mu);
% minimise y'Sy/2 subject to mu'y = 1 and y >= 0, then rescale to weights
y = qp(ones(n, 1) / n, S, zeros(n, 1), mu', 1, zeros(n, 1), []);
w = y / sum(y);
printf('Sharpe ratio %.2f with %d industries held\n', ...
       sqrt(12) * (w' * mu) / sqrt(w' * S * w), sum(w > 1e-9));
