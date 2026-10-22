% Octave cross-check for Chapter 8. Run from the repository root:
%   octave-cli --no-gui code/ch08/octave/ch08_check.m
% Writes code/ch08/out/octave_check.csv (name,value,tol), compared with Python in test_crosschecks.py.
% Methods differ from Python on purpose: the GARCH(1,1) likelihood by an explicit loop maximised with
% fminsearch, binomial zone probabilities through the incomplete beta function, the normal quantile
% through erfcinv, drawdowns by cummax, and smoothing and unsmoothing by filter().
1;
function r = add(r, name, value, tol)
  r(end + 1, :) = {name, value, tol};
end
function v = garch_nll(p, x)
  m = p(1); om = p(2); a = p(3); b = p(4);
  if om <= 0 || a < 0 || b < 0 || a + b >= 1, v = 1e12; return; end
  e = x - m; n = numel(e); h = zeros(n, 1); h(1) = mean(e .^ 2);
  for t = 2:n, h(t) = om + a * e(t - 1) ^ 2 + b * h(t - 1); end
  v = 0.5 * sum(log(2 * pi * h) + e .^ 2 ./ h);
end
function p = bcdf(k, n, q)   % binomial cdf P(X <= k)
  p = betainc(1 - q, n - k, k + 1);
end
function z = nq(p)
  z = -sqrt(2) * erfcinv(2 * p);
end
r = cell(0, 3);
D = dlmread('data/ff_daily.csv', ',', 1, 1); ret = D(:, 3) / 100; x = 100 * ret; n = numel(x);
% GARCH(1,1), normal: fminsearch from a generic start, then from its own answer
o = optimset('MaxFunEvals', 20000, 'MaxIter', 20000, 'TolX', 1e-10, 'TolFun', 1e-10);
p = fminsearch(@(p) garch_nll(p, x), [mean(x); 0.05 * var(x); 0.08; 0.90], o);
p = fminsearch(@(p) garch_nll(p, x), p, o);
r = add(r, 'gaAlphaN', p(3), 5e-3); r = add(r, 'gaBetaN', p(4), 5e-3); r = add(r, 'gaPersN', p(3) + p(4), 5e-4);
r = add(r, 'gaLogLikN', -garch_nll(p, x), 2e-6);
% zones and the Kupiec statistic
r = add(r, 'znGreen', 100 * bcdf(4, 250, 0.01), 1e-9);
r = add(r, 'znYellow', 100 * (bcdf(9, 250, 0.01) - bcdf(4, 250, 0.01)), 1e-9);
k = 8; N = 250; q = 0.01; ph = k / N;
lr = -2 * (((N - k) * log(1 - q) + k * log(q)) - ((N - k) * log(1 - ph) + k * log(ph)));
r = add(r, 'xdLR', lr, 1e-10);
% normal VaR and ES of the daily market
mu = mean(ret); s = std(ret);
r = add(r, 'vrNorN', 100 * -(mu + s * nq(0.01)), 1e-9);
r = add(r, 'esNorS', 100 * (-mu + s * exp(-nq(0.975) ^ 2 / 2) / sqrt(2 * pi) / 0.025), 1e-9);
% drawdowns
W = cumprod(1 + ret); P = cummax([1; W])(2:end); dd = 1 - W ./ P;
r = add(r, 'ddMax', 100 * max(dd), 1e-10);
% smoothing and unsmoothing of monthly market returns
F = dlmread('data/ff_factors_monthly.csv', ',', 1, 0); rm = F(:, 2) / 100 + F(:, 8) / 100; phi = 0.6;
rs = filter(1 - phi, [1 -phi], rm, phi * rm(1));
r = add(r, 'smVolS', 100 * sqrt(12) * std(rs), 1e-9);
z = rs - mean(rs); ph = sum(z(2:end) .* z(1:end - 1)) / sum(z .^ 2);
ru = (rs(2:end) - ph * rs(1:end - 1)) / (1 - ph);
r = add(r, 'smVolU', 100 * sqrt(12) * std(ru), 1e-9);
% generalised Pareto VaR and ES of exercise 5
u = 0.02; xi = 0.25; be = 0.007; vq = u + be / xi * ((10000 / 500 * 0.001) ^ (-xi) - 1);
r = add(r, 'xeVar', 100 * vq, 1e-12); r = add(r, 'xeEs', 100 * (vq + be - xi * u) / (1 - xi), 1e-12);
fid = fopen('code/ch08/out/octave_check.csv', 'w'); fprintf(fid, 'name,value,tol\n');
for i = 1:rows(r), fprintf(fid, '%s,%.15g,%g\n', r{i, 1}, r{i, 2}, r{i, 3}); end
fclose(fid); printf('%d Octave checks written\n', rows(r));
