% Octave cross-check for Chapter 7. Run from the repository root:
%   octave-cli --no-gui code/ch07/octave/ch07_check.m
% Writes code/ch07/out/octave_check.csv (name,value,tol), compared with Python in test_crosschecks.py.
% Methods differ from Python on purpose: the long-only portfolios by Octave's quadratic programming
% solver qp, the equal-risk-contribution weights by fsolve on the first-order conditions, the
% Black-Litterman posterior in information form, the Carino linking and the GRS statistic from the data.
1;
function r = add(r, name, value, tol)
  r(end + 1, :) = {name, value, tol};
end
r = cell(0, 3);
F = dlmread('data/ff_factors_monthly.csv', ',', 1, 0); I = dlmread('data/industry10_monthly.csv', ',', 1, 0);
rf = F(:, 8) / 100; mkt = F(:, 2) / 100; FF = F(:, 2:7) / 100;
X = I(:, 2:11) / 100 - rf; T = rows(X); n = 10;
mx = mean(X)'; S = cov(X);
% long-only tangency: min y'Sy subject to mx'y = 1, y >= 0
y = qp(ones(n, 1) / n, S, zeros(n, 1), mx', 1, zeros(n, 1), []);
w = y / sum(y); r = add(r, 'frLoSR', sqrt(12) * (w' * mx) / sqrt(w' * S * w), 1e-7);
g = qp(ones(n, 1) / n, S, zeros(n, 1), ones(1, n), 1, zeros(n, 1), []);
r = add(r, 'frLoGmvVol', 100 * sqrt(12 * g' * S * g), 1e-7);
% equal risk contributions of the six factors: (S y)_i = 1 / y_i
SF = cov(FF); opts = optimset('TolFun', 1e-16, 'TolX', 1e-16);
z = fsolve(@(z) SF * exp(z) - exp(-z), zeros(6, 1), opts); we = exp(z) / sum(exp(z));
r = add(r, 'rpErcCma', 100 * we(5), 1e-6); r = add(r, 'rpErcMkt', 100 * we(1), 1e-6);
% Black-Litterman, information form
S60 = cov(X(T - 59:T, :)); cap = I(T, 22:31)'; wm = cap / sum(cap); delta = mean(mkt) / var(mkt); tau = 1 / 60;
pi = delta * S60 * wm; P = zeros(2, n); P(1, 8) = 1; P(1, 5) = -1; P(2, 4) = 1; q = [0.02; 0.05] / 12;
Om = diag(diag(P * (tau * S60) * P'));
mu = (inv(tau * S60) + P' * inv(Om) * P) \ (inv(tau * S60) * pi + P' * inv(Om) * q);
r = add(r, 'blMuHiTec', 1200 * mu(5), 1e-8); r = add(r, 'blWblHiTec', 100 * ((delta * S60) \ mu)(5), 1e-7);
% GRS for the CAPM
Z = [ones(T, 1) mkt]; c = Z \ X; a = c(1, :)'; E = X - Z * c; Sig = E' * E / T; mb = mean(mkt); vb = mean((mkt - mb).^2);
J = (T - n - 1) / n * (a' * (Sig \ a)) / (1 + mb^2 / vb); r = add(r, 'grsCapm', J, 1e-9);
% Carino linking of the attribution example
rowsA = find(I(:, 1) >= 202601); capm = I(:, 22:31); ew = I(:, 12:21) / 100; vw = I(:, 2:11) / 100;
b0 = capm(rowsA(1), :) / sum(capm(rowsA(1), :)); w = b0; w(5) -= 0.10; w(8) += 0.04; w(4) += 0.03; w(9) += 0.03;
Rp = []; Rb = [];
for t = rowsA'
  b = capm(t, :) / sum(capm(t, :)); Rp(end + 1) = w * ew(t, :)'; Rb(end + 1) = b * vw(t, :)';
end
r = add(r, 'atActive', 100 * (prod(1 + Rp) - prod(1 + Rb)), 1e-10);
% Dimson beta of the smoothed Other industry, phi = 0.5, twelve lags
ro = zeros(T, 1); ro(1) = X(1, 10);
for t = 2:T, ro(t) = 0.5 * X(t, 10) + 0.5 * ro(t - 1); end
q = 12; Zl = ones(T - q, 1);
for j = 0:q, Zl = [Zl mkt(q + 1 - j:T - j)]; end
bd = Zl \ ro(q + 1:T); r = add(r, 'smoDimsonHalf', sum(bd(2:end)), 1e-9);
% exercise 7.2 tangency weights
mu3 = [0.04; 0.06; 0.08]; S3 = diag([0.1 0.15 0.2].^2); y3 = S3 \ mu3; r = add(r, 'xrTwoA', 100 * y3(1) / sum(y3), 1e-12);
fid = fopen('code/ch07/out/octave_check.csv', 'w'); fprintf(fid, 'name,value,tol\n');
for i = 1:rows(r), fprintf(fid, '%s,%.15g,%g\n', r{i, 1}, r{i, 2}, r{i, 3}); end
fclose(fid); printf('%d Octave checks written\n', rows(r));
