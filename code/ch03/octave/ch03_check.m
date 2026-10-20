% Octave cross-check for Chapter 3. Run from the repository root:
%   octave-cli --no-gui code/ch03/octave/ch03_check.m
% Writes code/ch03/out/octave_check.csv (name,value,tol), compared with Python in test_crosschecks.py.
% Methods differ from Python on purpose: exhaustive enumeration, Gaussian conditioning, quadrature,
% a sparse Crank-Nicolson solver and a finite difference boundary value problem.
addpath(fileparts(mfilename('fullpath')));
r = {};
add = @(r, name, v, tol) [r; {name, v, tol}];
% reflection principle by enumerating all 2^16 paths
n = 16; P = 2 * (dec2bin(0:2^n - 1) - '0') - 1; M = max(cumsum(P, 2), [], 2);
r = add(r, 'reflSixteen', mean(M >= 3), 1e-12);
% Levy construction: conditional variance of W(1/2) given W(1/4) and W(3/4), from the covariance min(s,t)
t = [0.25 0.5 0.75]; C = min(t', t);
cv = C(2, 2) - C(2, [1 3]) * (C([1 3], [1 3]) \ C([1 3], 2));
r = add(r, 'levyBridgeVar', cv, 1e-12);
% Ito isometry targets by quadrature
r = add(r, 'isoTExact', quadgk(@(s) s .^ 2, 0, 1), 1e-10);
r = add(r, 'isoEExact', quadgk(@(s) exp(2 * s), 0, 1), 1e-10);
% GBM moments by quadrature against the lognormal density
mu = 0.08; sg = 0.25; T = 30; m = (mu - sg ^ 2 / 2) * T; s = sg * sqrt(T);
f = @(x) exp(-(log(x) - m) .^ 2 / (2 * s ^ 2)) ./ (x * s * sqrt(2 * pi));
r = add(r, 'gbmMeanLow', 100 * quadgk(@(x) x .* f(x), 0, Inf, 'RelTol', 1e-12), 1e-6);
r = add(r, 'gbmProbLossLow', 100 * quadgk(f, 0, 1, 'RelTol', 1e-12), 1e-6);
% Euler weak error: expectation of the scheme, step by step
e = 100; for k = 1:12, e = e * (1 + 0.08 / 12); end
r = add(r, 'weakMeanTwelve', e, 1e-10);
% exit problems from the boundary value problem
[p, tau] = bvp_exit(0.1, 0.2, 0.1, 0.1, 2000);
r = add(r, 'exitPup', 100 * p, 1e-5); r = add(r, 'exitTau', tau, 1e-6);
[p, tau] = bvp_exit(0.05, 0.2, 0.1, 0.2, 2000);
r = add(r, 'exSixP', 100 * p, 1e-5); r = add(r, 'exSixTau', tau, 1e-6);
% Girsanov example by quadrature under the shifted law
r = add(r, 'girExact', quadgk(@(x) (x - 0.5) .* exp(-(x - 0.4) .^ 2 / 2) / sqrt(2 * pi), 0.5, Inf), 1e-9);
% Feynman-Kac: the same Crank-Nicolson grids as the Python solver, built independently
K = 100; R = 0.03; S = 0.2; x = linspace(log(1), log(1000), 1601);
u = fk_cn(x, 1, @(z) (R - S ^ 2 / 2) + 0 * z, @(z) S ^ 2 + 0 * z, R, @(z) max(exp(z) - K, 0), ...
          @(tau, xe) max(exp(xe) - K * exp(-R * tau), 0), 800, 4);
i = find(x > log(100), 1) - 2; c = polyfit(x(i:i+3)', u(i:i+3), 3);
r = add(r, 'fkPDE', polyval(c, log(100)), 1e-7);
ka = 1.5; tho = 0.2; so = 0.5; x = linspace(-3, 3, 1201);
mom = @(xx, tau) (tho + (xx - tho) .* exp(-ka * tau)) .^ 2 + so ^ 2 * (1 - exp(-2 * ka * tau)) / (2 * ka);
u = fk_cn(x, 2, @(z) ka * (tho - z), @(z) so ^ 2 + 0 * z, 0, @(z) z .^ 2, mom, 600, 4);
i = find(x > 1, 1) - 2; c = polyfit(x(i:i+3)', u(i:i+3), 3);
r = add(r, 'fkOUPDE', polyval(c, 1), 1e-9);
% yield case: OLS by backslash on month-end values
fid = fopen('data/daily_policy_bond.csv'); fgetl(fid);
Cc = textscan(fid, '%s %f %f', 'Delimiter', ',', 'EmptyValue', NaN); fclose(fid);
dts = Cc{1}; y = Cc{3}; ok = ~isnan(y) & cellfun(@(q) ~strcmp(q(1:7), '2026-10'), dts);
dts = dts(ok); y = y(ok);
ym = cellfun(@(q) q(1:7), dts, 'UniformOutput', false);
[~, last] = unique(ym, 'last'); yy = y(sort(last));
X = [ones(numel(yy) - 1, 1) yy(1:end-1)]; bh = X \ yy(2:end); phi = bh(2);
kh = -log(phi) * 12;
r = add(r, 'casePhi', phi, 1e-10); r = add(r, 'caseKappa', kh, 1e-9); r = add(r, 'caseHalf', log(2) / kh, 1e-8);
r = add(r, 'caseTheta', bh(1) / (1 - phi), 1e-8);
here = fileparts(mfilename('fullpath'));
f = fopen(fullfile(here, '..', 'out', 'octave_check.csv'), 'w');
fprintf(f, 'name,value,tol\n');
for i = 1:rows(r)
  fprintf(f, '%s,%.12f,%g\n', r{i, 1}, r{i, 2}, r{i, 3});
end
fclose(f);
printf('%d Octave checks written\n', rows(r));
