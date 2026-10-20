% Octave cross-check for Chapter 5. Run from the repository root:
%   octave-cli --no-gui code/ch05/octave/ch05_check.m
% Writes code/ch05/out/octave_check.csv (name,value,tol), compared with Python in test_crosschecks.py.
% Methods differ from Python on purpose: prices by quadrature of the payoff against the lognormal
% density, Greeks by Richardson-extrapolated differences, the implicit and Crank-Nicolson schemes with
% sparse matrices and backslash, implied volatility by fzero, the gamma peak by fminbnd, and the
% Heston price by quadgk on the Lewis integral.
1;
function v = ncdf(x)
  v = 0.5 * erfc(-x / sqrt(2));
end
function c = bsq(S, K, r, s, T, call)
  % discounted payoff integrated against the lognormal density of ln S_T
  m = log(S) + (r - s ^ 2 / 2) * T; v = s ^ 2 * T; sd = sqrt(v);
  dens = @(x) exp(-(x - m) .^ 2 / (2 * v)) / sqrt(2 * pi * v);
  if call
    c = exp(-r * T) * quadgk(@(x) (exp(x) - K) .* dens(x), log(K), m + 14 * sd, 'RelTol', 1e-13, 'AbsTol', 1e-12, 'MaxIntervalCount', 5000);
  else
    c = exp(-r * T) * quadgk(@(x) (K - exp(x)) .* dens(x), m - 14 * sd, log(K), 'RelTol', 1e-13, 'AbsTol', 1e-12, 'MaxIntervalCount', 5000);
  end
end
function c = bsf(S, K, r, s, T, q)
  d1 = (log(S / K) + (r - q + s ^ 2 / 2) * T) / (s * sqrt(T)); d2 = d1 - s * sqrt(T);
  c = S * exp(-q * T) * ncdf(d1) - K * exp(-r * T) * ncdf(d2);
end
function V = fdsparse(M, Nt, th)
  S0 = 100; K = 100; r = 0.05; s = 0.2; T = 1; L = 6 * s * sqrt(T);
  x = log(S0) + linspace(-L, L, M + 1)'; dx = x(2) - x(1); S = exp(x);
  a = s ^ 2 / 2; b = r - s ^ 2 / 2;
  lo = a / dx ^ 2 - b / (2 * dx); di = -2 * a / dx ^ 2 - r; up = a / dx ^ 2 + b / (2 * dx);
  n = M - 1; A = spdiags([lo * ones(n, 1), di * ones(n, 1), up * ones(n, 1)], [-1 0 1], n, n);
  I = speye(n); dt = T / Nt; V = max(S - K, 0);
  for k = 1:Nt
    b1old = V(end); b1 = S(end) - K * exp(-r * k * dt);
    bv = zeros(n, 1); bv(end) = up * ((1 - th) * b1old + th * b1);
    Vi = (I - th * dt * A) \ ((I + (1 - th) * dt * A) * V(2:M) + dt * bv);
    V = [0; Vi; b1];
  end
  V = V(M / 2 + 1);
end

addpath(fileparts(mfilename('fullpath')));
r = {};
add = @(r, name, v, tol) [r; {name, v, tol}];
S0 = 100; K = 100; rr = 0.05; s = 0.2; T = 1;
c0 = bsq(S0, K, rr, s, T, true); p0 = bsq(S0, K, rr, s, T, false);
r = add(r, 'callBase', c0, 1e-11); r = add(r, 'putBase', p0, 1e-11);
% Greeks by Richardson extrapolation of central differences of the closed form
h = 0.01; f = @(S) bsf(S, K, rr, s, T, 0);
D = @(h) (f(S0 + h) - f(S0 - h)) / (2 * h); G = @(h) (f(S0 + h) - 2 * f(S0) + f(S0 - h)) / h ^ 2;
r = add(r, 'deltaC', (4 * D(h / 2) - D(h)) / 3, 1e-9); r = add(r, 'gammaB', (4 * G(h / 2) - G(h)) / 3, 1e-7);
fv = @(sg) bsf(S0, K, rr, sg, T, 0); V = @(h) (fv(s + h) - fv(s - h)) / (2 * h);
r = add(r, 'vegaB', (4 * V(1e-3) - V(2e-3)) / 3, 1e-8);
fT = @(t) bsf(S0, K, rr, s, t, 0); Th = @(h) -(fT(T + h) - fT(T - h)) / (2 * h);
r = add(r, 'thetaC', (4 * Th(1e-3) - Th(2e-3)) / 3, 1e-8);
gam = @(x) -exp(-((log(x / K) + (rr + s ^ 2 / 2) * T) / (s * sqrt(T))) ^ 2 / 2) / (x * s * sqrt(2 * pi * T));
r = add(r, 'gammaPeakS', fminbnd(gam, 60, 140, optimset('TolX', 1e-12)), 1e-6);
% dividends, Black-76, Garman-Kohlhagen
r = add(r, 'divYieldCall', bsf(100, 100, rr, s, 1, 0.03), 1e-12);
r = add(r, 'blackCall', exp(-0.04 * 0.5) * bsf(70, 72, 0, 0.35, 0.5, 0), 1e-12);
r = add(r, 'gkCall', 1e6 * bsf(1.38, 1.40, 0.025, 0.07, 0.25, 0.04), 1e-10);
pv = 2 * exp(-rr * 0.25) + 2 * exp(-rr * 0.75); r = add(r, 'escCall', bsf(100 - pv, 100, rr, s, 1, 0), 1e-12);
% Monte Carlo theory: correlation of antithetic payoffs by quadrature
m = (rr - s ^ 2 / 2) * T; v = s * sqrt(T);
pay = @(z) exp(-rr * T) * max(S0 * exp(m + v * z) - K, 0);
EY2 = quadgk(@(z) pay(z) .^ 2 .* exp(-z .^ 2 / 2) / sqrt(2 * pi), -m / v, 14, 'RelTol', 1e-13, 'AbsTol', 1e-14);
EA = quadgk(@(z) pay(z) .* pay(-z) .* exp(-z .^ 2 / 2) / sqrt(2 * pi), -m / v, m / v, 'RelTol', 1e-13, 'AbsTol', 1e-15);
r = add(r, 'mcSdY', sqrt(EY2 - c0 ^ 2), 1e-9); r = add(r, 'mcCorrAnti', (EA - c0 ^ 2) / (EY2 - c0 ^ 2), 1e-8);
% finite differences with sparse matrices
r = add(r, 'fdCNValue', fdsparse(400, 200, 0.5), 1e-10);
r = add(r, 'fdErrImpl', abs(fdsparse(400, 200, 1) - c0), 1e-9);
% implied volatility by fzero
pex = bsf(100, 120, rr, 0.3, 0.5, 0); r = add(r, 'ivExPrice', pex, 1e-12);
r = add(r, 'xvEightSig', 100 * fzero(@(x) bsf(100, 105, 0.03, x, 0.75, 0) - 8, [0.01 2], optimset('TolX', 1e-15)), 1e-9);
% Heston by quadgk on the Lewis integral
v0 = 0.04; ka = 2; th = 0.04; xi = 0.5; rho = -0.7; Th = 0.5; F = 100 * exp(rr * Th); Kh = F;
cf = @(u) exp(((ka * th / xi ^ 2) * ((ka - rho * xi * 1i * u - sqrt((rho * xi * 1i * u - ka) .^ 2 + xi ^ 2 * (1i * u + u .^ 2))) * Th ...
  - 2 * log((1 - ((ka - rho * xi * 1i * u - sqrt((rho * xi * 1i * u - ka) .^ 2 + xi ^ 2 * (1i * u + u .^ 2))) ./ (ka - rho * xi * 1i * u + sqrt((rho * xi * 1i * u - ka) .^ 2 + xi ^ 2 * (1i * u + u .^ 2)))) .* exp(-sqrt((rho * xi * 1i * u - ka) .^ 2 + xi ^ 2 * (1i * u + u .^ 2)) * Th)) ...
  ./ (1 - ((ka - rho * xi * 1i * u - sqrt((rho * xi * 1i * u - ka) .^ 2 + xi ^ 2 * (1i * u + u .^ 2))) ./ (ka - rho * xi * 1i * u + sqrt((rho * xi * 1i * u - ka) .^ 2 + xi ^ 2 * (1i * u + u .^ 2)))))))) ...
  + (v0 / xi ^ 2) * (ka - rho * xi * 1i * u - sqrt((rho * xi * 1i * u - ka) .^ 2 + xi ^ 2 * (1i * u + u .^ 2))) ...
  .* (1 - exp(-sqrt((rho * xi * 1i * u - ka) .^ 2 + xi ^ 2 * (1i * u + u .^ 2)) * Th)) ...
  ./ (1 - ((ka - rho * xi * 1i * u - sqrt((rho * xi * 1i * u - ka) .^ 2 + xi ^ 2 * (1i * u + u .^ 2))) ./ (ka - rho * xi * 1i * u + sqrt((rho * xi * 1i * u - ka) .^ 2 + xi ^ 2 * (1i * u + u .^ 2)))) .* exp(-sqrt((rho * xi * 1i * u - ka) .^ 2 + xi ^ 2 * (1i * u + u .^ 2)) * Th)));
k = log(F / Kh);
val = quadgk(@(u) real(exp(1i * u * k) .* cf(u - 0.5i)) ./ (u .^ 2 + 0.25), 0, Inf, 'RelTol', 1e-12, 'AbsTol', 1e-14, 'MaxIntervalCount', 5000);
r = add(r, 'hesFourier', exp(-rr * Th) * (F - sqrt(F * Kh) / pi * val), 1e-8);
% leading-order hedging error, weekly rebalancing, drift 10%
mu = 0.10;
g2 = @(ta) (1 ./ (2 * pi * s ^ 2 * ta)) .* sqrt(pi * s ^ 2 * ta) .* exp(2 * (log(S0) + (mu - s ^ 2 / 2) * (T - ta)) + 2 * s ^ 2 * (T - ta)) ...
  .* exp(-0.5 * ((log(S0) + (mu - s ^ 2 / 2) * (T - ta) + 2 * s ^ 2 * (T - ta)) - (log(K) - (rr + s ^ 2 / 2) * ta)) .^ 2 ./ (s ^ 2 * (T - ta) + s ^ 2 * ta / 2)) ...
  ./ sqrt(2 * pi * (s ^ 2 * (T - ta) + s ^ 2 * ta / 2));
I = quadgk(@(u) 2 * u .* exp(-2 * rr * (T - u .^ 2)) * s ^ 4 .* g2(u .^ 2), 0, sqrt(T), 'RelTol', 1e-12, 'AbsTol', 1e-15);
r = add(r, 'hedgeThFiftyTwo', sqrt(T / 52 / 2 * I), 1e-8);
% exercises
r = add(r, 'xvOneCall', bsq(50, 55, 0.04, 0.25, 0.5, true), 1e-10);
r = add(r, 'xvTenGap', bsf(100, 100, rr, 0.25, 1, 0) - bsf(100, 100, rr, 0.2, 1, 0), 1e-12);

fid = fopen('code/ch05/out/octave_check.csv', 'w'); fprintf(fid, 'name,value,tol\n');
for i = 1:rows(r), fprintf(fid, '%s,%.15g,%g\n', r{i, 1}, r{i, 2}, r{i, 3}); end
fclose(fid); printf('%d Octave checks written\n', rows(r));
