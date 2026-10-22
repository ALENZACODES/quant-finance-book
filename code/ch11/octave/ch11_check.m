% Octave cross-check for Chapter 11. Run from the repository root:
%   octave-cli --no-gui code/ch11/octave/ch11_check.m
% Writes code/ch11/out/octave_check.csv (name,value,tol), compared with Python in test_crosschecks.py.
% Methods differ from Python on purpose: the leverage proposition by stepping net debt year by year for
% a given asset multiple, the value driver by summing the growing cash flows directly, the Miles and
% Ezzell rate by solving the one year APV recursion with fzero, the APV of the manufacturer by a
% forward sum of discounted tax shields, the lookback bound by quadgk over the density of the maximum,
% and the exercise answers from their formulas.
1;
function r = add(r, name, value, tol)
  r(end + 1, :) = {name, value, tol};
end
r = cell(0, 3);
% leverage: an asset path with free cash flow F each year and exit value set so that U is given
function m = levered(U, lam, i, tau, T)
  j = i * (1 - tau); P = 1; F = 0.05 * ones(1, T);
  EVT = U * P - sum(F .* (1 + j) .^ (T - (1:T)));
  N = lam * P;
  for t = 1:T, N = N * (1 + j) - F(t); end
  m = (EVT - N) / ((1 - lam) * P);
end
r = add(r, "lxMHi", levered(2.0, 0.6, 0.08, 0.25, 5), 1e-12);
r = add(r, "lxMLo", levered(1.1, 0.6, 0.08, 0.25, 5), 1e-12);
r = add(r, "yfMoic", levered(1.8, 0.6, 0.07, 0.25, 5), 1e-12);
r = add(r, "yfMoicLow", levered(1.2, 0.6, 0.07, 0.25, 5), 1e-12);
% value driver: sum the free cash flows of a firm reinvesting g/roc of NOPAT
function v = driver(W, g, roc)
  t = 1:20000; v = sum((1 - g / roc) / (1 + g) * ((1 + g) / (1 + W)) .^ t);
end
r = add(r, "yhV", 10 * driver(0.08, 0.03, 0.15), 1e-8);
r = add(r, "yhVhigh", 10 * driver(0.08, 0.05, 0.15), 1e-8);
r = add(r, "ynLow", driver(0.09, 0.02, 0.07), 1e-8);
r = add(r, "ynHigh", driver(0.09, 0.04, 0.07), 1e-8);
% Miles and Ezzell: the rate W at which a level perpetuity satisfies the one year APV recursion
function W = me(rU, rD, tau, L)
  f = @(w) (1 + 1 / w) / (1 + rU) + tau * rD * L * (1 / w) / (1 + rD) - 1 / w;
  W = fzero(f, [0.01 0.5], optimset("TolX", 1e-15));
end
r = add(r, "yiMe", 100 * me(0.09, 0.06, 0.25, 0.40), 1e-9);
% the manufacturer: inputs from the data, then the value as unlevered value plus discounted shields
fid = fopen("data/real_assets_monthly.csv"); hdr = strsplit(fgetl(fid), ","); fclose(fid);
txt = fileread("data/real_assets_monthly.csv"); lines = strsplit(strtrim(txt), "\n");
col = find(strcmp(hdr, "gs10"));
for k = 2:numel(lines)
  f = strsplit(lines{k}, ",");
  if strncmp(f{1}, "2026-09-01", 10), rf = str2double(f{col}) / 100; end
end
txt = fileread("data/damodaran_industry.csv"); lines = strsplit(strtrim(txt), "\n"); hdr = strsplit(lines{1}, ",");
for k = 2:numel(lines)
  f = strsplit(lines{k}, ",");
  if strcmp(f{1}, "Machinery"), row = f; end
end
beta = str2double(row{find(strcmp(hdr, "beta"))}); dw = str2double(row{find(strcmp(hdr, "debt_weight"))});
ew = str2double(row{find(strcmp(hdr, "equity_weight"))});
txt = fileread("data/damodaran_erp.csv"); lines = strsplit(strtrim(txt), "\n");
for k = 2:numel(lines)
  f = strsplit(lines{k}, ",");
  if strcmp(f{1}, "2025"), erp = str2double(f{2}); end
end
rd = rf + 0.0175; rU = rf + beta / (1 + dw / ew) * erp; L = 0.3; tau = 0.25;
W = me(rU, rd, tau, L); r = add(r, "wkWme", 100 * W, 1e-9); r = add(r, "wkRu", 100 * rU, 1e-10);
mg = (18.4 + 1.2 + 0.9 - 0.5 + 0.6 - 0.4) / 120; rev = 120; g = [0.05 0.05 0.04 0.04 0.03]; F = zeros(1, 5);
for y = 1:5
  r0 = rev; rev = rev * (1 + g(y));
  F(y) = rev * (mg - 0.03) * 0.75 + 0.03 * rev - 0.035 * rev - 0.12 * (rev - r0);
end
F6 = rev * 1.025 * (mg - 0.03) * 0.75 * (1 - 0.025 / 0.12);
% the value path V_t at the Miles and Ezzell rate, then the shields tau rD L V_{t-1} paid at t
N = 400; cf = [F, F6 * 1.025 .^ (0:N - 6)]; V = zeros(1, N + 1);
V(N + 1) = cf(N) * 1.025 / (W - 0.025);
for t = N:-1:1, V(t) = (cf(t) + V(t + 1)) / (1 + W); end
Vu = sum(cf ./ (1 + rU) .^ (1:N)) + cf(N) * 1.025 / (rU - 0.025) / (1 + rU) ^ N;
sh = tau * rd * L * V(1:N); pvs = sum(sh ./ ((1 + rU) .^ ((1:N) - 1) * (1 + rd)));
tailsh = sh(N) * 1.025 / (rU - 0.025) / ((1 + rU) ^ (N - 1) * (1 + rd));
r = add(r, "vlEv", V(1), 1e-9); r = add(r, "vlVu", Vu, 1e-9); r = add(r, "vlShield", pvs + tailsh, 1e-7);
% lookback bound by quadrature over the density of the maximum of the drifted Brownian motion
function b = look(T, s)
  v = s * sqrt(T); mu = -s ^ 2 / 2; c = 2 * mu / s ^ 2;
  phi = @(z) exp(-z .^ 2 / 2) / sqrt(2 * pi); Phi = @(z) 0.5 * erfc(-z / sqrt(2));
  % density of the maximum: minus the derivative of P(M > x)
  dens = @(x) phi((x - mu * T) / v) / v + exp(c * x) .* phi((x + mu * T) / v) / v ...
              - c * exp(c * x) .* Phi((-x - mu * T) / v);
  b = quadgk(@(x) exp(x) .* dens(x), 0, 20 * v + 5, "AbsTol", 1e-13, "RelTol", 1e-12) - 1;
end
r = add(r, "dmLong", 100 * look(2, 0.45), 1e-7); r = add(r, "dmLsThirtyOne", 100 * look(1, 0.3), 1e-7);
r = add(r, "dmLsSixtyFour", 100 * look(4, 0.6), 1e-7); r = add(r, "ymLs", 100 * look(2, 0.4), 1e-7);
% exercises from their formulas
r = add(r, "ycDur", log(1.8) / log(1.15), 1e-12); r = add(r, "ydPme", 160 / 130, 1e-12);
r = add(r, "ygG", 9.5 * 15, 1e-12); r = add(r, "ygM", 47.5, 1e-12); r = add(r, "ykV", 7 * (8.0 + 0.55 - 0.3), 1e-12);
r = add(r, "yjHam", 1.3 / (1 + 0.75 * 0.5), 1e-12); r = add(r, "yjHp", 1.3 / 1.5, 1e-12);
fid = fopen("code/ch11/out/octave_check.csv", "w"); fprintf(fid, "name,value,tol\n");
for j = 1:rows(r)
  fprintf(fid, "%s,%.15g,%g\n", r{j, 1}, r{j, 2}, r{j, 3});
end
fclose(fid); printf("%d checks written\n", rows(r));
