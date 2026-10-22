% Octave cross-check for Chapter 12. Run from the repository root:
%   octave-cli --no-gui code/ch12/octave/ch12_check.m
% Writes code/ch12/out/octave_check.csv (name,value,tol), compared with Python in test_crosschecks.py.
% Methods differ from Python on purpose: the first ridge forecast of the equity premium by the normal
% equations rather than the SVD, the first lasso forecast by proximal gradient (ISTA) rather than
% coordinate descent, the expected maximum of normals by quadgk, the Black-Scholes price by quadrature
% of the payoff against the lognormal density, Whalley and Wilmott's band, and the exercise answers.
1;
function r = add(r, name, value, tol)
  r(end + 1, :) = {name, value, tol};
end
r = cell(0, 3);
raw = jsondecode(fileread("code/ch12/out/numbers.json")).RAW;
% ---- equity premium, first out-of-sample forecast (skipped without the data)
if exist("data/goyal_welch_monthly.csv", "file")
  d = dlmread("data/goyal_welch_monthly.csv", ",", 1, 0, "emptyvalue", NaN);
  % columns: month index d12 e12 bm tbl aaa baa lty ntis rf infl ltr corpr svar ret
  mo = d(:, 1); P = d(:, 2); D = d(:, 3); E = d(:, 4); n0 = rows(d);
  lag = @(v) [NaN; v(1:end - 1)];
  X = [log(D) - log(P), log(D) - log(lag(P)), log(E) - log(P), log(D) - log(E), d(:, 15), d(:, 5), d(:, 10), ...
       d(:, 6), d(:, 9), d(:, 13), d(:, 9) - d(:, 6), d(:, 8) - d(:, 7), d(:, 14) - d(:, 13), lag(d(:, 12))];
  rr = log(1 + d(:, 16)) - log(1 + d(:, 11));
  ok = all(isfinite(X), 2);
  X = X(ok, :); rr = rr(ok); mo = mo(ok);
  j = find(mo >= 196501, 1);
  Xi = X(1:j - 2, :); yi = rr(2:j - 1); xj = X(j - 1, :); n = rows(Xi);
  mu = mean(Xi); sd = std(Xi, 1); Z = (Xi - mu) ./ sd; zj = (xj - mu) ./ sd; yc = yi - mean(yi);
  a = raw.eqRidgeAFirst;
  b = (Z' * Z + a * n * eye(columns(Z))) \ (Z' * yc);
  r = add(r, "eqRidgeFirst", mean(yi) + zj * b, 1e-8);
  lam = raw.eqLassoFFirst * max(abs(Z' * yc)) / n;
  Lp = max(eig(Z' * Z / n)); b = zeros(columns(Z), 1);
  for it = 1:200000
    g = Z' * (Z * b - yc) / n; bn = b - g / Lp; bn = sign(bn) .* max(abs(bn) - lam / Lp, 0);
    if max(abs(bn - b)) < 1e-14, b = bn; break; end
    b = bn;
  end
  r = add(r, "eqLassoFirst", mean(yi) + zj * b, 1e-7);
end
% ---- expected maximum of N standard normals
phi = @(z) exp(-z .^ 2 / 2) / sqrt(2 * pi); Phi = @(z) 0.5 * erfc(-z / sqrt(2));
emax = @(N) quadgk(@(z) z .* N .* phi(z) .* Phi(z) .^ (N - 1), -12, 12, "AbsTol", 1e-13, "RelTol", 1e-12);
r = add(r, "ovMaxTen", emax(10), 1e-8); r = add(r, "ovMaxHundred", emax(100), 1e-8);
r = add(r, "ovBoundHundred", sqrt(2 * log(100)), 1e-12);
% ---- Black-Scholes price by quadrature of the payoff, and the band at the money
s = 0.2; T = 0.25; v = s * sqrt(T);
c = quadgk(@(x) max(100 * exp(x) - 100, 0) .* phi((x + v ^ 2 / 2) / v) / v, -2, 2, "AbsTol", 1e-12, "RelTol", 1e-12);
r = add(r, "hdBs", c, 1e-9);
d1 = v / 2; gm = phi(d1) / (100 * v);
r = add(r, "hdWwAtm", (1.5 * 0.002 * 100 * gm ^ 2) ^ (1 / 3), 1e-10);
% ---- exercises
r = add(r, "zaLasso", 0.55, 1e-12); r = add(r, "zbMse", 1 / 3, 1e-12); r = add(r, "zdCorr", 2 / 3, 1e-12);
r = add(r, "zeYhat", 2 * tanh(0.5), 1e-12); r = add(r, "zfRtwo", 100 * (1 - 0.00190 / 0.00192), 1e-10);
r = add(r, "zkSe", sqrt((1 + 0.125 + 0.75 * 0.0625) / 119), 1e-12); r = add(r, "zlCut", 100 * 0.81 / 0.96, 1e-10);
fid = fopen("code/ch12/out/octave_check.csv", "w"); fprintf(fid, "name,value,tol\n");
for k = 1:rows(r)
  fprintf(fid, "%s,%.15g,%g\n", r{k, 1}, r{k, 2}, r{k, 3});
end
fclose(fid); printf("%d Octave checks written\n", rows(r));
