% Octave cross-check for Chapter 9. Run from the repository root:
%   octave-cli --no-gui code/ch09/octave/ch09_check.m
% Writes code/ch09/out/octave_check.csv (name,value,tol), compared with Python in test_crosschecks.py.
% Methods differ from Python on purpose: normal tails through erfc, the risk-share weight by fzero,
% the expected maximum by quad, the smile regression by the backslash operator, and drawdowns by cummax,
% all from the saved program file out/program.csv; the hurdle fees by quad, the closure formula through
% erfc, and the issuance sum.
1;
function r = add(r, name, value, tol)
  r(end + 1, :) = {name, value, tol};
end
function p = Phi(x)
  p = 0.5 * erfc(-x / sqrt(2));
end
function p = phi(x)
  p = exp(-x .^ 2 / 2) / sqrt(2 * pi);
end
function m = smean(mu, s, T, c)
  se = s / sqrt(T); a = (c - mu) / se; m = mu + se * phi(a) / (1 - Phi(a));
end
function v = cmean(mu, s)
  v = mu * Phi(mu / s) + s * phi(mu / s);
end
function d = mdd(x)
  w = [1; cumprod(1 + x(:))]; d = max(1 - w ./ cummax(w));
end
function s = sr(x)
  s = sqrt(12) * mean(x) / std(x);
end
r = cell(0, 3);
r = add(r, 'svBiasZeroTh', 100 * 12 * (smean(0.006, 0.03, 36, 0) - 0.006), 1e-9);
r = add(r, 'exDMean', 100 * 12 * smean(0.005, 0.04, 24, 0), 1e-9);
r = add(r, 'ffSepOne', 100 * 0.2 * cmean(0.08, 0.15), 1e-10);
r = add(r, 'ffPoolTwenty', 100 * 0.2 * cmean(0.08, 0.15 * sqrt(0.3 + 0.7 / 20)), 1e-10);
a = 0.01; b = 0.5625; c = 0.3 * 0.1 * 0.75;
sh = @(w) (w .^ 2 * b + w .* (1 - w) * c) ./ ((1 - w) .^ 2 * a + 2 * w .* (1 - w) * c + w .^ 2 * b);
r = add(r, 'exHW', 100 * fzero(@(w) sh(w) - 0.25, [1e-6, 0.5]), 1e-7);
r = add(r, 'exJMax', quad(@(z) z .* 20 .* phi(z) .* Phi(z) .^ 19, -12, 12), 1e-7);
r = add(r, 'exCLim', 0.2 / sqrt(0.1), 1e-12);
% from the saved program
D = dlmread('code/ch09/out/program.csv', ',', 1, 0);
mo = D(:, 1); gr = D(:, 2); net = D(:, 3); eq = D(:, 4); ust = D(:, 5);
r = add(r, 'tfSR', sr(gr), 1e-9); r = add(r, 'tfSRnet', sr(net), 1e-9);
r = add(r, 'tfMddEq', 100 * mdd(eq), 1e-9);
A = 0.6 * eq + 0.4 * ust; B = 0.9 * A + 0.1 * net;
r = add(r, 'csBSR', sr(B), 1e-9); r = add(r, 'csAMdd', 100 * mdd(A), 1e-9);
r = add(r, 'csCorrAT', corr(A, net), 1e-9);
q = floor(mo / 100) * 10 + floor((mod(mo, 100) - 1) / 3);
uq = unique(q); P = []; E = [];
for i = 1:numel(uq)
  k = q == uq(i);
  if sum(k) == 3, P(end + 1, 1) = prod(1 + gr(k)) - 1; E(end + 1, 1) = prod(1 + eq(k)) - 1; end
end
X = [ones(size(E)), E, E .^ 2]; bet = X \ P;
r = add(r, 'slQuad', bet(3), 1e-9); r = add(r, 'slLin', bet(2), 1e-9);
% fees: hard and soft hurdles by quad against the normal density, f = 0.2, h = 0.05, N(0.08, 0.12^2)
nd = @(x) exp(-((x - 0.08) / 0.12) .^ 2 / 2) / (0.12 * sqrt(2 * pi));
r = add(r, 'feHard', 100 * quad(@(x) 0.2 * max(x - 0.05, 0) .* nd(x), 0.05, 2), 1e-7);
r = add(r, 'feSoft', 100 * quad(@(x) min(max(x - 0.05, 0), 0.2 * max(x, 0)) .* nd(x), 0.05, 2), 1e-7);
% closure at a loss of 40%, zero drift, volatility 0.15: the survival formula through erfc
b = -log(0.6);
r = add(r, 'clSurvFive', 100 * (Phi(b / (0.15 * sqrt(5))) - Phi(-b / (0.15 * sqrt(5)))), 1e-9);
% the issuance schedule and the gambler's ruin
r = add(r, 'ocSupply', sum(210000 * floor(5e9 ./ 2 .^ (0:63))) / 1e8, 1e-12);
r = add(r, 'exNRuin', (0.2 / 0.8) ^ 6, 1e-12);
fid = fopen('code/ch09/out/octave_check.csv', 'w'); fprintf(fid, 'name,value,tol\n');
for i = 1:rows(r), fprintf(fid, '%s,%.15g,%g\n', r{i, 1}, r{i, 2}, r{i, 3}); end
fclose(fid); printf('%d checks written\n', rows(r));
