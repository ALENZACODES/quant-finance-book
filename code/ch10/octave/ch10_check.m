% Octave cross-check for Chapter 10. Run from the repository root:
%   octave-cli --no-gui code/ch10/octave/ch10_check.m
% Writes code/ch10/out/octave_check.csv (name,value,tol), compared with Python in test_crosschecks.py.
% Methods differ from Python on purpose: the IRR and the rate at which the loan tests switch by fzero,
% the loan balance by the closed form for an annuity's remaining payments, the Faustmann rotation by
% fminbnd on the land value, the concession by a closed form for the sum, the appraisal-era autocorrelation
% by its normal equations and the smoothing ratio from the filter's moving average weights, and the exercise answers from their formulas.
1;
function r = add(r, name, value, tol)
  r(end + 1, :) = {name, value, tol};
end
function k = mconst(rate, yrs)
  i = rate / 12; n = 12 * yrs; k = 12 * i / (1 - (1 + i) ^ (-n));
end
function b = remaining(rate, yrs, t)
  % present value of the payments still due, per unit of original loan
  i = rate / 12; n = 12 * yrs; m = 12 * t;
  b = mconst(rate, yrs) / 12 * (1 - (1 + i) ^ (-(n - m))) / i;
end
function v = q(t)
  v = 400 * (1 - exp(-0.06 * t)) .^ 3;
end
r = cell(0, 3);
noi = 2.40e6 * 0.93 + 0.05e6 - 0.82e6; cap = 0.05 / 0.9; V = noi * 0.9 / 0.05;
r = add(r, "pfSaleShare", 100 * (1.025 / 1.075) ^ 10, 1e-10);
t = 1:10; cf = [-V, noi * 1.025 .^ (t - 1) * 0.9]; cf(end) += noi * 1.025 ^ 10 / (cap + 0.0025) * 0.98;
r = add(r, "pfIrr", 100 * fzero(@(x) sum(cf ./ (1 + x) .^ (0:10)), [0.01 0.2], optimset("TolX", 1e-14)), 1e-7);
k = mconst(0.0625, 25); r = add(r, "lvK", 100 * k, 1e-10); r = add(r, "lvBal", 100 * remaining(0.0625, 25, 10), 1e-10);
r = add(r, "lvRateSwitch", 100 * fzero(@(x) noi / (1.25 * mconst(x, 25)) - 0.65 * V, [0.01 0.2], optimset("TolX", 1e-14)), 1e-7);
r = add(r, "xfBal", 100 * remaining(0.06, 25, 10), 1e-10); r = add(r, "xdK", 100 * mconst(0.055, 30), 1e-10);
r = add(r, "xdDscr", 0.06 / (0.7 * mconst(0.055, 30)), 1e-10);
for rr = [0.03 0.04 0.06]
  lev = @(T) -(40 * q(T) * exp(-rr * T) - 1500) / (1 - exp(-rr * T));
  T = fminbnd(lev, 10, 80, optimset("TolX", 1e-10));
  nm = {"Three", "Four", "Six"}{find(abs([0.03 0.04 0.06] - rr) < 1e-12)};
  r = add(r, ["tbF" nm], T, 1e-5); r = add(r, ["tbLev" nm], -lev(T), 1e-8);
end
r = add(r, "xkF", fminbnd(@(T) -(40 * q(T) * exp(-0.05 * T) - 1500) / (1 - exp(-0.05 * T)), 10, 80, optimset("TolX", 1e-10)), 1e-5);
x = 1.025 / 1.07;
for n = [30 50 99]
  v = 100 / 1.07 * (1 - x ^ n) / (1 - x);
  nm = {"Thirty", "Fifty", "NinetyNine"}{find([30 50 99] == n)};
  r = add(r, ["tbCon" nm], v, 1e-10); r = add(r, ["tbShare" nm], 100 * v / (100 / 0.045), 1e-10);
end
r = add(r, "tbDurPerp", 1.07 / 0.045, 1e-10);
% the appraisal-era property index: AR(1) by the normal equations, then the filter weights and unsmoothing
fid = fopen("data/real_assets_quarterly.csv"); C = textscan(fid, "%s %f %f %f %f", "Delimiter", ",", ...
  "HeaderLines", 1, "EmptyValue", NaN); fclose(fid);
d = C{1}; x = diff(log(C{2})); d = d(2:end);
pre = x(cellfun(@(z) str2double(z(1:4)) < 1996, d));
y = pre(2:end); z = pre(1:end - 1); a = sum((z - mean(z)) .* (y - mean(y))) / sum((z - mean(z)) .^ 2);
r = add(r, "smArPre", a, 1e-9);
w = (1 - a) * a .^ (0:5000); r = add(r, "smTheoryPre", 1 / sqrt(sum(w .^ 2)), 1e-8);
u = (pre(2:end) - a * pre(1:end - 1)) / (1 - a); r = add(r, "smUvolPre", 200 * std(u), 1e-9);
r = add(r, "smCumGap", 100 * a / (1 - a) * (pre(end) - pre(1)), 1e-9);
r = add(r, "smCorrShrink", sqrt(1 - a ^ 2), 1e-9);
r = add(r, "xhVol", 100 * 0.10 * sqrt(sum(((1 - 0.7) * 0.7 .^ (0:3000)) .^ 2)), 1e-8);
r = add(r, "xjSum", sum(0.5 * 0.5 .^ (0:3)), 1e-12);
r = add(r, "xaCap", 100 * 0.06 / 0.92, 1e-10); r = add(r, "xaValue", 0.74e6 * 0.92 / 0.06 / 1e6, 1e-10);
r = add(r, "xgRet", 100 * (0.07 - 0.6 * 0.055) / 0.4, 1e-10);
r = add(r, "xmBound", 70 * exp(0.03), 1e-10); r = add(r, "xmCy", 100 * (0.04 - log(71 / 70) / 0.5), 1e-10);
v = 50 / 1.08 * (1 - (1.03 / 1.08) ^ 25) / (1 - 1.03 / 1.08); r = add(r, "xlV", v, 1e-10);
r = add(r, "xnW", 0.6 * 0.03 ^ 2 / 0.01, 1e-12);
% intrinsic value of storage, exercise 10.15, by enumerating the pairs
F = [3.00 3.60 3.20]; t = [1 4 7] / 12; best = 0; v13 = 0;
for i = 1:3
  for j = i + 1:3
    v = exp(-0.04 * t(j)) * (F(j) - 0.05) - exp(-0.04 * t(i)) * (F(i) + 0.05); best = max(best, v);
    if i == 1 && j == 3, v13 = v; end
  end
end
r = add(r, "xoV", best, 1e-12); r = add(r, "xoLast", v13, 1e-12);
fid = fopen("code/ch10/out/octave_check.csv", "w"); fprintf(fid, "name,value,tol\n");
for j = 1:rows(r)
  fprintf(fid, "%s,%.15g,%g\n", r{j, 1}, r{j, 2}, r{j, 3});
end
fclose(fid); printf("%d checks written\n", rows(r));
