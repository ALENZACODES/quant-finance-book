% Octave cross-check for Chapter 6. Run from the repository root:
%   octave-cli --no-gui code/ch06/octave/ch06_check.m
% Writes code/ch06/out/octave_check.csv (name,value,tol), compared with Python in test_crosschecks.py.
% Methods differ from Python on purpose: the CIR bond price by integrating its Riccati equations with
% ode45, the Vasicek caplet by quadgk against the Gaussian law of the short rate, the Merton calibration
% by fsolve, the large-pool tranche losses by quadgk, and duration from the Macaulay sum.
1;
function v = ncdf(x)
  v = 0.5 * erfc(-x / sqrt(2));
end
function v = ninv(p)
  v = -sqrt(2) * erfcinv(2 * p);
end
function r = add(r, name, value, tol)
  r(end + 1, :) = {name, value, tol};
end
function M = csv2cell_fit(f)
  fid = fopen(f); fgetl(fid); M = [];
  while true
    l = fgetl(fid); if ~ischar(l), break; end
    p = strsplit(l, ','); M(end + 1, :) = str2double(p(3:6));
  end
  fclose(fid);
end
r = cell(0, 3);
fp = csv2cell_fit('code/ch06/out/fit_params.csv');
% rows: 1 now vasicek, 2 now cir, 3 inv vasicek, 4 inv cir
% CIR bond price from the Riccati equations  B' = 1 - k B - s^2 B^2 / 2,  (ln A)' = -k rb B
c = fp(2, :); k = c(2); rb = c(3); s = c(4); r0 = c(1);
opt = odeset('RelTol', 1e-12, 'AbsTol', 1e-14);
[~, Y] = ode45(@(t, y) [1 - k * y(1) - 0.5 * s ^ 2 * y(1) ^ 2; -k * rb * y(1)], [0 5 10], [0; 0], opt);
r = add(r, 'srMcExact', exp(Y(end, 2) - Y(end, 1) * r0), 1e-9);
% Vasicek caplet: integrate the discounted payoff over the joint Gaussian law of (r_T, int r)
v = fp(1, :); r0 = v(1); k = v(2); rb = v(3); s = v(4);
vasP = @(r, t) exp(((1 - exp(-k * t)) / k - t) * (k ^ 2 * rb - s ^ 2 / 2) / k ^ 2 - s ^ 2 * ((1 - exp(-k * t)) / k) .^ 2 / (4 * k) - (1 - exp(-k * t)) / k .* r);
T = 2; dl = 0.25; F = (vasP(r0, T) / vasP(r0, T + dl) - 1) / dl;
e1 = exp(-k * T); e2 = exp(-2 * k * T);
mr = rb + (r0 - rb) * e1; mI = rb * T + (r0 - rb) * (1 - e1) / k;
vr = s ^ 2 * (1 - e2) / (2 * k); vI = s ^ 2 / k ^ 2 * (T - 2 * (1 - e1) / k + (1 - e2) / (2 * k));
cRI = s ^ 2 / k * ((1 - e1) / k - (1 - e2) / (2 * k));
% E[e^{-I} | r_T] = exp(-m_{I|r} + v_{I|r}/2)
cm = @(x) mI + cRI / vr * (x - mr); cv = vI - cRI ^ 2 / vr;
pay = @(x) exp(-cm(x) + cv / 2) .* vasP(x, dl) .* dl .* max((1 ./ vasP(x, dl) - 1) / dl - F, 0) .* exp(-(x - mr) .^ 2 / (2 * vr)) / sqrt(2 * pi * vr);
B1 = (1 - exp(-k * dl)) / k; lnA1 = log(vasP(0, dl));
xs = (lnA1 + log(1 + F * dl)) / B1;   % the payoff is zero below this short rate
r = add(r, 'cpAtmTwo', 1e4 * quadgk(pay, xs, mr + 14 * sqrt(vr), 'RelTol', 1e-13, 'AbsTol', 1e-18), 1e-7);
% Merton calibration
fm = @(x) merton_res(x);
function f = merton_res(x)
  V = x(1); sv = x(2); D = 80; r = 0.03; T = 5;
  d1 = (log(V / D) + (r + sv ^ 2 / 2) * T) / (sv * sqrt(T)); d2 = d1 - sv * sqrt(T);
  E = V * 0.5 * erfc(-d1 / sqrt(2)) - D * exp(-r * T) * 0.5 * erfc(-d2 / sqrt(2));
  f = [E - 40; 0.5 * erfc(-d1 / sqrt(2)) * sv * V - 0.6 * 40];
end
x = fsolve(fm, [110; 0.3], optimset('TolFun', 1e-14, 'TolX', 1e-14));
r = add(r, 'mcV', x(1), 1e-7); r = add(r, 'mcSigV', 100 * x(2), 1e-7);
% large-pool tranche losses, with the five year default probability from the Python bootstrap
fid = fopen('code/ch06/out/hazard_cds.json'); txt = fread(fid, '*char')'; fclose(fid);
tok = regexp(txt, '"hzPdFive": ([0-9.eE+-]+)', 'tokens'); p = str2double(tok{end}{1}) / 100;
kink = @(rho, x) (ninv(p) - sqrt(1 - rho) * ninv(x / 0.6)) / sqrt(rho);
function w = inside(v)
  w = sort(v(v > -12 & v < 12));
end
function v = lhq(f, w)
  e = [-12, w, 12]; v = 0;
  for j = 1:numel(e) - 1
    v += quadgk(f, e(j), e(j + 1), 'RelTol', 1e-12, 'AbsTol', 1e-16, 'MaxIntervalCount', 5000);
  end
end
lhf = @(rho, a, d) @(m) min(max(0.6 * ncdf((ninv(p) - sqrt(rho) * m) / sqrt(1 - rho)) - a, 0), d - a) .* exp(-m .^ 2 / 2) / sqrt(2 * pi);
lh = @(rho, a, d) lhq(lhf(rho, a, d), inside([kink(rho, max(a, 1e-12)), kink(rho, min(d, 0.6 - 1e-9))])) / (d - a);
r = add(r, 'lhEqThree', 100 * lh(0.3, 0, 0.03), 1e-7); r = add(r, 'lhMzBThree', 100 * lh(0.3, 0.07, 0.10), 1e-7);
r = add(r, 'lhSnBSix', 100 * lh(0.6, 0.15, 0.30), 1e-7);
% duration and convexity of the ten year par bond, from the Macaulay sums
y = 0.0399; t = (1:20) / 2; cf = 100 * y / 2 * ones(1, 20); cf(end) += 100; df = (1 + y / 2) .^ (-2 * t);
P = sum(cf .* df); mac = sum(t .* cf .* df) / P;
r = add(r, 'bdMac', mac, 1e-10); r = add(r, 'bdDnum', mac / (1 + y / 2), 1e-10);
r = add(r, 'bdC', sum(cf .* df .* t .* (t + 0.5)) / P / (1 + y / 2) ^ 2, 1e-10);
% exercises
r = add(r, 'xsSeven', 100 * ncdf((ninv(0.02) + sqrt(0.2) * ninv(0.99)) / sqrt(0.8)), 1e-10);
r = add(r, 'xsFour', 1e4 * 0.5 * 1e-4 * 3 * 3.25, 1e-12);

fid = fopen('code/ch06/out/octave_check.csv', 'w'); fprintf(fid, 'name,value,tol\n');
for i = 1:rows(r), fprintf(fid, '%s,%.15g,%g\n', r{i, 1}, r{i, 2}, r{i, 3}); end
fclose(fid); printf('%d Octave checks written\n', rows(r));
