% Octave cross-check for Chapter 4. Run from the repository root:
%   octave-cli --no-gui code/ch04/octave/ch04_check.m
% Writes code/ch04/out/octave_check.csv (name,value,tol), compared with Python in test_crosschecks.py.
% Methods differ from Python on purpose: backward induction instead of binomial sums, the GLPK linear
% programming solver for price bounds, quadrature against Gaussian densities, and fzero for bootstrapping.
addpath(fileparts(mfilename('fullpath')));
r = {};
add = @(r, name, v, tol) [r; {name, v, tol}];

% bootstrap of the worked example: log-linear discount factors, coupon dates every six months
D05 = 1 / 1.02; D1 = 1 / 1.042;
lin = @(t1, D1_, t2, D2_, t) exp(log(D1_) + (log(D2_) - log(D1_)) * (t - t1) / (t2 - t1));
f2 = @(D2) 0.045 / 2 * (D05 + D1 + lin(1, D1, 2, D2, 1.5) + D2) + D2 - 1;
D2 = fzero(f2, [0.5 1.2], optimset('TolX', 1e-16));
f3 = @(D3) 0.047 / 2 * (D05 + D1 + lin(1, D1, 2, D2, 1.5) + D2 + lin(2, D2, 3, D3, 2.5) + D3) + D3 - 1;
D3 = fzero(f3, [0.5 1.2], optimset('TolX', 1e-16));
r = add(r, 'hxDTwo', D2, 1e-10); r = add(r, 'hxDThree', D3, 1e-10);
r = add(r, 'hxFwdOneTwo', 100 * (D1 / D2 - 1), 1e-8);

% price bounds by linear programming over state prices (glpk)
A = [1.02 120; 1.02 100; 1.02 80]; p = [1; 100]; X = [20; 0; 0];
lb = zeros(3, 1); ctype = 'SS'; vtype = 'CCC';
[~, lo] = glpk(X, A', p, lb, [], ctype, vtype, 1);
[~, hi] = glpk(X, A', p, lb, [], ctype, vtype, -1);
r = add(r, 'boundLo', lo, 1e-9); r = add(r, 'boundHi', hi, 1e-9);
% state prices in the complete market
psi = [1.02 120 20; 1.02 100 0; 1.02 80 0]' \ [1; 100; 7];
r = add(r, 'psiMid', psi(2), 1e-10); r = add(r, 'putFair', psi' * [0; 10; 30], 1e-10);

% futures less forward under a Vasicek short rate, by quadrature
ff = @(k, sr, ss, rho, T) sr ^ 2 * quadgk(@(u) ((1 - exp(-k * (T - u))) / k) .^ 2, 0, T, 'RelTol', 1e-13) + ...
     rho * ss * sr * quadgk(@(u) (1 - exp(-k * (T - u))) / k, 0, T, 'RelTol', 1e-13);
r = add(r, 'ffFivePos', 1e4 * ff(0.3, 0.01, 0.2, 0.5, 5), 1e-8);
r = add(r, 'ffTenPos', 1e4 * ff(0.3, 0.01, 0.2, 0.5, 10), 1e-8);
r = add(r, 'ffFiveNeg', 1e4 * ff(0.3, 0.01, 0.2, -0.5, 5), 1e-8);

% Schwartz futures as the risk-neutral mean of S_tau, by quadrature against the Gaussian law of ln S_tau
k = 1.2; s = 0.35; a = log(70);
for S0 = [50 100]
  e = exp(-k); m = e * log(S0) + (1 - e) * a; v = s ^ 2 * (1 - e ^ 2) / (2 * k);
  F = quadgk(@(x) exp(x - (x - m) .^ 2 / (2 * v)) / sqrt(2 * pi * v), m - 14 * sqrt(v), m + 14 * sqrt(v), 'RelTol', 1e-13);
  if S0 == 50, r = add(r, 'schOneLow', F, 1e-9); else, r = add(r, 'schOneHigh', F, 1e-9); end
end

% binomial trees by backward induction
r = add(r, 'crrHundredOne', crr_tree(100, 110, 0.05, 0.2, 1, 101, 0, 0), 1e-10);
r = add(r, 'crrFiveHundred', crr_tree(100, 110, 0.05, 0.2, 1, 500, 0, 0), 1e-10);
r = add(r, 'amPutThousand', crr_tree(100, 100, 0.05, 0.2, 1, 1000, 1, 1), 1e-10);
r = add(r, 'amCallDiv', crr_tree(100, 100, 0.05, 0.2, 1, 1000, 0, 1, 0.04), 1e-10);
r = add(r, 'euCallDiv', crr_tree(100, 100, 0.05, 0.2, 1, 1000, 0, 0, 0.04), 1e-10);
% Black-Scholes by quadrature of the discounted payoff against the lognormal density
S = 100; K = 110; rr = 0.05; sg = 0.2; T = 1; m = log(S) + (rr - sg ^ 2 / 2) * T; v = sg ^ 2 * T;
c = exp(-rr * T) * quadgk(@(x) (exp(x) - K) .* exp(-(x - m) .^ 2 / (2 * v)) / sqrt(2 * pi * v), log(K), m + 14 * sqrt(v), 'RelTol', 1e-13);
r = add(r, 'bsCall', c, 1e-9);
% exercises
r = add(r, 'exEightAm', crr_tree_uqd(50, 52, 1.1, 0.9, 1.02, 2), 1e-12);

fid = fopen('code/ch04/out/octave_check.csv', 'w'); fprintf(fid, 'name,value,tol\n');
for i = 1:rows(r), fprintf(fid, '%s,%.15g,%g\n', r{i, 1}, r{i, 2}, r{i, 3}); end
fclose(fid); printf('%d Octave checks written\n', rows(r));
