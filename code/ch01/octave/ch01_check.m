% Octave cross-check for Chapter 1. Run from the repository root:
%   octave-cli --no-gui code/ch01/octave/ch01_check.m
% Writes code/ch01/out/octave_check.csv (name,value), compared with Python in test_crosschecks.py.
% Methods differ on purpose: IRRs from polynomial roots, lognormal moments by quadrature,
% beta by least squares, normal tails from erfc, the JB critical value in closed form.
addpath(fileparts(mfilename('fullpath')));
r = {};
add = @(r, name, v) [r; {name, v}];
% compounding
r = add(r, 'earD', 100 * ((1 + 0.08 / 12) ^ 12 - 1));
r = add(r, 'earF', 100 * (exp(0.08) - 1));
% Fund A: final IRR and interim IRRs
[c, d, nav] = fund_a(); net = d - c;
x = irr_roots(net); r = add(r, 'fundIRR', 100 * x(end)); r = add(r, 'fundIRRcount', numel(x));
nm = {'Three', 'Five', 'Seven'}; ss = [3 5 7];
for i = 1:3
  f = net(1:ss(i) + 1); f(end) += nav(ss(i) + 1);
  x = irr_roots(f); r = add(r, ['interim' nm{i}], 100 * x(end));
end
x = irr_roots([-100 230 -132]); r = add(r, 'multiLo', 100 * x(1)); r = add(r, 'multiHi', 100 * x(2));
r = add(r, 'noneCount', numel(irr_roots([100 -300 250])));
r = add(r, 'irrLong', 100 * irr_roots([-100 0 0 200])); r = add(r, 'irrShort', 100 * irr_roots([-100 130]));
% crossover: the rate where the two NPVs are equal is an IRR of the difference of the flows
x = irr_roots([-100 0 0 200] - [-100 130 0 0]); r = add(r, 'crossRate', 100 * max(x));
r = add(r, 'irrX', 100 * irr_roots([-100 150])); r = add(r, 'irrY', 100 * irr_roots([-100 0 121]));
r = add(r, 'irrXY', 100 * max(irr_roots([-200 150 121])));
r = add(r, 'mirrMulti', 100 * mirr1([-100 230 -132], 0.08, 0.12));
r = add(r, 'mirrFundEight', 100 * mirr1(net, 0.08, 0.08));
r = add(r, 'mirrFundTwelve', 100 * mirr1(net, 0.08, 0.12));
r = add(r, 'twrPer', 100 * (sqrt(1.2 * 0.9) - 1));
r = add(r, 'dwrPer', 100 * max(irr_roots([-100 -200 288])));
% lognormal moments by quadrature
h = [1 5 20]; nm = {'One', 'Five', 'Twenty'};
for i = 1:3
  [m, sd, sk, ek] = ln_moments_quad(0.06 * h(i), 0.18 * sqrt(h(i)));
  r = add(r, ['lnMean' nm{i}], m); r = add(r, ['lnSkew' nm{i}], sk); r = add(r, ['lnExk' nm{i}], ek);
end
% beta by least squares on the worked example
mkt = [2.0 -1.5 3.0 0.5 -2.5 1.5]'; fnd = [1.2 -0.4 2.1 0.8 -1.0 1.1]';
b = [ones(6, 1) mkt] \ fnd; r = add(r, 'exBeta', b(2));
% Chebyshev and normal tails
nm = {'One', 'Two', 'Three'};
for k = 1:3
  r = add(r, ['cheb' nm{k}], 100 * min(1, 1 / k ^ 2));
  r = add(r, ['normOut' nm{k}], 100 * erfc(k / sqrt(2)));
end
r = add(r, 'jbCrit', -2 * log(0.05));
% JB on the daily Government of Canada five year yield changes
fid = fopen('data/daily_policy_bond.csv'); fgetl(fid);
C = textscan(fid, '%s %f %f', 'Delimiter', ',', 'EmptyValue', NaN); fclose(fid);
keep = ~isnan(C{3}) & cellfun(@(q) datenum(q, 'yyyy-mm-dd') <= datenum('2026-09-30', 'yyyy-mm-dd'), C{1});   % the book's data cut
y = C{3}(keep); dy = diff(y) * 100;
z = dy - mean(dy); m2 = mean(z .^ 2); S = mean(z .^ 3) / m2 ^ 1.5; K = mean(z .^ 4) / m2 ^ 2 - 3;
r = add(r, 'gocDskew', S); r = add(r, 'gocDexk', K); r = add(r, 'gocJB', numel(dy) / 6 * (S ^ 2 + K ^ 2 / 4));
here = fileparts(mfilename('fullpath'));
f = fopen(fullfile(here, '..', 'out', 'octave_check.csv'), 'w');
fprintf(f, 'name,value\n');
for i = 1:rows(r)
  fprintf(f, '%s,%.12f\n', r{i, 1}, r{i, 2});
end
fclose(f);
printf('%d Octave checks written\n', rows(r));
