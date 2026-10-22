% Equal risk contributions with fsolve on y .* (S y) = 1, then w = y / sum(y).
1;  % a script file, so the function below can be defined and then called
function w = erc(S)
  y0 = 1 ./ sqrt(diag(S));
  o = optimset('TolFun', 1e-14, 'TolX', 1e-14);
  y = fsolve(@(y) y .* (S * y) - 1, y0, o);
  w = y / sum(y);
end
S = [0.04, 0.002; 0.002, 0.0025];                  % vols 20% and 5%, correlation 0.2
w = erc(S);
printf('two assets: w1 = %.4f against s2 / (s1 + s2) = %.4f\n', w(1), 0.05 / 0.25);
F = dlmread('data/ff_factors_monthly.csv', ',', 1, 1)(:, 1:6) / 100;
w = erc(cov(F)); rc = w .* (cov(F) * w);
printf('factors: weights %s, largest over smallest contribution %.6f\n', ...
       mat2str(round(1000 * w') / 10), max(rc) / min(rc));
