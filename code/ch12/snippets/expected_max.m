% Expected maximum of N independent standard normals: exact by quadrature, the
% approximation of Bailey and Lopez de Prado, and the bound sqrt(2 log N).
g = 0.5772156649015329;                       % Euler's constant
Phi = @(x) 0.5 * erfc(-x / sqrt(2));
Phiinv = @(p) -sqrt(2) * erfcinv(2 * p);
approx = @(N) (1 - g) * Phiinv(1 - 1/N) + g * Phiinv(1 - 1/(N*e));
printf("    N   exact  approx   bound\n");
for N = [10 100 1000]
  f = @(x) x .* N .* exp(-x.^2/2) / sqrt(2*pi) .* Phi(x).^(N-1);
  printf("%5d %7.3f %7.3f %7.3f\n", N, quadgk(f, -12, 12), approx(N), sqrt(2*log(N)));
end
% Exercise 12.9: 100 strategies over five years, standard error about 1/sqrt(5)
printf("Exercise 12.9: %.3f standard errors, best annual Sharpe %.2f\n", ...
       approx(100), approx(100) / sqrt(5));
