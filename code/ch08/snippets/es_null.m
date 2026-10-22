% Null standard deviation and 5% critical value of Z2 for T independent
% days of Student t losses, scaled to unit variance, at confidence a.
nu = 5; a = 0.975; T = 14897;
c  = sqrt((nu - 2) / nu);                          % unit variance scale
f  = @(y) gamma((nu + 1) / 2) / (sqrt(nu * pi) * gamma(nu / 2)) ...
          * (1 + y .^ 2 / nu) .^ (-(nu + 1) / 2);  % t density
x  = betaincinv(2 * (1 - a), nu / 2, 0.5);         % t quantile at a
q  = sqrt(nu * (1 - x) / x);
ES = c * f(q) * (nu + q ^ 2) / ((nu - 1) * (1 - a));
% X = L I / ES is zero on a fraction a of days and L / ES beyond the VaR
m1 = c * integral(@(y) y .* f(y), q, Inf) / ES;    % equals 1 - a
m2 = c ^ 2 * integral(@(y) y .^ 2 .* f(y), q, Inf) / ES ^ 2;
v  = m2 - m1 ^ 2;                                  % the zeros count here
sd = sqrt(v / T) / (1 - a);
z  = -sqrt(2) * erfcinv(2 * 0.95);                 % normal 95% quantile
printf("E X %.6f, Var X %.6f, sd %.4f, critical value %.4f\n", ...
       m1, v, sd, z * sd);
