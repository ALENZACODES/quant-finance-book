function [p, tau] = bvp_exit(mu, s, a, b, n)
  % Exit probability and mean exit time of mu t + s W from (-b, a) at x = 0, by solving
  % s^2/2 u'' + mu u' = 0, u(-b) = 0, u(a) = 1 and s^2/2 v'' + mu v' = -1, v(-b) = v(a) = 0
  % with second order finite differences and Richardson extrapolation (n and 2n intervals).
  function [p0, t0] = solve(nn)
    x = linspace(-b, a, nn + 1)'; h = x(2) - x(1); m = nn - 1;
    lo = s^2 / 2 / h^2 - mu / (2 * h); di = -s^2 / h^2; up = s^2 / 2 / h^2 + mu / (2 * h);
    A = spdiags(repmat([lo di up], m, 1), [-1 0 1], m, m);
    r1 = zeros(m, 1); r1(end) = -up;
    u = A \ r1; v = A \ (-ones(m, 1));
    xi = x(2:end-1);
    p0 = interp1(xi, u, 0, 'spline'); t0 = interp1(xi, v, 0, 'spline');
  end
  [p1, t1] = solve(n); [p2, t2] = solve(2 * n);
  p = (4 * p2 - p1) / 3; tau = (4 * t2 - t1) / 3;
end
