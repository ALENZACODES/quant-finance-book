function u = fk_cn(x, T, a, b2, r, g, bnd, nt, nr)
  % Crank-Nicolson (with nr implicit half steps first) for u_t + a(x) u_x + b2(x)/2 u_xx - r u = 0,
  % u(T,x) = g(x), Dirichlet values bnd(tau, x_edge). Built with sparse matrices, independently of
  % the banded Python solver. Returns u(0, x).
  x = x(:); n = numel(x); dx = x(2) - x(1);
  A = a(x); B = 0.5 * b2(x);
  lo = B / dx^2 - A / (2 * dx); di = -2 * B / dx^2 - r; up = B / dx^2 + A / (2 * dx);
  m = n - 2; I = speye(m);
  Lm = spdiags([[lo(3:n-1); 0], di(2:n-1), [0; up(2:n-2)]], [-1 0 1], m, m);
  u = g(x); tau = 0;
  steps = [repmat([T / nt / 2, 1], nr, 1); repmat([T / nt, 0.5], nt - nr / 2, 1)];
  for s = 1:rows(steps)
    dt = steps(s, 1); th = steps(s, 2); tn = tau + dt;
    uL = bnd(tn, x(1)); uR = bnd(tn, x(n));
    rhs = u(2:n-1) + (1 - th) * dt * (Lm * u(2:n-1));
    rhs(1) += (1 - th) * dt * lo(2) * u(1) + th * dt * lo(2) * uL;
    rhs(end) += (1 - th) * dt * up(n-1) * u(n) + th * dt * up(n-1) * uR;
    u = [uL; (I - th * dt * Lm) \ rhs; uR];
    tau = tn;
  end
end
