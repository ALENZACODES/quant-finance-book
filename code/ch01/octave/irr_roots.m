function r = irr_roots(cf)
  % All real IRRs of cash flows cf(1..n+1) at dates 0..n, from the roots of the NPV polynomial.
  % With x = 1/(1+r), NPV = sum cf(k+1) x^k, so the roots in x > 0 give r = 1/x - 1.
  % This is a different method from the bracketing root finder used in Python and R.
  p = fliplr(cf(:)');                % polyval wants the highest power first
  x = roots(p);
  x = real(x(abs(imag(x)) < 1e-10 & real(x) > 0));
  r = sort(1 ./ x - 1)';
end
