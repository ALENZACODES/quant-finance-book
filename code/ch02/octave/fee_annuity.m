function pv = fee_annuity(aum, g, m, p, d, T)
  % Present value of T years of fees by direct summation (the closed form is checked against this).
  pv = 0; a = aum;
  for t = 1:T
    fm = m * a; fi = p * max(0, a * g - fm);
    pv = pv + (fm + fi) / (1 + d)^t;
    a = a * (1 + g) - fm - fi;
  end
end
