function v = carry_value(V0, K, H, T, r, s, carry, k)
  % Value of carried interest with a hurdle amount H and catch-up rate k > carry:
  % k calls struck at H minus (k - carry) calls struck at V* = (k H - carry K) / (k - carry).
  % k = 0 means a hard hurdle: carry calls struck at H.
  if k == 0
    v = carry * bs_call(V0, H, T, r, s);
  else
    vstar = (k * H - carry * K) / (k - carry);
    v = k * bs_call(V0, H, T, r, s) - (k - carry) * bs_call(V0, vstar, T, r, s);
  end
end
