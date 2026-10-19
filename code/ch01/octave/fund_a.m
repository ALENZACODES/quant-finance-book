function [c, d, nav] = fund_a()
  % Fund A of Chapter 1, rebuilt from its verbal description (Section 1.4).
  T = 9; inv = [20 22 20 15 10]; g = 0.15; fee = 2;
  c = zeros(1, T + 1); d = zeros(1, T + 1); nav = zeros(1, T + 1);
  for k = 1:5
    c(k) += inv(k); d(k + 5) += inv(k) * (1 + g) ^ 5;
  end
  for s = 0:T-1
    if s <= 4, c(s + 1) += fee; else, d(s + 1) -= fee; end
  end
  for s = 0:T
    for k = 1:5
      age = s - (k - 1);
      if age < 0 || age >= 5, continue; end
      if age == 0, nav(s + 1) += 0.9 * inv(k);
      elseif age <= 2, nav(s + 1) += inv(k);
      else, nav(s + 1) += inv(k) * (1 + g) ^ age; end
    end
  end
end
