1;
% Faustmann and single harvest rotations for Q(T) = 400 (1 - exp(-0.06 T))^3,
% a price of 40 dollars a cubic metre and planting cost of 1,500 dollars a hectare.
function v = lev(T, p, r, c, gp)
  % land value of an endless chain of rotations of length T, price growing at gp:
  % the harvests are discounted at r - gp, the planting costs at r
  Q = 400 * (1 - exp(-0.06 * T)) ^ 3;
  v = p * Q * exp(-(r - gp) * T) / (1 - exp(-(r - gp) * T)) - c / (1 - exp(-r * T));
end
p = 40; c = 1500;
for r = [0.03 0.04 0.06]
  [tf, nf] = fminbnd(@(T) -lev(T, p, r, c, 0), 5, 150);
  ts = fminbnd(@(T) -p * 400 * (1 - exp(-0.06 * T)) ^ 3 * exp(-r * T), 5, 150);
  printf("r %.0f%%: Faustmann %.1f, single harvest %.1f years, land %.0f\n", ...
         100 * r, tf, ts, -nf);
end
tg = fminbnd(@(T) -lev(T, p, 0.04, c, 0.01), 5, 150);
printf("price rising 1%% a year at r 4%%: %.1f years\n", tg);
