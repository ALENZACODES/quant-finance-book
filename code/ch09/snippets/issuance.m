% Total bitcoin issuance in satoshis: 210,000 blocks a period, the subsidy
% halved by a right shift each period.
total = 0; last = -1;
for j = 0:63
  sub = floor(5e9 / 2^j);            % 50 coins of 1e8 satoshis, shifted j times
  total = total + 210000 * sub;
  if sub > 0, last = j; end
end
printf('%d satoshis = %.8f coins, %.4f short of 21 million\n', ...
       total, total / 1e8, 21e6 - total / 1e8);
printf('last positive subsidy in period %d, zero after %d halvings\n', last, last + 1);
printf('below 2^53, so a double holds it exactly: %d\n', total < 2^53);
