function [inc, close] = hwm_fees(gross, mgmt, rate, low_rate, mult)
  % Incentive fees under a high-water mark. If low_rate and mult are given, a modified HWM applies:
  % after a loss, gains above the latest post-fee NAV earn low_rate until NAV (before incentive fee)
  % reaches trough + mult x (old HWM - trough); the full rate applies above that level.
  modified = nargin >= 5;
  n = numel(gross); inc = zeros(1, n); close = zeros(1, n);
  nav = 100; mark = 100; inrec = false; trough = 0; target = 0; old = 0;
  for t = 1:n
    pre = nav * (1 + gross(t)) - mgmt * nav;
    if inrec
      fee = low_rate * max(0, min(pre, target) - mark) + rate * max(0, pre - max(target, mark));
    else
      fee = rate * max(0, pre - mark);
    end
    c = pre - fee;
    if modified && ~inrec && c < mark
      inrec = true; old = mark; trough = c; target = c + mult * (old - c); mark = c;
    elseif modified && inrec
      if c < trough
        trough = c; target = c + mult * (old - c); mark = c;
      elseif pre >= target
        inrec = false; mark = c;
      else
        mark = max(mark, c);
      end
    else
      mark = max(mark, c);
    end
    inc(t) = fee; close(t) = c; nav = c;
  end
end
