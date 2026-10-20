% Octave cross-check for Chapter 2. Run from the repository root:
%   octave-cli --no-gui code/ch02/octave/ch02_check.m
% Writes code/ch02/out/octave_check.csv (name,value), compared with Python in test_ch02.py.
addpath(fileparts(mfilename('fullpath')));
H = 100 * 1.08^5;
r = {};
r(end+1, :) = {'carryFullThirty', carry_value(100, 100, H, 5, 0.03, 0.30, 0.2, 1.0)};
r(end+1, :) = {'carryHardThirty', carry_value(100, 100, H, 5, 0.03, 0.30, 0.2, 0)};
r(end+1, :) = {'carryFullFifteen', carry_value(100, 100, H, 5, 0.03, 0.15, 0.2, 1.0)};
r(end+1, :) = {'carryFullThirtyRich', carry_value(115, 100, H, 5, 0.03, 0.30, 0.2, 1.0)};
r(end+1, :) = {'spVstarFull', (H - 20) / 0.8};
[inc, cl] = hwm_fees([0.15 -0.10 0.08 0.20], 0.02, 0.20);
r(end+1, :) = {'hfIncOne', inc(1)}; r(end+1, :) = {'hfIncFour', inc(4)}; r(end+1, :) = {'hfCloseFour', cl(4)};
[inc, cl] = hwm_fees([-0.25 0.12 0.12 0.12 0.12 0.12], 0, 0.20, 0.10, 1.25);
r(end+1, :) = {'modModSum', sum(inc)}; r(end+1, :) = {'modModIncFive', inc(5)};
[inc, cl] = hwm_fees([-0.25 0.12 0.12 0.12 0.12 0.12], 0, 0.20);
r(end+1, :) = {'modTradSum', sum(inc)};
r(end+1, :) = {'annPV', fee_annuity(1000, 0.10, 0.02, 0.20, 0.10, 10)};
r(end+1, :) = {'annPVtwenty', fee_annuity(1000, 0.10, 0.02, 0.20, 0.10, 20)};
r(end+1, :) = {'exSixPV', fee_annuity(500, 0.08, 0.015, 0.15, 0.08, 5)};
r(end+1, :) = {'optDownSigForty', 0.2 * bs_call(80, 100, 1, 0.03, 0.40)};
here = fileparts(mfilename('fullpath'));
f = fopen(fullfile(here, '..', 'out', 'octave_check.csv'), 'w');
fprintf(f, 'name,value\n');
for i = 1:rows(r)
  fprintf(f, '%s,%.12f\n', r{i, 1}, r{i, 2});
end
fclose(f);
printf('%d Octave checks written\n', rows(r));
