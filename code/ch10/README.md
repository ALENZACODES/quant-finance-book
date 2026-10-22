# Chapter 10 code

Nine scripts compute the numbers in the chapter: `proforma.py`, `leverage.py`, `cycle.py`,
`smoothing.py`, `reits.py`, `timber.py`, `storage.py`, `inflation.py` and `exercises.py`. `run_all.py`
runs them in order and writes `chapters/ch10-numbers.tex`; `make code CH=10` does the same and then
runs the cross-checks. The tests in `test_*.py` check the library against simulation, finite
differences and numerical optimisation.

## Cross-checks

`crosscheck.R` recomputes the pro forma and its IRR with `uniroot` and steps through the loan's monthly
payments. It also marks the loan vintages to the property index, fits the autocorrelations, lead and
lag regressions and inflation regressions at every horizon with `lm` and a Newey and West variance
written for the purpose, finds the forest rotations with `optimize`, and recomputes the convenience
yields, inventory norms and gas spreads.

`octave/ch10_check.m` recomputes the IRR and the rate at which the loan tests switch with `fzero`, the
loan balance from the payments still due, the rotations with `fminbnd`, the concession values in
closed form, the smoothing ratios from the filter's weights, the intrinsic value of storage by
enumeration and the exercise answers.

`test_crosschecks.py` compares both sets of results with the Python values.
