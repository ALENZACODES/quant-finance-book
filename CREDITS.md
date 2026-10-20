# Credits

Open source projects studied for this book, with the chapters where each one is credited. Every URL
was checked against the GitHub API on 19 October 2026 and resolves. Licences are as reported by
GitHub; projects are credited, not copied, unless a chapter states otherwise.

## Academic writing, LaTeX and figure design (18)

| # | Repository | What it contributed | Chapters |
|---|---|---|---|
| 1 | [dspinellis/latex-advice](https://github.com/dspinellis/latex-advice) | Source layout, symbolic references, booktabs tables, siunitx, build automation | all |
| 2 | [Wookai/paper-tips-and-tricks](https://github.com/Wookai/paper-tips-and-tricks) | Notation macros, one script per data-driven figure, vector figures | all |
| 3 | [jbhuang0604/awesome-tips](https://github.com/jbhuang0604/awesome-tips) | Paper-writing checklist: claims, figures that carry the argument, related work | all |
| 4 | [writing-resources/awesome-scientific-writing](https://github.com/writing-resources/awesome-scientific-writing) | Survey of tools beyond LaTeX for reproducible writing | all |
| 5 | [suchow/Dissertate](https://github.com/suchow/Dissertate) | Book-length front matter and chapter structure | all |
| 6 | [kks32/phd-thesis-template](https://github.com/kks32/phd-thesis-template) | Chapter-per-file organisation and build scripts | all |
| 7 | [Pseudomanifold/latex-mimosis](https://github.com/Pseudomanifold/latex-mimosis) | Minimal, typographically clean thesis styling | all |
| 8 | [Tufte-LaTeX/tufte-latex](https://github.com/Tufte-LaTeX/tufte-latex) | Restraint in colour and rules, figure and caption design | all |
| 9 | [MartinThoma/LaTeX-examples](https://github.com/MartinThoma/LaTeX-examples) | Worked LaTeX patterns for tables, listings and diagrams | all |
| 10 | [amperser/proselint](https://github.com/amperser/proselint) | Prose checks: jargon, redundancy, hedging | all |
| 11 | [btford/write-good](https://github.com/btford/write-good) | Checks for passive voice, weasel words, adverbs | all |
| 12 | [vale-cli/vale](https://github.com/vale-cli/vale) | Markup-aware style linting rules | all |
| 13 | [sylvainhalle/textidote](https://github.com/sylvainhalle/textidote) | Grammar and spelling checks on LaTeX source | all |
| 14 | [rougier/scientific-visualization-book](https://github.com/rougier/scientific-visualization-book) | Figure design principles for matplotlib | all |
| 15 | [garrettj403/SciencePlots](https://github.com/garrettj403/SciencePlots) | Publication matplotlib styles (house style adapted) | all |
| 16 | [plk/biblatex](https://github.com/plk/biblatex) | Bibliography system used for per-chapter references | all |
| 17 | [JabRef/jabref](https://github.com/JabRef/jabref) | Reference management conventions for `references.bib` | all |
| 18 | [cmichi/latex-template-collection](https://github.com/cmichi/latex-template-collection) | Template patterns for title pages and front matter | front matter |

## Quantitative finance, statistics and numerical tools (16)

| # | Repository | Topics | Chapters |
|---|---|---|---|
| 1 | [lballabio/QuantLib](https://github.com/lballabio/QuantLib) | Compounding and day counts, term structures, stochastic processes, option engines | 3, 4 |
| 2 | [numpy/numpy-financial](https://github.com/numpy/numpy-financial) | IRR, MIRR and NPV, used in the Chapter 1 tests as a check on the chapter's own root finder | 1 |
| 3 | [statsmodels/statsmodels](https://github.com/statsmodels/statsmodels) | Jarque-Bera and Durbin-Watson statistics in the Chapter 1 tests, the Dickey-Fuller test in Chapter 3 and regressions with Newey-West standard errors in the Chapter 4 case study | 1, 3, 4 |
| 4 | [cantaro86/Financial-Models-Numerical-Methods](https://github.com/cantaro86/Financial-Models-Numerical-Methods) | Notebook treatments of distributions, PDEs, Monte Carlo, binomial trees | 3, 4 |
| 5 | [yhilpisch/py4fi2nd](https://github.com/yhilpisch/py4fi2nd) | Python for Finance code: statistics, simulation, valuation, binomial trees | 3, 4 |
| 6 | [wilsonfreitas/awesome-quant](https://github.com/wilsonfreitas/awesome-quant) | Index of quant libraries by topic | all |
| 7 | [scipy/scipy](https://github.com/scipy/scipy) | Root finding (including Brent's method), distributions, statistics, quadrature, banded solvers, HiGHS linear programming (used throughout) | all |
| 8 | [matplotlib/matplotlib](https://github.com/matplotlib/matplotlib) | All figures | all |
| 9 | [Anexen/pyxirr](https://github.com/Anexen/pyxirr) | Independent IRR and XIRR implementation, used in the Chapter 2 tests | 2 |
| 10 | [gnu-octave/octave](https://github.com/gnu-octave/octave) | GNU Octave, which runs the `.m` cross-checks in `code/ch01/octave` to `code/ch04/octave`, including quadrature and a sparse Crank-Nicolson solver for Chapter 3 and the GLPK linear programming solver, quadrature and fzero for Chapter 4 | 1, 2, 3, 4 |
| 11 | [crflynn/stochastic](https://github.com/crflynn/stochastic) | Generators for Brownian motion, GBM, Ornstein-Uhlenbeck and other processes (MIT) | 3 |
| 12 | [mattja/sdeint](https://github.com/mattja/sdeint) | Numerical integration of Ito and Stratonovich SDEs (GPL-3.0; studied, not copied) | 3 |
| 13 | [domokane/FinancePy](https://github.com/domokane/FinancePy) | Curve bootstrapping, forwards and futures, CRR and Leisen-Reimer trees, Black-Scholes, Black-76 and Heston pricers (GPL-3.0; studied, not copied) | 4 |
| 14 | [numpy/numpy](https://github.com/numpy/numpy) | Arrays and the seeded random number generator used by every simulation | all |
| 15 | [wch/r-source](https://github.com/wch/r-source) | Read-only mirror of the R sources; R runs the `crosscheck.R` script in each of `code/ch01` to `code/ch04` (GPL-2.0 or later) | 1, 2, 3, 4 |
| 16 | [pandas-dev/pandas](https://github.com/pandas-dev/pandas) | Reading and aligning the data files for the case studies | 1, 3, 4 |

## John Khan's repositories (data and case material, read only)

| Repository | Use |
|---|---|
| [ALENZACODES/canada-policy-rate-bond-yield-mortgage-rate](https://github.com/ALENZACODES/canada-policy-rate-bond-yield-mortgage-rate) | Chapter 1 and Chapter 3 case study data (Bank of Canada series snapshot, MIT licence) |

## Data

Bank of Canada Valet API, series `BD.CDN.5YR.DQ.YLD` and `V39079`: https://www.bankofcanada.ca/valet/docs

Chapter 4: Bank of Canada Valet API, Treasury bill series `TB.CDN.90D.MID`, `TB.CDN.180D.MID` and `TB.CDN.1Y.MID` and benchmark bond series `BD.CDN.2YR.DQ.YLD`, `BD.CDN.3YR.DQ.YLD`, `BD.CDN.5YR.DQ.YLD`, `BD.CDN.7YR.DQ.YLD`, `BD.CDN.10YR.DQ.YLD` and `BD.CDN.LONG.DQ.YLD`, retrieved 19 October 2026 and saved in `data/goc_bills_bonds.csv`.

## Other public sources for Chapter 2

Institutional Limited Partners Association, ILPA Principles 3.0 (2019):
https://ilpa.org/industry-guidance/principles-best-practices/ilpa-principles/
