# Credits

Open source projects studied for this book, with the chapters where each one is credited. Every URL
was checked against the GitHub API on 19 October 2026 and resolves. Licences are as reported by
GitHub; projects are credited, not copied, unless a chapter states otherwise.

## Acknowledgements

AI tools (ChatGPT and Claude) helped check the code and layout, and helped find and cite sources.

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

## Quantitative finance, statistics and numerical tools (31)

| # | Repository | Topics | Chapters |
|---|---|---|---|
| 1 | [lballabio/QuantLib](https://github.com/lballabio/QuantLib) | Compounding and day counts, term structures, stochastic processes, option engines | 3, 4, 5, 6 |
| 2 | [numpy/numpy-financial](https://github.com/numpy/numpy-financial) | IRR, MIRR, NPV (used in Chapter 1; a comparison for the Chapter 10 IRRs and loan payments and the Chapter 11 fund and deal IRRs) | 1, 10, 11 |
| 3 | [statsmodels/statsmodels](https://github.com/statsmodels/statsmodels) | Jarque-Bera, Durbin-Watson, regression, Dickey-Fuller, Newey-West standard errors (used in Chapter 1, 3, 4, 5; Newey-West errors for the Chapter 10 inflation regressions) | 1, 3, 4, 5, 7, 8, 10 |
| 4 | [bashtage/arch](https://github.com/bashtage/arch) | GARCH and volatility models | 8 |
| 5 | [cantaro86/Financial-Models-Numerical-Methods](https://github.com/cantaro86/Financial-Models-Numerical-Methods) | Notebook treatments of distributions, PDEs, Monte Carlo, binomial trees | 3, 4 |
| 6 | [yhilpisch/py4fi2nd](https://github.com/yhilpisch/py4fi2nd) | Python for Finance code: statistics, simulation, valuation, binomial trees | 3, 4 |
| 7 | [stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading) | ML workflows for trading | 12 |
| 8 | [PyPortfolio/PyPortfolioOpt](https://github.com/PyPortfolio/PyPortfolioOpt) | Mean-variance, Ledoit-Wolf shrinkage, Black-Litterman, HRP (MIT; studied, not copied) | 7 |
| 9 | [wilsonfreitas/awesome-quant](https://github.com/wilsonfreitas/awesome-quant) | Index of quant libraries by topic | all |
| 10 | [ranaroussi/quantstats](https://github.com/ranaroussi/quantstats) | Performance analytics | 8, 9 |
| 11 | [scipy/scipy](https://github.com/scipy/scipy) | Root finding (including Brent's method), distributions, statistics, quadrature, banded solvers, HiGHS linear programming (used throughout) | all |
| 12 | [matplotlib/matplotlib](https://github.com/matplotlib/matplotlib) | All figures | all |
| 13 | [goldmansachs/gs-quant](https://github.com/goldmansachs/gs-quant) | Industrial quant toolkit design | 6 |
| 14 | [Anexen/pyxirr](https://github.com/Anexen/pyxirr) | Independent IRR and XIRR implementation (used in Chapter 2; a comparison for the Chapter 10 and Chapter 11 IRRs) | 2, 10, 11 |
| 15 | [gnu-octave/octave](https://github.com/gnu-octave/octave) | GNU Octave, which runs the `.m` cross-checks in `code/chNN/octave` folders, including GLPK for Chapter 4, quadrature, sparse solvers and root finders for Chapter 5, quadratic programming for Chapter 7, the GARCH fit and zone probabilities for Chapter 8, and the trend programme, survivor bias, hurdle, closure and issuance checks for Chapter 9, and the IRR, loan, rotation, concession and smoothing checks for Chapter 10, and the leverage, value driver, Miles and Ezzell, APV and lookback quadrature checks for Chapter 11, and the ridge forecast by the normal equations, the lasso by proximal gradient, the expected maximum of normals by quadgk, the option price by quadrature and the no-trade band for Chapter 12 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 |
| 16 | [crflynn/stochastic](https://github.com/crflynn/stochastic) | Generators for Brownian motion, GBM, Ornstein-Uhlenbeck and other processes (MIT) | 3 |
| 17 | [mattja/sdeint](https://github.com/mattja/sdeint) | Numerical integration of Ito and Stratonovich SDEs (GPL-3.0; studied, not copied) | 3 |
| 18 | [domokane/FinancePy](https://github.com/domokane/FinancePy) | Curve bootstrapping, forwards and futures, CRR and Leisen-Reimer trees, Black-Scholes, Black-76 and Heston pricers (GPL-3.0; studied, not copied) | 4, 6 |
| 19 | [vollib/py_vollib](https://github.com/vollib/py_vollib) | Black, Black-Scholes and Black-Scholes-Merton prices, Greeks and implied volatility by Jaeckel's "Let's Be Rational" (MIT; studied, not copied) | 5 |
| 20 | [numpy/numpy](https://github.com/numpy/numpy) | Arrays and the seeded random number generator used by every simulation | all |
| 21 | [wch/r-source](https://github.com/wch/r-source) | Read-only mirror of the R sources; R runs the `crosscheck.R` scripts in each `code/chNN` folder (GPL-2.0 or later) | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 |
| 22 | [pandas-dev/pandas](https://github.com/pandas-dev/pandas) | Reading and aligning the data files for the case studies | 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 |
| 23 | [OpenSourceRisk/Engine](https://github.com/OpenSourceRisk/Engine) | Open Source Risk Engine: how a production library organises rates and credit instruments, built on QuantLib (studied, not copied) | 6 |
| 24 | [nodejs/node](https://github.com/nodejs/node) | Node.js, which runs the private credit screener's `calc.js` in the Chapter 6 | 6 |
| 25 | [dcajasn/Riskfolio-Lib](https://github.com/dcajasn/Riskfolio-Lib) | Risk budgeting, equal risk contribution and other risk measures (BSD-3-Clause; studied, not copied) | 7 |
| 26 | [bashtage/linearmodels](https://github.com/bashtage/linearmodels) | Linear factor model estimation with the joint test of alphas (NCSA; studied, not copied) | 7 |
| 27 | [pydata/pandas-datareader](https://github.com/pydata/pandas-datareader) | Downloader for the Kenneth French data library files (studied, not used) | 7 |
| 28 | [bitcoin/bitcoin](https://github.com/bitcoin/bitcoin) | Bitcoin Core: the block subsidy, halving interval and supply limit summed in Chapter 9 (MIT; read, not used) | 9 |
| 29 | [python-excel/xlrd](https://github.com/python-excel/xlrd) | Reads the `.xls` data sets of Aswath Damodaran in `scripts/pull_ch11.py` (BSD) | 11 |
| 30 | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | Regression trees, ridge and the lasso, used only in the Chapter 12 as a comparison for the implementations in `code/ch12/ch12lib.py` (BSD-3-Clause) | 12 |
| 31 | [openpyxl](https://foss.heptapod.net/openpyxl/openpyxl) (Heptapod) | Reads the `.xlsx` file of Amit Goyal in `scripts/pull_ch12.py` (MIT) | 12 |

## John Khan's repositories (data and case material, read only)

| Repository | Use |
|---|---|
| [ALENZACODES/canada-policy-rate-bond-yield-mortgage-rate](https://github.com/ALENZACODES/canada-policy-rate-bond-yield-mortgage-rate) | Chapter 1 and Chapter 3 case study data (Bank of Canada series snapshot, MIT licence) |
| [ALENZACODES/private-credit-screener](https://github.com/ALENZACODES/private-credit-screener) | Chapter 6 private credit screen: the example loans and `calc.js`, copied unchanged into `code/ch06/vendor` (MIT licence) |

## Data

Bank of Canada Valet API, series `BD.CDN.5YR.DQ.YLD` and `V39079`: https://www.bankofcanada.ca/valet/docs

Chapter 4: Bank of Canada Valet API, Treasury bill series `TB.CDN.90D.MID`, `TB.CDN.180D.MID` and `TB.CDN.1Y.MID` and benchmark bond series `BD.CDN.2YR.DQ.YLD`, `BD.CDN.3YR.DQ.YLD`, `BD.CDN.5YR.DQ.YLD`, `BD.CDN.7YR.DQ.YLD`, `BD.CDN.10YR.DQ.YLD` and `BD.CDN.LONG.DQ.YLD`, retrieved 19 October 2026 and saved in `data/goc_bills_bonds.csv`.

Chapter 5: Bank of Canada USD/CAD noon rate (`IEXE0101`, to 2016) and daily average rate (`FXUSDCAD`, from 2017) and one month Treasury bill yield (`TB.CDN.30D.MID`), via the Valet API, and the US one month Treasury yield from FRED (`DGS1MO`, https://fred.stlouisfed.org/series/DGS1MO), retrieved 19 October 2026, cut at 30 September 2026 and saved in `data/raw/`; `scripts/build_usdcad.py` builds `data/usdcad_rates.csv` from them.

Chapter 6: the Government of Canada five year benchmark yield `BD.CDN.5YR.DQ.YLD` and the other benchmark series in the Chapter 4 file `data/goc_bills_bonds.csv` (Bank of Canada Valet API, retrieved 19 October 2026). The CDS quotes, the firm in the Merton section and the loans are illustrations.

Chapter 7: Kenneth R. French Data Library, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html, files `F-F_Research_Data_5_Factors_2x3`, `F-F_Research_Data_Factors`, `F-F_Momentum_Factor` and `10_Industry_Portfolios` (release built from the CRSP database of August 2026), retrieved 19 October 2026. The files are not redistributed here: `python scripts/build_french.py --download` fetches them into `data/raw/french/` and builds `data/ff_factors_monthly.csv` and `data/industry10_monthly.csv`, July 1963 to August 2026. The put-writing fund, the simulated managers and the attribution portfolio are illustrations.

Chapter 8: Kenneth R. French Data Library, files `F-F_Research_Data_Factors_daily` (daily market excess return and one month bill rate, 1 July 1963 to 31 August 2026) and the monthly files of Chapter 7, release built from the CRSP database of August 2026, retrieved 19 October 2026. The files are not redistributed: `python scripts/build_french.py --download` fetches them and builds `data/ff_daily.csv`. The USD/CAD series is the Chapter 5 Bank of Canada file `data/usdcad_rates.csv`, used through 29 September 2026. The two loan example, the return laws of Figure 8.6, the simulated paths and the exercise inputs are illustrations.

Chapter 9: Kenneth R. French Data Library, the monthly file `F-F_Research_Data_Factors` of Chapter 7 (market excess return and one month bill rate, to 31 August 2026), not redistributed. FRED, Federal Reserve Bank of St. Louis, https://fred.stlouisfed.org/, series `DGS10`, `DCOILWTICO`, `DEXUSEU`, `DEXJPUS` and `DEXCAUS`, daily, 4 January 1999 to 30 September 2026, retrieved 19 October 2026; `python scripts/pull_fred.py` builds `data/fred_markets_daily.csv`. The Canadian 7 and 10 year benchmark yields and the 90 day bill rate are from the Chapter 4 Bank of Canada file `data/goc_bills_bonds.csv`, and the carry section uses the Chapter 5 file `data/usdcad_rates.csv` (USD/CAD and the Canadian and US one month bill yields) through September 2026. The yearly equity volatility uses the Chapter 8 daily French file, not redistributed. Coinbase bitcoin and ether prices in US dollars, FRED series `CBBTCUSD` and `CBETHUSD`, daily, 1 December 2014 to 30 September 2026, retrieved 19 October 2026; these are not redistributed in the public copy, and the same script rebuilds `data/crypto_daily.csv`. The strategy families table, the simulated fund databases and closures, the clones, the fee examples, the fund of funds, the oil curve and the exercise inputs are illustrations.

Chapter 10: FRED, Federal Reserve Bank of St. Louis, https://fred.stlouisfed.org/, public domain series of US federal agencies: `BOGZ1FL075035503Q` (Financial Accounts of the United States, Z.1, commercial real estate price index; from 1996 the Federal Reserve chains the CoStar U.S. Composite Index excluding multifamily, value weighted, and before 1996 it averages appraisal-based price series), `BOGZ1FL123164105Q` and `BOGZ1FR123164105Q` (Z.1 equity REIT market value and revaluation) and `USSTHPI` (FHFA all-transactions house price index), quarterly to 2026Q2; `CPIAUCSL`, `TB3MS`, `GS10`, `WPU01`, `WPU0851`, `WPU0811`, `MCOILWTICO` and `MHHNGSP`, monthly to September 2026; all retrieved 19 October 2026. US Energy Information Administration, https://www.eia.gov/dnav/, NYMEX crude oil and natural gas futures, contracts 1 to 4, monthly averages, to April 2024 (the EIA stopped publishing them then), US crude oil stocks excluding the SPR and working gas in underground storage, monthly, to July 2026, retrieved 19 October 2026. `python scripts/pull_ch10.py` builds `data/real_assets_quarterly.csv`, `data/real_assets_monthly.csv` and `data/eia_curves_monthly.csv`, which are shipped. The REIT and market correlations use the Chapter 7 monthly French file, not redistributed. The building, the loans, the forest, the concession and the exercise inputs are illustrations.

Chapter 11: California Public Employees' Retirement System, Private Equity Program Fund Performance Review, as of 31 March 2026, https://www.calpers.ca.gov/investments/about-investment-office/investment-organization/pep-fund-performance-print (all active partnerships: vintage, capital committed, cash in, cash out, cash out plus remaining value, net IRR and multiple), retrieved 19 October 2026. Aswath Damodaran, data sets at https://pages.stern.nyu.edu/~adamodar/, January 2026 update: US industry tables `vebitda`, `wacc` and `fundgrEB`, and the implied equity risk premium `histimpl` for 1960 to 2025, retrieved 19 October 2026. Both are free to read but not in the public domain, so neither the raw downloads nor the built files are redistributed; `python scripts/pull_ch11.py` downloads them and builds `data/calpers_pe_funds.csv`, `data/damodaran_industry.csv` and `data/damodaran_erp.csv`. The ten year Treasury yield `GS10` for September 2026 is from the shipped Chapter 10 file `data/real_assets_monthly.csv`. The fund, the buyout, the manufacturer and the exercise inputs are illustrations.

Chapter 12: Amit Goyal, data for Welch and Goyal (2008), "A Comprehensive Look at the Empirical Performance of Equity Premium Prediction", update to December 2025, monthly sheet (January 1926 to December 2025), linked as "Updated data (up to 2025)" from https://sites.google.com/view/agoyal145, retrieved 19 October 2026. The file is free to download but carries no licence for redistribution, so neither the download nor the built file nor the forecasts derived from it are redistributed; `python scripts/pull_ch12.py` downloads it and builds `data/goyal_welch_monthly.csv`. All other inputs to the chapter are simulated.

## Other public sources for Chapter 2

Institutional Limited Partners Association, ILPA Principles 3.0 (2019):
https://ilpa.org/industry-guidance/principles-best-practices/ilpa-principles/

## Other public sources for Chapter 9

Dao, T.-L., T.-T. Nguyen, C. Deremble, Y. Lempérière, J.-P. Bouchaud and M. Potters (2017), "Tail
Protection for Long Investors: Trend Convexity at Work", Journal of Investment Strategies 7(1), 61 to 84,
https://doi.org/10.21314/JOIS.2017.093, preprint https://arxiv.org/abs/1607.02410 (2016). Its equation
(7) is the straddle identity of Proposition 9.3.
