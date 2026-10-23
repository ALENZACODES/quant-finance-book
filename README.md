![Quantitative Finance for Alternative Investments](figures/cover.png)

# Quantitative Finance for Alternative Investments

*Mathematics, Models and Code*

This is a twelve chapter book on the quantitative methods behind alternative investments, written in
LaTeX with the code that produces it. Each chapter states a result, proves it, and checks it with a
number you can reproduce. No number in the text is typed by hand: the Python scripts in `code/` compute
them, write them to a generated macro file, and draw every figure. Independent cross-checks in R and
GNU Octave recompute the main results by different methods, and a test suite ties the three together.

The repository holds the source only. Run the build below to produce the PDFs.

## Chapters

| # | Chapter | What it covers | Status |
|---|---|---|---|
| 1 | Returns, Rates and the Shape of Risk | Compounding, simple and log returns, returns on collateralised positions, the IRR and the ways it fails, MIRR, time and money weighting, multiples and public market equivalents, the J-curve, lognormal returns, higher moments, correlation and beta, smoothing and unsmoothing, variance ratios and the size and power of normality tests. Case study: daily Government of Canada five year yields. | Released |
| 2 | Fund Structures, Fees and Distribution Waterfalls | Fund vehicles and master-feeder structures, open-end and drawdown funds, fund terms and governance, management fee bases, whole-fund and deal-by-deal waterfalls with hurdle, catch-up and clawback, carried interest as a portfolio of calls, a simulation of 6,000 funds, hedge fund incentive fees with high-water marks, fee annuities, funds of funds, real estate promotes and commitment pacing. | Released |
| 3 | Random Walks and Stochastic Calculus | Random walks and the reflection principle, a construction of Brownian motion, quadratic variation, the Ito integral and isometry, Ito's lemma, geometric Brownian motion and the Ornstein-Uhlenbeck process, Euler and Milstein schemes, martingales and optional stopping, Girsanov's theorem and importance sampling, Feynman-Kac and a Crank-Nicolson solver. Case study: how badly the speed of mean reversion is estimated from Canadian yields. | Released |
| 4 | Arbitrage, Forwards, Futures and Binomial Trees | Discount factors and bootstrapping, forward rates and FRAs, state prices and the one-period fundamental theorem, price bounds in incomplete markets, cost of carry and convenience yield, arbitrage bands with frictions, covered interest parity, futures against forwards under random rates, the Schwartz commodity model and roll yield, binomial replication, convergence to Black-Scholes, Leisen-Reimer trees, American options and the exercise boundary. Case study: whether Canadian forward rates forecast short rates. | Released |
| 5 | Black-Scholes: Derivation, Greeks and Numerical Checks | The Black-Scholes equation by delta hedging and by risk-neutral expectation, the closed form, put-call parity and bounds, the Greeks and the gamma-theta relation, dividend yields, cash dividends, Black-76 and currency options, Monte Carlo with antithetic and control variates, explicit, implicit and Crank-Nicolson schemes with von Neumann stability, implied volatility by Newton and Brent, jump and stochastic volatility smiles, risk-neutral densities, the profit and loss of hedging at the wrong volatility, discrete rebalancing and transaction costs. Case study: selling and delta hedging one month USD/CAD options, 2007 to 2026. | Released |
| 6 | Interest Rates, Credit Risk, Private Credit and Structured Products | Duration and convexity, short-rate models, swaps, structural and reduced-form credit models, private credit underwriting, tranching and copulas. Case study: whether five year commercial mortgages written each month could have been refinanced at maturity, using Government of Canada yields. | Released |
| 7 | Portfolio Theory, Factor Models, Alpha and Beta | Diversification, the mean-variance frontier and the two-fund and tangency results, long-only frontiers, estimation error and the bias of in-sample Sharpe ratios, the CAPM and the security market line, factor models and the APT, principal components, estimating alpha with Newey-West errors, the GRS test, detection times, screening many managers with false discovery control, smoothed returns and Dimson betas, option-like payoffs and timing regressions, Brinson attribution with Carino linking, Black-Litterman, risk budgets and equal risk contribution, rebalancing. Case study: seven portfolio rules on ten US industries, July 1968 to August 2026, estimated from the previous 60 months. | Released |
| 8 | Risk and Performance Measurement | Value at risk and expected shortfall, coherence and the scenario form of ES, heavy tails and Cornish-Fisher, the square root of time rule, GARCH(1,1) moments and forecasts, EWMA, generalised Pareto tails and the Hill estimator, Kupiec, Christoffersen, Basel zones and test power, the Acerbi-Szekely ES test, Sharpe, Treynor, Jensen, information ratio, M², Sortino, Omega and MPPM, the standard error of the Sharpe ratio and Lo's correction, the drawdown law of a random walk, smoothed returns and Geltner unsmoothing, GIPS composites and due diligence. Case study: seven daily VaR models backtested on the US market, June 1967 to August 2026, and on USD/CAD, April 2011 to September 2026. | Released |
| 9 | Hedge Funds, Managed Futures and Digital Assets | Strategy families and their exposures, the sign rule for trend signals, the martingale benchmark, the straddle identity of trend following, a seven market time series momentum programme with 2 and 20 fees and high-water marks, diversification across sleeves, carry and roll yield with the Canadian dollar and the Canada 10 year bond, survivorship and backfill bias, closure at a loss limit by the reflection principle, return attribution by regression, linear clones and out of sample error, hurdles, the incentive fee as an option and the high-water mark identity, the non-netting of fees in funds of funds, the bitcoin issuance schedule and Nakamoto's attacker calculation, a test for a change in correlation, the risk share of a small crypto allocation and selection bias among coins. Case study: a 60/40 stock and bond portfolio with a trend sleeve from January 2001 and a bitcoin sleeve from January 2015, both to August 2026. | Released |
| 10 | Real Assets and Commercial Real Estate | NOI, the capital reserve and the cap rate, discounted cash flow with an exit cap rate and its sensitivity, the mortgage constant and balance, the DSCR and LTV tests and the largest loan, leverage and the return and risk of equity, appraisal smoothing and the Z.1 property index before and after its switch to repeat sales, unsmoothing, REITs against private property with leverage removed and the lead of REIT prices, the Faustmann rotation, farmland and timber prices, infrastructure concessions of finite life, calendar spreads, the convenience yield and the Working curve for crude oil and natural gas, gas seasonality, inflation betas with the Fama-Schwert split and the least real variance mix, the intrinsic value of a storage facility and the spread fixed by the August gas curve, and inflation slopes over three, five and ten years. Case study: loan-to-value of loans written each quarter from 1996 marked to the Z.1 commercial property index. | Released |
| 11 | Private Equity and Business Valuation | The whole fund waterfall with hurdle, catch-up, carry and clawback over a fund's life, fees and carry from gross to net, the timing value of deal by deal carry, simulated funds and the J-curve, how well interim IRRs rank funds, the KS-PME and direct alpha and when they agree, the PME of a fund with beta above one, the CalPERS private equity portfolio by vintage with implied durations, leverage and the equity multiple of a buyout, a five year LBO model with the Shapley bridge, entry and exit multiples and simulated deals, free cash flow and the value driver formula, normalised earnings, the cost of capital with Hamada and Harris-Pringle betas and the Miles-Ezzell rate with its APV check, the DCF and its sensitivity, industry multiples against the value driver, and discounts for control and marketability with the protective put and the Longstaff bound. | Released |
| 12 | Machine Learning in Finance | Bias and variance, ridge as shrinkage of principal directions and its gain over least squares, the lasso with its optimality conditions and coordinate descent, cross-validation of overlapping labels with purging and an embargo, leakage from full-sample selection, regression trees, bagging, random forests and the correlation floor on their variance, gradient boosting as gradient descent in function space, backpropagation and a network learning option prices and deltas, the equity premium out of sample with fourteen predictors, combinations, ridge, the lasso, forests and boosting, the Clark-West test and the Campbell-Thompson value of a small R squared, quadratic hedging as a regression, a learned hedge against the Black-Scholes delta and the Whalley-Wilmott no-trade band with costs, the expected maximum of many trials, probabilistic and deflated Sharpe ratios, the probability of backtest overfitting, and Holm's procedure. | Released |

## Repository layout

```
main.tex                 book driver for the released chapters
preamble/quantbook.sty   house style (serif text, blue and red accents)
chapters/chNN.tex        chapter text; chNN-numbers.tex is generated by the code
figures/chNN/*.png       figures, generated by the code at 300 dpi
code/common/             shared plotting style and helpers
code/chNN/               chapter scripts, run_all.py, test_*.py, crosscheck.R, octave/*.m, snippets/
data/                    data snapshot used by the case studies, with its source
scripts/                 cover art and metadata stripping
references.bib           bibliography (biblatex, one consolidated reference list at the end of the book)
```

## Building

You need TeX Live with `latexmk` and `biber`, Python 3.10 or later, R and GNU Octave. `exiftool` is
optional and only used to check metadata.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt

make code CH=01      # numbers, figures and the R and Octave cross-checks for Chapter 1
make check CH=01     # the same, then the tests and the book PDF
make book            # the book from the current numbers and figures
```

The generated number files (`chapters/chNN-numbers.tex`) and the figures are committed, so `make book`
works straight after cloning. `make code` regenerates both from scratch. Output goes to `build/quant-finance-book.pdf`.

## Running the code

Each chapter folder can be used on its own.

```
.venv/bin/python code/ch01/run_all.py          # every script, numbers to code/ch01/out/numbers.json
.venv/bin/python code/ch01/distributions.py       # one script; random streams are fixed per script
.venv/bin/python -m pytest -q code/ch01        # tests, including agreement with R and Octave
Rscript code/ch01/crosscheck.R
octave-cli --no-gui code/ch01/octave/ch01_check.m
```

Every script uses a fixed seed, so the numbers in the text are reproduced exactly.

## Credits

The open source projects that shaped the writing, the figures and the code are listed with links in
[CREDITS.md](CREDITS.md) and credited by topic at the end of each chapter.

## Acknowledgements

AI tools (ChatGPT and Claude) helped check the code and layout, and helped find and cite sources.

## Licence

See [LICENSE](LICENSE).
