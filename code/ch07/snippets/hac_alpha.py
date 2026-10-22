import pandas as pd
import statsmodels.api as sm

f = pd.read_csv("data/ff_factors_monthly.csv", index_col=0) / 100
ind = pd.read_csv("data/industry10_monthly.csv", index_col=0) / 100
y = ind["Hlth"] - f["rf"]                               # health care, excess return
models = {"CAPM": ["mkt_rf"],
          "six factors": ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]}
for label, cols in models.items():
    X = sm.add_constant(f[cols])
    ols = sm.OLS(y, X).fit()                            # classical standard errors
    nw = {"maxlags": 6, "use_correction": True}  # Bartlett weights, factor T / (T - k)
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds=nw)  # Newey-West
    a = ols.params["const"]
    print(f"{label}: alpha {1200 * a:.2f}% a year, "
          f"t {ols.tvalues['const']:.2f} classical, "
          f"{hac.tvalues['const']:.2f} Newey-West")
