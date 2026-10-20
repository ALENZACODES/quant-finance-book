// Private credit deal screening math. Interest-only or amortizing senior loans on real estate.
(function (root) {
  function annualDebtService(loan, rate, amortYrs) {
    if (!amortYrs) return (loan * rate) / 100;
    const i = rate / 1200,
      n = amortYrs * 12;
    return (12 * loan * i) / (1 - Math.pow(1 + i, -n));
  }
  function metrics(d) {
    // loan, value, noi, rate, amort (0 = interest-only), termYrs, capRate, exitCapShift, noiGrowth, fees (%), sponsorScore 1-5
    const ltv = (d.loan / d.value) * 100,
      ds = annualDebtService(d.loan, d.rate, d.amort),
      dscr = d.noi / ds,
      debtYield = (d.noi / d.loan) * 100;
    let bal = d.loan;
    if (d.amort) {
      const i = d.rate / 1200,
        n = d.amort * 12,
        k = d.termYrs * 12,
        p = ds / 12;
      bal = d.loan * Math.pow(1 + i, k) - (p * (Math.pow(1 + i, k) - 1)) / i;
    }
    const exitNoi = d.noi * Math.pow(1 + d.noiGrowth / 100, d.termYrs),
      exitValue = exitNoi / ((d.capRate + d.exitCapShift) / 100),
      exitLtv = (bal / exitValue) * 100;
    const breakevenValueDrop = (1 - bal / exitValue) * 100,
      allIn = d.rate + (d.fees || 0) / d.termYrs;
    return { ltv, ds, dscr, debtYield, balloon: bal, exitValue, exitLtv, breakevenValueDrop, allInYield: allIn };
  }
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  function score(d) {
    const m = metrics(d); // 0 (worst) to 100 (best); weights sum to 1
    const s = {
      ltv: clamp((85 - m.ltv) / 35, 0, 1),
      dscr: clamp((m.dscr - 1) / 0.6, 0, 1),
      dy: clamp((m.debtYield - 6) / 8, 0, 1),
      exit: clamp((90 - m.exitLtv) / 40, 0, 1),
      sponsor: clamp((d.sponsorScore - 1) / 4, 0, 1),
    };
    const w = { ltv: 0.25, dscr: 0.25, dy: 0.15, exit: 0.25, sponsor: 0.1 };
    let t = 0;
    for (const k in w) t += w[k] * s[k];
    const total = Math.round(t * 100);
    return { total, parts: s, grade: total >= 75 ? "A" : total >= 60 ? "B" : total >= 45 ? "C" : "D", metrics: m };
  }
  const DRIVERS = [
    ["noi", "NOI", -15, 15, "pct"],
    ["value", "Appraised value", -15, 15, "pct"],
    ["rate", "Interest rate", -1.5, 1.5, "abs"],
    ["capRate", "Going-in cap rate", -0.75, 0.75, "abs"],
    ["exitCapShift", "Exit cap shift", -0.5, 1, "abs"],
    ["noiGrowth", "NOI growth", -3, 3, "abs"],
    ["sponsorScore", "Sponsor score", -1, 1, "abs"],
  ];
  function tornado(d, key = "total") {
    const base = score(d)[key];
    return DRIVERS.map(([k, label, lo, hi, t]) => {
      const f = (x) => {
        const e = { ...d };
        e[k] = t === "pct" ? d[k] * (1 + x / 100) : d[k] + x;
        if (k === "sponsorScore") e[k] = clamp(e[k], 1, 5);
        return score(e)[key];
      };
      return { label: label, low: f(lo), high: f(hi), swing: Math.abs(f(hi) - f(lo)) };
    })
      .sort((a, b) => b.swing - a.swing)
      .map((r) => ({ ...r, base }));
  }
  const api = { annualDebtService, metrics, score, tornado, DRIVERS };
  if (typeof module !== "undefined") module.exports = api;
  else root.Calc = api;
})(this);
