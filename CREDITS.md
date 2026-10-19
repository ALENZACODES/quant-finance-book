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

## Quantitative finance, statistics and numerical tools (8)

| # | Repository | Topics | Chapters |
|---|---|---|---|
| 1 | [numpy/numpy-financial](https://github.com/numpy/numpy-financial) | IRR, MIRR and NPV, checked against the chapter's own root finder | 1 |
| 2 | [statsmodels/statsmodels](https://github.com/statsmodels/statsmodels) | Jarque-Bera and Durbin-Watson statistics, used to check the chapter's own implementations | 1 |
| 3 | [wilsonfreitas/awesome-quant](https://github.com/wilsonfreitas/awesome-quant) | Index of quant libraries by topic | all |
| 4 | [scipy/scipy](https://github.com/scipy/scipy) | Brent's method for IRR roots, distributions and their moments | all |
| 5 | [matplotlib/matplotlib](https://github.com/matplotlib/matplotlib) | All figures | all |
| 6 | [gnu-octave/octave](https://github.com/gnu-octave/octave) | GNU Octave, which runs the `.m` cross-checks in `code/ch01/octave`, including the quadrature check of the lognormal moments | 1 |
| 7 | [numpy/numpy](https://github.com/numpy/numpy) | Arrays and the seeded random number generator used by every simulation | all |
| 8 | [wch/r-source](https://github.com/wch/r-source) | Read-only mirror of the R sources; R runs `code/ch01/crosscheck.R` (GPL-2.0 or later) | 1 |

## John Khan's repositories (data and case material, read only)

| Repository | Use |
|---|---|
| [ALENZACODES/canada-policy-rate-bond-yield-mortgage-rate](https://github.com/ALENZACODES/canada-policy-rate-bond-yield-mortgage-rate) | Chapter 1 case study data (Bank of Canada series snapshot, MIT licence) |

## Data

Bank of Canada Valet API, series `BD.CDN.5YR.DQ.YLD` and `V39079`: https://www.bankofcanada.ca/valet/docs

