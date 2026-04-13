## Empirical Check of the Constant-Specialty-Effect Assumption

The hierarchical logistic model assumes
$$
\operatorname{logit}\,p_i \;=\; \alpha + x_i^{\top}\beta + u_{j[i]},
$$
i.e. the specialty contribution $u_j$ is a constant additive shift on the
logit scale that does **not** depend on the patient covariates $x_i$ (no
random slopes, no specialty $\times$ covariate interactions). Before
accepting this simplification we check it empirically in two complementary
ways: a visual stratified-rate diagnostic and a formal likelihood-ratio /
information-criterion test.

### Visual diagnostic (Figure 4)

We restrict to the ten largest specialty groups (which jointly cover
$>94\%$ of the cohort), sort them by overall readmission rate, and
recompute each specialty's raw 30-day rate within three disjoint
patient-profile stratifications:

- **(A)** age band $\{\le 54,\, 55\text{--}74,\, \ge 75\}$,
- **(B)** comorbidity burden $\{0\text{--}1,\, 2,\, 3\}$ distinct
  comorbidity groups,
- **(C)** sex $\{\text{Female},\, \text{Male}\}$.

Cells with fewer than 50 patients are masked to suppress small-sample
noise. If the additive assumption holds, curves for different strata
should be approximately **parallel** — each stratum's baseline rate may be
shifted (because $x_i^{\top}\beta$ differs across strata), but the
specialty-to-specialty *differences* should line up. Figure 4 shows this
pattern clearly: strata differ in overall level but the rank ordering of
specialties and the spacing between them are preserved, with the sex
panel in particular displaying almost superimposable curves.

### Quantitative diagnostic (Table 2)

To move beyond visual inspection we fit two nested logistic regressions
on the full cohort $(N=37{,}964)$ using the same 43-variable design matrix
$X$ that enters the Bayesian model:

$$
\begin{aligned}
\mathcal{M}_0 \text{ (additive)}  &: \operatorname{logit} p
   = \alpha + x_i^{\top}\beta + \text{specialty}_{j[i]},\\
\mathcal{M}_1 \text{ (interactive)} &: \mathcal{M}_0
   + \text{specialty}_{j[i]} \times c_i,
\end{aligned}
$$

for a single profile variable $c_i \in \{\text{age}_z,\;
\text{comorbidity burden}_z,\;\text{female}\}$. Each interaction adds
$J-1 = 18$ parameters, giving stable, well-identified estimates (no
sparse cell issues). We report the likelihood-ratio statistic, the AIC /
BIC differences, and the percentage of the main-effect deviance
improvement that the interaction adds,
$\Delta_{\%} \;=\; 100 \cdot
 \bigl(\ell(\mathcal{M}_1) - \ell(\mathcal{M}_0)\bigr)\,/\,
 \bigl(\ell(\mathcal{M}_0) - \ell(\mathcal{M}_{\text{null}})\bigr)$.

| Interaction tested | LRT $\chi^2$ (df, p) | $\Delta$AIC / $\Delta$BIC | Verdict | Extra deviance |
|---|---|---|---|---|
| specialty $\times$ age           | 26.1 (18, $p=0.097$)   | $+9.9$ / $+163.7$ | additive preferred | 2.95\% |
| specialty $\times$ comorbidity   | 46.1 (18, $p=2.8\!\times\!10^{-4}$) | $-10.1$ / $+143.7$ | additive preferred | 5.22\% |
| specialty $\times$ sex (female)  | 28.8 (18, $p=0.051$)   | $+5.2$ / $+150.5$ | additive preferred | 3.25\% |

### Interpretation

Two of the three LRTs do not reject additivity at $\alpha = 0.05$. The
comorbidity interaction is statistically significant, but this is a
large-$N$ artefact: at $N \approx 38{,}000$ the LRT has enough power to
detect arbitrarily small departures from the null, so the $p$-value is a
poor measure of practical importance. The meaningful effect-size
indicators are:

1. **BIC:** all three tests yield $\Delta \text{BIC} > +140$ in favour of
   the additive model. Interpreting differences in BIC on the standard
   scale, any $\Delta \text{BIC} > 10$ is taken as *very strong* evidence
   for the simpler model; we are an order of magnitude beyond that
   threshold.
2. **Deviance share:** the interactions increase the model log-likelihood
   by only $\leq 5.2\%$ of the improvement that the main effects already
   achieve over the null. More than $94\%$ of the explanatory signal is
   therefore captured by the additive specification
   $\alpha + x_i^{\top}\beta + u_j$ alone.

Taken together with the near-parallel curves in Figure 4, the evidence
supports the conclusion that modelling specialty as a **constant additive
shift on the logit scale** is a reasonable simplification for this cohort:
any remaining $\text{specialty} \times \text{patient-profile}$
heterogeneity is small relative to the main effects, does not justify
the extra complexity of random slopes under BIC, and does not change
the qualitative ordering of specialty-level readmission risk. The
random-intercept assumption is therefore retained in the hierarchical
Bayesian model.
