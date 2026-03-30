# Speaker Script — Bayesian Hierarchical Logistic Regression Presentation

> Total time: ~30 minutes
> Reminder: No reading from notes during the actual presentation. Use this to rehearse.

---

## Slide 1: Title Slide (~30s)

Hi everyone. Our project is on Bayesian Hierarchical Logistic Regression for 30-Day Hospital Readmission. Specifically, we're looking at the role of medical specialty among diabetic patients. My name is Xingyu, and this is my partner Zhouhan.

---

## Slide 2: Section Divider — Motivation & Research Question

*(click through quickly, no narration needed)*

---

## Slide 3: Why Study Readmission? (~1 min)

So why do we care about hospital readmissions? In the US, unplanned readmissions cost over 26 billion dollars annually. This is a huge burden on the healthcare system.

To address this, CMS — the Centers for Medicare and Medicaid Services — launched the Hospital Readmissions Reduction Program, which financially penalizes hospitals with excess readmission rates. So 30-day readmission has become one of the most important quality metrics in healthcare.

Among all patient populations, diabetic patients are particularly high-risk. They tend to have multiple comorbidities, require frequent hospitalizations, and their care often spans multiple medical specialties. This cross-specialty nature of diabetic care is what motivated our central question: does the admitting specialty itself affect whether a patient gets readmitted?

---

## Slide 4: Research Question (~1 min)

Our research question is: how much specialty-level heterogeneity exists in 30-day readmission risk, and can we reliably estimate it despite severe group imbalance?

Let me explain why this is not a simple question. When we look at the raw data, readmission rates across specialties range from about 4 percent to over 32 percent — that's a huge spread. But here's the catch: the sample sizes are extremely imbalanced. Internal Medicine has almost 11,000 encounters, while Physical Medicine and Rehab has only 288. So if we just compare raw rates, we can't tell whether the differences are real or just noise from small samples.

We need a framework that can borrow strength across groups while properly quantifying uncertainty — and that's exactly what a Bayesian hierarchical model gives us.

---

## Slide 5: Why Not Standard Logistic Regression? (~1 min)

Before jumping into our model, let's consider the alternatives and why they fall short.

A pooled logistic regression ignores group structure entirely — it treats all specialties as identical. That clearly misses the point.

We could add fixed-effect dummy variables for each specialty. But for small groups like Physical Medicine with only 288 patients, the maximum likelihood estimates are going to be very unstable and noisy.

Even a frequentist mixed model, while it does model group structure, only gives us a point estimate for the variance component. It doesn't give us a full posterior distribution — so we can't properly quantify uncertainty about how much heterogeneity exists.

A Bayesian hierarchical model solves all three problems. It provides partial pooling — small groups borrow strength from larger ones — and it gives us full uncertainty quantification through the posterior distribution.

---

## Slide 6: Data Overview (~45s)

Our data comes from the UCI Diabetes 130-US Hospitals dataset, covering encounters from 1999 to 2008 across 130 hospitals. After cleaning, our analytic cohort has about 38,000 encounters, spread across 19 medical specialties, with 43 covariates. The overall 30-day readmission rate is around 9.6 percent.

On the left you can see the distribution of encounters across specialties — notice the severe imbalance, with Internal Medicine dominating. On the right, you can see the raw readmission rates by specialty with 95 percent Wilson confidence intervals. The wide intervals for small groups illustrate exactly why we need partial pooling.

---

## Slide 7: Data Processing Pipeline (~45s)

Let me quickly walk through our data processing pipeline. We started with about 102,000 raw encounters. We removed duplicate encounter IDs, excluded patients who were discharged to hospice or expired, and excluded encounters with unknown specialty — that was about 45 percent of the data, which is a limitation we'll discuss later.

We also dropped variables with very high missingness, like weight which was missing 97 percent of the time.

For specialty regrouping, we consolidated 73 raw specialties into 19 groups using a threshold of at least 250 encounters per group.

For feature engineering, we standardized 10 continuous variables to z-scores and created 33 binary and categorical features from demographics, ICD-9 diagnosis codes, and medication indicators. This gives us our final design matrix: N equals 37,964, J equals 19, K equals 43.

---

## Slide 8: Section Divider — Overall Analysis Framework

*(click through)*

---

## Slide 9: Analysis Pipeline Overview (~30s)

This slide gives you the big picture of our entire analysis pipeline — from data cleaning through feature engineering, EDA, model specification, posterior derivation, MCMC sampling, diagnostics, inference, model evaluation, and finally sensitivity analysis. I'll walk through each of these in the following sections.

---

## Slide 10: Section Divider — Model Structure

*(click through)*

---

## Slide 11: Model Specification — Likelihood (~1.5 min)

Now let's get into the model. Our outcome y_i is binary — whether a patient was readmitted within 30 days — so we model it as Bernoulli with probability pi_i.

The linear predictor eta_i has three components: alpha, the global intercept; x_i transpose beta, which captures the 43 patient-level fixed effects; and u_j, the random intercept for specialty j that patient i belongs to.

We link eta to pi through the logistic function.

An important assumption here is conditional independence: given the linear predictor, outcomes for different patients are independent. This is standard for generalized linear models, but it's worth stating explicitly because it's what allows us to write the likelihood as a simple product.

Alpha captures the baseline log-odds of readmission, beta captures how patient characteristics affect risk, and u_j captures the specialty-specific deviation from the population average.

---

## Slide 12: Prior Specification (~1.5 min)

For priors, we use weakly informative choices throughout.

Alpha gets a Normal zero, twenty-five prior. On the log-odds scale, this is quite diffuse — it allows the baseline readmission probability to range essentially from near zero to near one.

Each beta_k gets a Normal zero, 6.25 prior. This corresponds to allowing odds ratios up to about exp(5), which is around 150. So we're not constraining the effects much, but we do prevent completely unreasonable values.

The key to our model is the hierarchical prior on u_j. Conditional on tau, each random intercept follows a Normal zero, tau-squared distribution. These are exchangeable — we assume all specialties are drawn from the same population distribution. This is the mechanism that produces partial pooling: specialties with small sample sizes get shrunk toward the population mean.

Finally, tau — the between-group standard deviation — gets a Half-Normal prior with scale 1. This is our hyperparameter. It controls how much variation exists across specialties. If tau is near zero, all specialties are essentially the same. If tau is large, there's substantial heterogeneity.

---

## Slide 13: DAG / Model Hierarchy (~30s)

This directed acyclic graph summarizes our model structure. You can see the three levels clearly.

At the top, the hyperparameter level: tau controls the spread of specialty effects. In the middle, the group level: each u_j is drawn from the distribution governed by tau. At the bottom, the observation level: each patient's outcome depends on the fixed effects alpha and beta, plus the random intercept for their specialty.

The dashed plates indicate the indexing — j ranges over 19 specialties, i ranges over all 38,000 observations.

---

## Slide 14: Section Divider — Key Derivation

*(click through)*

---

## Slide 15: Joint Posterior Distribution (~1.5 min)

Moving to the derivation. By Bayes' theorem, the joint posterior of all parameters — alpha, beta, u, and tau — given the data is proportional to the product of the likelihood and all the priors.

Let me walk through each factor. The first term is the product of Bernoulli likelihoods across all N observations. Then we have the prior on alpha, the independent priors on each beta_k, the hierarchical prior on each u_j conditional on tau, and finally the hyperprior on tau.

Now, a critical point: this posterior is non-conjugate. The logistic likelihood does not have a conjugate prior in the Gaussian family. This means we cannot derive a closed-form expression for the posterior. There is no analytical solution. And this is precisely what motivates our use of MCMC sampling — we need to approximate this posterior numerically.

---

## Slide 16: Log-Posterior Decomposition (~1.5 min)

Taking the log of the joint posterior gives us a sum of five interpretable terms.

Term one is the log-likelihood — this is the only term that depends on the observed data. It's the standard logistic log-likelihood: y_i times eta_i minus log one plus exp eta_i.

Terms two and three are the Gaussian log-priors on alpha and beta — they act as quadratic regularization, pulling parameters toward zero.

Term four is the hierarchical prior on u given tau. Notice that tau appears here in two places: negative J log tau, and u_j squared over two tau squared. This is important.

Term five is the Half-Normal log-prior on tau.

Here's the key structural insight: tau does NOT appear anywhere in term one — the likelihood. Tau only appears in terms four and five. This means tau is informed by the data only indirectly, through the posterior draws of u_j. And practically, this means when we update tau in MCMC, we don't need to evaluate the likelihood at all — making the update very cheap, order J instead of order N.

---

## Slide 17: Reparameterization phi = log(tau) (~1 min)

One technical detail: tau must be positive, which makes random-walk proposals on tau inefficient near the boundary.

So we reparameterize: let phi equal log tau. This maps tau from the positive reals to the entire real line, where standard Gaussian proposals work much better.

The transformation introduces a Jacobian correction. Since d-tau-d-phi equals exp phi, the log-Jacobian is simply plus phi. You can see this term explicitly in the transformed log-target.

And again, notice that this target function does not involve the likelihood — it only depends on the current draws of u_j. So the computational cost of updating phi is O(J) — that's just 19 operations per iteration.

---

## Slide 18: Section Divider — Sampling Strategy

*(click through)*

---

## Slide 19: Metropolis-within-Gibbs Block Structure (~1.5 min)

Our sampler is a Metropolis-within-Gibbs algorithm with five blocks updated sequentially in each iteration.

Block one updates alpha with a symmetric random walk. We use an incremental update — delta eta equals alpha-star minus alpha — so we only need to add a scalar to each element. This is O(N).

Block two updates each beta_k one at a time, component-wise. For each coordinate, we compute delta eta as the difference times the k-th column of X. Each update is O(N), but because we do it incrementally rather than recomputing the full linear predictor, we avoid the naive O(N times K) cost.

Block three updates each u_j component-wise. The key efficiency here is that for group j, we only need to evaluate the likelihood contribution from patients in that group — n_j observations, not all N. For a small specialty with 288 patients, this is much cheaper than a full-data evaluation.

Block four updates log-tau. As we discussed, this does not involve the likelihood at all — it only depends on the current u_j draws. The cost is O(J), which is just 19 operations.

The overall computational cost per iteration scales linearly in N, not N times K.

---

## Slide 20: Joint alpha-u Translation Move (~1.5 min)

This is the most important algorithmic contribution of our sampler.

The idea is simple but powerful. We propose a shift delta from a Gaussian. Then we simultaneously move alpha up by delta and every u_j down by delta.

Now look at what happens to the linear predictor: eta-star equals alpha plus delta plus x transpose beta plus u_j minus delta. The deltas cancel. Eta-star equals eta. The linear predictor is completely unchanged.

This means the likelihood ratio in the Metropolis-Hastings acceptance probability is exactly one. The acceptance ratio depends only on the priors — specifically the prior on alpha and the hierarchical prior on u. And evaluating those is O(J), just 19 operations.

Why does this work so well? It exploits the non-identifiability between alpha and u_j. In the likelihood, alpha and u are confounded — you can shift one up and the other down without changing the fit. Standard component-wise updates explore this ridge very slowly. The translation move traverses it directly.

We repeat this move 10 times per iteration to maximize the mixing benefit, and because each repetition is so cheap — O(J) — the overhead is negligible.

---

## Slide 21: MCMC Settings & Convergence Diagnostics (~1 min)

We ran 4 independent chains, each for 30,000 iterations. We discarded the first 8,000 as burn-in and thinned by 5, giving us 4,400 saved samples per chain — about 17,600 total posterior draws.

Proposal standard deviations were adaptively tuned per coordinate to achieve acceptance rates between 20 and 50 percent.

On the left you can see representative trace plots and autocorrelation functions. The chains mix well and show good convergence.

On the right, the diagnostics summary: all split R-hat values are below 1.05, confirming convergence. Effective sample sizes are reasonable — tau has an ESS of about 2,400, and the random effects u_j average around 3,400. Alpha has the lowest ESS at about 306, which is expected given its correlation with the u_j's, but the translation move keeps it manageable.

---

## Slide 22: Section Divider — Inferential Findings

*(click through)*

---

## Slide 23: Fixed Effects — Key Predictors (~2 min)

Now to the results. This table shows the top 12 fixed effects sorted by the absolute value of the posterior mean. Remember, these are on the log-odds scale, so you can interpret them as multiplicative changes in odds through the odds ratio column.

The highlighted rows are variables whose 95 percent credible intervals exclude zero — meaning we have strong posterior evidence of a non-zero effect.

The strongest predictor is comorbid neoplasms — a cancer comorbidity — with a posterior mean of 0.415 on the log-odds scale, corresponding to an odds ratio of about 1.51. So patients with a cancer diagnosis in their record have about 50 percent higher odds of readmission.

The primary diagnosis of respiratory disease is associated with lower readmission risk, with an OR of about 0.70. A1C being measured has a positive association — but this likely reflects measurement bias: sicker patients are more likely to have their A1C checked.

Number of prior inpatient visits is the most precisely estimated predictor, with a very narrow credible interval. Each standard-deviation increase in prior visits corresponds to about a 22 percent increase in odds. This makes clinical sense — patients who've been hospitalized frequently before are at higher risk of coming back.

Age and number of diagnoses also show clear positive effects.

---

## Slide 24: Specialty Random Effects & Shrinkage (~1.5 min)

This forest plot shows the posterior means and 95 percent credible intervals for each specialty's random intercept u_j.

It's important to interpret these correctly: each u_j represents the deviation from the population-average log-odds. It's not a causal effect — it's the residual specialty-level variation after adjusting for all 43 patient-level covariates.

Physical Medicine and Rehabilitation has the highest random effect, at plus 1.32 on the logit scale — the only specialty whose 95 percent credible interval is entirely above zero. This means that, even after controlling for patient characteristics, patients admitted under this specialty have notably higher readmission risk. On the other end, Surgery-Cardiovascular/Thoracic shows a significantly negative effect at minus 0.58, and Obstetrics/Gynecology also has a negative random intercept.

Notice the shrinkage effect. Specialties with smaller sample sizes have wider credible intervals and their point estimates are pulled closer to zero — closer to the population mean. This is partial pooling in action. The model doesn't naively trust the noisy raw rates from small groups; instead, it borrows information from the larger groups through the hierarchical structure.

---

## Slide 25: Heterogeneity Scale tau — Core Finding (~1.5 min)

This is arguably the most important slide. Tau is the between-specialty standard deviation — it directly answers our research question about how much heterogeneity exists.

The posterior mean of tau is 0.437, with a 95 percent credible interval from 0.298 to 0.644. On the logit scale, a tau of 0.44 means that specialty effects vary by about 0.44 log-odds units on average — this is a meaningful amount of variation.

Critically, the posterior mass is clearly away from zero. The entire 95 percent credible interval is above 0.29. This provides strong evidence that between-specialty heterogeneity is real — it's not just noise.

So to directly answer our research question: yes, specialty-level heterogeneity exists and we can reliably estimate it. The hierarchical model successfully separates the signal from the noise even under severe group imbalance.

That said, we should put this in perspective. The specialty random effects, while significant, are smaller in magnitude than many of the patient-level fixed effects. Patient-level covariates — like prior inpatient visits, comorbidities, and medication use — remain the primary drivers of readmission risk. Specialty contributes additional variation on top of that.

---

## Slide 26: Posterior Predictive Check & Model Comparison (~1.5 min)

To evaluate model adequacy, we performed posterior predictive checks.

On the left, you can see the PPC results. We simulated replicated datasets from the posterior and compared two key statistics to the observed data: the overall readmission rate and the standard deviation of readmission rates across specialties. In both cases, the observed statistic falls well within the distribution of replicated statistics, confirming that our model captures the data-generating process adequately.

The model captures both first-order structure — the mean readmission rate — and second-order structure — the variability across specialties. This is important because a model could get the average right while completely missing the group-level variation.

On the right, the model comparison table. The hierarchical model achieves an AUC of 0.641 compared to 0.629 for the no-specialty baseline. The improvement is modest in terms of raw discrimination, but the DIC of 23,254 favors the hierarchical model. More importantly, the hierarchical model's value isn't just predictive — it's inferential. It quantifies the specialty effect and its uncertainty, which a simple logistic regression cannot do.

---

## Slide 27: Section Divider — Sensitivity & Robustness Analysis

*(click through)*

---

## Slide 28: Sensitivity Analysis — Prior Robustness (~2 min)

For sensitivity analysis, we re-ran the sampler under three alternative prior configurations.

In A1, we doubled the fixed-effect prior standard deviation from 2.5 to 5.0, making the beta priors even more diffuse.

In B1, we widened the tau hyperprior to Half-Normal with scale 2.5 instead of 1.0.

In B2, we switched the tau hyperprior to a Half-Cauchy distribution with scale 2.5, which has heavier tails and puts more prior mass on large values of tau.

On the left, you can see the tau posterior densities under all four configurations overlaid. They are nearly identical. The range of posterior means across configurations is less than 0.05 — essentially negligible.

The key fixed-effect estimates also remain stable, with beta variation less than 0.02 across configs. And the specialty rankings — which specialty has the highest or lowest random effect — are highly consistent, with a Spearman correlation above 0.95.

The conclusion is clear: our posterior is likelihood-dominated for all key parameters. The data, not the prior, is driving the results. Our findings are robust to the specific choice of prior.

---

## Slide 29: Section Divider — Conclusion

*(click through)*

---

## Slide 30: Summary (~1 min)

Let me summarize our main findings.

First, specialty heterogeneity is real and reliably estimated. The posterior of tau is clearly bounded away from zero, providing strong Bayesian evidence of between-specialty variation in readmission risk.

Second, patient-level factors — particularly prior inpatient visits, comorbid neoplasms, diabetes medication use, and number of diagnoses — remain the primary drivers of readmission.

Third, the hierarchical model achieves principled partial pooling. Small specialties benefit from shrinkage toward the population mean, giving us stable estimates even for groups with fewer than 300 observations.

Fourth, posterior predictive checks confirm that our model captures both the mean structure and the group-level variance. DIC favors the hierarchical model over the no-specialty baseline.

And fifth, all conclusions are robust across multiple prior specifications, confirming that the posterior is likelihood-dominated.

---

## Slide 31: Limitations & Future Work (~1 min)

We should acknowledge some limitations. The dataset is from 1999 to 2008, and clinical practices have evolved since then. The ICD-9 coding system has been replaced by ICD-10. So our specific coefficient estimates may not directly apply to current practice.

We also excluded 45 percent of encounters with unknown specialty, which could introduce selection bias if the missing-specialty mechanism is non-random.

And our model uses only random intercepts — it assumes the specialty effect is the same regardless of patient profile. A random-slopes model could capture, for example, whether age affects readmission risk differently across specialties.

For future work, we would consider adding temporal effects, incorporating hospital-level hierarchy — so patients nested in hospitals nested in specialties — and extending to random slopes on key covariates.

Thank you! We're happy to take any questions.

---

## Timing Checkpoint Summary

| Part | Slides | Target Time | Cumulative |
|------|--------|-------------|------------|
| Title | 1 | 0:30 | 0:30 |
| Motivation & Data | 2-7 | 4:30 | 5:00 |
| Framework | 8-9 | 0:30 | 5:30 |
| Model Structure | 10-13 | 4:00 | 9:30 |
| Key Derivation | 14-17 | 4:30 | 14:00 |
| Sampling Strategy | 18-21 | 5:00 | 19:00 |
| Inference | 22-26 | 7:00 | 26:00 |
| Sensitivity | 27-28 | 2:00 | 28:00 |
| Conclusion | 29-31 | 2:00 | 30:00 |
