# From bootstrap to guided particle filters

A narrative explainer for `examples/12_model_d_gdm.py` and
`examples/13_model_c_on_gdm_data.py`. The setup is short enough to fit in
your head and is structured to walk a talk audience through three things:

1. The **model** we actually want to fit (contact-tracing outbreak with
   exact cohort-budget bookkeeping and a Generalised Dirichlet Multinomial
   reporting delay).
2. The **non-bootstrap PF** we used to fit it — a guided proposal that
   exactly matches each day's observation, with an auxiliary-variable
   Rao-Blackwellisation step that prevents the IS weights from blowing up.
3. A **structural-failure complement**: applying a model that doesn't have
   the contact-tracing mechanism to the same data. It fits the cases but
   gives an operationally wrong story about Rt.

---

# Part I — The model

## 1. What we are trying to represent

A short outbreak (50 days) of a disease where contact tracing is real:
when an infected person is reported, they are *also removed from
circulation* — they no longer contribute to onward transmission. This is
a Hantavirus-style high-ascertainment regime, not COVID-style noisy
surveillance.

Three mechanisms have to be in the model:

- **Transmission**. Continues whenever there are still-infectious people
  in circulation.
- **Reporting delay**. Cases take some days to be detected and recorded,
  with both shape (mean delay) and over-dispersion (cohort-to-cohort
  variability in how fast reporting proceeds).
- **Contact-tracing removal**. Reporting *is* removal — the moment a
  person is reported, they stop being part of the renewal pool.

The third mechanism is what makes this example structurally different
from Models A–D in the repo. It introduces a *self-limiting feedback*
that depends on the observation process, not on transmission rate itself.

## 2. Latent dynamics

```
v_R[t]      =  v_R[t-1]  +  σ_vR · η_R[t]                    velocity walk
log Rt[t]   =  log Rt[t-1]  +  v_R[t-1]                       integrated BM
g_conv      =  Σ_k g(k) · U_buf[k]                            renewal sum
λ_t         =  exp(log Rt[t]) · g_conv                        Poisson rate
N_t         ~  Poisson(λ_t)                                   new infections today
```

Two things to notice:

- The renewal kernel reads from `U_buf` (still-circulating-and-not-yet-
  reported) not from "total infections per cohort". Under
  100% ascertainment these are the same concept under alternating
  add/subtract events — at each step we *add* `N_t` to slot 0 and
  *subtract* reported cases across all slots. A single buffer suffices.

- There is no F-feedback term (susceptibility depletion is a long-
  timescale phenomenon; on a 50-day outbreak the bending comes from
  contact tracing). There is no immigration term (single-seed outbreak).
  These dimensions of Models A–D are gone.

## 3. Observation model — Generalised Dirichlet Multinomial delay

Following Stoner et al, each cohort's eventually-observable cases are
stick-broken across reporting stages via per-stage Beta-Binomials. For
each stage `s ∈ {1, …, L−1}` (minimum delay 1 day; stage 0 forced to
probability zero):

```
q_s   ~  Beta(α_s, β_s)                 reporting fraction at this stage
O_s   ~  Binomial(U_s, q_s)             how many of this cohort report at age s
Φ⁻¹(p_s)  =  b_0  +  b_1 · (s − 1)      probit-linear delay shape
α_s = p_s · M,  β_s = (1 − p_s) · M     M = exp(log_M) is shared concentration
```

Marginally, `O_s ~ BetaBin(U_s, α_s, β_s)`. The per-stage Beta is what
introduces over-dispersion above plain Binomial.

The observed count is the simple sum:

```
y_t  =  Σ_s O_s
```

— with `y_t` exactly equal to the sum of cohort contributions, no extra
NegBin noise on top. This is the indicator-like aspect of the likelihood
that will break the bootstrap PF.

## 4. The buffer update

At the end of each step, the reported cases are removed from their
respective cohort slots, and the buffer shifts forward (today's new
cohort enters age 0; the oldest cohort age L−1 is dropped, modulo the
negligible tail-truncation mass):

```
U_prop      =  [N_t,  U_buf_prev[0],  …,  U_buf_prev[L-2]]      (new cohort enters)
U_buf_new   =  U_prop  −  O                                       (remove reported)
```

`U_prop[0] = N_t` is the just-sampled cohort. With min-delay 1 day, no
observations apply to slot 0 today, so at the end of this step
`U_buf[0] = N_t` — meaning the freshest-slot trace `U_buf[:, :, 0]` over
time *is* the latent-infection time series `N_t`. We use this for plots
without needing a separate `I_buf`.

## 5. The full model in one line

```
state    =  (log_Rt, v_R, log_I0, U_buf)
params   =  (log σ_vR, b_0, b_1, log_M)            Liu-West cloud, 4-D
```

Truth values used in `examples/12_model_d_gdm.py`:
`log_Rt = 0.7` (Rt ≈ 2), `v_R = 0` (constant Rt — the bending comes from
tracing, not from declining transmission), `log_I0 = 1.0` (small seed),
`b_0 = −1.5, b_1 = 0.4` (mean reporting delay ~ 3.5 days),
`log_M = 3.5` (moderate Beta concentration). Outbreak peaks around day
18 at ~ 25 cases/day and declines from there as tracing drains the
U-buffer.

---

# Part II — The non-bootstrap PF

Now we have a model to fit. The standard PF — bootstrap — fails badly,
and the fix walks through the three things any guided PF design has to
get right.

## 6. The particle filter, fast

State-space recursion:

```
x_t  ~  p(x_t | x_{t-1}, θ)         transition (the dynamics + buffer update)
y_t  ~  p(y_t | x_t, θ)             observation (deterministic sum here)
```

Per-step update is importance sampling:

```
proposal:  x_t^(i)  ~  q(x_t | x_{t-1}^(i), y_t, θ)
weight:    w_t^(i)  ∝  w_{t-1}^(i)  ·  p(x_t^(i) | x_{t-1}^(i), θ) · p(y_t | x_t^(i), θ)
                                       ─────────────────────────────────────────────────
                                                  q(x_t^(i) | x_{t-1}^(i), y_t, θ)
```

The **bootstrap** PF chooses `q = transition prior`. The transition then
cancels, leaving just the likelihood as the weight. Beautifully simple,
universally taught — and dead on arrival here.

## 7. Why bootstrap dies on this model

Run the model forward starting from a particle's state. You sample `N_t`,
form `U_prop`, draw `O_s` from each cohort's BetaBin independently, and
take their sum.

The probability that this sum *happens to equal* the observed `y_t` is
effectively zero. Even at moderate scale (~30 cases/day), with L ≈ 14
cohorts each contributing a BetaBin draw, the joint distribution over
`Σ O_s` is too spread out for exact hits to occur. Every particle gets
zero weight. ESS collapses to 1 immediately.

This is the same failure mode as a sharp likelihood: bootstrap proposes
without using `y_t`, and the data demands an exact match.

## 8. Designing the proposal

The fix is to **propose so `y_t` is matched exactly**, then correct via
IS. Factor the proposal as

```
q(x_t | x_{t-1}, y_t)  =  q_dyn(z_t | x_{t-1})  ·  q_part(O_t | z_t, y_t)
```

- `q_dyn` keeps the model prior for the dynamics state `z_t = (log_Rt, v_R, N_t)`.
- `q_part` is the new ingredient. It must:
  1. **Hit the constraint** `Σ_s O_s = y_t` exactly.
  2. **Respect per-cohort budgets** `0 ≤ O_s ≤ U_s`.
  3. **Match the target's marginal variance** well enough that IS weights
     have bounded variance.

The target on `O_t` (conditional on `z_t` and `y_t`) is

```
p(O_t | z_t, y_t)  ∝  ∏_s BetaBin(O_s; U_s, α_s, β_s)  ·  𝟙[Σ_s O_s = y_t]
```

— a product of independent BetaBins restricted to the simplex.

### Three candidate proposals

| Proposal | Hits constraint? | Respects caps? | Marginal mean? | Variance vs target | IS weight tractable? |
|---|---|---|---|---|---|
| Multinomial(y_t, π_s) with `π_s ∝ U_s · p_s` | yes | no (occasional escape) | matches | similar | clean |
| Multivariate hypergeometric | yes | yes | mean-mismatched (ignores `p_s`) | lighter | very clean (binomial coefs cancel) |
| Wallenius noncentral hypergeometric (weights = `p_s`) | yes | yes | matches | lighter than BetaBin | needs multinomial-coef correction |

We picked **Wallenius** as the structural skeleton — it respects
constraints, matches the target's mean per cohort, and the IS weight has
a tractable closed form (modulo the well-known ordering→outcome-space
multinomial-coefficient correction). Sampling is sequential
weighted-without-replacement, with `y_t` ball draws weighted by
`U_curr · p_s` at each step.

## 9. The IS pitfall: tail mismatch

Wallenius is *lighter-tailed* than the target. The per-stage target
marginal is BetaBin, with over-dispersion factor `1 + (U−1)/(M+1)`
relative to Binomial. The Wallenius proposal (with fixed weights `p_s`)
is sub-Binomial because of the without-replacement structure. So
target / proposal can be several times heavier per stage, compounding
across all L stages. Some draws get log-weights tens of nats above the
cloud average, and one particle eats the whole posterior mass.

This is the classic IS failure mode: a low-variance proposal against a
high-variance target gives heavy-tailed IS weights. ESS collapses; the
PF runs but gives a degenerate cloud.

## 10. Rao-Blackwellisation: introduce `q_s` as an auxiliary

The cure is to undo the marginalisation that creates the BetaBin in the
first place. The model says:

```
q_s   ~  Beta(α_s, β_s)
O_s   ~  Binomial(U_s, q_s)
```

so `BetaBin(U_s, α_s, β_s) = ∫ Bin(U_s, q_s) Beta(q_s; α_s, β_s) dq_s`.
The over-dispersion of BetaBin comes entirely from `q_s` being random.

**Don't marginalise `q_s` out**. Sample `q_s` as part of the proposal,
and weight against the joint `(O, q)` target.

```
Augmented target:
    p_aug(O, q | y_t)  ∝  ∏_s [ Bin(O_s; U_s, q_s) · Beta(q_s; α_s, β_s) ]  ·  𝟙[ΣO=y_t]

Proposal:
    q_s    ~  Beta(α_s, β_s)                                  per stage, per particle
    O      ~  Wallenius(U, weights = q_s, y_t)                conditional on q_s

IS weight (Beta factors cancel between target and proposal):
    w(O, q)  =  ∏_s Bin(O_s; U_s, q_s)
              ──────────────────────────────────────────────────
              Wallenius(O; U, q_s, y_t)  ·  multinomial-coef(y_t; O)
```

The Beta priors are in both numerator and denominator and drop out. What
remains is **per-cohort Bin against Wallenius weighted by the same q_s
on both sides**. These match in marginal mean and have comparable
variance. Tails match. IS weights stay bounded.

The over-dispersion that was killing us is now produced *by the proposal
itself* — each particle draws its own `q_s` per stage, so different
particles sit on different points of the BetaBin's support, and the IS
reweighting just corrects between the per-stage Bin and Wallenius
conditionals (with the same q_s on both sides).

Empirically on the contact-tracing outbreak: coverage on `log Rt` and
`I(t)` both hit ~ 0.95–0.98 (near nominal 0.9), min ESS settles around
10 with N = 8000, and parameter posteriors on `b_0, b_1, log_M` visibly
move from prior toward truth. The filter recovers a constant Rt ≈ 2.0
correctly, and the I(t) trace tracks the truth despite the noisy cohort
partition.

## 11. The algorithm, end to end

```
for each particle i:
    # --- Dynamics from the prior ---
    sample velocity update η_R
    advance log_Rt  ←  log_Rt + v_R
    advance v_R     ←  v_R + σ_vR · η_R
    g_conv          ←  Σ_k g(k) · U_buf[k]               contact-tracing renewal
    λ_t             ←  exp(log_Rt) · g_conv
    N_t  ~  Poisson(λ_t)

    # --- Auxiliary q draws (the GDM stick-breaking variables) ---
    α_s, β_s  =  gdm_beta_params(b_0, b_1, log_M)
    q_s       ~  Beta(α_s, β_s)         for s = 1, …, L−1   (q_0 ≡ 0, min delay 1 day)

    # --- Form the working U-buffer (new cohort enters at age 0) ---
    U_prop  =  [N_t, U_buf_prev[0], …, U_buf_prev[L-2]]

    # --- Zero-weight escape for infeasible particles ---
    if y_t > Σ U_prop:
        log_w_inc = -∞
        continue

    # --- Guided partition: sequential weighted without-replacement (Wallenius) ---
    U_curr = U_prop
    log_q_ordered = 0
    for k = 1, …, y_t:
        weights      = U_curr · q_s
        π            = weights / Σ weights
        s_drawn      ~ Categorical(π)
        U_curr[s_drawn] -= 1
        log_q_ordered += log π[s_drawn]
    O = count of cohorts drawn

    # --- IS log-weight (Bin / Wallenius ratio, Beta cancels) ---
    log_target = Σ_s log Bin(O_s; U_s, q_s)
    log_mc     = gammaln(y_t + 1) - Σ_s gammaln(O_s + 1)
    log_w_inc  = log_target - log_mc - log_q_ordered

    # --- Update state buffer (single buffer in this model) ---
    U_buf_new  =  U_prop - O

# After all particles:
log_w  =  log_w + log_w_inc
normalize, compute ESS
if ESS < threshold:
    resample (multinomial)
Liu-West shrink-jitter on the parameter cloud
```

The Liu-West kernel + ESS-triggered multinomial resampling come from
`pf/_runner_core.py` and are unchanged from the other PF variants. Only
the per-step proposal + weight differ. The 4-D Liu-West cloud is
`(log σ_vR, b_0, b_1, log_M)`.

---

# Part III — Structural failure when the mechanism is missing

`examples/13_model_c_on_gdm_data.py` applies **Model C** — same
velocity-driven log Rt dynamics, but with a delay-conv + NegBin
observation model and **no contact-tracing depletion in the renewal
kernel** — to the same locked synthetic dataset that example 12
generates. To isolate the comparison to the dynamics layer, Model C's
delay PMF is set to the GDM's marginal lag distribution so the delay
shapes match.

## 12. The numbers

| Metric                | Model E (contact tracing in model) | Model C (no contact tracing)   |
|---|---|---|
| log Rt 90% coverage   | 0.98 (near nominal)                | **0.36** (badly under nominal)  |
| log Rt filter-median trajectory | tracks truth (Rt ≈ 2.12, *constant*) | drifts from +0.5 to −1.5  |
| I(t)    90% coverage  | 0.92 (near nominal)                | 1.00 (over-wide)                 |
| log φ posterior       | n/a                                | +3.1 (NegBin absorbing variance) |
| Min ESS               | ~ 10                               | ~ 590 (healthy)                  |

Model C *fits the cases* fine — NegBin overdispersion soaks up the
cohort noise. What it gets wrong is **the inferred Rt trajectory**.

## 13. Why this happens

Truth `log Rt ≈ 0.75` (Rt ≈ 2.12) is *pinned* essentially constant —
the synthetic data was generated with very small `σ_vR` and rejected
unless `log Rt > 0` throughout the first 25 days. **Transmission never
actually slows.** The reason cases peak and decline is that contact
tracing depletes the still-circulating-and-infectious pool faster than
new infections replenish it.

Model C doesn't have that mechanism, so to fit the observed decline it
is forced to attribute the bending to a *falling Rt*: its filter median
walks from above 1 to well below 1 over the 50 days. NegBin's `log φ`
absorbs the cohort-level noise; it cannot represent the depletion
feedback. The model has to put the bending somewhere, and the only place
it has is Rt.

The forecast figures (`12_model_d_gdm_forecasts.png` and
`13_model_c_on_gdm_data_forecasts.png`, both on log y) make this
concrete. **At the day-15 origin** neither model has yet seen any
decline data, so both forecast continued growth — but with characteristic
differences in *uncertainty*:

- Model C's 95% upper PI stretches to ~10,000 cases by the end of the
  14-day horizon. With no depletion mechanism, the model genuinely
  doesn't know how high the outbreak might go.
- Model E's 95% upper PI tops out around ~800 — the contact-tracing
  mechanism caps how high the outbreak can plausibly go *even if* Rt
  stays high, because the U-buffer can only carry so many people before
  reporting catches up.

**At days 25 and 35**, both models have seen enough decline data to
forecast decline going forward. Model E's median tracks the observed
decline cleanly; Model C also declines, but because its filter has
attributed the bending to a falling Rt, it extrapolates whatever
declining Rt trend the cloud now sees. Headline forecasts look similar;
the inferred Rt story is what's different.

## 14. The operational consequence

The two stories are **operationally opposite**.

- **Model E's story**: Rt is still ~2 but tracing is faster than
  transmission. *Action*: maintain tracing intensity; relaxing it would
  re-ignite the outbreak immediately.

- **Model C's story**: Rt has crashed below 1, transmission has slowed
  dramatically. *Action*: relax interventions, the epidemic is dying.

A model that fits the data fine and predicts the next 14 days
reasonably can still recommend the wrong intervention. The fit is not
the test of the model; the *mechanism* is.

---

# Part IV — Three takeaways for a talk

1. **Bootstrap PF is the special case `q = transition`, not the only
   PF.** It works when the likelihood is forgiving. When the likelihood
   is an indicator (high ascertainment, exact accounting, contact
   tracing), bootstrap PF is dead — you have to design a custom proposal
   that uses the observation.

2. **A guided proposal is just IS per step.** The mechanics are
   mechanical: factor the joint, pick a proposal, write down the
   weight. The art is choosing a proposal that respects constraints,
   matches marginals, and matches tails. Skip any of these and the PF
   fails in a characteristic way (constraint violations → wasted
   particles; mean mismatch → high IS variance; tail mismatch → ESS
   collapse). **Auxiliary-variable Rao-Blackwellisation is the standard
   fix for tail mismatch**: if your target is
   `p(O) = ∫ p(O | q) p(q) dq` and your proposal forces you to commit
   to a fixed `q`, lift `q` into the proposal instead. The integral
   disappears; the priors cancel; tails match.

3. **A correct proposal fixes the variance problem; only a correct
   *model* fixes the interpretation problem.** Example 13 demonstrates
   that a model lacking a structural mechanism can still fit the data
   well in marginal-variance terms while getting the dynamics
   operationally wrong. NegBin overdispersion absorbs the
   cohort-budget noise, but it cannot represent the depletion feedback
   that makes Rt constant during a tracing-driven outbreak — so
   Model C tells you transmission has fallen when in fact only the
   circulating pool has. The cost of mis-specifying mechanism is a
   wrong recommendation for the public health user, not a wrong fit
   to the data.

---

## See also

- The structural model: `examples/12_model_d_gdm.py`.
- The structural-failure complement: `examples/13_model_c_on_gdm_data.py`.
- The locked synthetic dataset: `examples/data/12_truth.npz`.
- Implementation: `src/smc_renewal/pf/model_gdm.py`,
  `src/smc_renewal/pf/runner_gdm.py`,
  `src/smc_renewal/observation_gdm.py`.
- The Stoner et al GDM nowcasting paper inspired the cohort
  decomposition used here; this example uses the GDM for delay
  modelling rather than nowcasting, pairs it with contact-tracing
  depletion in the renewal kernel, and fits it with a guided PF rather
  than a Gibbs / HMC fit.
