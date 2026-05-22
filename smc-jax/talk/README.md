# SMC talk for the pyrenew team

A 60-minute talk introducing SMC methods to the CDC pyrenew team, who are
deeply experienced with NUTS / HMC and largely new to SMC. The
informational goal is to make the audience aware of where SMC has clear
upsides — sequential updates, fully discrete latents, principled
parameter-learning frontiers — without dissuading them from NUTS where
NUTS is the right tool.

## Render

```bash
cd smc-jax/talk
quarto render talk.qmd
```

This produces `talk.pptx` (PowerPoint) in this directory. Open in
PowerPoint / Keynote / LibreOffice Impress.

## Layout

- `talk.qmd` — the deck.
- `diagrams/*.py` — Python scripts that generate the new methodology
  diagrams (density evolution, filter family tree, bootstrap PF step,
  pMCMC vs SMC², SMC² nested clouds, Liu-West geometry, discrete-latent
  gradient).
- `assets/*.png` — the generated diagrams.
- `assets/figures_ref/*.png` — copies of selected example figures from
  `examples/figures/`, so this directory is self-contained for sharing.

To regenerate the methodology diagrams after editing a script:

```bash
python diagrams/01_density_evolution.py
# ... or all of them:
for f in diagrams/*.py; do python "$f"; done
```

## What the talk references

- `FROM_BOOTSTRAP_TO_GUIDED.md` (repo root) — full derivation of the
  guided-proposal PF that powers Examples 12–14.
- `PYRENEW_FRICTION.md` (repo root) — honest writeup of where pyrenew
  did not slot in unchanged, and the two SMC² inner-loop approximations
  this project ships.
- `src/smc_renewal/pf/liu_west.py` — the Liu-West shrink-jitter
  implementation referenced in slide 18 / the geometry diagram.
- `examples/12_model_d_gdm.py`, `13_model_c_on_gdm_data.py`,
  `14_counterfactual_tracing_stops.py` — the three examples driving
  the "fully discrete latent" section.
