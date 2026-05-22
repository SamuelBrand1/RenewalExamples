# SMC talk for the pyrenew team

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
