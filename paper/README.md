# Paper: When Density Beats Prediction

arXiv-format academic paper. Self-supervised latent density estimation for anomaly detection in sparse mmWave point clouds.

## Compilation

### Option 1: Local (requires LaTeX)

```bash
# macOS
brew install --cask mactex

# Compile
cd paper
pdflatex main.tex
pdflatex main.tex  # second pass for references
```

### Option 2: Overleaf (recommended)

1. Create new project at https://overleaf.com
2. Upload `main.tex` and `references.bib`
3. Set compiler to pdfLaTeX
4. Compile

## Figures Needed

The paper references these figures. Place them in `paper/figures/`:

| Reference | Filename | Source |
|-----------|----------|--------|
| Figure 1: t-SNE projection | `fig_tsne.pdf` | `ai/eval/plot_report.py` (to be generated) |
| Figure 2: Drift score trajectory | `fig_drift.pdf` | `ai/eval/plot_report.py` (to be generated) |
| Figure 3: PCA dimension sweep | `fig_pca_sweep.pdf` | Ablation results (Section 5.4) |
| Figure 4: Per-action confusion | `fig_confusion.pdf` | Per-action analysis (Section 5.5) |

Existing figures in `docs/figures/`:
- `fig1_hypotheses.png` — H1-H4 summary (for Phase B0 training report)
- `fig2_error_dist.png` — Prediction error distribution
- `fig3_window_auc.png` — Window AUC curve

## Author Info

Fill in author names and affiliations in `main.tex` before submission.

## Target Venues

- arXiv preprint (immediate)
- UbiComp / IMWUT 2027
- IEEE EMBC 2027
- SenSys 2027
