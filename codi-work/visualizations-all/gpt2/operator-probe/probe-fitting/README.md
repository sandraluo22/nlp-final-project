# operator-probe/probe-fitting/ — LDA probes that fit on operator labels

All slideshows here fit Linear Discriminant Analysis (LDA) supervised on the
4-class operator label (Add / Sub / Mul / Div) and then visualize the resulting
projection. PCA visualizations live one level up in `visualizations-all/gpt2/`
because they're unsupervised.

## SVAMP (full set)
- `lda_slideshow.{py,pdf}` — per (layer, latent step) LDA on SVAMP latent residuals.
  5 colorings: plain / faithful / problem_type / magnitude / correct.
- `lda_slideshow_colon.{py,pdf}` — `:`-residual analog, per layer. Same 5 colorings + log_answer.

## cf_balanced (counterfactual)
- `cf_lda_slideshow.{py,pdf}` — LDA on cf_balanced latent acts, colored by problem_type +
  output_bucket + correct.
- `cf_lda_slideshow_colon.{py,pdf}` — `:`-residual analog.

## 80/20 split + cross-dataset transfer
- `cf_lda_80_20_slideshow.{py,pdf}` — LDA on 80% of cf_balanced; project held-out 20%
  and full SVAMP. k ∈ {1, 2, 3} components.
- `cf_lda_80_20_slideshow_colon.{py,pdf}` — `:` analog.
- `cf_lda_80_20_dim1.{py,pdf}` — 1D-only LDA along each LD axis, scored under 4
  grouping schemes (indiv / AS|M|D / AS|MD / AM|SD).
- `cf_lda_80_20_dim1_colon.{py,pdf}` — `:` analog.

## Reverse transfer
- `cf_lda_compare.{py,pdf}` — LDA fit on SVAMP, project cf_balanced onto SVAMP's axes.
- `cf_lda_compare_colon.{py,pdf}` — `:` analog.

## Stats JSONs
Probe accuracies (in-sample + held-out + transfer) at each (layer, latent step)
or (layer) cell, for each variant above.
