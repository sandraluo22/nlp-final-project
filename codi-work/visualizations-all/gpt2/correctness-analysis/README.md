# correctness-analysis/ — correctness probes, faithfulness, 1→2 deep dive, ablations

## Probes
- `correctness_probe.{py,json,pdf}` — latent-loop probe; held-out 80/20 logreg.
- `correctness_probe_colon.{py,json,pdf}` — ':' analog.
- `faithfulness_probe.{py,json,pdf}` (v1) and `faithfulness_probe_v2.{py,json,pdf}` (5-fold CV + C-sweep + null/random baselines).
- `pca_bifurcation_diagnose.{py,json,pdf}` — which features drive PC1 of latent activations.

## Step 1→2 deep dive
- `step1to2_deep_dive.{py,json,pdf}` — trajectory diff per layer + wr vs rw cohorts.
- `step1to2_feature_projection.{py,json,pdf}` — project delta onto operator/magnitude/correctness/faithfulness/answer-token directions.
- `step1to2_correctness_distribution.{py,json,pdf}` — probe-confidence P(correct) per cohort at each layer.
- `step2_layer_ablate.{py,json,pdf}` — zero-ablate one layer of step 2 at a time.

## Ablation sweeps
- `patch_recovery_sweep.{py,json,pdf}` — 4-tier recovery rate sweep.
- `patch_sanity_check.{py,json}` — zero-ablate + std/‖mean‖ sanity.
- `patch_cf_mean.{py,json}` — mean-CF patching per cell.
- `ablate_mlp_attn.py` — earlier mean-ablation script (superseded by patch_cf_mean).
- `plot_recovery_sweep.py` — bar-chart renderer.

## Answer-direction
- `answer_direction_projection.{py,npz,pdf}` — per (step, layer) projection onto each example's answer token row.

## Slideshows
- `latent_loop_compute_slideshow.pdf` (10 slides) — synthesizes force-decode + patch + sanity.
- `latent_compute_master_slideshow.pdf` (17 slides) — master deck combining 6 probes.
