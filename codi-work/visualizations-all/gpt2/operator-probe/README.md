# operator-probe/ — operator probe + multi-position probes

## Subdirectories
- `probe-fitting/` — all LDA-on-operator slideshows (per-(step, layer) supervised
  LDA, 80/20 train/test, dim-sweeps, cross-dataset transfer). See its own README.

## Outputs
- `operator_probe_colon.{py,json,pdf}` — 4-class operator probe at ':' residual per layer.

## Multi-position decode probes
- `gpt2_multipos_probes.json` — per (position × layer) probe accuracy for units/tens/operator.
- `gpt2_multipos_probes.png`, `gpt2_multipos_peaks.png` — heatmaps.
- `gpt2_multipos_probes_modelans.py`, `probes_multipos.py` — probe runners.

## Other GPT-2 probes (legacy)
- `gpt2_decode_probes.{json,png}`, `gpt2_logit_lens.json`, `gpt2_var_probes.{json,png}`
- `gpt2_narrowing_{decode,prompt}.png`, `gpt2_narrowing_summary.json`, `gpt2_answer_proj.png`
- `gpt2_corrected_lens.npz` — corrected logit lens output.

## Scripts
- `probes_decode_acts.py`, `probes_gpt2.py`, `probes_multipos.py`, `probes_multipos_modelans.py`
- `per_head_ablation_probe.py`, `per_head_ablation_canonical.py`
- `redo_probes_and_op_dir.py`

NOTE: Most legacy scripts reference data files at `experiments/computation_probes/`.
