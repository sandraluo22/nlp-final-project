# steering/ — all steering experiments

## Outputs
- `operator_steering_slideshow.pdf` — original operator-steering deck (latent + 'The' decode).
- `steering_slideshow.pdf` — legacy meta-deck (model-label / cossim / magnitude).
- `steering_slideshow_colon.pdf` — `:`-position steering deck.

## Result JSONs
- `steering_correctness.json` — α-sweep on the correctness direction (latent step 2, layer 6).
- `steering_operator_all.json` — A_latent / A_decode / C_latent / C_decode at original cells.
- `steering_operator_colon.json` — operator steering at `:` (decode pos 4, layer 9): A / C / DAS rank-4.
- `steering_operator_steps.json` — operator steering across all 6 latent steps at layer 10.
- `steering_operator_causality.json` — legacy operator-steering (LDA direction).
- `steering_cossim_magnitude.json`, `steering_modellabel_results.json` — legacy direction sweeps.
- `bare_math_steering.json`, `steer_mlp_and_perhead.json` — auxiliary steering experiments.

## Attention captures (used by `operator_steering_slideshow.pdf`)
- `attention_operator_latent.{npz,_meta.json}` — 200 examples, 12 heads, attention from latent (step 4, layer 10).
- `attention_operator_decode.{npz,_meta.json}` — same at decode pos 1, layer 8.

## Scripts
- `steer_operator_all.py` / `_colon.py` / `_steps.py` — operator-steering runners (used to produce the JSONs above).
- `steer_correctness.py` — correctness-direction α sweep.
- `steer_codi_gpt2.py`, `steer_clean_op_fixed.py`, `steer_modellabel.py`, `steer_cossim_magnitude.py`,
  `steer_mlp_and_perhead.py`, `steer_latent_loop.py`, `steer_op_analysis.py`,
  `steer_odd_even_layers.py`, `steer_orthogonal_cell.py`, `steer_svamp_like_dir.py` — legacy steering scripts.
- `build_steering_slideshow.py`, `build_steering_slideshow_colon.py`, `build_operator_steering_slideshow.py` — slideshow builders.

NOTE: Scripts may reference data files (activations, JSONs) by relative paths from the
original `experiments/computation_probes/` location. After this reorganization the
output paths still write to that location; manual path updates required to re-run
from this directory.
