# Fit targets (notebook / paper-brain-wide-map)

Canonical copies used by `run_fit_weights`, `run_fit_joint`, `run_fit_retinal`,
and plot/bench helpers. Same files as `paper-brain-wide-map/model_test.ipynb`.

| File | Role |
|------|------|
| `mean_data_results.npy` | Nested I/M `{stim, choice}` traj targets for `L_w` |
| `data_act_block_duringstim.npy` | Prior-effect data (stim window) |
| `data_act_block_duringchoice.npy` | Prior-effect data (choice window) |
| `avg_mean_R.npy` | Right-stim S curves for `L_S` |

Do **not** replace these with ONE `manifold/res` flat mean_data or
`manifold/figs` prior curves (see journals 2026-08-12c).
