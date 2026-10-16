# Standard splits

Catalog of the regional-distance splits that count as the current standard analysis. Figure 2, Figure 4 prior, and the ITI split are the shuffle set. Alternative nulls are at the end. Do not treat an older `manifold/res` or `res/new` file as this standard until it is listed here and rerun.

## Pipeline

Order, for every split in this catalog:

1. Action kernel (α=0.2) on the **full session** choice sequence. Bad trials, saturation-dropped trials, and true `probabilityLeft==0.5` stay in the EMA.
2. Then the existing bad-trial and saturation mask.
3. Then drop trials whose **true** `probabilityLeft` is 0.5. The kernel has already seen them.
4. `min_trials_per_side=5` on the two sides of the split, after that drop.
5. Plain within-stratum label shuffle, `nrand=2000`, except `act_block_only`, whose null is `generate_pseudo_blocks` (0.8/0.2 runs, same length as the kept trials, no t−1 lag). That null is the Figure 2, Figure 4, and ITI sections. The alternative-null section below keeps these labels and changes the null.

Outputs go to `manifold/res_standard` (and `_stream_acc` under it). Existing `manifold/res` and `res/new` files are left in place. Alternative nulls use the same directory with a suffix, so they do not replace `{split}.npy`.

Flag: `--standard-act-prior` on `scripts/run_goal2_splits.py`, or `STANDARD_ACT_PRIOR=1`. Default off, so other campaigns keep the masked-then-kernel path. Shuffle driver: `scripts/submit_standard_fig2_orcd.sh` (preset `standard`). Alternative nulls: `scripts/submit_standard_alt_nulls_orcd.sh`.

### Dropping the unbiased block

The drop does not move any remaining trial between act-prior sides. The kernel already ran on the full sequence, including the unbiased block, so a kept trial’s 0.8/0.2 label is fixed before the drop. Figure 2 strata are that binary act prior, plus stim side or choice side. They are not a true-block contrast, so removing true-0.5 trials only changes which trials enter the pools.

The drop is not a no-op on the pools. A true-0.5 trial would otherwise be scored on whichever act side the kernel assigned. After the overwrite, `probabilityLeft` is only 0.8 or 0.2, so a filter on the overwritten column would remove nothing. The drop uses the saved true block.

A trial with no choice (`0` or NaN) stays in the sequence and does not move the prior, matching IBL `ActionKernel`. A left choice steps the left prior up; a right choice steps it down. A bad trial that does have ±1 is updated from that choice. The standard path uses this through `_standard_act_pack` → `action_kernel_priors` on the full session, before the mask.

### No-choice and the bad-trial mask (2026-10-15)

`exclude_nochoice` is off in `load_trials_and_mask`. A choice of 0 is not itself a bad trial. On the alyx BWM trials table (295920 trials) there are no NaN choices; 1148 trials have `choice==0`. Of those, 1077 already fail the mask, mostly because `firstMovement_times` is missing. 71 pass `bwm_include`. On the sessions that contain those 71, treating them as a right step instead of a hold flips 33 of 21016 kept labels.

So a mask-first run barely sees no-choice trials, and the hold does not move its labels. The gap between a full-session kernel and a mask-first kernel is the bad trials that still have a ±1 choice: those update the full-session EMA and are skipped when the kernel runs after the mask (about 8.3% of kept biased ±1 labels).

Findling’s ephys fit (`prior-localization` `fit_session_ephys`) passes the full trial table into `ActionKernel`, then applies the mask only to the decoded samples. That mask uses `exclude_nochoice=True` and `max_rt=None` (no 2 s cap). Inside the kernel, NaN choices are turned into 0 and a 0 holds the prior. No-choice trials also drop out of the likelihood (`p=1`). They are not deleted before the update. Remaining differences: we use a fixed α=0.2 and then binarize ≥0.5 to 0.8/0.2; they fit α per session (median decay 5.45 trials) and decode the continuous prior (`binarization_value` null). Our mask still caps reaction time at 2 s.

## 2026-10-14 — Figure 2 (18 splits)

Not run yet. Submit with `scripts/submit_standard_fig2_orcd.sh` (plain shuffle, `res_standard`).

| Figure use | Splits | Align / window | What is contrasted | Stratum |
|---|---|---|---|---|
| a, early stim | `stim_block_l_act`, `stim_block_r_act` | stimOn, 0–80 ms | stim L vs R | act prior |
| b, during stim | `stim_choice_{r,l}_block_{r,l}_act` (4) | stimOn, 0–150 ms | stim L vs R | choice × act prior |
| b, around move | `stim_duringchoice_{r,l}_block_{r,l}_act` (4) | firstMovement, 150 ms before | stim L vs R | choice × act prior |
| c, during stim | `choice_duringstim_{r,l}_block_{r,l}_act` (4) | stimOn, 0–150 ms | choice L vs R | stim × act prior |
| c, around move | `choice_stim_{r,l}_block_{r,l}_act` (4) | firstMovement, 150 ms before | choice L vs R | stim × act prior |

Full names, in preset order:

- `stim_block_l_act`
- `stim_block_r_act`
- `stim_choice_r_block_r_act`
- `stim_choice_l_block_l_act`
- `stim_choice_r_block_l_act`
- `stim_choice_l_block_r_act`
- `stim_duringchoice_r_block_r_act`
- `stim_duringchoice_l_block_l_act`
- `stim_duringchoice_r_block_l_act`
- `stim_duringchoice_l_block_r_act`
- `choice_duringstim_r_block_r_act`
- `choice_duringstim_l_block_l_act`
- `choice_duringstim_r_block_l_act`
- `choice_duringstim_l_block_r_act`
- `choice_stim_r_block_r_act`
- `choice_stim_l_block_l_act`
- `choice_stim_r_block_l_act`
- `choice_stim_l_block_r_act`

## 2026-10-15 — Figure 4 prior (10 splits)

Same pipeline. Not run yet. The submit preset `standard` is these plus the 18 Figure 2 splits and `act_block_only`.

`prior_80ms` in the figure is the first 80 ms of the stim-side unsplit curves. The job is the full 0–150 ms split. There is no separate 80 ms window.

| Figure use | Splits | Align / window | What is contrasted | Stratum |
|---|---|---|---|---|
| prior_stim | `act_block_duringstim_{l,r}_choice_{l,r}_{f1,f2}` (4) | stimOn, 0–150 ms | act prior L vs R | choice × feedback |
| prior_choice | `act_block_stim_{l,r}_duringchoice_{l,r}_{f1,f2}` (4) | firstMovement, 150 ms before | act prior L vs R | stim × feedback |
| prior_80ms | `act_block_duringstim_l`, `act_block_duringstim_r` | stimOn, 0–150 ms (row uses 0–80 ms) | act prior L vs R | stim side |

- `act_block_duringstim_r_choice_r_f1`
- `act_block_duringstim_l_choice_l_f1`
- `act_block_duringstim_l_choice_r_f2`
- `act_block_duringstim_r_choice_l_f2`
- `act_block_stim_r_duringchoice_r_f1`
- `act_block_stim_l_duringchoice_l_f1`
- `act_block_stim_l_duringchoice_r_f2`
- `act_block_stim_r_duringchoice_l_f2`
- `act_block_duringstim_l`
- `act_block_duringstim_r`

## 2026-10-15 — intertrial prior (`act_block_only`)

Same pipeline. One split, no stim/choice stratum. stimOn window `[0.4, −0.1]` (400 ms to 100 ms before stimulus). The label is `priors[t]`, the action kernel of choices through trial t−1. There is no extra lag.

The null is `generate_pseudo_blocks` (`first5050=0`, length equal to the kept trials). This is the only split in the standard set that uses it. The other 28 permute the observed labels inside their stratum. This is the historical `{split}.npy` null, not the action-kernel `_pseudosession`.

- `act_block_only`

## 2026-10-15 — alternative nulls

Same label pipeline as the sections above. Not run yet. Driver: `scripts/submit_standard_alt_nulls_orcd.sh` (`ARM=sticky|harris_choice|harris_prior_stim|harris_prior_none|all`). `nrand=2000`. Files land in `manifold/res_standard` with the suffix below.

### Act prior, action-kernel pseudo-session + choice stickiness

The 10 Figure 4 prior splits. Each null draw is a new pseudo-session (option 1): fitted ActionKernel on the real session, simulated choices on that full schedule (the 0.5 warm-up stays in the sequence), copy-last so the post-0.5 quintile mean run matches that session. The act-binary is the α=0.2 kernel on that full pseudo, then the 0.5 trials are dropped, then the stim×choice or stim-side stratum is remade on the kept trials. Pseudo length starts at 1× the real kept session and grows up to 16× if the stratum is short. Suffix `_pseudo_strat_sticky`. Preset `standard_prior_sticky`. Runs without `--standard-act-prior` still cold-start the kernel after the drop.

### Harris session transplant

Unique donor draws, suffix `_harris_unique`. The donor bank is `manifold/choice_donors_standard.npy`: the kernel runs on the donor’s full session, then that split’s saturation mask, then the true-0.5 drop. Stim-aligned splits use the stim view; movement-aligned splits use the move view. `REBUILD_DONORS=1` rewrites the bank.

| Arm | Preset | Splits | What is contrasted | Stratum |
|---|---|---|---|---|
| choice | `standard_choice_harris` | the 8 `choice_*_act` splits | choice L vs R | stim × act prior |
| prior, stim | `standard_prior_harris_stim` | `act_block_duringstim_{l,r}` | act prior L vs R | stim side |
| prior, none | `standard_prior_harris_none` | `act_block_duringstim_fully_unsplit`, `act_block_duringchoice_fully_unsplit` | act prior L vs R | none |

Choice during stim is stimOn, 0–150 ms. Choice around movement is firstMovement, 150 ms before. Prior with a stim stratum is stimOn, 0–150 ms. Prior with no stratum is the same two windows: stimOn 0–150 ms, and firstMovement 150 ms before. The f1/f2 prior splits stay on the sticky arm above.
