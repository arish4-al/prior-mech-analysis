# Region lists for variance-partition analysis (git-tracked)

| File | Role |
|------|------|
| `stimchoice_act_regtype_regions_p_mean_c_0.01.csv` | Full SC classification + Σ / `mixed_stim_choice` flags (default `--regtype-csv`) |
| `var_partition_mixed_stim_choice_regions.csv` | Compact list of the 19 mixed stim×choice regions |

**Remote:** pull repo, then::

```bash
python scripts/run_var_partition.py --target mixed
```

Uses the default ONE cache (override with `--one-cache-dir` / `$ONE_CACHE_DIR` only if needed).
Do not re-export from openalyx on the cluster.

**Local regen only** (if SC results change):

```bash
python scripts/export_stimchoice_regtypes.py --skip-out-cache
```
