# Fig 4.5 cell types (git-tracked)

| File | Role |
|------|------|
| `fig45/cell_types.npy` | Rastermap C / M / S-pool cell picks (uuid, pid, cluster_id, insertions); default `--cell-types` / `CELL_TYPES` for `scripts/fig45_*` and the ORCD jobs |
| `fig45/cell_types_picks.json` | Cluster ranges used to build it |

Built locally from `~/Downloads/concat_cvTrue.npy` (not on ORCD) with
`python scripts/fig45_define_cell_types.py save --pick … --union …`; commit the
regenerated file to change the picks.

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
