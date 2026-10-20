# Sensory responding areas

**Goal.** Replace the Figure 2 stim-type labels as the definition of a sensory responding area. An area is sensory responding when its population distinguishes stimulus left from right in the first 80 ms, inside an action-kernel prior stratum, on the standard split set. Onset offsets that do not survive subtraction are treated as confounds and dropped. A region is also dropped when choice is already significant inside that same 80 ms (a run of at least five choice bins; see 2026-10-20).

This category is what Figures 2 and 4 should use. Those figures are not redrawn yet. See [paper main figures](paper_main_figures.md).

## 2026-10-19 — standard 80 ms prior-stratum stim

Split: `stim_duringstim1_act` = `stim_block_{l,r}_act` in alyx `manifold/res_standard`. Stimulus left versus right, stimOn 0–80 ms, stratum is the binary action-kernel prior only (no choice stratum). Time-mean FDR `p_mean_c` ≤ 0.01. 17 of 207 regions.

Curves against shuffles (40 of 2000), mean of the two prior sides:

`<oa>/manifold/paper_figs/fig2/res_standard/earlystim80_prior_only_sig.png`

`<oa> = ~/Downloads/ONE/openalyx.internationalbrainlab.org`.

All 17 sit on the shuffle floor (raw p = 1/2001). Benjamini–Hochberg assigns that tie one corrected value, 0.0061. The traces are not copies of each other.

In Beryl order:

VISal, VISam, VISp, VISpm, CP, LGd, LP, PO, PF, ZI, MRN, SCm, PAG, NOT, PRNr, DCO, SIM.

Most of these look like a stimulus response that builds after onset. Three are already separated from the shuffles in the first bins.

### Onset offset

Same test as the prior curves: mean of the first five bins (0–7.8 ms) against the shuffles. Subtract that constant from the observed curve when the uncorrected p is ≤ 0.01, then retest the mean of the remaining bins for a gain.

| region | offset | uncorrected p | gain p after subtraction |
|---|---:|---:|---:|
| PRNr | 0.104 | 0.0005 | 0.995 |
| DCO | 0.150 | 0.0005 | 0.993 |
| SIM | 0.116 | 0.0005 | 0.922 |

None of the other 14 pass the uncorrected onset test. PO is the next, at 0.011. Brain-wide FDR on the onset p-values puts the three-way tie at 0.035.

After the subtraction the late window is inside the shuffles, so the early-stim hit does not survive. Dashed = data, solid = data minus the onset offset:

`<oa>/manifold/paper_figs/fig2/res_standard/earlystim80_prior_only_offset_gain.png`

The offset is a separation between left-stimulus and right-stimulus trials that is already present at stimOn, too early for a visual volley. It reads as a confound: previous movement in PRNr and SIM (premotor and cerebellar), and an auditory difference in DCO. Those three are excluded.

### Sensory responding category

14 regions:

VISal, VISam, VISp, VISpm, CP, LGd, LP, PO, PF, ZI, MRN, SCm, PAG, NOT.

## 2026-10-19 — old res-standard stim class outside the new list

Ignore the openalyx type table. On the res-standard `sc_duringstim_regtype` table the stim class (type 0; none of these are stim-early) has five regions that are not in the 14 above:

FRP, VISli, MH, LDT, DCO.

Plots, same shuffle overlay as the 17:

| window | contrast | stratum | file |
|---|---|---|---|
| 0–80 ms | stim L vs R | prior only | `earlystim80_regtype_only.png` |
| 0–150 ms | stim L vs R | choice × prior | `stim150_choice_prior_regtype_only.png` |
| 0–150 ms | choice L vs R | stim × prior | `choice150_stim_prior_regtype_only.png` |

All under `<oa>/manifold/paper_figs/fig2/res_standard/`.

| region | 80 ms stim, prior only | 150 ms stim, choice × prior | 150 ms choice, stim × prior |
|---|---|---|---|
| FRP | 0.40 | raw 0.0005, FDR 0.0034 | raw 0.012, FDR 0.024 |
| VISli | 0.020 | raw 0.0005, FDR 0.0034 | raw 0.0045, FDR 0.010 |
| MH | 0.14 | raw 0.0005, FDR 0.0034 | raw 0.026, FDR 0.046 |
| LDT | 0.047 | raw 0.0005, FDR 0.0034 | raw 0.064, FDR 0.11 |
| DCO | 0.0061 | raw 0.0005, FDR 0.0034 | raw 0.014, FDR 0.028 |

The shared 150 ms stim FDR is the floor tie: all five beat every shuffle on the time mean, along with 25 other regions.

**DCO** is out. The onset offset is there at 80 ms (uncorrected p = 0.0005; gain gone after subtraction) and again at 150 ms (uncorrected p = 0.0005, FDR 0.020). The curve is high from the first sample in both windows. Auditory confound, as above.

**FRP** (frontal pole, rostral prefrontal cortex) is not an early sensory area: the 80 ms curve stays in the shuffles. Over 0–150 ms the stimulus curve leaves the shuffles late and is still climbing at 150 ms. Choice in that window is marginal (FDR 0.024). Around movement both contrasts pass (stim FDR 0.0027, choice FDR 0.0038), choice is about half the combined amplitude (`sc_duringchoice` = 0.54), and the movement-aligned label is integrator rather than a movement ramp. So during the stimulus this is a late stimulus representation, not integration. The integrator label belongs on the movement-aligned row, where the choice signal is actually there. Open: whether the during-stim row should carry any label, or stay unlabeled because it fails the 80 ms sensory test and fails the choice test.

**MH** is the medial habenula (epithalamus). It is known for aversion, anxiety, and behavioral suppression, via the fasciculus retroflexus to the interpeduncular nucleus, not for visual decisions. **LDT** is the laterodorsal tegmental nucleus, a cholinergic pontine nucleus next to the pedunculopontine nucleus, tied to arousal, REM sleep, and attention through projections to the midbrain dopamine groups and thalamus. On the 150 ms stimulus contrast both are FDR-significant, but the traces stay close to the shuffle cloud and do not ramp. Onset bias is not significant (MH p = 0.046, LDT p = 0.048). Choice does not pass. What the late stimulus average is picking up is unclear.

**VISli** (laterointermediate visual area) misses the 80 ms cutoff at 0.020. At 150 ms the stimulus curve ramps and choice is just over the line (FDR 0.010). Not resolved.

## 2026-10-19 — choice sensitivity of the 14

Choice left versus right for all 14 sensory responding regions, stimOn 0–150 ms, stratum is stimulus side × action-kernel prior (`choice_duringstim_act`, mean of the four splits). Titles are the 150 ms time-mean FDR (`p_mean_c`). A dotted line marks 80 ms. Blue marks are bins whose uncorrected p is ≤ 0.01. The 80 ms window FDR is not on the figure.

`<oa>/manifold/paper_figs/fig4/res_standard/choice150_sensory14_stim_prior.png`

Eight of the 14 pass FDR ≤ 0.01:

| region | 150 ms choice FDR |
|---|---:|
| VISp | 0.0018 |
| CP | 0.0018 |
| LP | 0.0018 |
| ZI | 0.0018 |
| MRN | 0.0018 |
| SCm | 0.0018 |
| PAG | 0.0030 |
| PO | 0.0054 |

The six at 0.0018 are a floor tie: raw p = 1/2001, shared with 55 of 202 regions. The other six of the 14 do not pass. Closest are VISpm (0.013) and NOT (0.022); then PF 0.081, LGd 0.101, VISal 0.136, VISam 0.184.

On the res-standard type table, the during-stim integrators and movement regions inside the 14 are exactly those eight:

| during stim | regions |
|---|---|
| integrator (`sc_duringstim_regtype` 0.5) | VISp, LP, PO, PAG |
| movement (1.0) | CP, ZI, MRN, SCm |

The remaining six are stim or stim-early on that row: VISal and VISam are stim-early (0.1); VISpm, LGd, PF, and NOT are stim (0).

Around movement the integrator set is larger. `sc_duringchoice_regtype` 0.5 adds LGd and PF to VISp, LP, PO, and PAG. The movement label is the same four (CP, ZI, MRN, SCm). VISal, VISam, VISpm, and NOT have no movement-aligned type. LGd and PF are integrators only on that row, and neither passes the 150 ms choice FDR above.

## 2026-10-20 — drop regions with early choice

Choice left versus right, stimOn 0–150 ms, stratum stimulus side × action-kernel prior (`choice_duringstim_act`, mean of the four splits). A bin is significant when its uncorrected shuffle p is ≤ 0.01. **Early choice** is a run of at least five such bins, about 10 ms, lying at t ≤ 80 ms (the first 38 bins; the last of those is at 78 ms). The 80 ms time-mean FDR is not this test: brain-wide, that mean sits on a floor tie at 0.011 and does not separate curves that are visibly above the shuffles.

Longest early run inside the 14:

| region | bins |
|---|---:|
| CP, MRN | 14 |
| SCm | 11 |
| PAG | 7 |
| LP | 4 |
| VISp | 3 |
| VISpm, ZI | 2 |
| PO, NOT | 1 |
| VISal, VISam, LGd, PF | 0 |

`<oa>/manifold/paper_figs/fig4/res_standard/early_choice80_run_sensory14.png`

CP, MRN, and SCm each have one run that starts near 50–60 ms and stays significant through 150 ms. PAG has a burst of seven bins at 19–32 ms. Those four are out. ZI’s marks before 80 ms are one- and two-bin crossings; its sustained choice run starts at 114 ms. LP’s 0–80 ms mean is significant (raw p = 0.002) but the longest run is four bins, so LP stays.

### Sensory responding category

10 regions:

VISal, VISam, VISp, VISpm, LGd, LP, PO, PF, ZI, NOT.

### Prior aggregate (Figure 4c and 4i left)

Same rule as panel c: cell-weighted 80 ms stim-stratum prior (`act_block_duringstim_{l,r}`, mean of the two sides), restricted to members whose 80 ms prior FDR is ≤ 0.01. Five of the 10:

VISal, VISam, LP, PO, ZI.

4,858 cells (VISal 48, VISam 359, LP 1,793, PO 2,316, ZI 342). The other five miss that prior cutoff (LGd 0.0102, VISp 0.018, PF 0.050, VISpm 0.050, NOT 0.083).

Dashed = data, solid = data minus the onset offset. The inset is the shuffle-subtracted excess on the same 0–0.2 scale as the earlier nine-region panel.

`<oa>/manifold/paper_figs/fig4/res_standard/panel_c_prior80_sensory_no_earlychoice.png`

`<oa>/manifold/paper_figs/fig4/res_standard/panel_i_left_sensory_no_earlychoice.png`

On the aggregate, against its own shuffles: onset offset p = 0.002 (effect 0.017), and gain after that subtraction p = 0.004. Both are ≤ 0.01. The time-mean of the aggregate is on the floor (p = 0.0005). The excess over the shuffle mean stays between 0.014 and 0.040. On the left half of panel i, C and M are unchanged (integrator union 85 regions, movement union 30); S is this five-region excess, drawn through 78 ms.

## Open

1. Redraw Figure 2 and Figure 4 on the 10-region list. The current `sc_duringstim_regtype` sensory and sensory-early labels are not this set. Panel c and the left half of panel i above are diagnostics, not the full figure.
2. The old openalyx early-stim split (35/208) is not used here.
3. Whether FRP gets a during-stim label. Movement-aligned FRP is an integrator. During stim it is a late stimulus representation with no choice significance.
4. MH and LDT: significant late stimulus average, no ramp, no choice, no onset offset. Not assigned.
