# SGOED Code Repository

Analysis scripts backing the SGOED Phase 1–2 manuscript (record A) and its artifact-test record.

**Hosted at:** <https://github.com/dhammawatthumpra-coder/sgoed-analysis>

> **Zenodo–GitHub integration:** to mint a DOI, enable the Zenodo GitHub hook for this repo (zenodo.org → GitHub → settings) and create a release. Zenodo will archive the tagged snapshot and assign a DOI; that DOI is the "software" record B to cite alongside record A. The script→paper mapping below mirrors the "Data & Code Availability" section of `SGOED_manuscript.md`.

## Script → paper section mapping

| Script | Backs (SGOED_manuscript.md) |
|---|---|
| `sgoed_phase1_v2.py` | §2 The Framework (matrix construction, I_C) |
| `sgoed_phase1_gate_comparison.py` | §2.3–2.4 two-gate structure |
| `sgoed_phase1_crystallize_prescriptions.py` | §2 crystallization prescriptions |
| `sgoed_phase2_v2.py`, `sgoed_phase2_v3_kinetic.py` | §4.1 kinetic aggregation model |
| `sgoed_phase2_analytic_merge.py` | §4.3 analytic merge characterization |
| `sgoed_phase2_first_passage.py` | §4.4 first-passage / renewal |
| `sgoed_coagulation_steady_state.py` | §4.5 coagulation characterization |
| `sgoed_phase12_interface_check.py` | Phase 1–2 interface |
| `sgoed_phase2_exp1_fixed_p_null.py` … `exp5_operator_ablation.py`, `test_phase2_artifact_sanity.py` | §5 Artifact-Test Record (Exp 1–5) |
| `sgoed_phase2_realistic_margins.py` | §5.9 realistic margins |
| `sgoed_phase2_recursive.py`, `robustness_sweep.py` | robustness checks |
| `sgoed_structural_time.py` | STF-related structural check |
| `sgoed_merge_graph_analysis.py` | merge-graph analysis |
| `sgoed_k_analogue_*.py` | K-analogue prototype (record C) |
| `sgoed_phase2_exp3b_blend_margin_mediation.py` | §5 blend-margin mediation |

## Requirements

Python 3.11+, numpy, scipy, matplotlib (standard scientific stack). All scripts are self-contained (no external data files).

## Citation

Cite this code alongside record A (the main manuscript). See `CITATION.cff` in this folder.
