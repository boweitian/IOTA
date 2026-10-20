# ANLI R3: exploratory direction scan

Qwen3.5-9B, complete official R3 test split, 1,200 items. All 61 configurations use the same detailed-analysis prompt, a final `答案：A/B/C` line, and a 1,024-token generation limit. The scan contains 73,200 actual benchmark generations.

| Configuration or candidate set | Correct / covered | Rate |
|---|---:|---:|
| Unsteered baseline | 803/1200 | 66.92% |
| Best fixed named direction: confidence +1× | 843/1200 | 70.25% |
| Best existing mixture | 801/1200 | 66.75% |
| At least one of 40 named single-direction settings correct | 1184/1200 | 98.67% |
| Including 12 existing mixtures and baseline | 1186/1200 | 98.83% |
| Including all 8 random-control settings as well | 1186/1200 | 98.83% |

The best fixed setting repairs 137 baseline errors while damaging 97 previously correct answers. It improves entailment and contradiction scores but reduces correct neutral predictions from 272 to 204. Its 3.33 percentage-point gain is exploratory: paired McNemar p=0.0106 before selection correction, Bonferroni-adjusted p=0.553 over 52 main candidates.

Oracle coverage uses the official label to select a correct configuration after generation. **It is not achieved Navigator accuracy.** This experiment does not train a Navigator. All 1,200 items have now been used for search and diagnosis; future routing claims require independent evaluation.

The grid is finite: 10 named axes at −2, −1, +1, +2; 12 existing mixtures; baseline; and 2 norm-matched random axes at 4 strengths each. It does not cover the entire continuous intervention space.

Important observations:

- Stronger interventions are often worse. Both positive and negative 2× settings have substantially more incomplete or invalid outputs.
- Precision −2× scores 554/1200 under this detailed protocol, with 390 invalid outputs, including 382 truncations. Earlier small-sample results used a different response protocol.
- Errors include excessive literalism, unsupported inference, entity and scope confusion, string interpretation, and time or modality mistakes. They do not support a single universal “loosen the rules” remedy.
- All 14 items unsolved by every configuration were manually inspected. Some exhibit clear model errors; others expose tensions between question wording and the dataset author's explanation. Official scoring was preserved throughout.
- Post-evaluation same-model diagnostic annotations can repeat the original error. They are candidate explanations, not independent adjudication or confirmed internal mechanisms.

The independent audit reconstructed all 73,200 decoded answers and scores, checked prompt consistency and intervention records, and confirmed that truncated or invalid outputs count as wrong. The source pipeline is in [`network_pilot/fullspace_detailed.py`](../network_pilot/fullspace_detailed.py), [`fullspace_analysis.py`](../network_pilot/fullspace_analysis.py), and [`audit_fullspace_detailed.py`](../network_pilot/audit_fullspace_detailed.py).

Large run outputs, benchmark downloads, model checkpoints, and local GPU-session files are excluded from this source repository. Frozen run manifests bind source and artifact hashes; moving or editing a source file requires a new experiment specification rather than silently reusing an old run.
