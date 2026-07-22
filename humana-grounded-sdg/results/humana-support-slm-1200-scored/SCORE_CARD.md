# Humana Support SLM — Evaluation & Scores

Generated: 2026-07-22T17:55:46.716706+00:00

## See the data
1. Open: `C:\Users\yashk\HumanaDataView\index.html`
2. GitHub: https://github.com/Yash-Kavaiya/Hackathon_Cluster/tree/feat/humana-nemo-sdg/humana-grounded-sdg/results/humana-support-slm-1200
3. Scored JSONL: `support_conversations_scored.jsonl`

## Local evaluation (passed=True)
| Metric | Rate |
|---|---:|
| Turn structure | 1.0 |
| Citation valid | 1.0 |
| Grounded quote | 1.0 |
| Safety compliance | 1.0 |
| No personal PII | 1.0 |
| Unique | 1.0 |

## Scores on every record
| Field | All 1200? | Mean | Notes |
|---|---|---:|---|
| local_quality_score_0_10 | YES | 10.0 | Project metric |
| local_privacy_score_0_10 | YES | 10.0 | Project metric |
| synthetic_data_quality_score (official SQS) | field present / value null | n/a | Needs NeMo Safe Synthesizer |
| data_privacy_score (official DPS) | field present / value null | n/a | Needs NeMo Safe Synthesizer |

## Official SQS/DPS blocker
No NMP_BASE_URL/token; localhost:8080 Safe Synthesizer API unreachable. Official SQS/DPS remain null (fail-closed, never fabricated).

Official NeMo SQS/DPS are never fabricated.
