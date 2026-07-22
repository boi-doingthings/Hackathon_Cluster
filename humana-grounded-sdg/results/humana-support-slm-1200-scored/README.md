# Humana Support SLM — Scored Viewer Pack

This folder makes the 1,200 generated conversations **visible and scored**.

## Open the data
- Browser viewer: open [`index.html`](./index.html)
- Full scored dataset: [`support_conversations_scored.jsonl`](./support_conversations_scored.jsonl)
- Per-record score table: [`per_record_scores.json`](./per_record_scores.json)
- Evaluation report: [`full_evaluation_report.json`](./full_evaluation_report.json)
- Score card: [`SCORE_CARD.md`](./SCORE_CARD.md)

## Score fields on every record
| Field | Meaning |
|---|---|
| `scores.local_quality_score_0_10` | Project quality score (0-10) from grounding/structure/safety gates |
| `scores.local_privacy_score_0_10` | Project privacy score (0-10) from personal-PII/synthetic flags |
| `scores.synthetic_data_quality_score` | **Official NeMo SQS** — null until Safe Synthesizer service is available |
| `scores.data_privacy_score` | **Official NeMo DPS** — null until Safe Synthesizer service is available |

Official NeMo SQS/DPS are never fabricated. See `full_evaluation_report.json` → `official_nemo_safe_synthesizer`.

## Local evaluation result
All local gates passed on 1,200/1,200 conversations (citation, grounding, structure, safety, personal-PII, uniqueness).
