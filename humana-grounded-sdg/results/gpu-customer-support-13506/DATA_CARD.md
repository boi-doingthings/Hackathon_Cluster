# Humana customer-support synthetic transcript dataset

- Records: 1200
- Turns: 7200
- Use cases: 10, balanced at 120 records each
- Grounding: public Humana/CMS citations with exact grounded excerpts
- Privacy: synthetic-only coarse profiles; PII findings: 0
- Intended use: customer-support SLM training and evaluation preparation
- Prohibited use: member-level decisions, coverage/claim/eligibility outcomes, authorization decisions, or medical advice
- Human verification: required for every record
- Executed commit: `350905ddc9e1e31c89962b436a5bcd2882b62358`
- Slurm job: `13506` (`COMPLETED`, `0:0`, GPU node `gpu004`)

See `nemo-service-provenance.json` for the fail-closed SQS/DPS and NeMo Evaluator service status.
