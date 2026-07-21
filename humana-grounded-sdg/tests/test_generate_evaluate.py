from humana_sdg.evaluate import EvaluationThresholds, evaluate_records
from humana_sdg.generate import generate_deterministic_records
from humana_sdg.models import GroundingChunk


def _chunk() -> GroundingChunk:
    return GroundingChunk(
        chunk_id="chunk-1",
        source_id="humana_prior_auth_2026_07",
        source_title="Humana Medicare Prior Authorization List",
        source_url="https://assets.humana.com/is/content/humana/referencepdf",
        source_sha256="a" * 64,
        publisher="Humana",
        year=2026,
        category="prior_authorization",
        page=7,
        text=(
            "Prior authorization is required for selected services before they are furnished. "
            "Providers should verify current requirements using the plan's official lookup resources."
        ),
        citation_label="Humana Medicare Prior Authorization List, p. 7",
    )


def test_deterministic_generator_is_grounded_and_reproducible() -> None:
    first = generate_deterministic_records([_chunk()], records_per_chunk=2)
    second = generate_deterministic_records([_chunk()], records_per_chunk=2)

    assert first == second
    assert len(first) == 2
    assert all(record.task_type == "prior_authorization" for record in first)
    assert all(record.citations[0].quote in record.answer for record in first)
    assert all(record.is_synthetic for record in first)


def test_evaluation_passes_fully_grounded_records() -> None:
    records = generate_deterministic_records([_chunk()], records_per_chunk=2)

    report = evaluate_records(records, [_chunk()], EvaluationThresholds())

    assert report.passed
    assert report.citation_valid_rate == 1.0
    assert report.grounded_quote_rate == 1.0
    assert report.pii_findings == 0


def test_evaluation_rejects_unsupported_answer() -> None:
    record = generate_deterministic_records([_chunk()])[0]
    record.answer = "Humana guarantees approval for every service without review."

    report = evaluate_records([record], [_chunk()], EvaluationThresholds())

    assert not report.passed
    assert report.answer_support_rate == 0.0


def test_evaluation_distinguishes_personal_and_public_contact_email() -> None:
    personal = generate_deterministic_records([_chunk()])[0]
    personal.answer += " Send the member record to person@example.com."
    personal_report = evaluate_records([personal], [_chunk()], EvaluationThresholds())
    assert personal_report.pii_findings == 1
    assert not personal_report.passed

    public = generate_deterministic_records([_chunk()])[0]
    public.answer += " Public agency contact: support@jfs.ohio.gov."
    public_report = evaluate_records([public], [_chunk()], EvaluationThresholds())
    assert public_report.pii_findings == 0
    assert public_report.passed
