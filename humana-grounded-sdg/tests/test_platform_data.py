from humana_sdg.embedding import build_embedding_triplets
from humana_sdg.generate import generate_deterministic_records
from humana_sdg.models import GroundingChunk
from humana_sdg.safe_synth import records_to_safe_synth_frame
from humana_sdg.tool_data import generate_tool_call_records


def _chunk(source_id: str, category: str, page: int) -> GroundingChunk:
    return GroundingChunk(
        chunk_id=f"{source_id}-{page}",
        source_id=source_id,
        source_title=f"Guide for {category}",
        source_url="https://www.cms.gov/reference.pdf",
        source_sha256="a" * 64,
        publisher="CMS",
        year=2026,
        category=category,
        page=page,
        text=(
            f"This {category.replace('_', ' ')} guidance explains the documented operational "
            "rule for providers. Users must verify current requirements in the cited official "
            "source before taking action."
        ),
        citation_label=f"Guide for {category}, p. {page}",
    )


def test_safe_synth_frame_contains_only_governed_columns() -> None:
    records = generate_deterministic_records([_chunk("one", "claims_billing", 1)])

    frame = records_to_safe_synth_frame(records)

    assert list(frame.columns) == [
        "record_id",
        "task_type",
        "question",
        "answer",
        "citation_source_ids",
        "citation_pages",
        "is_synthetic",
        "disclaimer",
    ]
    assert frame.iloc[0]["citation_source_ids"] == "one"


def test_embedding_triplets_have_distinct_positive_and_negative_documents() -> None:
    chunks = [
        _chunk("one", "claims_billing", 1),
        _chunk("two", "prior_authorization", 2),
        _chunk("three", "appeals_grievances", 3),
    ]

    triplets = build_embedding_triplets(chunks)

    assert len(triplets) == 3
    assert all(item.pos_doc not in item.neg_doc for item in triplets)
    assert all(len(item.neg_doc) == 1 for item in triplets)


def test_tool_call_records_are_valid_openai_single_call_examples() -> None:
    records = generate_tool_call_records(12)

    assert len(records) == 12
    for record in records:
        tool_names = {tool["function"]["name"] for tool in record["tools"]}
        calls = record["messages"][1]["tool_calls"]
        assert len(calls) == 1
        assert calls[0]["function"]["name"] in tool_names
        assert "SYN-" in str(calls[0]["function"]["arguments"])
