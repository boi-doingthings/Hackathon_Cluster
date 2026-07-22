from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_member_eligibility",
            "description": "Check eligibility for a synthetic member reference on a service date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "member_reference": {"type": "string"},
                    "service_date": {"type": "string", "description": "ISO-8601 date"},
                },
                "required": ["member_reference", "service_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_prior_authorization_status",
            "description": "Get status for a synthetic prior authorization reference.",
            "parameters": {
                "type": "object",
                "properties": {"authorization_reference": {"type": "string"}},
                "required": ["authorization_reference"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_claim_status",
            "description": "Get status for a synthetic claim reference.",
            "parameters": {
                "type": "object",
                "properties": {"claim_reference": {"type": "string"}},
                "required": ["claim_reference"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_covered_benefits",
            "description": "Search plan benefits without making a coverage determination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_reference": {"type": "string"},
                    "service_code": {"type": "string"},
                },
                "required": ["plan_reference", "service_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_appeal_intake",
            "description": (
                "Open an appeal intake record for synthetic references; does not decide the appeal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "case_reference": {"type": "string"},
                    "decision_reference": {"type": "string"},
                },
                "required": ["case_reference", "decision_reference"],
            },
        },
    },
]


SCENARIOS = (
    (
        "Check whether synthetic member reference {member} is eligible on 2030-01-15.",
        "check_member_eligibility",
        {"member_reference": "{member}", "service_date": "2030-01-15"},
    ),
    (
        "What is the status of synthetic authorization {auth}?",
        "get_prior_authorization_status",
        {"authorization_reference": "{auth}"},
    ),
    (
        "Look up synthetic claim {claim}.",
        "get_claim_status",
        {"claim_reference": "{claim}"},
    ),
    (
        "Search synthetic plan {plan} for service code SYN-SVC-100.",
        "search_covered_benefits",
        {"plan_reference": "{plan}", "service_code": "SYN-SVC-100"},
    ),
    (
        "Open appeal intake for synthetic case {case} and decision {decision}.",
        "submit_appeal_intake",
        {"case_reference": "{case}", "decision_reference": "{decision}"},
    ),
)


def generate_tool_call_records(count: int) -> list[dict]:
    if count < 1:
        raise ValueError("count must be positive")

    records = []
    for index in range(count):
        template, tool_name, arguments_template = SCENARIOS[index % len(SCENARIOS)]
        references = {
            "member": f"SYN-MEMBER-{index:06d}",
            "auth": f"SYN-AUTH-{index:06d}",
            "claim": f"SYN-CLAIM-{index:06d}",
            "plan": f"SYN-PLAN-{index:06d}",
            "case": f"SYN-CASE-{index:06d}",
            "decision": f"SYN-DECISION-{index:06d}",
        }
        arguments = {key: value.format(**references) for key, value in arguments_template.items()}
        records.append(
            {
                "messages": [
                    {"role": "user", "content": template.format(**references)},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {"name": tool_name, "arguments": arguments},
                            }
                        ],
                    },
                ],
                "tools": deepcopy(TOOLS),
                "is_synthetic": True,
            }
        )
    return records


def write_tool_call_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
