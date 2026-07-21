from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ParameterDef(BaseModel):
    name: str = Field(description="Parameter name")
    type: str = Field(description="JSON Schema type")
    description: str


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: list[ParameterDef]


class ToolDefinitions(BaseModel):
    tools: list[FunctionDef] = Field(min_length=2, max_length=4)


class ArgumentDef(BaseModel):
    name: str
    value: str = Field(description="JSON-encoded argument value")


class ToolCallArg(BaseModel):
    name: str
    arguments: list[ArgumentDef]


class ExpectedToolCalls(BaseModel):
    tool_calls: list[ToolCallArg] = Field(min_length=1, max_length=1)


class PlatformConnection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    base_url: str = "http://localhost:8080"
    workspace: str = "default"
    access_token: str | None = None


class DataDesignerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = "default/nvidia-build"
    model: str = "nvidia/nemotron-3-nano-30b-a3b"
    model_alias: str = "nemotron"
    num_records: int = Field(default=500, ge=1)


class EmbeddingCustomizationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_name: str = "humana-embedding-dataset"
    model_name: str = "humana-nemotron-embed-base"
    hf_repo_id: str = "nvidia/llama-nemotron-embed-1b-v2"
    hf_secret_name: str = "hf-token"
    epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=5e-6, gt=0)
    max_seq_length: int = Field(default=512, ge=64)


def run_data_designer_tool_calling(
    connection: PlatformConnection,
    settings: DataDesignerSettings,
    output_path: Path,
) -> Path:
    """Generate insurance tool-calling data with the official NeMo Data Designer SDK."""
    try:
        import data_designer.config as dd
        from nemo_platform import NeMoPlatform
    except ImportError as exc:
        raise RuntimeError("Install NeMo Platform/Data Designer: uv sync --extra platform") from exc

    sdk = NeMoPlatform(
        base_url=connection.base_url,
        workspace=connection.workspace,
        access_token=connection.access_token,
    )
    model_configs = [
        dd.ModelConfig(
            provider=settings.provider,
            model=settings.model,
            alias=settings.model_alias,
            inference_parameters=dd.ChatCompletionInferenceParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=2048,
            ),
        )
    ]
    builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)
    builder.add_column(
        dd.SamplerColumnConfig(
            name="domain",
            sampler_type="category",
            params={
                "values": [
                    "eligibility",
                    "benefits_lookup",
                    "prior_authorization_status",
                    "claim_status",
                    "appeal_intake",
                    "provider_lookup",
                ]
            },
        )
    )
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="tools",
            prompt=(
                "Generate 2-4 realistic read-only healthcare payer API function definitions "
                "for '{{ domain }}'. Use snake_case names and JSON Schema types. Never include "
                "medical advice, coverage decisions, real persons, or real member identifiers."
            ),
            output_format=ToolDefinitions,
            model_alias=settings.model_alias,
        )
    )
    builder.add_column(
        dd.LLMTextColumnConfig(
            name="user_query",
            prompt=(
                "Given these tools: {{ tools }}, write one realistic user query requiring "
                "exactly one tool call. All identifiers must begin with SYN-. Output only the query."
            ),
            model_alias=settings.model_alias,
        )
    )
    builder.add_column(
        dd.LLMStructuredColumnConfig(
            name="expected_tool_call",
            prompt=(
                "For '{{ user_query }}' and tools {{ tools }}, produce exactly one matching call. "
                "Arguments must be precise and all identifiers must begin with SYN-."
            ),
            output_format=ExpectedToolCalls,
            model_alias=settings.model_alias,
        )
    )

    preview = sdk.data_designer.preview(builder, num_records=min(4, settings.num_records))
    if preview.dataset.empty:
        raise RuntimeError("Data Designer preview returned no records")
    job = sdk.data_designer.create(builder, num_records=settings.num_records)
    job.wait_until_done()
    dataset = job.download_artifacts().load_dataset()
    records = [_to_openai_tool_record(row) for row in dataset.to_dict("records")]
    records = [record for record in records if len(record["messages"][1]["tool_calls"]) == 1]
    if not records:
        raise RuntimeError("Data Designer returned no valid single-tool-call records")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
    return output_path


def submit_embedding_customization(
    connection: PlatformConnection,
    settings: EmbeddingCustomizationSettings,
    dataset_directory: Path,
) -> dict[str, str]:
    """Upload triplets and submit the official NeMo Customizer embedding SFT job."""
    try:
        from nemo_platform import ConflictError, NeMoPlatform
        from nemo_platform.types.customization import (
            CustomizationJobInputParam,
            ParallelismParamsParam,
            SftTrainingParam,
        )
        from nemo_platform.types.files import HuggingfaceStorageConfigParam
    except ImportError as exc:
        raise RuntimeError("Install NeMo Platform: uv sync --extra platform") from exc

    for required in (dataset_directory / "training.jsonl", dataset_directory / "validation.jsonl"):
        if not required.exists():
            raise FileNotFoundError(required)

    sdk = NeMoPlatform(
        base_url=connection.base_url,
        workspace=connection.workspace,
        access_token=connection.access_token,
    )
    try:
        sdk.files.filesets.create(
            workspace=connection.workspace,
            name=settings.dataset_name,
            description="Humana/CMS page-grounded embedding triplets",
        )
    except ConflictError:
        pass
    sdk.files.upload(
        local_path=dataset_directory.absolute(),
        remote_path="",
        fileset=settings.dataset_name,
        workspace=connection.workspace,
    )

    try:
        sdk.files.filesets.create(
            workspace=connection.workspace,
            name=settings.model_name,
            description="NVIDIA embedding base model for Humana grounding",
            storage=HuggingfaceStorageConfigParam(
                type="huggingface",
                repo_id=settings.hf_repo_id,
                repo_type="model",
                token_secret=settings.hf_secret_name,
            ),
        )
    except ConflictError:
        pass

    try:
        base_model = sdk.models.create(
            workspace=connection.workspace,
            name=settings.model_name,
            fileset=f"{connection.workspace}/{settings.model_name}",
            trust_remote_code=True,
        )
    except ConflictError:
        base_model = sdk.models.update(
            workspace=connection.workspace,
            name=settings.model_name,
            fileset=f"{connection.workspace}/{settings.model_name}",
            trust_remote_code=True,
        )

    deadline = time.monotonic() + 120
    while not base_model.spec:
        if time.monotonic() > deadline:
            raise TimeoutError("ModelSpec was not populated within 120 seconds")
        time.sleep(2)
        base_model = sdk.models.retrieve(workspace=connection.workspace, name=settings.model_name)

    job_name = f"humana-embed-finetune-{int(time.time())}"
    job = sdk.customization.jobs.create(
        name=job_name,
        workspace=connection.workspace,
        spec=CustomizationJobInputParam(
            model=f"{connection.workspace}/{base_model.name}",
            dataset=f"fileset://{connection.workspace}/{settings.dataset_name}",
            training=SftTrainingParam(
                type="sft",
                epochs=settings.epochs,
                batch_size=settings.batch_size,
                learning_rate=settings.learning_rate,
                max_seq_length=settings.max_seq_length,
                micro_batch_size=1,
                parallelism=ParallelismParamsParam(
                    num_gpus_per_node=1,
                    num_nodes=1,
                    tensor_parallel_size=1,
                    pipeline_parallel_size=1,
                ),
            ),
        ),
    )
    return {"job_name": job.name, "output_model": job.spec.output.name}


def _to_openai_tool_record(row: dict[str, Any]) -> dict[str, Any]:
    tools = []
    for tool in row["tools"]["tools"]:
        properties = {
            parameter["name"]: {
                "type": parameter["type"],
                "description": parameter["description"],
            }
            for parameter in tool["parameters"]
        }
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": {"type": "object", "properties": properties},
                },
            }
        )

    calls = []
    for call in row["expected_tool_call"]["tool_calls"]:
        arguments = {}
        for argument in call["arguments"]:
            try:
                arguments[argument["name"]] = json.loads(argument["value"])
            except (json.JSONDecodeError, TypeError):
                arguments[argument["name"]] = argument["value"]
        calls.append(
            {
                "type": "function",
                "function": {"name": call["name"], "arguments": arguments},
            }
        )
    return {
        "messages": [
            {"role": "user", "content": row["user_query"]},
            {"role": "assistant", "content": "", "tool_calls": calls},
        ],
        "tools": tools,
    }
