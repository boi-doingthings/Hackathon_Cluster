import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from humana_sdg.nemo_platform_jobs import (
    DataDesignerSettings,
    PlatformConnection,
    run_data_designer_tool_calling,
)
from humana_sdg.safe_synth import SafeSynthSettings, run_safe_synthesis


def test_safe_synthesizer_builder_matches_public_platform_extra() -> None:
    pytest.importorskip("nemo_safe_synthesizer_plugin")
    from nemo_safe_synthesizer_plugin.sdk.job_builder import SafeSynthesizerJobBuilder

    assert hasattr(SafeSynthesizerJobBuilder, "with_data_source")
    assert hasattr(SafeSynthesizerJobBuilder, "with_replace_pii")
    assert hasattr(SafeSynthesizerJobBuilder, "create_job")


def test_data_designer_08_config_and_sdk_adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("data_designer.config")
    import nemo_platform

    dataset = pd.DataFrame(
        [
            {
                "domain": "claim_status",
                "tools": {
                    "tools": [
                        {
                            "name": "get_claim_status",
                            "description": "Read a synthetic claim status.",
                            "parameters": [
                                {
                                    "name": "claim_reference",
                                    "type": "string",
                                    "description": "Synthetic claim reference.",
                                    "required": True,
                                }
                            ],
                        }
                    ]
                },
                "user_query": "Check synthetic claim SYN-CLAIM-0001.",
                "expected_tool_call": {
                    "tool_calls": [
                        {
                            "name": "get_claim_status",
                            "arguments": [
                                {
                                    "name": "claim_reference",
                                    "value": '"SYN-CLAIM-0001"',
                                }
                            ],
                        }
                    ]
                },
            }
        ]
    )

    class FakeJob:
        def wait_until_done(self) -> None:
            return None

        def download_artifacts(self, path: Path):
            path.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(load_dataset=lambda: dataset)

    class FakeDataDesigner:
        def preview(self, builder, **kwargs):
            assert kwargs["workspace"] == "default"
            built = builder.build()
            assert len(built.columns) == 4
            return SimpleNamespace(dataset=dataset)

        def create(self, builder, **kwargs):
            assert kwargs["num_records"] == 1
            assert kwargs["workspace"] == "default"
            builder.build()
            return FakeJob()

    class FakePlatform:
        def __init__(self, **kwargs):
            assert kwargs["workspace"] == "default"
            self.data_designer = FakeDataDesigner()

    monkeypatch.setattr(nemo_platform, "NeMoPlatform", FakePlatform)
    output = tmp_path / "tool_calls.jsonl"
    result = run_data_designer_tool_calling(
        PlatformConnection(base_url="http://nmp.invalid", workspace="default"),
        DataDesignerSettings(num_records=1),
        output,
    )

    record = json.loads(result.read_text(encoding="utf-8"))
    assert record["messages"][1]["tool_calls"][0]["function"]["name"] == "get_claim_status"
    assert record["is_synthetic"] is True


def test_safe_synthesizer_adapter_writes_all_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("nemo_safe_synthesizer_plugin")
    import nemo_platform
    from nemo_safe_synthesizer_plugin.sdk import job_builder

    class FakeProjects:
        def create(self, **kwargs):
            return SimpleNamespace(name=kwargs["name"])

    class FakeJobs:
        def list(self, **kwargs):
            return []

    class FakeClient:
        def __init__(self, **kwargs):
            self.projects = FakeProjects()
            self.safe_synthesizer = SimpleNamespace(jobs=FakeJobs())

    class FakeJob:
        job_name = "safe-synth-test"

        def wait_for_completion(self) -> None:
            return None

        def fetch_data(self) -> pd.DataFrame:
            return pd.DataFrame([{"synthetic": True}])

        def fetch_summary(self):
            return SimpleNamespace(
                synthetic_data_quality_score=0.91,
                data_privacy_score=0.98,
                num_valid_records=1,
                num_prompts=1,
            )

        def save_report(self, path: str) -> None:
            Path(path).write_text("<html>verified</html>", encoding="utf-8")

    class FakeBuilder:
        def __init__(self, client, workspace: str):
            assert workspace == "default"

        def __getattr__(self, name):
            if name.startswith("with_") or name == "synthesize":
                return lambda *args, **kwargs: self
            raise AttributeError(name)

        def create_job(self, **kwargs):
            return FakeJob()

    monkeypatch.setattr(nemo_platform, "NeMoPlatform", FakeClient)
    monkeypatch.setattr(job_builder, "SafeSynthesizerJobBuilder", FakeBuilder)

    result = run_safe_synthesis(
        pd.DataFrame([{"seed": index} for index in range(200)]),
        tmp_path,
        SafeSynthSettings(
            base_url="http://nmp.invalid",
            workspace="default",
            access_token="test-token",
            provider_name="default/test-provider",
            hf_secret_name="hf-token",
        ),
    )

    assert result.job_name == "safe-synth-test"
    assert result.synthetic_csv.exists()
    assert result.evaluation_report.exists()
    assert result.summary_json.exists()
