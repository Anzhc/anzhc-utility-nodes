from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .lm_studio_llm_node import (
    DEFAULT_MODEL_OPTION,
    LM_STUDIO_MODELS_URL,
    call_lm_studio_chat,
    get_lm_studio_model_options,
)


PIPELINES_ROOT = Path(__file__).resolve().parent / "LLM templates"
PIPELINE_CONFIG_NAME = "pipeline.json"
TEXT_INPUT_POINTER = "<text_input>"
AGENT_CONTEXT_POINTER = "<agent_context>"
DEFAULT_PIPELINE_OPTION = "<no LM Studio pipelines found>"
MAX_SEED = 0xFFFFFFFF


def get_pipeline_options(pipelines_root: Path | None = None) -> list[str]:
    root = Path(pipelines_root) if pipelines_root is not None else PIPELINES_ROOT
    try:
        choices = [
            child.name
            for child in root.iterdir()
            if child.is_dir() and (child / PIPELINE_CONFIG_NAME).is_file()
        ]
    except OSError:
        choices = []

    choices.sort(key=str.casefold)
    return choices or [DEFAULT_PIPELINE_OPTION]


def _require_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Pipeline field '{field}' must be a non-empty string.")
    return value.strip()


def _require_non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"Pipeline field '{field}' must be a non-negative integer.")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"Pipeline field '{field}' must be a positive integer.")
    return value


def _resolve_pipeline_file(pipeline_dir: Path, relative_path: Any, field: str) -> Path:
    relative_path = _require_non_empty_string(relative_path, field)
    pipeline_root = pipeline_dir.resolve()
    resolved_path = (pipeline_root / relative_path).resolve()

    try:
        resolved_path.relative_to(pipeline_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Pipeline field '{field}' must point to a file inside '{pipeline_dir.name}'."
        ) from exc

    if not resolved_path.is_file():
        raise RuntimeError(f"Pipeline file for '{field}' was not found: {resolved_path}")
    return resolved_path


def _read_utf8(path: Path, field: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"Could not read pipeline file for '{field}': {path}") from exc


def load_pipeline(pipeline_name: str, pipelines_root: Path | None = None) -> dict[str, Any]:
    root = Path(pipelines_root) if pipelines_root is not None else PIPELINES_ROOT
    if pipeline_name == DEFAULT_PIPELINE_OPTION:
        raise RuntimeError(
            f"No LM Studio pipelines were found in {root}. Add a subfolder containing pipeline.json."
        )
    if (
        not isinstance(pipeline_name, str)
        or not pipeline_name.strip()
        or Path(pipeline_name).name != pipeline_name
        or pipeline_name in {".", ".."}
    ):
        raise RuntimeError("Pipeline selection must be an immediate LLM templates subfolder name.")

    pipeline_dir = _resolve_pipeline_file(
        root,
        f"{pipeline_name}/{PIPELINE_CONFIG_NAME}",
        "pipeline",
    ).parent
    config_path = pipeline_dir / PIPELINE_CONFIG_NAME

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Pipeline configuration is invalid JSON: {config_path}") from exc
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"Could not read pipeline configuration: {config_path}") from exc

    if not isinstance(config, dict):
        raise RuntimeError("Pipeline configuration must be a JSON object.")

    user_template_path = _resolve_pipeline_file(
        pipeline_dir,
        config.get("user_template"),
        "user_template",
    )
    user_template = _read_utf8(user_template_path, "user_template")
    if TEXT_INPUT_POINTER not in user_template:
        raise RuntimeError(f"Pipeline user template must contain {TEXT_INPUT_POINTER}.")
    if AGENT_CONTEXT_POINTER not in user_template:
        raise RuntimeError(f"Pipeline user template must contain {AGENT_CONTEXT_POINTER}.")

    raw_agents = config.get("agents")
    if not isinstance(raw_agents, list) or not raw_agents:
        raise RuntimeError("Pipeline field 'agents' must be a non-empty array.")

    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        raise RuntimeError("Pipeline field 'defaults' must be an object when provided.")
    default_max_tokens = defaults.get("max_output_tokens")

    agents: list[dict[str, Any]] = []
    agents_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_agent in enumerate(raw_agents):
        field_prefix = f"agents[{index}]"
        if not isinstance(raw_agent, dict):
            raise RuntimeError(f"Pipeline field '{field_prefix}' must be an object.")

        agent_id = _require_non_empty_string(raw_agent.get("id"), f"{field_prefix}.id")
        if agent_id in agents_by_id:
            raise RuntimeError(f"Pipeline agent id '{agent_id}' is duplicated.")

        name = _require_non_empty_string(raw_agent.get("name"), f"{field_prefix}.name")
        tier = _require_non_negative_int(raw_agent.get("tier"), f"{field_prefix}.tier")
        system_path = _resolve_pipeline_file(
            pipeline_dir,
            raw_agent.get("system"),
            f"{field_prefix}.system",
        )

        depends_on = raw_agent.get("depends_on", [])
        if not isinstance(depends_on, list) or any(
            not isinstance(dependency, str) or not dependency.strip()
            for dependency in depends_on
        ):
            raise RuntimeError(
                f"Pipeline field '{field_prefix}.depends_on' must be an array of non-empty agent ids."
            )
        depends_on = [dependency.strip() for dependency in depends_on]
        if len(set(depends_on)) != len(depends_on):
            raise RuntimeError(f"Pipeline agent '{agent_id}' contains duplicate dependencies.")

        max_output_tokens = raw_agent.get("max_output_tokens", default_max_tokens)
        max_output_tokens = _require_positive_int(
            max_output_tokens,
            f"{field_prefix}.max_output_tokens",
        )

        agent = {
            "id": agent_id,
            "name": name,
            "tier": tier,
            "depends_on": depends_on,
            "system_prompt": _read_utf8(system_path, f"{field_prefix}.system"),
            "max_output_tokens": max_output_tokens,
        }
        agents.append(agent)
        agents_by_id[agent_id] = agent

    final_agent_id = _require_non_empty_string(config.get("final_agent"), "final_agent")
    final_agent = agents_by_id.get(final_agent_id)
    if final_agent is None:
        raise RuntimeError(f"Pipeline final_agent '{final_agent_id}' does not exist.")
    if final_agent["tier"] != 0:
        raise RuntimeError(f"Pipeline final_agent '{final_agent_id}' must be in tier 0.")

    tier_zero_agents = [agent["id"] for agent in agents if agent["tier"] == 0]
    if tier_zero_agents != [final_agent_id]:
        raise RuntimeError(
            "Pipeline must contain exactly one tier-0 agent, and it must match final_agent."
        )

    for agent in agents:
        for dependency_id in agent["depends_on"]:
            dependency = agents_by_id.get(dependency_id)
            if dependency is None:
                raise RuntimeError(
                    f"Pipeline agent '{agent['id']}' depends on unknown agent '{dependency_id}'."
                )
            if dependency["tier"] <= agent["tier"]:
                raise RuntimeError(
                    f"Pipeline agent '{agent['id']}' dependency '{dependency_id}' must be in a higher tier."
                )

    return {
        "name": pipeline_name,
        "user_template": user_template,
        "final_agent": final_agent_id,
        "agents": agents,
    }


def format_agent_context(
    agent: dict[str, Any],
    agents_by_id: dict[str, dict[str, Any]],
    outputs: dict[str, str],
) -> str:
    blocks = []
    for dependency_id in agent["depends_on"]:
        dependency = agents_by_id[dependency_id]
        blocks.append(
            f"Agent: {dependency['name']} ({dependency_id})\n{outputs[dependency_id]}"
        )
    return "\n\n".join(blocks)


def build_agent_instruction(
    user_template: str,
    text_input: str,
    agent_context: str,
) -> str:
    return user_template.replace(TEXT_INPUT_POINTER, text_input).replace(
        AGENT_CONTEXT_POINTER,
        agent_context,
    )


def execute_pipeline(
    pipeline: dict[str, Any],
    *,
    model: str,
    temperature: float,
    seed: int,
    text_input: str,
) -> str:
    agents = pipeline["agents"]
    agents_by_id = {agent["id"]: agent for agent in agents}
    outputs: dict[str, str] = {}

    for agent in sorted(agents, key=lambda item: -item["tier"]):
        agent_context = format_agent_context(agent, agents_by_id, outputs)
        instruction = build_agent_instruction(
            pipeline["user_template"],
            text_input,
            agent_context,
        ).strip()

        try:
            outputs[agent["id"]] = call_lm_studio_chat(
                model=model,
                system_prompt=agent["system_prompt"],
                session_instruction=instruction,
                temperature=temperature,
                seed=seed,
                max_tokens=agent["max_output_tokens"],
            )
        except Exception as exc:
            try:
                import comfy.model_management as model_management

                if isinstance(exc, model_management.InterruptProcessingException):
                    raise
            except (ImportError, AttributeError):
                pass
            raise RuntimeError(
                f"LM Studio pipeline '{pipeline['name']}' agent "
                f"'{agent['id']}' ({agent['name']}) failed: {exc}"
            ) from exc

    return outputs[pipeline["final_agent"]]


class AnzhcLMStudioPipeline:
    @classmethod
    def INPUT_TYPES(cls):
        pipelines = get_pipeline_options()
        models = get_lm_studio_model_options()
        return {
            "required": {
                "pipeline": (
                    pipelines,
                    {
                        "default": pipelines[0],
                        "tooltip": "Pipeline folder under the Anzhc utility nodes LLM templates directory.",
                    },
                ),
                "model": (
                    models,
                    {
                        "default": models[0],
                        "tooltip": f"Active LM Studio model from {LM_STUDIO_MODELS_URL}.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "round": 0.01,
                        "tooltip": "Sampling temperature used for every pipeline agent.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": MAX_SEED,
                        "control_after_generate": True,
                        "tooltip": "Reproducible seed shared by every pipeline agent.",
                    },
                ),
                "text_input": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": "Required source text inserted at <text_input> for every pipeline agent.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "anzhc/utility"
    DESCRIPTION = (
        "Runs a tiered LM Studio workflow from an LLM templates subfolder. "
        "Higher-tier agent responses are supplied only to agents that declare them in depends_on. "
        "The configured final tier-0 agent response is returned."
    )

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # External API node: always re-execute when queued.
        return float("nan")

    def generate(
        self,
        pipeline: str,
        model: str,
        temperature: float,
        seed: int,
        text_input: str,
    ):
        if model == DEFAULT_MODEL_OPTION:
            raise RuntimeError(
                f"No active LM Studio models were found at {LM_STUDIO_MODELS_URL}. "
                "Load a model in LM Studio, then refresh the node."
            )

        loaded_pipeline = load_pipeline(pipeline)
        response = execute_pipeline(
            loaded_pipeline,
            model=model,
            temperature=float(temperature),
            seed=int(seed),
            text_input=str(text_input or ""),
        )
        return (response,)


NODE_CLASS_MAPPINGS = {
    "Anzhc LM Studio Pipeline": AnzhcLMStudioPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Anzhc LM Studio Pipeline": "LMStudio Pipeline (Anzhc)",
}
