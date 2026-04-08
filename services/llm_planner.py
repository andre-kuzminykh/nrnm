"""LLM-driven planner — translates a natural-language goal into a
StructuredPlan that can be compiled by `services.graph_runtime`.

## Трассируемость
Feature: Telegram AI Platform v1.1
Requirements: FR-18, FR-19, NFR-14, Rule 3

The contract:

1. **Input** — `goal: str`, `available_tools: tuple[str, ...]`,
   optional `attached_memory` summary.
2. **Output** — `StructuredPlan` with steps, dependency edges
   (`depends_on`), parallel groups, and optional conditional edges.
   Tool bindings are constrained to `available_tools` — the planner
   must NEVER hallucinate a tool name (FR-13/Rule 3 enforcement).

When `config.LLM_API_KEY` is empty (offline dev / CI / first boot
before secrets), the planner returns a deterministic stub plan whose
shape still satisfies all FR-18..22 contracts (NFR-14).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Iterable

import config

logger = logging.getLogger(__name__)


# ── Public types ─────────────────────────────────────────────────


@dataclass
class PlanStep:
    """One node in the structured plan."""
    id: str
    description: str
    tool: str | None  # None == pure LLM reasoning step
    tool_args: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    expected_result: str = ""
    acceptance_criteria: str = ""


@dataclass
class ConditionalEdge:
    """An if/else branching point.

    `from_step` produces a value (or its critic verdict does); the
    runtime evaluates `condition` against the step result and routes
    to either `true_target` or `false_target`.
    """
    from_step: str
    condition: str
    true_target: str
    false_target: str


@dataclass
class StructuredPlan:
    plan_id: str
    goal: str
    steps: list[PlanStep]
    parallel_groups: list[list[str]] = field(default_factory=list)
    conditional_edges: list[ConditionalEdge] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def step_ids(self) -> list[str]:
        return [s.id for s in self.steps]


# ── Public API ───────────────────────────────────────────────────


def build_plan(
    goal: str,
    available_tools: Iterable[str] = (),
    *,
    attached_memory: list | None = None,
    mcp_catalog: list | None = None,
) -> StructuredPlan:
    """FR-18 entry point. Returns a StructuredPlan ready for
    compilation by `services.graph_runtime.compile_plan`.

    Parameters:
    - `goal` — natural-language user objective.
    - `available_tools` — legacy tuple of tool names; still honoured
      as the strict whitelist the planner must stay inside.
    - `attached_memory` — resolved memory objects from `[контекст]`
      refs in the goal (FR-14).
    - `mcp_catalog` — FR-25: list of `MCPEntry` objects for the
      active domain. When supplied, the LLM prompt includes each
      entry's description so the model can pick tools by capability
      instead of just name. `available_tools` is still enforced as
      the hard whitelist — names outside it are stripped post-hoc.

    Routing:
    - If `config.LLM_API_KEY` is set, call OpenAI with a strict
      JSON-schema prompt. On any failure (network, parse, validation),
      degrade to the stub.
    - Otherwise, return the deterministic stub plan immediately.
    """
    tools = tuple(available_tools)
    catalog = list(mcp_catalog or [])
    if config.LLM_API_KEY:
        try:
            plan = _build_plan_via_llm(goal, tools, attached_memory or [], catalog)
            logger.info(
                "planner(openai): %d steps, %d parallel groups, %d conditional edges | tools=%s | goal=%r",
                len(plan.steps), len(plan.parallel_groups),
                len(plan.conditional_edges), list(tools), goal[:80],
            )
            return plan
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM planner failed, falling back to stub: %s", exc)
    plan = _build_plan_stub(goal, tools, attached_memory or [])
    logger.info(
        "planner(stub): %d steps, tools=%s | goal=%r",
        len(plan.steps), list(tools), goal[:80],
    )
    return plan


# ── LLM provider ─────────────────────────────────────────────────


_PLAN_SYSTEM_PROMPT = """\
You are a planning assistant. Your job: turn a user goal into a
structured plan as JSON. Each step is one concrete action. You may
call ONLY the tools the user gives you — never invent a new tool name.

Output schema (return a single JSON object, no prose, no markdown):

{
  "steps": [
    {
      "id": "step-1",
      "description": "one short imperative sentence",
      "tool": "web_search" | "pdf_parser" | null,
      "tool_args": { "...": "..." },
      "depends_on": ["step-id", ...],
      "expected_result": "what success looks like",
      "acceptance_criteria": "how to recognise it"
    }
  ],
  "parallel_groups": [["step-2", "step-3"]],
  "conditional_edges": [
    {
      "from_step": "step-2",
      "condition": "result is non-empty",
      "true_target": "step-4",
      "false_target": "step-5"
    }
  ]
}

Rules:
- 2..6 steps. Keep it tight.
- depends_on edges form a DAG (no cycles).
- A step in a parallel_groups list must NOT depend on its siblings.
- conditional_edges is optional — only when there's a real branching point.
- tool=null means a reasoning / synthesis step done by the LLM itself.
"""


def _build_plan_via_llm(
    goal: str,
    available_tools: tuple[str, ...],
    attached_memory: list,
    mcp_catalog: list | None = None,
) -> StructuredPlan:
    from openai import OpenAI  # local import: only needed in this path

    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)

    # FR-25: include MCP descriptions when the catalog is supplied so
    # the LLM picks tools by capability, not just by name. Falls back
    # gracefully to bare tool names when no catalog is provided.
    if mcp_catalog:
        tool_lines = []
        for entry in mcp_catalog:
            tool_lines.append(
                f"- {entry.name}: {entry.description}"
            )
        tools_block = "Available tools (with descriptions):\n" + "\n".join(tool_lines)
    else:
        tools_block = (
            f"Available tools: {list(available_tools) or '(none — reasoning only)'}"
        )

    user_prompt = (
        f"Goal: {goal}\n"
        f"{tools_block}\n"
        f"Attached memory: "
        f"{[getattr(o, 'filename', '?') for o in attached_memory] or '(none)'}\n\n"
        "Return the JSON plan."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    plan = _parse_plan_dict(data, goal=goal, available_tools=available_tools)
    plan.metadata = {"provider": "openai", "model": "gpt-4o-mini"}
    return plan


def _parse_plan_dict(
    data: dict,
    *,
    goal: str,
    available_tools: tuple[str, ...],
) -> StructuredPlan:
    """Validate and coerce the LLM output into our dataclasses.

    The validator is the FR-13 enforcement seam — any tool name not in
    `available_tools` is silently rewritten to None (reasoning step) so
    a hallucinating model can't escape the whitelist.
    """
    raw_steps = data.get("steps") or []
    if not raw_steps:
        raise ValueError("planner returned 0 steps")

    steps: list[PlanStep] = []
    for i, raw in enumerate(raw_steps):
        sid = str(raw.get("id") or f"step-{i + 1}")
        tool = raw.get("tool")
        if tool not in (None, *available_tools):
            logger.info("planner stripped disallowed tool %r from %s", tool, sid)
            tool = None
        steps.append(PlanStep(
            id=sid,
            description=str(raw.get("description") or sid),
            tool=tool,
            tool_args=dict(raw.get("tool_args") or {}),
            depends_on=[str(d) for d in (raw.get("depends_on") or [])],
            expected_result=str(raw.get("expected_result") or "step output"),
            acceptance_criteria=str(raw.get("acceptance_criteria") or "non-empty result"),
        ))

    parallel_groups = [
        [str(s) for s in group]
        for group in (data.get("parallel_groups") or [])
        if group
    ]
    conditional_edges = [
        ConditionalEdge(
            from_step=str(c["from_step"]),
            condition=str(c.get("condition") or "true"),
            true_target=str(c["true_target"]),
            false_target=str(c["false_target"]),
        )
        for c in (data.get("conditional_edges") or [])
        if "from_step" in c and "true_target" in c and "false_target" in c
    ]
    return StructuredPlan(
        plan_id=uuid.uuid4().hex[:12],
        goal=goal,
        steps=steps,
        parallel_groups=parallel_groups,
        conditional_edges=conditional_edges,
    )


# ── Stub provider (NFR-14) ───────────────────────────────────────


def _build_plan_stub(
    goal: str,
    available_tools: tuple[str, ...],
    attached_memory: list,
) -> StructuredPlan:
    """Deterministic stub used when LLM is unavailable. Returns a
    plausible plan that exercises every shape FR-19 promises:
    - sequential dependency (collect -> work -> synthesise)
    - parallel group (when 2+ tools available, search + parse fan out)
    - empty conditional_edges (the field exists; LLM path can populate)
    """
    use_search = "web_search" in available_tools and _looks_like_research(goal)
    use_parser = "pdf_parser" in available_tools and bool(attached_memory)

    steps: list[PlanStep] = [
        PlanStep(
            id="step-1",
            description=f"Расшифровать цель и собрать вход: {goal[:80]}",
            tool=None,
            depends_on=[],
            expected_result="goal parsed and inputs catalogued",
            acceptance_criteria="non-empty goal, attached memory enumerated",
        ),
    ]
    parallel_group: list[str] = []
    if use_search:
        steps.append(PlanStep(
            id="step-2",
            description="Поискать релевантные источники в вебе",
            tool="web_search",
            tool_args={"query": goal},
            depends_on=["step-1"],
            expected_result="non-empty hits list",
            acceptance_criteria="at least one hit",
        ))
        parallel_group.append("step-2")
    if use_parser:
        steps.append(PlanStep(
            id="step-3",
            description="Извлечь контент из подключённой Памяти",
            tool="pdf_parser",
            tool_args={"content": ""},
            depends_on=["step-1"],
            expected_result="extracted text",
            acceptance_criteria="text length > 0",
        ))
        parallel_group.append("step-3")

    last_id = steps[-1].id
    deps_for_synth = parallel_group or [last_id]
    steps.append(PlanStep(
        id="step-final",
        description="Свести найденное в финальный ответ",
        tool=None,
        depends_on=deps_for_synth,
        expected_result="final answer addressing the goal",
        acceptance_criteria="answer references inputs",
    ))

    parallel_groups: list[list[str]] = [parallel_group] if len(parallel_group) > 1 else []
    return StructuredPlan(
        plan_id=uuid.uuid4().hex[:12],
        goal=goal,
        steps=steps,
        parallel_groups=parallel_groups,
        conditional_edges=[],
        metadata={"provider": "stub"},
    )


_RESEARCH_HINTS = re.compile(
    r"(?:найди|найти|поищ|поиск|search|research|сравни|compare|свод|обзор|news)",
    re.IGNORECASE,
)


def _looks_like_research(goal: str) -> bool:
    return bool(_RESEARCH_HINTS.search(goal or ""))
