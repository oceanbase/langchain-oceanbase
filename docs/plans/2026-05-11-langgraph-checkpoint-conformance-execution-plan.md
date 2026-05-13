# OceanBase LangGraph Checkpoint Conformance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `OceanBaseCheckpointSaver` pass the current LangGraph checkpoint conformance base suite and align the public OceanBase checkpoint surface with the current `BaseCheckpointSaver` contract.

**Architecture:** Keep `langchain_oceanbase/checkpointer.py` as the single production checkpoint implementation and make it conform to the upstream async-first LangGraph contract by adding async methods, fixing metadata/blob/list semantics, and wiring the official conformance suite into this repo. Treat `langchain_oceanbase/checkpoint/saver.py` as legacy compatibility code and either deprecate it clearly or remove it from the recommended public path so users are not routed to the non-conformant implementation.

**Tech Stack:** Python 3.11+, Poetry, LangGraph, `langgraph-checkpoint`, `langgraph-checkpoint-conformance`, `pyobvector`, SQLAlchemy, pytest, pytest-asyncio

---

## Scope Clarification

This repo currently has:

- `langchain_oceanbase/checkpointer.py`: the main LangGraph checkpoint saver
- `langchain_oceanbase/checkpoint/saver.py`: an older legacy checkpoint saver

This repo does **not** currently contain a LangGraph Store API implementation based on `langgraph.store`. If "store saver" means the LangGraph Store abstraction, that is a separate project and should not block checkpoint conformance. This plan covers:

- full **checkpoint saver** base conformance
- cleanup of the **legacy saver** surface so it does not masquerade as the standard path
- optional extended capability work only after base conformance is green

## File Map

**Primary code**

- Modify: `langchain_oceanbase/checkpointer.py`
- Modify: `langchain_oceanbase/__init__.py`
- Modify: `langchain_oceanbase/checkpoint/__init__.py`
- Modify: `langchain_oceanbase/checkpoint/saver.py`

**Tests**

- Modify: `tests/integration_tests/test_checkpointer.py`
- Create: `tests/integration_tests/test_checkpoint_conformance.py`

**Packaging and docs**

- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `docs/migration_guide.md`

## Execution Order

1. Add the official conformance harness to the repo and get a red baseline.
2. Make `OceanBaseCheckpointSaver` pass the 5 required base capabilities.
3. Clean up the legacy saver surface so standard imports and docs point to the conformant implementation.
4. Decide whether to implement optional extended capabilities now or explicitly defer them.
5. Update docs and CI/test instructions.

### Task 1: Add an Upstream Conformance Harness

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/integration_tests/test_checkpoint_conformance.py`

- [ ] **Step 1: Add the upstream conformance package to test dependencies**

```toml
[tool.poetry.group.test.dependencies]
langgraph-checkpoint-conformance = ">=0.1.0"
```

Notes:
- Keep this in the test group, not the runtime dependencies.
- If the lockfile needs regeneration, do it in the implementation PR, not in the planning-only commit.

- [ ] **Step 2: Register a test factory that creates a fresh saver per capability suite**

```python
import pytest
from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.checkpoint.conformance.report import ProgressCallbacks

from langchain_oceanbase import OceanBaseCheckpointSaver


@checkpointer_test(name="OceanBaseCheckpointSaver")
async def oceanbase_checkpointer():
    saver = OceanBaseCheckpointSaver(connection_args=get_connection_args())
    saver.setup()
    try:
        yield saver
    finally:
        # no explicit close hook today
        pass


@pytest.mark.asyncio
async def test_checkpoint_conformance_base():
    report = await validate(
        oceanbase_checkpointer,
        capabilities={"put", "put_writes", "get_tuple", "list", "delete_thread"},
        progress=ProgressCallbacks.verbose(),
    )
    report.print_report()
    assert report.passed_all_base()
```

- [ ] **Step 3: Run only the new conformance test to capture the real failure baseline**

Run: `poetry run pytest tests/integration_tests/test_checkpoint_conformance.py -q`

Expected:
- initial failure on missing async methods
- after async wrappers are added, failures should narrow to semantic mismatches in `put`, `get_tuple`, `list`, and metadata handling

### Task 2: Make the Primary Saver Expose the Current BaseCheckpointSaver Contract

**Files:**
- Modify: `langchain_oceanbase/checkpointer.py`
- Test: `tests/integration_tests/test_checkpoint_conformance.py`

- [ ] **Step 1: Import current upstream metadata helpers and version constant**

```python
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    LATEST_VERSION,
    get_checkpoint_id,
    get_serializable_checkpoint_metadata,
)
```

- [ ] **Step 2: Add async wrappers for all 5 required capabilities**

```python
async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
    return self.get_tuple(config)


async def alist(
    self,
    config: Optional[RunnableConfig],
    *,
    filter: Optional[Dict[str, Any]] = None,
    before: Optional[RunnableConfig] = None,
    limit: Optional[int] = None,
):
    for item in self.list(config, filter=filter, before=before, limit=limit):
        yield item


async def aput(
    self,
    config: RunnableConfig,
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata,
    new_versions: ChannelVersions,
) -> RunnableConfig:
    return self.put(config, checkpoint, metadata, new_versions)


async def aput_writes(
    self,
    config: RunnableConfig,
    writes: Sequence[tuple[str, Any]],
    task_id: str,
    task_path: str = "",
) -> None:
    self.put_writes(config, writes, task_id, task_path)


async def adelete_thread(self, thread_id: str) -> None:
    self.delete_thread(thread_id)
```

Notes:
- This is the minimum required for capability detection.
- Do not implement fake async DB I/O yet; conformance only requires the async surface to exist.

- [ ] **Step 3: Run the base conformance suite again to confirm capability detection is fixed**

Run: `poetry run pytest tests/integration_tests/test_checkpoint_conformance.py -q`

Expected:
- capability detection should now include all 5 base capabilities
- failures should now be semantic, not `NotImplementedError`

### Task 3: Fix Conformance Semantics in put/get/list/metadata

**Files:**
- Modify: `langchain_oceanbase/checkpointer.py`
- Modify: `tests/integration_tests/test_checkpointer.py`
- Test: `tests/integration_tests/test_checkpoint_conformance.py`

- [ ] **Step 1: Replace `_prepare_metadata()` with upstream-compatible metadata serialization**

```python
def _prepare_metadata(
    self, config: RunnableConfig, metadata: CheckpointMetadata
) -> Dict[str, Any]:
    return dict(get_serializable_checkpoint_metadata(config, metadata))
```

Why:
- preserves `run_id`
- merges supported config metadata
- strips null characters the same way LangGraph does
- removes internal config keys and `writes`

- [ ] **Step 2: Store explicit empty markers for versioned channels without inline/blob values**

```python
for channel, version in new_versions.items():
    if channel in blob_values:
        type_str, blob = self.serde.dumps_typed(blob_values[channel])
    else:
        type_str, blob = "empty", b""

    conn.execute(
        text(UPSERT_CHECKPOINT_BLOBS_SQL),
        {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "channel": channel,
            "version": str(version),
            "type": type_str,
            "blob": blob,
        },
    )
```

Why:
- preserves incremental channel version history
- allows later checkpoints to distinguish "no stored blob for this version" from "channel intentionally empty/no inline payload"

- [ ] **Step 3: Stop hardcoding stale checkpoint version defaults**

```python
checkpoint: Checkpoint = {
    "v": checkpoint_data.get("v", LATEST_VERSION),
    "id": checkpoint_data.get("id", checkpoint_id),
    ...
}
```

Notes:
- upstream `generate_checkpoint()` still uses `v=1` today, so this is mostly a correctness cleanup
- do not mutate stored checkpoint versions on write unless there is a real migration requirement

- [ ] **Step 4: Make `list()` metadata filtering compare JSON values correctly**

Preferred direction:

```python
conditions.append(
    f"JSON_UNQUOTE(JSON_EXTRACT(c.metadata, '$.{key}')) = :{param_name}"
)
params[param_name] = str(value)
```

Fallback if OceanBase JSON behavior differs:
- normalize only primitive values in SQL
- for non-primitives, narrow SQL by thread/namespace and finish filtering in Python

Guardrails:
- keep `thread_id`, namespace, `before`, and `limit` in SQL
- only move custom metadata comparison in-memory if OceanBase JSON semantics are inconsistent

- [ ] **Step 5: Preserve existing namespace and parent semantics while tightening ordering assumptions**

Checks to satisfy:
- `get_tuple(config_without_checkpoint_id)` returns newest checkpoint in the namespace
- `list()` returns newest-first by monotonic checkpoint ID
- `before` excludes the cursor checkpoint and all newer checkpoints
- namespace isolation remains strict in both checkpoints and writes

- [ ] **Step 6: Add direct regression tests for the known semantic gaps**

Add focused tests to `tests/integration_tests/test_checkpointer.py` for:

```python
def test_put_preserves_run_id(...): ...
def test_put_incremental_channel_update(...): ...
def test_put_channel_removed(...): ...
def test_list_filter_by_source(...): ...
def test_list_before_cursor(...): ...
def test_put_writes_namespace_isolation(...): ...
```

Why:
- the upstream conformance suite is the acceptance gate
- local targeted tests make debugging faster than re-reading a full 50+ test report

- [ ] **Step 7: Re-run targeted tests until the 5 base capability suites pass**

Run:

```bash
poetry run pytest tests/integration_tests/test_checkpointer.py -q
poetry run pytest tests/integration_tests/test_checkpoint_conformance.py -q
```

Expected:
- `test_checkpoint_conformance_base` passes
- no regressions in the existing integration tests

### Task 4: Resolve the Legacy Saver Surface

**Files:**
- Modify: `langchain_oceanbase/checkpoint/saver.py`
- Modify: `langchain_oceanbase/checkpoint/__init__.py`
- Modify: `README.md`
- Modify: `docs/migration_guide.md`

- [ ] **Step 1: Decide the product stance**

Recommended stance:
- `OceanBaseCheckpointSaver` is the only standard LangGraph checkpoint saver
- `OceanBaseSaver` is legacy and not recommended for new code

- [ ] **Step 2: Mark the legacy saver as deprecated in code and docs**

Example:

```python
import warnings

warnings.warn(
    "OceanBaseSaver is deprecated; use OceanBaseCheckpointSaver instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

Notes:
- keep the class importable short-term to avoid abrupt breakage
- do not spend time making the pickle/f-string implementation conformant

- [ ] **Step 3: Update public docs so examples and migration text only point to the primary saver**

README updates should state:
- install instructions for LangGraph checkpoint use
- `checkpointer.setup()` requirement
- `OceanBaseCheckpointSaver` is the supported path
- `langchain_oceanbase.checkpoint.OceanBaseSaver` is legacy

- [ ] **Step 4: Add a lightweight test that the legacy import still resolves but warns**

```python
def test_legacy_saver_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning):
        OceanBaseSaver(...)
```

Only add this if the constructor can be exercised without fragile external state. If not, test the import surface and document the deprecation instead.

### Task 5: Optional Extended Capability Phase

**Files:**
- Modify: `langchain_oceanbase/checkpointer.py`
- Extend: `tests/integration_tests/test_checkpoint_conformance.py`

- [ ] **Step 1: Treat `prune` as the only realistic extended capability for this cycle**

Reasoning:
- upstream marks `prune`, `copy_thread`, and `delete_for_runs` as optional
- `prune` is small and already part of the modern `BaseCheckpointSaver` surface
- `copy_thread` and `delete_for_runs` need more product semantics and are easy to get subtly wrong

- [ ] **Step 2: If implemented now, add sync + async prune only**

```python
def prune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
    ...

async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
    self.prune(thread_ids, strategy=strategy)
```

Minimum acceptable semantics:
- `strategy="keep_latest"` keeps the newest checkpoint per `(thread_id, checkpoint_ns)`
- keep the surviving checkpoint's pending writes
- delete all older checkpoints, writes, and no-longer-referenced blobs for that thread/namespace
- `strategy="delete"` removes all rows for the given threads

- [ ] **Step 3: Explicitly defer `copy_thread` and `delete_for_runs` unless there is a product requirement**

Document in the PR:
- not required for base conformance
- better left unimplemented than shipped with unsafe semantics

- [ ] **Step 4: If prune is implemented, extend the conformance test to include it**

```python
report = await validate(
    oceanbase_checkpointer,
    capabilities={"put", "put_writes", "get_tuple", "list", "delete_thread", "prune"},
)
```

### Task 6: Final Verification and Release Readiness

**Files:**
- Verify: `langchain_oceanbase/checkpointer.py`
- Verify: `tests/integration_tests/test_checkpoint_conformance.py`
- Verify: docs touched above

- [ ] **Step 1: Run the full targeted verification stack**

Run:

```bash
poetry run pytest tests/integration_tests/test_checkpointer.py -q
poetry run pytest tests/integration_tests/test_checkpoint_conformance.py -q
poetry run ruff check langchain_oceanbase tests
poetry run mypy langchain_oceanbase
```

- [ ] **Step 2: If the local OceanBase environment is stable, run the broader integration test slice**

Run:

```bash
poetry run pytest tests/integration_tests/ -q
```

Notes:
- if broader integration tests are flaky or environment-dependent, keep the release gate focused on checkpoint tests plus lint/typecheck

- [ ] **Step 3: Record the acceptance criteria in the PR description**

Required acceptance criteria:
- base conformance suite passes
- existing checkpoint integration tests pass
- README and migration docs point to the conformant implementation
- legacy saver is clearly deprecated and no longer presented as the standard path

## Recommended Delivery Split

**PR 1: Base conformance**
- async methods
- metadata fix
- blob/version fix
- list/filter cleanup
- official conformance test

**PR 2: Surface cleanup**
- deprecate `OceanBaseSaver`
- doc cleanup
- optional warning test

**PR 3: Optional extensions**
- `prune` only, if desired

This split keeps the risky storage-behavior changes isolated from doc and deprecation work.

## Risks and Decision Points

- **No LangGraph Store implementation exists here today.**
  If you want `langgraph.store` support, plan a separate workstream instead of burying it inside checkpoint conformance.

- **Async wrappers are enough for conformance detection, but not ideal long-term I/O behavior.**
  They should be accepted for this cycle; true async DB support can be a later improvement.

- **Metadata filtering may be the only OceanBase-specific query trap.**
  If `JSON_EXTRACT` behavior is inconsistent, prefer correctness first and push complex metadata matching into Python after narrowing rows in SQL.

- **Do not upgrade the legacy saver into a second standard implementation.**
  That path is insecure and divergent. Deprecate it instead.

## First Implementation Move

Start with Task 1 and Task 2 in one branch. The first meaningful checkpoint is:

1. add `langgraph-checkpoint-conformance`
2. add `tests/integration_tests/test_checkpoint_conformance.py`
3. add async wrappers in `langchain_oceanbase/checkpointer.py`
4. run the conformance test and capture the next set of real failures

That gives the shortest path from analysis to a trustworthy red/green loop.
