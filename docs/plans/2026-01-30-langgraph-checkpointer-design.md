# OceanBase LangGraph Checkpointer Design

## 1. Overview
This document outlines the design for implementing a LangGraph-compatible `Checkpointer` for OceanBase, replacing the deprecated `ChatMessageHistory` pattern.

## 2. Architecture

### 2.1 Core Components
*   **`OceanBaseSaver`**: Main class in `langchain_oceanbase/checkpoint/saver.py`.
*   **Dependency**: `langgraph` as an **optional dependency**.

### 2.2 Data Model (Schema)

#### Table 1: `langchain_checkpoints`
Stores state snapshots.
*   `thread_id`, `checkpoint_ns`, `checkpoint_id` (PKs)
*   `parent_checkpoint_id`
*   `type`, `checkpoint` (BLOB), `metadata` (BLOB), `created_at`

#### Table 2: `langchain_checkpoint_writes`
Stores pending writes.
*   `thread_id`, `checkpoint_ns`, `checkpoint_id`, `task_id`, `idx` (PKs)
*   `channel`, `type`, `value` (BLOB)

### 2.3 Serialization
*   Protocol: Binary serialization (Python's standard object serialization) for BLOBs.
*   Metadata columns for indexing.

## 3. Implementation Plan

### 3.1 Phase 1: Foundation
1.  Update `pyproject.toml` to add `langgraph` optional dependency.
2.  Create directory structure.

### 3.2 Phase 2: Core Logic
1.  Implement `OceanBaseSaver.__init__` and `_create_tables_if_not_exists`.
2.  Implement `put` (save checkpoint).
3.  Implement `put_writes` (save pending writes).
4.  Implement `get_tuple` (load checkpoint).
5.  Implement `list` (list history).

### 3.3 Phase 3: Migration
1.  Add deprecation warning to `OceanBaseChatMessageHistory`.
2.  Update README with migration guide.
3.  Add integration tests.
