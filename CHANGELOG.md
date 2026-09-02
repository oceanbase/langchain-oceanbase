# Changelog

All notable changes to this project are documented in this file.

## [0.6.3]

- Raise seekDB dependency floors to `pyseekdb >=1.4.0.post1,<3`, `pylibseekdb >=1.4.0,<2`, and `pyobvector >=0.2.29`.
- Stabilize the embedded seekDB hybrid-search performance test for native-runtime cold starts while retaining the tighter server-backed threshold.
- Include merged production and development dependency refreshes for LangChain/LangGraph, test tooling, `dashscope`, and `virtualenv`.

## [0.6.2]

- Bump `pylibseekdb` to `1.3.0.post4` in `pyproject.toml` for stable Unix-socket async I/O support.
- Bump CI workflows to use `actions/setup-python@v7`.
- Refresh dependency lockfile for maintenance dependency updates (`dashscope`, `aiohttp`, `virtualenv`, production/development dependency groups).

## [0.6.1]

- Embedded SeekDB HNSW writes made through `OceanbaseVectorStore.add_texts()` are immediately available to ANN search by requiring the released `pyobvector` index-refresh behavior.
- Require `pyobvector >=0.2.29` for both standard and `pyseekdb`-extra installations.
- Validate the embedded SeekDB stack with `pylibseekdb 1.3.0.post3`, including native `pyseekdb` async-index smoke coverage and LangChain HNSW read-after-write coverage.
- Continue to exclude `pylibseekdb 1.3.0.post1`, which has a separate embedded-client lifecycle hang.

## [0.6.0]

- Added `copy_thread` and `delete_for_runs` support to `OceanBaseCheckpointSaver`, including asynchronous counterparts and LangGraph checkpoint conformance coverage.
- Made asynchronous checkpointer operations non-blocking and resolved concurrent-access and performance issues.
- Raised the supported baseline to LangChain Core 1.x, LangGraph 1.x, and LangGraph Checkpoint 4.x.
- Excluded `pylibseekdb 1.3.0.post1` after identifying its embedded-client lifecycle hang.

## [0.5.2]

- Raised the embedded `pylibseekdb` floor to 1.3.0 to avoid a segfault when embedded SeekDB data directories are reused.

## [0.5.1]

- Added embedded SeekDB mode for `OceanBaseCheckpointSaver` and `OceanBaseStore` when `connection_args` provides a local `path`.
- Handled vector-search result rows without primary keys.

## [0.5.0]

- Added LangGraph Store support through `OceanBaseStore`, including semantic search, TTL handling, and embedded SeekDB coverage.
- Moved embedded SeekDB dependencies into the optional `pyseekdb` extra so standard installations do not require the native embedded runtime.
