# Changelog

All notable changes to this project are documented in this file.

## [0.6.1] - Unreleased

### Fixed

- Embedded SeekDB HNSW writes made through `OceanbaseVectorStore.add_texts()` are immediately available to ANN search by requiring the released `pyobvector` index-refresh behavior.

### Changed

- Require `pyobvector >=0.2.29` for both standard and `pyseekdb`-extra installations.
- Validate the embedded SeekDB stack with `pylibseekdb 1.3.0.post3`, including native `pyseekdb` async-index smoke coverage and LangChain HNSW read-after-write coverage.

### Compatibility

- Continue to exclude `pylibseekdb 1.3.0.post1`, which has a separate embedded-client lifecycle hang.
