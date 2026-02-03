# Journal - sjtumrgx (Part 1)

> AI development session journal
> Started: 2026-02-03

---


## Session 1: Support 4 Qwen3-VL Models

**Date**: 2026-02-03
**Task**: Support 4 Qwen3-VL Models

### Summary

(Add summary)

### Main Changes

## Summary
Added support for all 4 Qwen3-VL models on the RTX5000 server and fixed KV cache OOM issue.

## Changes

| Feature | Description |
|---------|-------------|
| 4 Model Support | Added Qwen3-VL-8B, 30B, 30B-Thinking, 32B models |
| Fix OOM | Added `--max-model-len` parameter (default 32768) |
| CPU Offload | Added `--cpu-offload-gb` for memory-based KV cache |

## Updated Files
- `inference/config.py` - Added 4 model configurations with snapshot paths
- `inference/start_server.py` - Added `--max-model-len` and `--cpu-offload-gb` parameters
- `inference/manage.sh` - Updated help documentation

## Technical Notes
- Original error: KV cache needed 16GB for 262K max_model_len, only 7.4GB available
- Solution: Limit max_model_len to 32K (sufficient for VLN tasks)
- CPU offload available for longer contexts if needed

### Git Commits

| Hash | Message |
|------|---------|
| `9a113c6` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
