# FSDP Version for Dual A100 40GB Setup

## Overview
`sequential_validation_v4_fsdp.py` is a modified version of the original validation script specifically designed for dual A100 40GB GPU setup using PyTorch FSDP (Fully Sharded Data Parallel).

## Key Differences from Original
- **FSDP Integration**: Uses PyTorch's FSDP for model sharding across 2 GPUs
- **Memory Optimization**: CPU offloading for parameters to fit in 40GB×2=80GB total
- **Distributed Setup**: Automatic distributed environment configuration
- **Same Functionality**: Identical inference logic, prompting, and output format

## Requirements
- 2× A100 40GB GPUs
- PyTorch with FSDP support
- NCCL backend for distributed communication

## Usage

### Single Node, Dual GPU:
```bash
# Automatic setup (recommended)
python sequential_validation_v4_fsdp.py --start 0 --num 10 --tokens 30000

# Manual distributed launch
torchrun --nproc_per_node=2 sequential_validation_v4_fsdp.py --start 0 --num 10
```

### Arguments
- `--start`: Starting problem index (default: 0)
- `--num`: Number of problems to solve (default: 10)
- `--tokens`: Max tokens per generation (default: 30000)

## Technical Details

### FSDP Configuration
- **Auto-wrap policy**: 1M parameters minimum per shard
- **CPU offload**: Parameters offloaded to CPU when not in use
- **Mixed precision**: Uses bfloat16 for memory efficiency
- **Sync module states**: Ensures consistency across GPUs

### Memory Management
- Model parameters: ~20B → distributed across 2 GPUs
- CPU offloading for inactive parameters
- Aggressive garbage collection after each problem
- CUDA cache clearing between inferences

### Output Files
- Progress files: `arc_v4_fsdp_progress_YYYYMMDD_HHMMSS.json`
- Final results: `SUCCESS_channel_switching_result_fsdp_YYYYMMDD_HHMMSS.json`
- Same format as original, with additional FSDP metadata

## Expected Performance
- **Memory usage**: ~30-35GB per GPU (within 40GB limit)
- **Speed**: Similar to single GPU due to inference workload
- **Reliability**: Better stability on memory-constrained systems

## Troubleshooting

### Common Issues
1. **OOM errors**: Reduce batch size or enable more aggressive CPU offloading
2. **NCCL timeouts**: Check network connectivity between GPUs
3. **Distributed init failures**: Verify MASTER_ADDR/MASTER_PORT settings

### Environment Variables (auto-set by script)
```bash
MASTER_ADDR=localhost
MASTER_PORT=29500
WORLD_SIZE=2
LOCAL_RANK=0/1
```

## Compatibility
- ✅ Same ARC problem format
- ✅ Same Harmony prompting
- ✅ Same 30k token generation
- ✅ Same channel switching detection
- ✅ Same accuracy evaluation
- ✅ Compatible output format for analysis tools