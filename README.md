# GPT-OSS 20B ARC DAPO Training

Final clean implementation of GPT-OSS 20B training on ARC tasks using Unsloth's native DAPO.

## 📁 Project Structure

```
/home/ubuntu/gpt_oss_arc_final/
├── main_native_dapo.py     # Main training script
├── README.md               # This file
└── requirements.txt        # Dependencies

/data/gpt_oss_final/
├── logs/
│   ├── training.log        # Training logs
│   └── responses/          # Detailed response data
│       ├── step_001.json   # Step 1 responses
│       ├── step_002.json   # Step 2 responses
│       └── ...
├── checkpoints/            # Model checkpoints (every 50 steps)
└── final_model/           # Final trained model
```

## 🎯 Training Details

### Model Configuration
- **Model**: `unsloth/gpt-oss-20b-BF16` (20B parameters)
- **Quantization**: 4-bit (MXFP4 via NF4)
- **LoRA**: r=16, alpha=32, ~4M trainable parameters
- **Context**: 2048 tokens

### DAPO Configuration
- **Steps**: 500 maximum (early stopping enabled)
- **Candidates**: 8 per step with varied temperature
- **Problems**: 5 ARC problems (cyclical)
- **Early stopping**: After 3 perfect solutions

### Hardware
- **GPU**: 6 (NVIDIA A100-SXM4-80GB)
- **Memory**: ~20GB usage with 4-bit quantization

## 🔄 Training Process

### 1. Harmony Prompt Structure (analysis + final)
```
<|start|>user<|message|>Solve this pattern:
Example 1: Input:\n0 1 2\n1 0 1\nOutput:\n1 0 2\n0 1 0\n
Test Input:
1 2 0
0 1 2
Test Output:<|end|>
<|start|>assistant<|channel|>analysis<|message|>Let me analyze the pattern step by step.<|end|>
<|start|>assistant<|channel|>final<|message|>
```

### 2. Model Response Format (Harmony Channels)
The model responds using structured channels:

**Analysis Channel:**
```
<|channel|>analysis<|return|>
Looking at the examples, I can see the pattern is...
The transformation rule appears to be...

<|channel|>final<|return|>
2 0 1
1 2 0
<|end|>
```

**Or simplified format:**
```
Based on the pattern analysis:
2 0 1  
1 2 0
```

### 3. Evaluation
- **Pixel accuracy**: Exact match percentage
- **Size match**: Correct grid dimensions
- **Perfect match**: 100% pixel accuracy

### 4. DAPO Process
1. Generate 8 candidates with different temperatures
2. Evaluate all candidates against target grid
3. Find best and worst responses
4. Apply preference loss: increase likelihood of best, decrease worst
5. Weight loss by accuracy difference

## 📊 Response Logging

Each step saves detailed JSON with:
```json
{
  "step": 1,
  "problem_id": "af902bf9",
  "prompt": "Full prompt text...",
  "target_grid": [[0,1,2], [1,0,1]],
  "candidates": [
    {
      "candidate_id": 1,
      "response_text": "Model response...",
      "predicted_grid": [[0,1,1], [1,0,1]],
      "accuracy": 0.83,
      "has_grid": true,
      "correct_size": true,
      "perfect_match": false
    }
  ],
  "best_candidate_id": 3,
  "best_accuracy": 0.95,
  "timestamp": "2025-08-20T07:45:00"
}
```

## 🎮 Usage

### Start Training
```bash
cd /home/ubuntu/gpt_oss_arc_final
conda activate oss_fixed
python main_native_dapo.py
```

### Monitor Progress
```bash
# Real-time logs
tail -f /data/gpt_oss_final/logs/training.log

# Check latest responses
ls -la /data/gpt_oss_final/logs/responses/

# View specific step
cat /data/gpt_oss_final/logs/responses/step_001.json
```

### Resume from Checkpoint
```bash
# Checkpoints saved every 50 steps in:
/data/gpt_oss_final/checkpoints/step_50/
/data/gpt_oss_final/checkpoints/step_100/
# etc.
```

## 🏆 Success Criteria

### Early Stopping Conditions
- **Perfect Solution**: 100% pixel accuracy on test grid
- **Multiple Perfect**: 3+ perfect solutions across different problems
- **Manual Stop**: Any time via Ctrl+C

### Expected Outcomes
- Model learns to extract patterns from ARC examples
- Generates correct test output grids
- Improves through preference optimization
- Saves all responses for analysis

## 🔧 Technical Notes

### ARC Problem Format
- **Training pairs**: Input-output examples showing pattern
- **Test pairs**: Input only, model must predict output
- **Evaluation**: Compare prediction vs actual test output

### Grid Format
- Numbers 0-9 representing different colors/elements
- Space-separated within rows
- Newline-separated between rows
- Variable sizes (typically 3x3 to 30x30)

### Memory Optimization
- BFloat16 precision for activations
- 4-bit quantization for weights
- Gradient checkpointing enabled
- LoRA for parameter-efficient fine-tuning