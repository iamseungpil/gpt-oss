#!/usr/bin/env python3
"""
Test Unsloth GRPOTrainer interface
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Import and patch
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from trl import GRPOTrainer, GRPOConfig
import inspect

# Check GRPOTrainer signature
print("=== GRPOTrainer.__init__ signature ===")
sig = inspect.signature(GRPOTrainer.__init__)
print(sig)

print("\n=== GRPOTrainer.__init__ parameters ===")
for name, param in sig.parameters.items():
    print(f"  {name}: {param}")

# Check GRPOConfig signature
print("\n=== GRPOConfig.__init__ signature ===")
sig = inspect.signature(GRPOConfig.__init__)
print(sig)

print("\n=== GRPOConfig.__init__ parameters ===")
for name, param in sig.parameters.items():
    print(f"  {name}: {param}")