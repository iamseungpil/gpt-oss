#!/usr/bin/env python3
"""
Test TRL GRPOTrainer interface (without Unsloth)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from trl import GRPOTrainer, GRPOConfig
import inspect

# Check GRPOTrainer signature
print("=== TRL GRPOTrainer.__init__ signature ===")
sig = inspect.signature(GRPOTrainer.__init__)
print(sig)

print("\n=== TRL GRPOTrainer.__init__ parameters ===")
for name, param in sig.parameters.items():
    print(f"  {name}: {param}")

# Check GRPOConfig signature
print("\n=== TRL GRPOConfig.__init__ signature ===")
sig = inspect.signature(GRPOConfig.__init__)
print(sig)

print("\n=== TRL GRPOConfig.__init__ parameters ===")
for name, param in sig.parameters.items():
    print(f"  {name}: {param}")