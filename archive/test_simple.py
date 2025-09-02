#!/usr/bin/env python3
"""
Simple test to verify tmux + conda output
"""
import sys
import time

print("🚀 Starting simple test...", flush=True)
sys.stdout.flush()

for i in range(10):
    print(f"⏰ Step {i+1}/10: Testing output...", flush=True)
    sys.stdout.flush()
    time.sleep(2)

print("✅ Test completed!", flush=True)
sys.stdout.flush()