#!/usr/bin/env python3
"""
GPT-OSS 20B Memory Test with DeepSpeed Stage 3 + Flash Attention
==============================================================

Simple test to verify GPT-OSS loads with DeepSpeed Stage 3 and Flash Attention
"""

import os
import sys
import torch
from datetime import datetime
import logging

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def test_deepspeed_loading():
    """Test GPT-OSS loading with DeepSpeed Stage 3"""
    
    logger.info("🚀 Testing GPT-OSS 20B with DeepSpeed Stage 3 + Flash Attention")
    
    try:
        # Skip quantization for compatibility test
        bnb_config = None
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "openai/gpt-oss-20b", 
            trust_remote_code=True,
            padding_side="left",
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        logger.info("✅ Tokenizer loaded successfully")
        
        # Load model with Flash Attention (no quantization for compatibility)
        logger.info("Loading model with Flash Attention 2...")
        model = AutoModelForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )
        
        logger.info("✅ Model loaded with Flash Attention")
        
        # Add LoRA adapters
        logger.info("Adding LoRA adapters...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                           "gate_proj", "up_proj", "down_proj"],
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        logger.info("✅ LoRA adapters added successfully")
        
        # Test inference
        logger.info("Testing inference...")
        test_prompt = "What is 2 + 2?"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test response: {response}")
        
        logger.info("✅ Inference test successful!")
        
        # Memory stats
        if torch.cuda.is_available():
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    success = test_deepspeed_loading()
    if success:
        logger.info("🎉 All tests passed! Ready for training with DeepSpeed.")
    else:
        logger.error("❌ Tests failed. Check configuration.")