#!/usr/bin/env python3

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_generation():
    """Quick test to verify model can generate with our settings"""
    
    try:
        logger.info("🔍 Loading model for diagnostic test...")
        
        # Load model and tokenizer - same as in main_hf_trl_dapo.py
        model_name = "openai/gpt-oss-20b"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Add padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("✅ Model loaded successfully")
        
        # Create a simple ARC prompt
        test_prompt = """<|system|>
You are an AI assistant that solves ARC (Abstract Reasoning Corpus) puzzles. Analyze the input-output patterns and generate the solution grid.

<|user|>
Analyze this ARC puzzle and provide the solution.

Input grid:
[[0, 0, 0], [0, 1, 0], [0, 0, 0]]

Output grid should transform the pattern. What is the output?

<|assistant|>
<|channel|>analysis<|message|>
Let me analyze this pattern:
"""

        logger.info("🚀 Testing generation with improved settings...")
        
        # Tokenize
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        # Generate with our improved settings
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=500,  # Shorter for diagnostic
                temperature=0.5,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.8,
                no_repeat_ngram_size=3,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode response
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        response_only = tokenizer.decode(outputs.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        
        logger.info(f"✅ Generation completed successfully!")
        logger.info(f"📝 Generated tokens: {len(outputs.sequences[0]) - inputs.input_ids.shape[1]}")
        logger.info(f"🔤 Response preview: {response_only[:200]}...")
        
        # Save result
        result = {
            "status": "success",
            "generated_tokens": len(outputs.sequences[0]) - inputs.input_ids.shape[1],
            "full_response": generated_text,
            "response_only": response_only
        }
        
        with open("diagnostic_result.json", "w") as f:
            json.dump(result, f, indent=2)
        
        logger.info("💾 Results saved to diagnostic_result.json")
        
    except Exception as e:
        logger.error(f"❌ Error during diagnostic test: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Save error result
        result = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }
        
        with open("diagnostic_result.json", "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    test_model_generation()