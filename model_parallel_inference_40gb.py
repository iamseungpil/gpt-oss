#!/usr/bin/env python3
"""
GPT-OSS Model Parallel Inference - Optimized for 40GB A100 x2
40GB A100 2장에서 모델을 나눠서 빠른 추론 구현 (메모리 최적화)
"""

import os
import sys
import json
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from arc import train_problems

def log_with_timestamp(message):
    """Enhanced logging with timestamp and rank."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    rank = dist.get_rank() if dist.is_initialized() else 0
    print(f"[{timestamp}] [Rank {rank}] {message}")
    sys.stdout.flush()

def setup_model_parallel_distributed():
    """Setup distributed for model parallel."""
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0" 
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "2"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29900"
    
    # Optimized NCCL settings for A100
    os.environ['NCCL_TIMEOUT'] = '300'
    os.environ['NCCL_DEBUG'] = 'WARN'
    
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    try:
        dist.init_process_group(backend="nccl")
        log_with_timestamp(f"🚀 40GB A100 Model Parallel setup complete - Rank: {dist.get_rank()}/{dist.get_world_size()}")
        return local_rank
    except Exception as e:
        log_with_timestamp(f"❌ Distributed init failed: {e}")
        return local_rank

def get_memory_info(device):
    """Get GPU memory information"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        return memory_allocated, memory_reserved, memory_total
    return 0, 0, 0

class ModelParallelGPTOSS40GB:
    """
    Model Parallel GPT-OSS optimized for 40GB A100 GPUs
    각 GPU가 모델의 절반을 담당하여 메모리 효율적이면서도 빠른 추론
    """
    
    def __init__(self, model_name="openai/gpt-oss-20b", quantization="fp16"):
        self.model_name = model_name
        self.quantization = quantization
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.cuda.current_device()
        
        log_with_timestamp(f"🔧 Initializing 40GB A100 Model Parallel GPT-OSS on rank {self.rank}")
        log_with_timestamp(f"🧠 Using {quantization.upper()} quantization for memory optimization")
        
        # Load model configuration first
        from transformers import AutoConfig
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        log_with_timestamp(f"📊 Model config: {self.config.num_hidden_layers} layers, {self.config.hidden_size} hidden")
        
        # Calculate optimal layer split for 40GB GPUs
        total_layers = self.config.num_hidden_layers  # 24 for GPT-OSS 20B
        
        if self.rank == 0:
            # GPU 0: Embedding + First 12 layers
            self.layer_start = 0
            self.layer_end = total_layers // 2  # 12
            self.has_embedding = True
            self.has_head = False
            log_with_timestamp(f"📍 Rank 0: Embedding + Layers 0-{self.layer_end-1}")
            
        else:  # rank == 1
            # GPU 1: Last 12 layers + Head
            self.layer_start = total_layers // 2  # 12
            self.layer_end = total_layers  # 24
            self.has_embedding = False
            self.has_head = True
            log_with_timestamp(f"📍 Rank 1: Layers {self.layer_start}-{self.layer_end-1} + Head")
        
        # Load model components with optimization
        self._load_optimized_model_components()
        
        log_with_timestamp(f"✅ 40GB A100 Model parallel setup complete on rank {self.rank}")
        self._log_memory_usage()
    
    def _load_optimized_model_components(self):
        """Load only the required model components for this rank with memory optimization"""
        log_with_timestamp(f"📦 Loading optimized model components for rank {self.rank}...")
        
        # Load full model temporarily to extract components with quantization
        if self.quantization == "int8":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            full_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map=None  # Load on CPU first
            )
        elif self.quantization == "fp16":
            full_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map=None  # Load on CPU first
            )
        else:  # bf16 (default)
            full_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=None  # Load on CPU first
            )
        
        if self.rank == 0:
            # Load embedding and first half of layers
            self.embed_tokens = full_model.model.embed_tokens.to(self.device)
            self.layers = nn.ModuleList([
                full_model.model.layers[i].to(self.device) 
                for i in range(self.layer_start, self.layer_end)
            ])
            
        else:  # rank == 1
            # Load second half of layers and head
            self.layers = nn.ModuleList([
                full_model.model.layers[i].to(self.device) 
                for i in range(self.layer_start, self.layer_end)
            ])
            self.norm = full_model.model.norm.to(self.device)
            self.lm_head = full_model.lm_head.to(self.device)
        
        # Clear full model from memory
        del full_model
        torch.cuda.empty_cache()
        
        log_with_timestamp(f"✅ Optimized model components loaded on rank {self.rank}")
    
    def _log_memory_usage(self):
        """Log current GPU memory usage"""
        mem_alloc, mem_reserved, mem_total = get_memory_info(self.device)
        usage_percent = mem_alloc / mem_total * 100
        
        log_with_timestamp(f"🧠 GPU {self.rank} Memory: {mem_alloc:.1f}GB/{mem_total:.1f}GB ({usage_percent:.1f}%)")
        
        if usage_percent > 95:
            log_with_timestamp(f"⚠️  WARNING: Very high memory usage on GPU {self.rank}!")
        elif usage_percent > 85:
            log_with_timestamp(f"⚠️  High memory usage on GPU {self.rank} - consider INT8 quantization")
        else:
            log_with_timestamp(f"✅ Safe memory usage on GPU {self.rank}")
    
    def forward_rank0(self, input_ids, attention_mask=None, position_ids=None):
        """Forward pass for rank 0 (embedding + first layers)"""
        # Embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # Create position_ids if not provided
        if position_ids is None:
            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # First half layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False  # Disable cache for simplicity
            )
            hidden_states = layer_outputs[0]
        
        return hidden_states
    
    def forward_rank1(self, hidden_states, attention_mask=None, position_ids=None):
        """Forward pass for rank 1 (second layers + head)"""
        # Create position_ids if not provided
        if position_ids is None:
            batch_size, seq_len, _ = hidden_states.shape
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Second half layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False
            )
            hidden_states = layer_outputs[0]
        
        # Final norm and head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def generate_token_by_token_optimized(self, input_ids, max_new_tokens=2000, temperature=0.7, top_p=0.9):
        """
        Generate tokens one by one with model parallelism - 40GB A100 optimized
        """
        if self.rank == 0:
            batch_size, seq_len = input_ids.shape
            current_tokens = input_ids.clone()
            log_with_timestamp(f"🚀 Starting 40GB A100 model parallel generation: {max_new_tokens} tokens")
        else:
            # Rank 1 needs to know batch_size and seq_len from rank 0
            batch_size = 1  # Will be synchronized
            seq_len = 0     # Will be synchronized
            current_tokens = None
        
        # Synchronize batch_size and seq_len across ranks
        if self.rank == 0:
            tensor_info = torch.tensor([batch_size, seq_len], dtype=torch.long, device=self.device)
            dist.broadcast(tensor_info, src=0)
        else:
            tensor_info = torch.zeros(2, dtype=torch.long, device=self.device)
            dist.broadcast(tensor_info, src=0)
            batch_size, seq_len = tensor_info[0].item(), tensor_info[1].item()
        
        start_time = time.time()
        generated_tokens = 0
        
        for step in range(max_new_tokens):
            step_start_time = time.time()
            
            # Clear cache periodically for memory optimization
            if step % 100 == 0:
                torch.cuda.empty_cache()
            
            if self.rank == 0:
                # GPU 0: Embedding + first layers
                with torch.no_grad():
                    hidden_states = self.forward_rank0(current_tokens)
                
                # Send hidden states to GPU 1
                dist.send(hidden_states.contiguous(), dst=1)
                
                # Receive next token from GPU 1
                next_token = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
                dist.recv(next_token, src=1)
                
                # Append new token
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
                generated_tokens += 1
                
                step_time = time.time() - step_start_time
                if step % 100 == 0:  # Log every 100 tokens
                    mem_alloc, _, mem_total = get_memory_info(self.device)
                    log_with_timestamp(f"🔄 Step {step}: token {next_token.item()}, {1/step_time:.2f} tok/sec, mem: {mem_alloc:.1f}GB")
                
                # Check for EOS tokens
                if next_token.item() in [200002, 199998]:  # <|return|> or <|eos|>
                    log_with_timestamp(f"🛑 EOS token reached at step {step}")
                    break
                    
            else:  # rank == 1
                # GPU 1: Receive hidden states from GPU 0
                current_seq_len = seq_len + step
                hidden_states = torch.zeros(
                    (batch_size, current_seq_len, self.config.hidden_size),
                    dtype=torch.bfloat16 if self.quantization == "bf16" else torch.float16, 
                    device=self.device
                )
                dist.recv(hidden_states, src=0)
                
                # GPU 1: Second layers + generate next token
                with torch.no_grad():
                    logits = self.forward_rank1(hidden_states)
                    next_token_logits = logits[:, -1, :]  # Last position
                    
                    # Apply temperature and top-p sampling
                    if temperature > 0:
                        next_token_logits = next_token_logits / temperature
                        
                        # Top-p (nucleus) sampling
                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            next_token_logits[indices_to_remove] = float('-inf')
                        
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Send token back to GPU 0
                dist.send(next_token.contiguous(), dst=0)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if self.rank == 0:
            tokens_per_second = generated_tokens / total_time
            log_with_timestamp(f"✅ 40GB A100 Model parallel generation completed!")
            log_with_timestamp(f"📊 Generated {generated_tokens} tokens in {total_time:.2f}s")
            log_with_timestamp(f"⚡ Speed: {tokens_per_second:.2f} tokens/second")
            log_with_timestamp(f"🎛️ Quantization: {self.quantization.upper()}")
            self._log_memory_usage()
            return current_tokens
        
        return None

def create_arc_prompt(problem, tokenizer):
    """Create optimized ARC prompt"""
    # Use fewer examples for faster processing on 40GB GPUs
    train_examples_str = []
    for i, train_pair in enumerate(problem.train_pairs[:3]):  # Only first 3 examples
        input_str = '\n'.join(' '.join(str(int(cell)) for cell in row) for row in train_pair.x)
        output_str = '\n'.join(' '.join(str(int(cell)) for cell in row) for row in train_pair.y)
        train_examples_str.append(f"Example {i+1}:\nInput:\n{input_str}\n\nOutput:\n{output_str}")
    
    test_input = problem.test_pairs[0].x
    test_input_str = '\n'.join(' '.join(str(int(cell)) for cell in row) for row in test_input)
    
    user_content = f"""# ARC Pattern Analysis

## Examples:
{chr(10).join(train_examples_str)}

## Test:
{test_input_str}

Analysis and Solution:"""
    
    messages = [
        {"role": "system", "content": "You are an expert pattern recognition system. Analyze quickly and provide solutions."},
        {"role": "user", "content": user_content}
    ]
    
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="high"
        )
    except:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    return prompt + "<|channel|>"

def run_model_parallel_inference_40gb(quantization="fp16"):
    """Run 40GB A100 optimized model parallel inference"""
    rank = dist.get_rank()
    
    if rank == 0:
        log_with_timestamp("🚀 STARTING 40GB A100 MODEL PARALLEL INFERENCE")
        log_with_timestamp("⚡ Optimized for 40GB A100 x2 - Memory Efficient!")
        log_with_timestamp(f"🧠 Using {quantization.upper()} quantization")
    
    # Load tokenizer (both ranks need it)
    if rank == 0:
        log_with_timestamp("📦 Loading tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "openai/gpt-oss-20b",
        trust_remote_code=True,
        padding_side='left'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize optimized model parallel system
    model_parallel = ModelParallelGPTOSS40GB(quantization=quantization)
    
    if rank == 0:
        # Create and process problem
        problem = train_problems[0]  # First training problem
        prompt = create_arc_prompt(problem, tokenizer)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model_parallel.device)
        
        log_with_timestamp(f"📝 Input tokens: {input_ids.shape[1]}")
        log_with_timestamp("🎯 Starting 40GB A100 model parallel generation...")
        
        # Generate with optimized model parallelism
        output_tokens = model_parallel.generate_token_by_token_optimized(
            input_ids,
            max_new_tokens=2000,
            temperature=0.7,
            top_p=0.9
        )
        
        if output_tokens is not None:
            # Decode result
            result = tokenizer.decode(output_tokens[0], skip_special_tokens=False)
            
            # Save result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"model_parallel_40gb_{quantization}_result_{timestamp}.txt"
            
            with open(output_file, "w") as f:
                f.write(result)
            
            log_with_timestamp(f"✅ 40GB A100 generation completed! Length: {len(result)} characters")
            log_with_timestamp(f"💾 Result saved to {output_file}")
            
            # Log final memory usage
            model_parallel._log_memory_usage()
    else:
        # Rank 1 participates in generation
        model_parallel.generate_token_by_token_optimized(
            None,  # Rank 1 doesn't need input_ids
            max_new_tokens=2000
        )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="40GB A100 Model Parallel Inference")
    parser.add_argument("--tokens", type=int, default=2000, help="Max tokens to generate")
    parser.add_argument("--quantization", type=str, default="fp16",
                       choices=["int8", "fp16", "bf16"],
                       help="Quantization method for memory optimization")
    
    args = parser.parse_args()
    
    # Setup distributed
    local_rank = setup_model_parallel_distributed()
    
    try:
        run_model_parallel_inference_40gb(args.quantization)
    except Exception as e:
        log_with_timestamp(f"❌ 40GB A100 model parallel inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)