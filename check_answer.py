#!/usr/bin/env python3
"""
ARC Answer Checker - Manual grid comparison tool
"""

import json
import numpy as np
from arc import validation_problems

def string_to_grid(grid_str: str) -> np.ndarray:
    """Convert string grid to numpy array."""
    try:
        lines = grid_str.strip().split('\n')
        grid = []
        for line in lines:
            if line.strip():
                row = [int(x) for x in line.split()]
                if row:  # Only add non-empty rows
                    grid.append(row)
        return np.array(grid)
    except Exception as e:
        print(f"Error parsing grid: {e}")
        return None

def extract_final_grid(response_text: str) -> np.ndarray:
    """Extract grid from final channel."""
    try:
        final_idx = response_text.find('<|channel|>final<|message|>')
        if final_idx == -1:
            return None
        
        final_content = response_text[final_idx + len('<|channel|>final<|message|>'):]
        end_idx = final_content.find('<|return|>')
        if end_idx == -1:
            end_idx = final_content.find('<|end|>')
        
        if end_idx != -1:
            final_content = final_content[:end_idx]
        
        return string_to_grid(final_content)
    except Exception as e:
        print(f"Error extracting grid: {e}")
        return None

def check_single_answer(problem_idx: int = 0, result_file: str = None):
    """Check answer for a specific problem."""
    
    # Get target answer
    problem = validation_problems[problem_idx]
    target_grid = problem.test_pairs[0].y
    
    print(f"Problem: {problem.uid}")
    print(f"Target shape: {target_grid.shape}")
    print("Target grid:")
    print(target_grid)
    print()
    
    if result_file:
        # Load from result file
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        if 'validation_results' in data:
            # Batch validation result
            result = data['validation_results'][problem_idx]
            response = result.get('response', '')
        else:
            # Single inference result
            response = data['response']
        
        predicted_grid = extract_final_grid(response)
        
        if predicted_grid is not None:
            print(f"Predicted shape: {predicted_grid.shape}")
            print("Predicted grid:")
            print(predicted_grid)
            print()
            
            # Compare
            if predicted_grid.shape == target_grid.shape:
                is_correct = np.array_equal(predicted_grid, target_grid)
                print(f"✅ Match: {is_correct}")
                if not is_correct:
                    print("Differences:")
                    diff = predicted_grid != target_grid
                    print(f"Different cells: {np.sum(diff)}")
            else:
                print("❌ Shape mismatch!")
        else:
            print("❌ No grid extracted from response")
    
    return target_grid

def compare_grids_visual(grid1: np.ndarray, grid2: np.ndarray):
    """Visual grid comparison."""
    print("Side by side comparison:")
    print("TARGET" + " " * (grid1.shape[1]*2-6) + "PREDICTED")
    print("-" * (grid1.shape[1]*2 + 10))
    
    max_rows = max(grid1.shape[0], grid2.shape[0])
    for i in range(max_rows):
        # Target row
        if i < grid1.shape[0]:
            target_row = " ".join(str(x) for x in grid1[i])
        else:
            target_row = " " * (grid1.shape[1]*2-1)
        
        # Predicted row  
        if i < grid2.shape[0]:
            pred_row = " ".join(str(x) for x in grid2[i])
        else:
            pred_row = ""
        
        print(f"{target_row}    {pred_row}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        problem_idx = int(sys.argv[1])
    else:
        problem_idx = 0
    
    if len(sys.argv) > 2:
        result_file = sys.argv[2]
    else:
        result_file = None
    
    check_single_answer(problem_idx, result_file)