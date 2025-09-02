#!/usr/bin/env python3
"""
Visualize ARC grid comparison: Target vs Model Output
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple
import glob

# ARC color palette (0-9)
COLORS = [
    (0, 0, 0),       # 0: Black
    (0, 116, 217),   # 1: Blue
    (255, 65, 54),   # 2: Red
    (46, 204, 64),   # 3: Green
    (255, 220, 0),   # 4: Yellow
    (127, 219, 255), # 5: Gray/Light Blue
    (255, 0, 255),   # 6: Pink/Magenta
    (255, 127, 0),   # 7: Orange
    (128, 0, 128),   # 8: Purple
    (135, 65, 21),   # 9: Brown
]

def render_grid(grid: np.ndarray, cell_size: int = 30, border: int = 2) -> Image.Image:
    """Render a single grid as an image"""
    h, w = grid.shape
    img_w = w * cell_size + (w + 1) * border
    img_h = h * cell_size + (h + 1) * border
    
    img = Image.new('RGB', (img_w, img_h), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    
    for i in range(h):
        for j in range(w):
            x = j * (cell_size + border) + border
            y = i * (cell_size + border) + border
            color = COLORS[int(grid[i, j]) % 10]
            draw.rectangle([x, y, x + cell_size, y + cell_size], fill=color)
            
            # Add grid lines for better visibility
            if grid[i, j] == 0:
                # Draw subtle grid lines for zeros
                draw.line([x, y, x + cell_size, y + cell_size], fill=(50, 50, 50), width=1)
                draw.line([x + cell_size, y, x, y + cell_size], fill=(50, 50, 50), width=1)
    
    return img

def create_empty_grid(shape: Tuple[int, int]) -> np.ndarray:
    """Create an empty (all zeros) grid of given shape"""
    return np.zeros(shape, dtype=int)

def visualize_comparison(results_file: str, output_dir: str = "visualizations", reasoning_level: str = "high"):
    """Create side-by-side comparison of target vs model output"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read results JSON
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    problem_id = results.get('problem_id', 'unknown')
    target_grid = np.array(results['target_output'])
    
    # Handle different result formats
    if 'reasoning_level_results' in results:
        # New harmony library format
        level_result = results['reasoning_level_results'].get(reasoning_level, {})
        model_output = level_result.get('model_output')
        if model_output is not None:
            model_grid = np.array(model_output)
            accuracy = level_result.get('grid_extraction', {}).get('accuracy', 0.0)
        else:
            model_grid = create_empty_grid(target_grid.shape)
            accuracy = 0.0
        
        # Check channel usage
        channels = level_result.get('channels', {})
        final_channel_used = channels.get('has_final', False)
        
    else:
        # Legacy format
        if results.get('model_output') is not None:
            model_grid = np.array(results['model_output'])
            accuracy = results.get('accuracy', 0.0)
        else:
            model_grid = create_empty_grid(target_grid.shape)
            accuracy = 0.0
        
        channels = results.get('channels_used', {})
        final_channel_used = channels.get('final_final', False)
    
    # Render grids
    target_img = render_grid(target_grid)
    model_img = render_grid(model_grid)
    
    # Create composite image
    padding = 40
    gap = 60
    
    # Calculate dimensions
    max_w = max(target_img.width, model_img.width)
    max_h = max(target_img.height, model_img.height)
    
    total_w = 2 * max_w + gap + 2 * padding
    total_h = max_h + 2 * padding + 80  # Extra space for labels and info
    
    composite = Image.new('RGB', (total_w, total_h), color='white')
    draw = ImageDraw.Draw(composite)
    
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_info = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font_title = None
        font_info = None
    
    # Title
    title = f"Problem: {problem_id}"
    if 'reasoning_level_results' in results:
        title += f" | Reasoning: {reasoning_level.upper()}"
    if font_title:
        draw.text((padding, 10), title, fill='black', font=font_title)
    else:
        draw.text((padding, 10), title, fill='black')
    
    # Channel info
    channel_info = f"Final channel used: {'✅' if final_channel_used else '❌'} | Accuracy: {accuracy:.1%}"
    if font_info:
        draw.text((padding, 35), channel_info, fill='gray', font=font_info)
    else:
        draw.text((padding, 35), channel_info, fill='gray')
    
    # Draw labels
    y_offset = 70
    
    # Target grid
    label_target = f"Target ({target_grid.shape[0]}×{target_grid.shape[1]})"
    if font_title:
        draw.text((padding, y_offset), label_target, fill='green', font=font_title)
    else:
        draw.text((padding, y_offset), label_target, fill='green')
    
    # Model output grid
    label_model = f"Model Output ({model_grid.shape[0]}×{model_grid.shape[1]})"
    if results.get('model_output') is None:
        label_model += " [No grid extracted]"
    
    x_model = padding + max_w + gap
    if font_title:
        draw.text((x_model, y_offset), label_model, fill='blue' if accuracy > 0 else 'red', font=font_title)
    else:
        draw.text((x_model, y_offset), label_model, fill='blue' if accuracy > 0 else 'red')
    
    # Paste grid images
    y_grids = y_offset + 30
    composite.paste(target_img, (padding, y_grids))
    composite.paste(model_img, (x_model, y_grids))
    
    # Draw comparison arrows if shapes match
    if target_grid.shape == model_grid.shape and accuracy > 0:
        arrow_x = padding + max_w + gap // 2
        arrow_y = y_grids + max_h // 2
        draw.text((arrow_x - 10, arrow_y - 10), "→", fill='black')
        if font_info:
            draw.text((arrow_x - 20, arrow_y + 10), f"{accuracy:.1%}", fill='black', font=font_info)
    
    # Save visualization
    base_name = os.path.basename(results_file).replace('.json', '')
    if 'reasoning_level_results' in results:
        output_path = os.path.join(output_dir, f"{base_name}_{reasoning_level}_comparison.png")
    else:
        output_path = os.path.join(output_dir, f"{base_name}_comparison.png")
    composite.save(output_path)
    
    print(f"✅ Saved comparison visualization to: {output_path}")
    print(f"   Problem: {problem_id}")
    print(f"   Target shape: {target_grid.shape}")
    print(f"   Model output shape: {model_grid.shape}")
    print(f"   Final channel used: {'Yes' if final_channel_used else 'No'}")
    print(f"   Accuracy: {accuracy:.1%}")
    
    return output_path

if __name__ == "__main__":
    import sys
    
    # Find latest results file
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
        reasoning_level = sys.argv[2] if len(sys.argv) > 2 else "high"
    else:
        # Find most recent harmony library results
        json_files = glob.glob("/tmp/harmony_library_results_*.json")
        if not json_files:
            # Fallback to no_injection results
            json_files = glob.glob("/tmp/no_injection_results_*.json")
        if json_files:
            results_file = max(json_files, key=os.path.getctime)
            reasoning_level = "high"
        else:
            print("No results file found")
            sys.exit(1)
    
    if os.path.exists(results_file):
        # Create visualizations for all reasoning levels if harmony library format
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        if 'reasoning_level_results' in results:
            for level in results['reasoning_level_results'].keys():
                print(f"\n🎨 Creating visualization for reasoning level: {level.upper()}")
                visualize_comparison(results_file, reasoning_level=level)
        else:
            visualize_comparison(results_file)
    else:
        print(f"File not found: {results_file}")