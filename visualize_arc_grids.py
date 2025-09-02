#!/usr/bin/env python3
"""
Visualize ARC grids from GPT-OSS multi-turn outputs
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Optional, Tuple
import re

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

def parse_grid_from_text(text: str, max_size: int = 30) -> Optional[np.ndarray]:
    """Parse a grid from text response"""
    lines = text.strip().split('\n')
    grid_rows = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip non-grid lines
        if any(skip in line.lower() for skip in ['example', 'input:', 'output:', 'grid:', '```', 'channel', 'message']):
            continue
        
        # Try to parse as grid row
        parts = line.split()
        if parts:
            try:
                row = []
                for part in parts:
                    if part.isdigit():
                        num = int(part)
                        if 0 <= num <= 9:
                            row.append(num)
                
                if row and len(row) <= max_size:
                    grid_rows.append(row)
                elif grid_rows:
                    # Stop if we hit invalid line after starting grid
                    break
                    
            except:
                if grid_rows:
                    break
                continue
    
    if grid_rows:
        # Check if all rows have same length
        row_lengths = [len(row) for row in grid_rows]
        if len(set(row_lengths)) == 1 and len(grid_rows) <= max_size:
            return np.array(grid_rows)
    
    return None

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
    
    return img

def visualize_multiturn_responses(conversation_file: str, output_dir: str = "visualizations"):
    """Visualize all grids from a multi-turn conversation"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read conversation
    with open(conversation_file, 'r') as f:
        content = f.read()
    
    # Parse different sections
    turns = content.split("TURN")
    
    grids_found = []
    images = []
    
    for i, turn in enumerate(turns):
        if not turn.strip():
            continue
            
        # Extract grids from this turn
        grid = parse_grid_from_text(turn)
        if grid is not None:
            grids_found.append((f"Turn {i}", grid))
            img = render_grid(grid)
            images.append((f"Turn {i}", img))
            print(f"Found grid in Turn {i}: shape {grid.shape}")
    
    # Also look for target output
    if "TARGET OUTPUT:" in content:
        target_section = content.split("TARGET OUTPUT:")[1]
        target_grid = parse_grid_from_text(target_section)
        if target_grid is not None:
            grids_found.append(("Target", target_grid))
            img = render_grid(target_grid)
            images.append(("Target", img))
            print(f"Found target grid: shape {target_grid.shape}")
    
    # Create composite image
    if images:
        # Calculate layout
        cell_size = 30
        padding = 20
        gap = 40
        
        # Arrange in rows
        cols = min(4, len(images))
        rows = (len(images) + cols - 1) // cols
        
        max_w = max(img.width for _, img in images)
        max_h = max(img.height for _, img in images)
        
        total_w = cols * max_w + (cols - 1) * gap + 2 * padding
        total_h = rows * max_h + (rows - 1) * gap + 2 * padding + 30 * rows  # Extra space for labels
        
        composite = Image.new('RGB', (total_w, total_h), color='white')
        draw = ImageDraw.Draw(composite)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = None
        
        for idx, (label, img) in enumerate(images):
            row = idx // cols
            col = idx % cols
            
            x = padding + col * (max_w + gap)
            y = padding + row * (max_h + gap + 30)
            
            # Draw label
            if font:
                draw.text((x, y), label, fill='black', font=font)
            else:
                draw.text((x, y), label, fill='black')
            
            # Paste grid image
            composite.paste(img, (x, y + 20))
        
        # Save composite
        output_path = os.path.join(output_dir, os.path.basename(conversation_file).replace('.txt', '_vis.png'))
        composite.save(output_path)
        print(f"\n✅ Saved visualization to: {output_path}")
        
        return output_path, grids_found
    else:
        print("❌ No grids found in conversation")
        return None, []

def compare_grids(grid1: np.ndarray, grid2: np.ndarray) -> float:
    """Compare two grids and return accuracy"""
    if grid1.shape != grid2.shape:
        return 0.0
    return np.mean(grid1 == grid2)

if __name__ == "__main__":
    # Test with the CoT multi-turn conversation
    import sys
    
    if len(sys.argv) > 1:
        conversation_file = sys.argv[1]
    else:
        conversation_file = "/tmp/cot_multiturn_feca6190.txt"
    
    if os.path.exists(conversation_file):
        output_path, grids = visualize_multiturn_responses(conversation_file)
        
        if grids and len(grids) > 1:
            print("\n📊 Grid comparisons:")
            target = None
            for label, grid in grids:
                if "Target" in label:
                    target = grid
                    break
            
            if target is not None:
                for label, grid in grids:
                    if "Target" not in label and grid.shape == target.shape:
                        accuracy = compare_grids(grid, target)
                        print(f"   {label} vs Target: {accuracy:.1%} accuracy")
    else:
        print(f"File not found: {conversation_file}")