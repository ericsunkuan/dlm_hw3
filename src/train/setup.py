# src/utils/setup.py
import os
from typing import List

def create_directory_structure(base_dir: str = 'outputs'):
    """Create the directory structure for outputs"""
    # Task 1 directories
    task1_temps = ['temp_0.8', 'temp_1.0', 'temp_1.2']
    task1_dirs = [
        os.path.join(base_dir, 'task1', 'midi', temp) for temp in task1_temps
    ] + [
        os.path.join(base_dir, 'task1', 'wav', temp) for temp in task1_temps
    ]
    
    # Task 2 directories
    task2_prompts = ['prompt1', 'prompt2', 'prompt3']
    task2_dirs = [
        os.path.join(base_dir, 'task2', 'midi', prompt) for prompt in task2_prompts
    ] + [
        os.path.join(base_dir, 'task2', 'wav', prompt) for prompt in task2_prompts
    ]
    
    # Create all directories
    for directory in task1_dirs + task2_dirs:
        os.makedirs(directory, exist_ok=True)
        
    return base_dir