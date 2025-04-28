# Model Format Requirements

## Overview
This directory should contain your trained reinforcement learning models for playing Splendor. 
The WebGUI and evaluation scripts expect certain model formats and dimensions.

## Required Models
- `ppo_checkpoint_final.pt` - This is the main model file expected by the WebGUI by default
- Other model files can be specified manually through the WebGUI interface

## Model Requirements
1. Models must be saved as `.pt` files (PyTorch format)
2. Input dimension: 2300 (state vector size)
3. Output dimension: 100 (action space size)

## Issues with Model Loading
If you encounter errors when loading a model:

1. **Architecture mismatch**: Your model architecture needs to match one of the supported architectures:
   - Standard Actor-Critic network with shared layers
   - Simple feed-forward network with action and value outputs

2. **Dimension mismatch**: Check that your model's input and output dimensions match the expected values (2300 and 100)

3. **State dictionary format**: The model should be saved either as a complete model or as a state dictionary that can be loaded into the expected architecture

## Debugging
- Look for error messages in the console output when starting the WebGUI
- Use the "Load Model" button in the WebGUI to specify a path to your model and see detailed error messages
- No fallback models are created - you must have a valid model to use the Model Bot feature 