import os
import glob
import json
import torch


def load_model_config(checkpoint_path):
    """
    Load model configuration from checkpoint directory
    
    Args:
        checkpoint_path: Path to checkpoint directory or file
        
    Returns:
        dict: Model configuration, or None if not found
    """
    # If path is a file, get its directory
    if os.path.isfile(checkpoint_path):
        checkpoint_path = os.path.dirname(checkpoint_path)
    
    config_file = os.path.join(checkpoint_path, 'model_config.json')
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Warning: Failed to load model config: {e}")
            return None
    
    return None


def find_checkpoint_file(checkpoint_path):
    """
    Find a checkpoint file given a path (file or directory)
    
    Args:
        checkpoint_path: Path to checkpoint file or directory
        
    Returns:
        str: Path to checkpoint file, or None if not found
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    
    # If it's a file, return it directly
    if os.path.isfile(checkpoint_path):
        return checkpoint_path
    
    # If it's a directory, search for checkpoint files
    if os.path.isdir(checkpoint_path):
        # Try common checkpoint names in order of preference
        possible_names = [
            'best_model.pt',
            'best_model.pth',
            'checkpoint.pth',
            'checkpoint.pt',
        ]
        
        for name in possible_names:
            ckpt_file = os.path.join(checkpoint_path, name)
            if os.path.exists(ckpt_file):
                return ckpt_file
        
        # Look for any .pt or .pth file
        ckpt_files = glob.glob(os.path.join(checkpoint_path, '*.pt')) + \
                     glob.glob(os.path.join(checkpoint_path, '*.pth'))
        if ckpt_files:
            return ckpt_files[0]  # Return first found
    
    return None


def load_checkpoint(model, checkpoint_path, device='cpu', strict=False):
    """
    Load model checkpoint with automatic file discovery and prefix handling
    
    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to checkpoint file or directory
        device: Device to load checkpoint on
        strict: Whether to strictly enforce state dict keys match
        
    Returns:
        bool: True if loaded successfully, False otherwise
    """
    if not checkpoint_path:
        return False
    
    # Find checkpoint file
    ckpt_file = find_checkpoint_file(checkpoint_path)
    
    if not ckpt_file:
        print(f"Warning: No checkpoint found at {checkpoint_path}")
        return False
    
    print(f"Loading checkpoint from {ckpt_file}")
    
    try:
        checkpoint = torch.load(ckpt_file, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Handle 'module.' prefix from DataParallel
        # Check if state_dict has 'module.' prefix but model doesn't expect it
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        
        has_module_prefix = any(k.startswith('module.') for k in checkpoint_keys)
        model_expects_module = any(k.startswith('module.') for k in model_keys)
        
        if has_module_prefix and not model_expects_module:
            # Remove 'module.' prefix from checkpoint keys
            print("  Removing 'module.' prefix from checkpoint keys...")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif not has_module_prefix and model_expects_module:
            # Add 'module.' prefix to checkpoint keys
            print("  Adding 'module.' prefix to checkpoint keys...")
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        
        # Load with strict=False to allow partial loading
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        
        if missing_keys or unexpected_keys:
            if len(missing_keys) == 0 and len(unexpected_keys) == 0:
                print("✓ Checkpoint loaded successfully (all keys matched)")
            elif len(missing_keys) > 0 and len(unexpected_keys) > 0:
                print(f"⚠ Checkpoint loaded with issues:")
                print(f"    - {len(missing_keys)} missing keys (model has, checkpoint doesn't)")
                print(f"    - {len(unexpected_keys)} unexpected keys (checkpoint has, model doesn't)")
                if len(missing_keys) <= 5:
                    print(f"    Missing: {missing_keys}")
                if len(unexpected_keys) <= 5:
                    print(f"    Unexpected: {unexpected_keys}")
            elif len(missing_keys) > 0:
                print(f"⚠ Checkpoint loaded: {len(missing_keys)} missing keys")
            else:
                print(f"⚠ Checkpoint loaded: {len(unexpected_keys)} unexpected keys")
        else:
            print("✓ Checkpoint loaded successfully (all keys matched)")
        
        return len(missing_keys) == 0  # Return True only if all keys loaded
        
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        return False
