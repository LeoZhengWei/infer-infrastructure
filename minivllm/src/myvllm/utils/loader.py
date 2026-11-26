import torch
from torch import nn
import os
from safetensors import safe_open
from transformers import AutoConfig
import re

def default_weight_loader(param, weight):
    """默认权重加载器：用于将权重数据复制到参数中"""
    if param.shape != weight.shape:
        raise ValueError(f"Shape mismatch: param {param.shape} vs weight {weight.shape}")
    param.data.copy_(weight)

def load_weights_from_checkpoint(model: nn.Module, model_name_or_path: str):
    """
    从huggingface模型导入权重到自定义模型
    为优化层处理QKV与gate_up权重合并
    参数：
        model:待加载权重的目标模型
        model_name_or_path: 本地检查点路径或huggingFace模型名称
    """
    from huggingface_hub import snapshot_download
    checkpoint_path = None
    # First, try local paths
    if model_name_or_path.startswith('~'):
        checkpoint_path = os.path.expanduser(model_name_or_path)
    elif os.path.isdir(model_name_or_path):
        checkpoint_path = model_name_or_path

    # If not a local path, try to download from HuggingFace
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        try:
            checkpoint_path = snapshot_download(
                repo_id=model_name_or_path,
                allow_patterns=["*.safetensors", "*.json"],
                ignore_patterns=["*.msgpack", "*.h5", "*.bin"]
            )
        except Exception as e:
            raise ValueError(
                f"Could not find or download model '{model_name_or_path}'. "
                f"Error: {e}\n"
                f"Please ensure the model name is correct or provide a valid local path. "
            )

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path not found: {checkpoint_path}")

    # Load all safetensors files in the checkpoint directory
    safetensor_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors')]

    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {checkpoint_path}")

    # Collect all weights from HF model
    hf_weights = {}
    for file in sorted(safetensor_files):
        file_path = os.path.join(checkpoint_path, file)
        with safe_open(file_path, framework='pt', device='cpu') as f:
            for weight_name in f.keys():
                hf_weights[weight_name] = f.get_tensor(weight_name)

    # Now map and load weights into custom model
    loaded_params = set()
    skipped_params = []

    # Process each HF weight
    for hf_name, hf_weight in hf_weights.items():
        try:
            # 1. Handle QKV merge (q_proj + k_proj + v_proj -> qkv_projection)
            if '.self_attn.q_proj.weight' in hf_name:
                layer_match = re.search(r'layers\.(\d+)', hf_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    k_name = hf_name.replace('q_proj', 'k_proj')
                    v_name = hf_name.replace('q_proj', 'v_proj')
                    
                    if k_name in hf_weights and v_name in hf_weights:
                        q_weight = hf_weight
                        k_weight = hf_weights[k_name]
                        v_weight = hf_weights[v_name]

                        # Concatenate q, k, v along output dimension
                        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

                        custom_name = f"model.layers.{layer_idx}.self_attn.qkv_projection.weight"
                        try:
                            param = model.get_parameter(custom_name)
                            param.data.copy_(qkv_weight)
                            loaded_params.add(custom_name)
                            loaded_params.add(hf_name)
                            loaded_params.add(k_name)
                            loaded_params.add(v_name)
                        except AttributeError:
                            skipped_params.append((custom_name, "Parameter not found"))
            
            # 2. Handle gate_up merge (gate_proj + up_proj -> gate_up)
            elif '.mlp.gate_proj.weight' in hf_name:
                layer_match = re.search(r'layers\.(\d+)', hf_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    up_name = hf_name.replace('gate_proj', 'up_proj')
                    
                    if up_name in hf_weights:
                        gate_weight = hf_weight
                        up_weight = hf_weights[up_name]

                        # Concatenate gate and up along output dimension
                        gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)

                        custom_name = f"model.layers.{layer_idx}.mlp.gate_up.weight"
                        try:
                            param = model.get_parameter(custom_name)
                            param.data.copy_(gate_up_weight)
                            loaded_params.add(custom_name)
                            loaded_params.add(hf_name)
                            loaded_params.add(up_name)
                        except AttributeError:
                            skipped_params.append((custom_name, "Parameter not found"))
            
            # 3. Handle gate_up merge for bias (if present)
            elif '.mlp.gate_proj.bias' in hf_name:
                layer_match = re.search(r'layers\.(\d+)', hf_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    up_bias_name = hf_name.replace('gate_proj', 'up_proj')

                    if up_bias_name in hf_weights:
                        gate_bias = hf_weight
                        up_bias = hf_weights[up_bias_name]
                        gate_up_bias = torch.cat([gate_bias, up_bias], dim=0)

                        custom_name = f"model.layers.{layer_idx}.mlp.gate_up.bias"
                        try:
                            param = model.get_parameter(custom_name)
                            param.data.copy_(gate_up_bias)
                            loaded_params.add(custom_name)
                            loaded_params.add(hf_name)
                            loaded_params.add(up_bias_name)
                        except AttributeError:
                            skipped_params.append((custom_name, "Parameter not found"))

            # 4. 跳过已经合并的 k_proj, v_proj, up_proj (already merged)
            elif any(x in hf_name for x in ['.k_proj', '.v_proj', '.up_proj']):
                if hf_name not in loaded_params:
                    skipped_params.append((hf_name), "Merged into qkv_projection or gate_up")

            # 5. 所有其他参数：直接加载（名字匹配HF）
            else:
                try:
                    param = model.get_parameter(hf_name)
                    if param.shape != hf_weight.shape:
                        if len(param.shape) > 0 and len(hf_weight.shape) > 0:
                            min_size = min(param.shape[0], hf_weight.shape[0])
                            param.data[:min_size].copy_(hf_weight[:min_size])
                        else:
                            param.data.copy_(hf_weight)
                    else:
                        param.data.copy_(hf_weight)
                    loaded_params.add(hf_name)
                except AttributeError:
                    skipped_params.append((hf_name, "Parameter not found"))
        
        except Exception as e:
            skipped_params.append((hf_name, f"Error: {str(e)}"))

    # 检查未加载的参数
    unloaded_params = []
    for name, param in model.named_parameters():
        if name not in loaded_params:
            unloaded_params.append(name)

    print(f"\n{'='*80}")
    print(f"Weight Loading Summary:")
    print(f"{'='*80}")
    print(f"Successfully loaded: {len([p for p in loaded_params if not any(x in p for x in ['.k_proj.', '.v_proj.', '.up_proj.'])])} parameter groups")
    
    if unloaded_params:
        print(f"\n WARNING: {len(unloaded_params)} model parameters NOT loaded from checkpoint:")
        for name in unloaded_params[:15]:
            param = dict(model.named_parameters())[name]
            print(f" - {name} (shape: {param.shape}, mean: {param.data.mean():.6f})")
        if len(unloaded_params) > 15:
            print(f" ... and {len(unloaded_params) - 15} more")

    if skipped_params:
        merged_skips = [s for s in skipped_params if "Merged" in s[1]]
        not_found_skips = [s for s in skipped_params if "not found" in s[1]]
        no_mapping_skips = [s for s in skipped_params if "No mapping" in s[1]]

        if merged_skips:
            print(f"Skipped (merged into other weights) : {len(merged_skips)}")
        if not_found_skips:
            print(f"Skipped (not found in model) : {len(not_found_skips)}")
            for name, reason in not_found_skips[:5]:
                print(f" - {name}")
            if len(not_found_skips) > 5:
                print(f" ... and {len(not_found_skips) - 5} more")
        if no_mapping_skips:
            print(f"Skipped (no mapping rule) : {len(no_mapping_skips)}")
            for name, reason in no_mapping_skips[:5]:
                print(f" - {name}")
            if len(no_mapping_skips) > 5:
                print(f" ... and {len(no_mapping_skips) - 5} more")

    print(f"{'='*80}")
    return loaded_params
                    