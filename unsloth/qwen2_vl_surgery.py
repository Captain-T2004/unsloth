import argparse
import os
from typing import Dict

import torch
import numpy as np
from gguf import *
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    AutoProcessor,
    Qwen2VLConfig
)


VISION = "clip.vision"


def k(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)


def to_gguf_name(name: str) -> str:
    og = name
    name = name.replace("text_model", "t").replace("vision_model", "v")
    name = name.replace("blocks", "blk").replace("embeddings.", "")
    name = name.replace("attn.", "attn_")
    name = name.replace("mlp.fc1", "ffn_down").replace("mlp.fc2", "ffn_up").replace("proj.", "out.")
    # name = name.replace("layrnorm", "ln").replace("layer_norm", "ln").replace("layernorm", "ln")
    name = name.replace("norm1", "ln1").replace("norm2", "ln2")
    name = name.replace("merger.mlp", 'mm')
    print(f"[to_gguf_name] {og} --> {name}")
    return name


def find_vision_tensors(qwen2vl, dtype) -> Dict[str, np.ndarray]:
    vision_model = qwen2vl.visual
    tensor_map = {}
    for name, ten in vision_model.state_dict().items():
        # Ensure we're working with CPU tensors before converting to numpy
        ten = ten.cpu().detach()
        ten_np = ten.numpy()  # Convert to numpy array

        if 'qkv' in name:
            if ten_np.ndim == 2: # weight
                c3, _ = ten_np.shape
            else:             # bias
                c3 = ten_np.shape[0]
            assert c3 % 3 == 0
            c = c3 // 3
            wq = ten_np[:c].copy()  # Make sure to create a copy
            wk = ten_np[c: c * 2].copy()
            wv = ten_np[c * 2:].copy()
            tensor_map[to_gguf_name(f"vision_model.{name}").replace("qkv", "q")] = wq
            tensor_map[to_gguf_name(f"vision_model.{name}").replace("qkv", "k")] = wk
            tensor_map[to_gguf_name(f"vision_model.{name}").replace("qkv", "v")] = wv
        elif 'merger' in name:
            if name.endswith("ln_q.weight"):
                tensor_map['v.post_ln.weight'] = ten_np.copy()
            elif name.endswith("ln_q.bias"):
                tensor_map['v.post_ln.bias'] = ten_np.copy()
            else:
                # "merger.mlp.%d.weight/bias" --> "mm.%d.weight/bias"
                tensor_map[to_gguf_name(name)] = ten_np.copy()
        elif 'patch_embed.proj.weight' in name:
            # NOTE: split Conv3D into Conv2Ds
            c1, c2, kt, kh, kw = ten_np.shape
            assert kt == 2, "Current implementation only supports temporal_patch_size of 2"
            tensor_map["v.patch_embd.weight"] = ten_np[:, :, 0, ...].copy()
            tensor_map["v.patch_embd.weight.1"] = ten_np[:, :, 1, ...].copy()
        else:
            tensor_map[to_gguf_name(f"vision_model.{name}")] = ten_np.copy()

    # Convert data types carefully
    for new_name, ten in list(tensor_map.items()):  # Use list() to avoid dictionary size change during iteration
        if ten.ndim <= 1 or new_name.endswith("_norm.weight"):
            tensor_map[new_name] = np.asarray(ten, dtype=np.float32)
        else:
            tensor_map[new_name] = np.asarray(ten, dtype=dtype)

    # Add dummy tensor with explicit numpy array creation
    tensor_map["v.position_embd.weight"] = np.zeros([10, 10], dtype=np.float32)

    return tensor_map


def main(args):
    if args.d_type == 'fp32':
        dtype = torch.float32
        np_dtype = np.float32
        ftype = 0
    elif args.d_type == 'fp16':
        dtype = torch.float32  # Keep PyTorch dtype as float32
        np_dtype = np.float16  # Set numpy dtype as float16
        ftype = 1
    else:
        raise ValueError(f"Unsupported data type: {args.d_type}")

    local_model = False
    model_path = args.model_name
    original_model_name = args.model_name

    # Determine if model_name is a local directory or a model ID
    if os.path.isdir(model_path):
        local_model = True
        if model_path.endswith(os.sep):
            model_path = model_path[:-1]
        model_name = os.path.basename(model_path)
    else:
        model_name = model_path.replace('/', '-')

    print(f"Model name: {model_name}")
    print(f"Model path: {model_path if local_model else 'HuggingFace Hub'}")

    # Set output directory to same as input model if it's a local directory
    if local_model:
        output_dir = model_path
    else:
        # If output_dir is specified, use it; otherwise use current directory
        output_dir = args.output_dir if args.output_dir else os.getcwd()

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Full path for output file
    fname_out = os.path.join(output_dir, f"{model_name.lower()}-vision.gguf")

    # Load the model
    try:
        print(f"Loading model from {original_model_name}...")
        qwen2vl = Qwen2VLForConditionalGeneration.from_pretrained(
            original_model_name, torch_dtype=dtype, device_map="auto"
        )
        cfg: Qwen2VLConfig = qwen2vl.config
        vcfg = cfg.vision_config
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        print(f"Creating GGUF file: {fname_out}")
        fout = GGUFWriter(path=fname_out, arch="clip")
        fout.add_description("image encoder for Qwen2VL")

        fout.add_file_type(ftype)
        fout.add_bool("clip.has_text_encoder", False)
        fout.add_bool("clip.has_vision_encoder", True)
        fout.add_bool("clip.has_qwen2vl_merger", True)
        fout.add_string("clip.projector_type", "qwen2vl_merger")

        print("Vision configuration:", cfg.vision_config)
        if 'silu' in cfg.vision_config.hidden_act.lower():
            fout.add_bool("clip.use_silu", True)
            fout.add_bool("clip.use_gelu", False)
        elif 'gelu' in cfg.vision_config.hidden_act.lower():
            fout.add_bool("clip.use_silu", False)
            fout.add_bool("clip.use_gelu", 'quick' not in cfg.vision_config.hidden_act.lower())
        else:
            raise ValueError(f"Unsupported activation function: {cfg.vision_config.hidden_act}")

        print("Finding vision tensors...")
        tensor_map = find_vision_tensors(qwen2vl, np_dtype)

        print(f"Adding {len(tensor_map)} tensors to GGUF...")
        for name, data in tensor_map.items():
            # Ensure data is a valid numpy array with correct dtype
            if not isinstance(data, np.ndarray):
                print(f"Warning: {name} is not a numpy array, converting...")
                data = np.array(data, dtype=np_dtype)
            fout.add_tensor(name, data)

        fout.add_uint32("clip.vision.patch_size", vcfg.patch_size)
        fout.add_uint32("clip.vision.image_size", 14 * 40)  # some reasonable size that is divisible by (14*2)
        fout.add_uint32(k(KEY_EMBEDDING_LENGTH, VISION), vcfg.embed_dim)
        fout.add_uint32("clip.vision.projection_dim", vcfg.hidden_size)
        fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, VISION), vcfg.num_heads)
        fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, VISION), 1e-6)
        fout.add_uint32(k(KEY_BLOCK_COUNT, VISION), vcfg.depth)
        fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, VISION), 0)  # not sure what this does, put 0 here as a placeholder
        fout.add_name(model_name)
        """
        HACK: Since vision rope related parameter aren't stored in the `Qwen2VLConfig,
                it will be hardcoded in the `clip_image_build_graph` from `clip.cpp`.
        """

        print("Loading processor...")
        if local_model:
            processor: Qwen2VLProcessor = AutoProcessor.from_pretrained(model_path)
        else:
            processor: Qwen2VLProcessor = AutoProcessor.from_pretrained(original_model_name)

        # Convert image_mean and image_std to numpy arrays if they aren't already
        image_mean = np.array(processor.image_processor.image_mean, dtype=np.float32)
        image_std = np.array(processor.image_processor.image_std, dtype=np.float32)

        fout.add_array("clip.vision.image_mean", image_mean)
        fout.add_array("clip.vision.image_std", image_std)

        print("Writing GGUF file...")
        fout.write_header_to_file()
        fout.write_kv_data_to_file()
        fout.write_tensors_to_file()
        fout.close()
        print(f"Model saved successfully as: {fname_out}")

    except Exception as e:
        print(f"Error during conversion: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Qwen2VL model to GGUF format")
    parser.add_argument("model_name", nargs='?', default="Qwen/Qwen2-VL-2B-Instruct",
                        help="Path to local model directory or HuggingFace model ID")
    parser.add_argument("--d_type", choices=['fp32', 'fp16'], default="fp16",
                        help="Data type for the output GGUF file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save the output GGUF file (defaults to same as input for local models)")
    args = parser.parse_args()
    main(args)