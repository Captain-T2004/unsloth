import os
import gc
import argparse
import subprocess
import tempfile
from pathlib import Path
import torch
import numpy as np
from gguf import GGUFWriter
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLConfig, AutoProcessor

CONVERSION_SCRIPT = "./llama.cpp/convert_hf_to_gguf.py"

def to_gguf_name(name: str) -> str:
    name = name.replace("text_model", "t").replace("vision_model", "v")
    name = name.replace("blocks", "blk").replace("embeddings.", "")
    name = name.replace("attn.", "attn_").replace("mlp.fc1", "ffn_down").replace("mlp.fc2", "ffn_up")
    name = name.replace("proj.", "out.").replace("norm1", "ln1").replace("norm2", "ln2")
    name = name.replace("merger.mlp", 'mm')
    return name

def process_single_tensor(name, tensor, fout, np_dtype):
    ten = tensor.cpu().detach().numpy()
    tensor_name = to_gguf_name(name)
    data = ten.astype(np.float32) if ten.ndim <= 1 else ten.astype(np_dtype)
    fout.add_tensor(tensor_name, data)
    del ten, data
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

def convert_vision_model(qwen2vl, np_dtype, fout):
    vision_model = qwen2vl.visual
    for name, tensor in vision_model.state_dict().items():
        print(f"Processing vision tensor: {name}")
        process_single_tensor(name, tensor, fout, np_dtype)
    fout.add_tensor("v.position_embd.weight", np.zeros([10, 10], dtype=np.float32))

def convert_llm(model_path, outdir):
    model_name = Path(model_path).stem
    model_dir = Path(model_path).parent

    pattern = "*.bin"
    model_files = list(model_dir.glob(pattern))
    if not model_files:
        raise Exception("No .bin model files found in the specified directory.")

    fp16 = Path(outdir) / f"{model_name}.fp16.gguf"
    result = subprocess.run([
        "python", CONVERSION_SCRIPT, model_path, "--outtype", "f16", "--outfile", fp16
    ], shell=False, capture_output=True)
    if result.returncode != 0:
        raise Exception(f"Error converting to fp16: {result.stderr.decode('utf-8')}")
    print("LLM converted to fp16 successfully!")
    return fp16

def main(args):
    dtype, np_dtype, ftype = (torch.float32, np.float32, 0) if args.data_type == 'fp32' else (torch.float16, np.float16, 1)
    model_path = os.path.abspath(args.model_name)
    fname_out = f"{os.path.basename(model_path).lower()}-full.gguf"

    print("Loading Qwen2VL model...")
    qwen2vl = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, device_map="auto")
    config = Qwen2VLConfig.from_pretrained(model_path)

    with tempfile.TemporaryDirectory(dir="outputs") as outdir:
        fp16_llm = convert_llm(model_path, outdir)

        fout = GGUFWriter(path=fname_out, arch="clip")
        fout.add_description("Full Qwen2-VL Model (Vision + LLM) GGUF Conversion")
        fout.add_file_type(ftype)
        fout.add_bool("clip.has_vision_encoder", True)
        fout.add_bool("clip.has_text_encoder", True)
        fout.add_uint32("clip.vision.patch_size", config.vision_config.patch_size)

        convert_vision_model(qwen2vl, np_dtype, fout)

        fout.add_file(fp16_llm, "llm_model")

        fout.write_header_to_file()
        fout.write_kv_data_to_file()
        fout.write_tensors_to_file()
        fout.close()
        print(f"Full model conversion completed: {fname_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--data_type", choices=['fp32', 'fp16'], default="fp16")
    args = parser.parse_args()
    main(args)