import os
import glob
import json
import argparse
from transformers import AutoTokenizer, AutoModel, AutoConfig, PreTrainedTokenizerFast
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType, quant_pre_process
import torch


ap = argparse.ArgumentParser(
    description="Export SentenceTransformers Models to ONNX for use with letsearch"
)
ap.add_argument("-m", "--model", required=True, help="Model to export")
ap.add_argument("-o", "--output", required=True, help="Where to save the ONNX model")
ap.add_argument(
    "-d",
    "--description",
    required=False,
    default="",
    help="Description to add to the metadata file",
)
args = ap.parse_args()
model_path = args.model
output_path = args.output

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(output_path)
required_files = glob.glob(f"{output_path}/**")
required_files = [os.path.basename(path) for path in required_files]

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(
    model_path,
)
model.to(device)
model.eval()

onnx_path = f"{output_path}/model-f32.onnx"
onnx_f16_path = f"{output_path}/model-f16.onnx"
onnx_infer_path = f"{output_path}/model-infer.onnx"
onnx_int8_path = f"{output_path}/model-i8.onnx"


dummy_model_input = tokenizer("Using BERT with ONNX Runtime!", return_tensors="pt")
inputs = tuple(dummy_model_input.values())
input_names = tuple(dummy_model_input.keys())
dynamic_axes = {
    input_name: {0: "batch_size", 1: "sequence"} for input_name in input_names
}
dynamic_axes["last_hidden_state"] = {0: "batch_size", 1: "sequence"}
dynamic_axes["pooler_output"] = {0: "batch_size", 1: "sequence"}

torch.onnx.export(
    model,
    inputs,
    onnx_path,
    input_names=input_names,
    output_names=("last_hidden_state", "pooler_output"),
    dynamic_axes=dynamic_axes,
    do_constant_folding=True,
    opset_version=14,
    artifacts_dir="./artifacts",
    external_data=False,
)

print(f"saved f32 model to {onnx_path}")

model.half()
torch.onnx.export(
    model,
    inputs,
    onnx_f16_path,
    input_names=input_names,
    output_names=("last_hidden_state", "pooler_output"),
    dynamic_axes=dynamic_axes,
    do_constant_folding=True,
    opset_version=14,
    artifacts_dir="./artifacts",
    external_data=False,
)
print(f"Saved f16 model to {onnx_f16_path}")


quant_pre_process(onnx_path, onnx_infer_path, auto_merge=True)

quantize_dynamic(
    model_input=onnx_infer_path,
    model_output=onnx_int8_path,
    weight_type=QuantType.QInt8,
)

print(f"Saved i8 model to {onnx_int8_path}")

metadata = {
    "letsearch_version": 1,
    "converted_from": model_path,
    "description": args.description,
    "variants": [
        {"variant": "f32", "path": "model-f32.onnx"},
        {"variant": "f16", "path": "model-f16.onnx"},
        {"variant": "i8", "path": "model-i8.onnx"},
    ],
}

with open(f"{output_path}/metadata.json", "w") as f:
    f.write(json.dumps(metadata))
