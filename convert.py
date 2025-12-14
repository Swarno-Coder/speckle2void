# ==========================================
# 🛠️ EXPORT PIPELINE: PyTorch -> Quantized ONNX
# ==========================================
import torch.onnx
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from unet import BlindSpotUNet

print("⚙️ Starting Export Process...")

# 1. Load your Best Trained Weights
# Re-init model structure
model = BlindSpotUNet().to('cpu') 
model.load_state_dict(torch.load("best_speckle2void.pth", map_location='cpu'))
model.eval()

# 2. Export to Standard ONNX (Float32)
dummy_input = torch.randn(1, 1, 256, 256) # Batch size 1, 1 Channel
onnx_path = "speckle2void_raw.onnx"

torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path, 
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print(f"✅ Exported to {onnx_path}")

# 3. Quantize to INT8 (The Speed Boost)
# This converts 32-bit floats to 8-bit integers, making the model 4x smaller and faster on CPU.
quantized_path = "speckle2void_quantized.onnx"
quantize_dynamic(
    onnx_path,
    quantized_path,
    weight_type=QuantType.QUInt8
)

print(f"⚡ Quantized Model Saved: {quantized_path}")