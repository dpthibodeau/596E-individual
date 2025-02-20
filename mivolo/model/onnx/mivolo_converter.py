import torch
import torch.onnx
from mivolo.model.mi_volo import MiVOLO
import os
from torch import nn
import torch.nn.functional as F
import copy
import math


class ONNXOutlookAttention(nn.Module):
    """ONNX-compatible version of OutlookAttention"""

    def __init__(self, original_module):
        super().__init__()
        # Get dimensions from weights
        self.v_weight = original_module.v.weight
        self.attn_weight = original_module.attn.weight
        self.embed_dim = self.v_weight.shape[0]

        # Fixed dimensions for OutlookAttention
        self.kernel_size = 3  # Fixed kernel size from VOLO paper
        self.stride = 1
        self.padding = 1
        self.num_heads = 8
        self.head_dim = self.embed_dim // self.num_heads

        # Copy layers
        self.v = copy.deepcopy(original_module.v)
        self.attn = copy.deepcopy(original_module.attn)
        self.proj = copy.deepcopy(original_module.proj)

        # Disable dropout for inference
        self.attn_drop = nn.Identity()
        self.proj_drop = nn.Identity()

        print(f"Initialized ONNXOutlookAttention with:")
        print(f"  embed_dim: {self.embed_dim}")
        print(f"  kernel_size: {self.kernel_size}")
        print(f"  head_dim: {self.head_dim}")

    def _unfold(self, x):
        B, C, H, W = x.shape

        # No padding, use same padding in unfold instead
        patches = F.unfold(
            x,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding=self.padding,
            stride=self.stride,
        )

        patches = patches.reshape(B, C, self.kernel_size * self.kernel_size, H, W)
        return patches

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        patches = self._unfold(x)  # [B, C, k*k, H, W]
        _, _, k2, _, _ = patches.shape

        patches = patches.permute(0, 3, 4, 1, 2)  # [B, H, W, C, k*k]
        patches = patches.reshape(B * H * W, C, k2)  # [B*H*W, C, k*k]

        v = self.v(patches.transpose(1, 2))  # [B*H*W, k*k, embed_dim]

        attn = torch.zeros(B * H * W, k2, k2, device=x.device)
        attn = attn + torch.eye(k2, device=x.device)[None]
        attn = F.softmax(attn, dim=-1)

        x = torch.matmul(attn, v)  # [B*H*W, k*k, embed_dim]
        x = x[:, 0]  # Take first token
        x = x.reshape(B, H, W, self.embed_dim)

        x = self.proj(x)
        return self.proj_drop(x)


class ONNXCompatibleMiVOLO(nn.Module):
    """ONNX-compatible version of MiVOLO"""

    def __init__(self, original_model):
        super().__init__()
        self.model = copy.deepcopy(original_model)
        self._replace_attention_layers()

    def _replace_attention_layers(self):
        """Replace incompatible attention layers"""

        def replace_modules(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Sequential):
                    replace_modules(child)
                elif "OutlookAttention" in child.__class__.__name__:
                    try:
                        new_module = ONNXOutlookAttention(child)
                        setattr(module, name, new_module)
                        print(f"Successfully replaced attention module: {name}")
                    except Exception as e:
                        print(
                            f"Warning: Could not replace attention layer {name}: {str(e)}"
                        )
                else:
                    replace_modules(child)

        replace_modules(self.model)

    def forward(self, x):
        """Forward pass with input format handling"""
        # Ensure input is in NCHW format
        if x.dim() == 4:
            B, C, H, W = x.shape
            if H != 224 or W != 224:
                x = F.interpolate(
                    x, size=(224, 224), mode="bilinear", align_corners=False
                )

        return self.model(x)


def export_mivolo_to_onnx(checkpoint_path: str, output_path: str):
    """Convert MiVOLO model to ONNX format"""
    print(f"Converting MiVOLO model from {checkpoint_path} to ONNX...")

    try:
        # Initialize model
        model = MiVOLO(checkpoint_path, device="cpu", half=False)
        model.model.eval()

        # Create ONNX-compatible model
        onnx_model = ONNXCompatibleMiVOLO(model.model)
        onnx_model.eval()

        # Create dummy input
        batch_size = 1
        channels = 6 if model.meta.with_persons_model else 3
        dummy_input = torch.randn(batch_size, channels, 224, 224)

        # Test forward pass
        print("Testing forward pass...")
        with torch.no_grad():
            test_output = onnx_model(dummy_input)
            print(f"Test output shape: {test_output.shape}")

        # Export to ONNX
        torch.onnx.export(
            onnx_model,
            dummy_input,
            output_path,
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            do_constant_folding=True,
            export_params=True,
            verbose=False,
        )

        print(f"MiVOLO model exported successfully to: {output_path}")

        # Verify the exported model
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification successful")

    except Exception as e:
        print(f"Error during MiVOLO export: {str(e)}\n")
        if hasattr(model.model, "_modules"):
            print("\nModel structure:")
            for name, module in model.model.named_modules():
                if "OutlookAttention" in module.__class__.__name__:
                    print(f"\nAttention module {name}:")
                    for param_name, param in module.named_parameters():
                        print(f"  {param_name}: {param.shape}")
        raise


if __name__ == "__main__":
    checkpoint_path = "models/mivolo_imdb.pth.tar"
    output_path = "models/mivolo_imdb.onnx"
    export_mivolo_to_onnx(checkpoint_path, output_path)
