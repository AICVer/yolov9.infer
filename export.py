import torch

checkpoint_model = 'weights/gelan-c.pt'
checkpoint = torch.load(checkpoint_model)
model = checkpoint['model']
model.float()
model.eval()

for name, module in model.named_modules():
    if hasattr(module, 'fuse_convs'):
        module.fuse_convs()
        module.forward = module.forward_fuse

inputs = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model,
    inputs, 
    'weights/gelan-c.onnx'
)
