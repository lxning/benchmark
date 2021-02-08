import torch
from torchsummary import summary

# https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/
# https://nvidia.github.io/TRTorch/_notebooks/ssd-object-detection-demo.html

# List of available models in PyTorch Hub from Nvidia/DeepLearningExamples
torch.hub.list('NVIDIA/DeepLearningExamples:torchhub')
['checkpoint_from_distributed',
 'nvidia_ncf',
 'nvidia_ssd',
 'nvidia_ssd_processing_utils',
 'nvidia_tacotron2',
 'nvidia_waveglow',
 'unwrap_distributed']

# load SSD model pretrained on COCO from Torch Hub
precision = 'fp32'
ssd300 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)

# Next, we run object detection
model = ssd300.eval().to("cuda")
summary(model, (3,300,300))
traced_model = torch.jit.trace(model, [torch.randn((1,3,300,300)).to("cuda")])
# This is just an example, and not required for the purposes of this demo
torch.jit.save(traced_model, "model.pt")
