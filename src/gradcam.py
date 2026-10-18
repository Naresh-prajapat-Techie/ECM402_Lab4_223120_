"""Task 6: Grad-CAM from scratch (hook on model.layer4[-1], block OUTPUT after final ReLU)."""
import torch, torch.nn.functional as F


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self.acts = out.detach()
        out.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x, cls=None):
        self.model.eval(); self.model.zero_grad()
        x = x.clone().requires_grad_(True)           # guarantees a graph even if backbone frozen
        logits = self.model(x)
        cls = int(logits.argmax(1)) if cls is None else cls
        logits[0, cls].backward()
        w = self.grads.mean((2, 3), keepdim=True)    # channel weights = GAP of gradients
        cam = F.relu((w * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), cls, logits.softmax(1)[0].detach().cpu()
