[![Project Page](https://img.shields.io/badge/Project%20Page-darkred)](https://tobias.nauen-it.de/publication/taylor-shift/)
[![ICPR Abstract](https://img.shields.io/badge/ICPR%202024-Abstract-forest)](https://link.springer.com/chapter/10.1007/978-3-031-78172-8_1)
[![arXiv](https://img.shields.io/badge/arXiv-2403.02920-b31b1b?logo=arxiv)](https://arxiv.org/abs/2403.02920)
![Two Modes](https://img.shields.io/badge/Modes-Efficient%20&%20Direct-purple)
[![Static Badge](https://img.shields.io/badge/Citation-BibTeX-blue)](https://tobias.nauen-it.de/publication/taylor-shift/cite.bib)

# TaylorShift: Shifting the Complexity of Self-Attention from Squared to Linear and Back using Taylor-Softmax

This is the code appendix for the paper [*TaylorShift: Shifting the Complexity of Self-Attention from Squared to Linear (and Back) using Taylor-Softmax*](https://arxiv.org/abs/2403.02920).

For now, we can only include the code for the TaylorShift attention mechanism and models.
We will include the code for the experiments and analysis later on.
Our code is based on the [Timm](https://github.com/huggingface/pytorch-image-models) package, which is a PyTorch package for computer vision models.

## Updates
- [03.12.2024] *TaylorShift* was published in the [ICPR 2024 proceedings](https://link.springer.com/chapter/10.1007/978-3-031-78172-8_1) :book:
- [07.09.2024] We have created a [blog post on *TaylorShift*](https://tobias.nauen-it.de/publication/taylor-shift/), containing a condensed version of the paper and all the important links.
- [01.08.2024] We are happy to anounce that *TaylorShift* has been **accepted for publication at ICPR 2024** :tada: :fireworks:
- [18.07.2024] We have updated the paper on [ArXiv](https://arxiv.org/abs/2403.02920) :newspaper_roll:
- [17.07.2024] We release the code for the [training](training/README.md) :test_tube:
- [17.07.2024] We release the code for [validating out theoretical analysis](analysis/README.md) :mag:


## Models

You can instantiate a Transformer or ViT model with the TaylorShift attention mechanism by importing the corresponding classes from `taylor_shift`:

### Transformer Architecture

```python
import torch
from taylor_shift import TaylorShiftTransformer

max_seq_len = 4096
input_dim = 256
model = TaylorShiftTransformer(max_seq_len, input_dim, num_classes=1000,
                               num_heads=8, depth=6, embed_dim=256)

bs = 32
seq_len = 1024  # or any other sequence length <= max_seq_len
x = torch.randn(bs, seq_len, input_dim)
pred = model(x)  # (bs, num_classes)
```

### ViT Architecture

```python
import torch
from taylor_shift import TaylorShiftViT
from utils import vit_sizes

size = vit_sizes['S']
image_size = 224
model = TaylorShiftViT(image_size, num_classes=1000, patch_size=16, **size)

bs = 32
x = torch.randn(bs, 3, image_size, image_size)
pred = model(x)  # (bs, num_classes)
```

### Implementation Versions

To switch between using direct-TaylorShift and efficient-TaylorShift, you can set the threshold value N0.
The model will automatically use efficient-TaylorShift when the sequence length is greater than N0, otherwise it will use direct-TaylorShift.

```python
model = ...  # as above
print(model.N0)  # by default, we set N0 to the theoretical threshold value based on the dimension d = embed_dim//num_heads

model.N0 = -1  # to use efficient-TaylorShift for all sequence lengths

model.N0 = 2**64  # to use direct-TaylorShift for (almost) all sequence lengths
```

### Difference Between Implementations

To validate that the difference between the direct-TaylorShift and efficient-TaylorShift implementations is only numerical, we provide a script to compare the two implementations.

```python
import torch
from taylor_shift import TaylorShiftAttention

# example parameters
bs = 128
d_embed = 256
heads = 8
seq_len = 1024

attn = TaylorShiftAttention(d_embed, num_heads=heads)

q, k, v = torch.rand(3, bs, heads, seq_len, d_embed//heads).unbind(0)
y_dir = attn._direct_attention(q, k, v)
y_eff = attn._efficient_attention(q, k, v)

print(f"Difference: abs={(y_dir - y_eff).abs().mean(dim=0).max()} -> rel={(2* (y_dir - y_eff)/(y_dir + y_eff)).abs().mean(dim=0).max()}")
```

## Experiments
See [training/README.md](training/README.md) for training code. 


# Citation
If you use this code in your project, please cite:
```BibTex
@inproceedings{Nauen2024TaylorShift,
  title     = {TaylorShift: Shifting the Complexity of Self-Attention from Squared to Linear (and Back) using Taylor-Softmax},
  author    = {Tobias Christian Nauen and Sebastian Palacio and Andreas Dengel},
  note      = {ICPR 2024 (oral)},
  editor    = {Antonacopoulos, Apostolos and Chaudhuri, Subhasis and Chellappa, Rama and Liu, Cheng-Lin and Bhattacharya, Saumik and Pal, Umapada},
  booktitle = {Pattern Recognition},
  year      = {2024},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  pages     = {1--16},
  isbn      = {978-3-031-78172-8},
  doi       = {10.1007/978-3-031-78172-8_1}
}
```
