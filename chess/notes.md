# Model Sizes
Table 1 Configurations of ViT models

| Model       | Layers | Width |  MLP | Heads | Params |
| ----------- | :----: | :---: | :--: | :---: | :----: |
| ViT-Ti (39) |   12   |  192  |  768 |   3   |  5.8 M |
| ViT-S (39)  |   12   |  384  | 1536 |   6   | 22.2 M |
| ViT-B (13)  |   12   |  768  | 3072 |   12  |  86 M  |
| ViT-L (13)  |   24   |  1024 | 4096 |   16  |  307 M |

Patch Size: 16/32

---

**Citation**
[Steiner et al., “How to Train Your ViT? Data, Augmentation, and Regularization in Vision Transformers,” TMLR 2022](https://arxiv.org/abs/2106.10270)

# Choices
1. Chinchila optimal training regime says
  Param:Data :: 1:20
  For out small 2.5-3 M param model, data: 50-60M unique tokens
  Each board has ctx length of 77, thus we need: 50-50M / 77 = 650-780k boards
  **(Currently we have 640k board positions tokenized which we will use for the experiments)**
2. RecordBatch in dataset is 1k. So one step means trained on 1k board positions.
  For our small model optimal batch size will always be smaller than this, so no problem while loading data in memory

3. Constant LR after warmup is perfectly reasonable if you pick a good base rate (e.g. the Chinchilla-rule — 0.003 × batch_size/256).
5. Training regime: warmup during first 5% and then constant learning rate
4. Number of checkpoints, 1 after warmup and 1 at the end. In between 4 evenly spaced checkpoints.
  5% 640k boards = 32k

  | Phase        | Step    | Notes                 |
| ------------ | ------- | --------------------- |
| Warmup end   | 32,000  | ✔️ First checkpoint   |
| Stable early | 160,000 | 20% into stable phase |
| Stable mid 1 | 288,000 | 40% into stable phase |
| Stable mid 2 | 416,000 | 60% into stable phase |
| Stable late  | 544,000 | 80% into stable phase |
| Final        | 640,000 | ✔️ End of training    |

