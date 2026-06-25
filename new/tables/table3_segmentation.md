**Table 3.** Glacial-lake segmentation: fine-tuning Prithvi-EO-2.0-300M-TL.

| Config | Input | Decoder | Final loss | Test mIoU | Test F1 |
|---|---|---|---:|---:|---:|
| Two-stage baseline | 128px | — | — | 0.293 | 0.862 |
| Seg v1 | 448px | ConvTranspose | 0.293 | 0.29 | 0.83 |
| Seg v2 (final) | 224px | FPN (taps 5/11/17/23) | 0.292 | 0.34 | 0.86 |

Backbone frozen; LoRA (r=16, α=32) on QKV; loss = 0.5·BCE + 0.5·Dice.