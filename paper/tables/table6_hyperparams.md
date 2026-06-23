**Table 6.** Key hyper-parameters and GRPO reward design.

| Component | Setting |
|---|---|
| Backbone | Prithvi-EO-2.0-300M-TL (ViT-L, 1024-d, 24 blocks) |
| Adaptation | LoRA r=16, α=32 on QKV; backbone otherwise frozen |
| Seg decoder | FPN, taps {5,11,17,23}, 256-d laterals |
| Seg loss | 0.5·BCE + 0.5·Dice |
| Seg optim | AdamW, lr 3e-5 (LoRA) / 5e-4 (decoder), cosine, 5% warmup |
| Reasoning head | visual(1024→256) ⊕ physics(3→256) → 3-way classifier |
| Physics features | lake area (km²), annual change (%), distance to known GLOF (km) |
| RL algorithm | GRPO, group G=2, advantage = (R−μ)/σ, no critic |
| Reward | 0.5·label + 0.2·area-consistency + 0.2·change-consistency + 0.1·format |
| RL optim | AdamW, lr 1e-4, 0.7·PG + 0.3·CE, 20 epochs |
