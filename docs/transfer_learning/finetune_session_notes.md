# Finetune Session Notes

- Base model: lightweight CNN backbone
- Strategy: freeze backbone, train classifier head first
- Optimizer: AdamW, lr 1e-4
- Epoch warmup: 3
