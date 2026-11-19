# 600M-Parameter Llama Transformer (From Scratch)

A decoder-only transformer language model built from scratch in PyTorch and trained on multi-GPU Kaggle environments.

## Architecture & Features
- **Model Components:** RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU feed-forward networks, and Grouped-Query Attention (GQA).
- **Training Pipeline:** Multi-GPU execution (`nn.DataParallel`), FP16 mixed-precision (`GradScaler`), gradient clipping, weight tying, and automated checkpointing.
- **Data Streaming:** Multithreaded background worker tokenizing and batching `FineWeb-Edu`, `Cosmopedia`, and Python code in real time.
