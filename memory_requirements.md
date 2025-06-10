# Memory Requirements for Meta-Learning Baselines

## Overview
The memory requirements vary significantly between different meta-learning algorithms. Here's a detailed analysis of each method's memory usage and limitations.

## 1. FullMAML (Second-Order MAML)
**Memory Complexity: HIGH** ⚠️

### Key Memory Consumers:
- **Second-order gradients**: Uses `create_graph=True` in `torch.autograd.grad()` (line 133)
- **Computation graph retention**: Maintains full computational graph through inner loop updates
- **Parameter cloning**: Creates copies of all LoRA parameters for each task in batch

### Memory Requirements:
```
Memory ≈ O(num_params × inner_steps × batch_size × 2)
```
- Factor of 2 comes from storing both first and second-order gradient information

### Limitations:
- May cause OOM (Out of Memory) errors on consumer GPUs (< 16GB VRAM)
- Batch size and inner steps directly multiply memory usage
- Not recommended for large models or many inner adaptation steps

### Mitigation Strategies:
- Reduce `meta_batch_size` (currently 4 in config)
- Reduce `inner_steps` (currently 1 in config)
- Use gradient checkpointing (already partially implemented)
- Consider using mixed precision training

## 2. FOMAML (First-Order MAML)
**Memory Complexity: MODERATE** ✓

### Key Memory Consumers:
- **First-order gradients only**: Regular `backward()` without graph retention
- **Parameter cloning**: Stores original parameters and accumulates gradients
- **Gradient accumulation**: Stores gradients for each task before averaging

### Memory Requirements:
```
Memory ≈ O(num_params × batch_size)
```
- No multiplication by inner_steps since gradients aren't retained through adaptation

### Advantages:
- 50-70% less memory than FullMAML
- Suitable for most consumer GPUs (8-16GB VRAM)
- Performance often comparable to FullMAML

### Current Implementation:
- Only optimizes LoRA parameters (line 295), not full model
- Uses gradient accumulation pattern (lines 377-389)
- Properly manages memory by cloning parameters only once

## 3. Reptile
**Memory Complexity: LOW** ✅

### Key Memory Consumers:
- **Parameter storage**: Only stores initial and final parameters
- **No gradient accumulation**: Updates are computed on-the-fly
- **Single task processing**: Processes one task at a time

### Memory Requirements:
```
Memory ≈ O(num_params × 2)
```
- Factor of 2 for storing initial and adapted parameters

### Advantages:
- Most memory-efficient meta-learning method
- Can run on GPUs with as little as 4-6GB VRAM
- Scales well to larger models
- Simple implementation reduces overhead

### Current Implementation:
- Processes single task per iteration (line 487)
- Direct parameter interpolation (line 521)
- No gradient graph retention

## 4. Memory Comparison Table

| Method | Relative Memory | GPU Requirement | Recommended Use Case |
|--------|----------------|-----------------|---------------------|
| FullMAML | 100% (baseline) | 16GB+ | Research, small models |
| FOMAML | 30-50% | 8GB+ | Production, medium models |
| Reptile | 10-20% | 4GB+ | Large models, limited resources |

## 5. Configuration Impact on Memory

### Current Configuration (from config.yml):
```yaml
meta_learning:
  default_meta_batch_size: 4
  inner_loop_steps: 1
  
lora:
  default_rank: 8
```

### Memory Scaling Factors:
1. **meta_batch_size**: Linear scaling for MAML methods
2. **inner_loop_steps**: Linear scaling for FullMAML, minimal impact on others
3. **lora_rank**: Quadratic impact on parameter count (rank × hidden_dim × 2)
4. **sequence_length**: Quadratic impact due to attention mechanisms

## 6. Practical Recommendations

### For Limited Memory (< 8GB):
- Use Reptile
- Set `meta_batch_size: 1`
- Set `lora_rank: 4`
- Use smaller model variants

### For Moderate Memory (8-16GB):
- Use FOMAML (recommended)
- Set `meta_batch_size: 2-4`
- Keep `lora_rank: 8`
- Enable gradient checkpointing

### For High Memory (16GB+):
- Can use FullMAML if needed
- Set `meta_batch_size: 4-8`
- Can increase `lora_rank: 16`
- Still recommend FOMAML for efficiency

## 7. Memory Optimization Features Already Implemented

### In MELoRAModel:
- **LoRA**: Reduces trainable parameters by 90%+
- **Selective checkpointing**: Reduces activation memory
- **Frozen base model**: Only LoRA parameters in optimizer

### In Baselines:
- **FOMAML/Reptile**: Only optimize LoRA parameters
- **Gradient clipping**: Prevents gradient explosion
- **Efficient parameter management**: Reuses parameter dictionaries

## 8. Additional Memory Optimization Options

If memory is still an issue:
1. **Mixed Precision Training**: Add `torch.cuda.amp` (can save 30-50% memory)
2. **Gradient Accumulation**: Split batch processing across steps
3. **CPU Offloading**: Move inactive parameters to CPU
4. **Dynamic Batching**: Adjust batch size based on available memory
5. **Quantization**: Use 8-bit or 4-bit LoRA adapters

## Conclusion

The current implementation is already well-optimized for memory efficiency:
- LoRA significantly reduces parameter count
- FOMAML and Reptile provide memory-efficient alternatives to FullMAML
- Configuration allows easy tuning for different memory constraints

For most use cases, **FOMAML is the recommended choice** as it provides a good balance between memory efficiency and meta-learning performance. 