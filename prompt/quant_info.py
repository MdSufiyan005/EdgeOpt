"""
Knowledge base for GGUF quantization suffixes and performance trade-offs.
"""

context = """
### GGUF Quantization Technical Reference

#### 1. Suffix Nomenclature (The S-M-L / K System)
- **Q**: Indicates the model is **Quantized**.
- **Digit (2-8)**: The target **average bits per weight** (BPW).
- **K**: Modern **K-Quants** implementation. Uses grouped (block-wise) scaling factors rather than a single global scale for the whole tensor.
- **0 / 1**: **Legacy** formats. Q4_0 (per-tensor/global) and Q4_1 (per-row). Generally less accurate than K-Quants.
- **S / M / L**: Sub-variants indicating internal precision:
    - **S (Small)**: Aggressive compression, smallest footprint, highest speed.
    - **M (Medium)**: The "Sweet Spot." Balanced precision and memory usage.
    - **L (Large)**: Prioritizes accuracy over file size within the bit-depth.

#### 2. Quantitative Trade-offs (Baseline: Vicuna-13B)
| Quant Type | BPW (Approx) | Perplexity (PPL) ∆ | Recommendation |
| :--- | :--- | :--- | :--- |
| **F16** | 16.00 | +0.0000 | Baseline for quantization. |
| **Q8_0** | 8.50 | +0.0004 | Use for near-lossless deployment. |
| **Q6_K** | 6.59 | +0.0044 | Excellent fidelity for logical reasoning. |
| **Q5_K_M** | 5.50 | +0.0142 | High accuracy, moderate memory overhead. |
| **Q4_K_M** | 4.80 | +0.0535 | **Standard Recommendation** for edge/RAM limits. |
| **Q3_K_M** | 3.90 | +0.2437 | Use only on high-parameter models (30B+). |
| **Q2_K** | 2.67 | +0.8698 | High risk of hallucination and logic failure. |

#### 3. MLOps Deployment Insights
- **Memory Estimation**: VRAM needed $\approx \frac{Parameters(B) \times BPW}{8} \times 1.15$ (includes 15% overhead for KV cache/activations).
- **K-Quants Mechanism**: Weights are split into blocks (e.g., 256). Each block has its own scale and zero-point. This prevents "outlier" weights from skewing the precision of the entire tensor.
- **Selective Precision**: `llama.cpp` scripts often keep critical tensors (like output and attention layers) at slightly higher bit-depths than the specified quant type to preserve coherence.
"""
