# RiverScope: Research Evolution & Benchmarking Log

This log documents the technical milestones, resolution strategies, and benchmarking results for river segmentation on PlanetScope imagery.

## 📊 Summary of Results

| Approach | Branch | Spatial Strategy | Native Res | Val IoU | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pre-Downsampled Prototype** | `feature/first` | `rasterio` Resizing on Load | 10m (224x224) | **0.5020** | ⚠️ Simplified |
| **Dynamic Transform (Olmo)** | `feature/dynamic_transform` | GPU Interpolation (3m $\leftrightarrow$ 10m) | 3m (Native) | **0.5155** | ✅ Rigorous |
| **Native U-Net (Baseline)** | `feature/baseline` | Native Convolutional Flow | 3m (Native) | **0.5453** | 🏆 Champion |

---

## 🔬 Approach Breakdown

### 1. Pre-Downsampled Prototype (`feature/first`)
*   **The Logic**: Modified `dataset.py` to physically shrink PlanetScope 3m images down to 10m (224x224) using `Resampling.bilinear` during disk I/O.
*   **The Problem**: It "blurred" the truth. Evaluating at 10m resolution hides the difficult edge-cases of river segmentation, making the score (0.5020) look higher than the task actually was relative to the HD reality.

### 2. Dynamic Resolution FM (`feature/dynamic_transform`)
*   **The Logic**: Dataset restored to **Native 3m**. The `EndToEndOlmoSegmenter` acts as an elastic wrapper.
    *   Input (3m) $\rightarrow$ Squash (10m) $\rightarrow$ OlmoEarth Encode $\rightarrow$ Probing Head Decode $\rightarrow$ Re-inflate Mask (3m).
*   **The Insight**: Proven that OlmoEarth is powerful but limited by a "Resolution Bottleneck" (10m patch size) which makes it miss ultra-thin braided channels.

### 3. Native U-Net Baseline (`feature/baseline`)
*   **The Logic**: Standard U-Net architecture trained directly on the 12-channel 3m PlanetScope stack.
*   **The Advantage**: Zero resolution loss. Skip-connections preserve fine-grained features that Foundation Models currently lose during patchification.
*   **The Result**: Current performance leader for High-Definition river segmentation.

---

## 🛠️ Key Technical Stability Patches (MPS / Apple Silicon)
*   **Flash Attention Bypass**: All forward passes wrapped in `sdpa_kernel(SDPBackend.MATH)` to prevent shader crashes on M-series chips.
*   **Memory Contiguity**: Injected `.contiguous()` after tensor permutations to prevent GPU memory stride errors.
*   **Dummy Time Dimensions**: Unified dataloader to provide `[B, 1, 12, H, W]` to satisfy FM requirements while remaining compatible with standard Baseline U-Nets.
