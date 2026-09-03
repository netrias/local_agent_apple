# Model Selection & Quantization Guide (oMLX + Apple Silicon)

Research notes for the Netrias local-inference stack (oMLX + pi coding agent). Covers which open-weight models fit which unified-memory tier, and how oMLX's `oQ` data-driven quantization works. Compiled from public sources in September 2026 — this space moves fast, re-check before committing to a model for production use.

**Rule of thumb used throughout:** keep model weights + KV cache/context under ~70–75% of total unified RAM, leaving the rest for macOS, oMLX's own overhead, and any other apps running. MLX quants generally use ~10% less memory than equivalent GGUF quants and run faster on Apple GPUs, so MLX (native or `mlx-community` conversions) is the default assumption below.

## Part 1 — Best open-weight models by RAM tier

### 48 GB tier (top MacBook Pro configs)
Usable budget for weights + context: roughly 33–36 GB.

| Model | Params | Quant | Approx. footprint | Role |
|---|---|---|---|---|
| Qwen3.8-27B | 27B dense | 4-bit | ~15 GB | General/agentic default — big headroom left for long context |
| Qwen3.6-35B-A3B | 35B MoE (3B active) | 4-bit | ~19 GB | Faster throughput than the 27B dense model at similar quality |
| GLM-4.7-Flash-PRISM | — | 8-bit | ~30 GB | Higher-fidelity agentic/coding option if you want less quality loss than 4-bit |
| Qwen3-Coder-Next | 80B MoE (3B active) | 4-bit | ~45 GB | **Tight fit** — community reports running this in ~46 GB RAM; leaves very little headroom for context or other apps. Treat 48 GB as the bare minimum, not comfortable, for this one. |

### 24 GB tier
Usable budget: roughly 17 GB.

| Model | Params | Quant | Approx. footprint | Role |
|---|---|---|---|---|
| Qwen3.8-27B | 27B dense | 4-bit | ~15 GB | Still fits, but leaves less headroom than at 48 GB |
| Ternary "Bonsai" 27B | 27B | 2-bit ternary | ~7.9 GB | Unusual but notable — near-27B-class capability at a very low footprint via ternary quantization |
| GLM-4.7-Flash-PRISM | — | 4-bit | ~16 GB | Coding/agentic option sized for this tier |
| Qwen3.5-9B | 9B dense | 4-bit | ~5.6 GB | Fast, leaves plenty of headroom for context |

### 16 GB tier
Usable budget: roughly 11 GB.

| Model | Params | Quant | Approx. footprint | Role |
|---|---|---|---|---|
| Qwen3.5-9B | 9B dense | 4-bit | ~5.6 GB | Best all-around "smart" pick at this tier |
| Qwen3.5-9B | 9B dense | 8-bit | ~9.8 GB | Higher fidelity, less context headroom |
| Qwen3.5-4B | 4B dense | 4-bit | ~2.9 GB | "Fast" pick, most headroom for context |

### 8 GB tier (base MacBook Air / small devices)
Usable budget: roughly 5.5 GB.

| Model | Params | Quant | Approx. footprint | Role |
|---|---|---|---|---|
| LFM2.5-2.6B | 2.6B | 4-bit | ~3.0 GB | "Smart" pick — best capability that still fits |
| LFM2.5-1B | 1B | 4-bit | ~1.9 GB | "Fast" pick, minimal footprint |
| Qwen3.5-0.8B | 0.8B | 4-bit | ~0.6 GB | Smallest general-purpose option with broad ecosystem/tooling support |

### Where Ornith-1.5-9B fits
At 9B params, an oQ6 (~6-bit) quant lands at roughly **6.5–7 GB** on disk/in memory (9B × ~6 bits ÷ 8 ≈ 6.75 GB, back-of-envelope — not vendor-published). That comfortably clears the 16 GB tier budget and is essentially free on 24 GB+ machines, which lines up with it being this team's default model.

### Notes
- Family landscape as of Sept 2026: **Qwen** (3.5 / 3.6 / 3.8 / Coder-Next) has the deepest `mlx-community` coverage and is the safest default across tiers. **GLM** (4.7-Flash-PRISM, 5.x) and **DeepSeek** (V4, and R1-style distills) are strong agentic/coding alternatives at the larger tiers. **LFM2.5**, **Gemma**, and **Phi** lead the small-device tier. **Llama 4** and **Mistral Small 4** remain relevant general options but had thinner MLX-specific coverage in the sources checked.
- Always check `mlx-community` on Hugging Face for an existing quant before quantizing yourself — most mainstream releases land there within days.
- Figures above are disk/weights size, not total runtime memory — add KV cache, which grows with context length and batch size (see oMLX's `--memory-guard` flag and TurboQuant KV-cache option, below).

## Part 2 — oMLX's oQ data-driven quantization

### What it is
`oQ` ("oMLX Universal Dynamic Quantization") is oMLX's built-in **mixed-precision, data-driven** quantizer. Instead of applying a fixed bit-width to every layer (what plain `mlx_lm.convert`-style static quantization does), oQ runs a calibration pass and measures each layer's actual sensitivity to quantization:

```
sensitivity = MSE(float_output, quantized_output) / mean(float_output²)
```

Layers with high measured sensitivity get bits boosted (e.g. 4→8-bit); low-sensitivity layers stay at the base bit-width. This produces a **model-specific** bit-allocation profile rather than a one-size-fits-all rule — e.g. embeddings and the LM head are typically auto-promoted to 8-bit while most transformer layers stay at 4-bit. Output is a standard MLX-compatible safetensors model that loads in oMLX, `mlx-lm`, or anything else that reads MLX quantized weights — no custom loader required.

There's also an enhanced path, **oQ+/oQe**, which adds GPTQ-style weight optimization (Hessian-based error compensation) and, in the `oQe` variant, an activation-importance calibration pass with MoE expert-coverage tracking — reported to modestly outperform base oQ on MoE models (one benchmark: 83.88% vs 82.83% on Qwen3.6-35B-A3B, source below).

### Supported levels and bit-widths

| Level | Base bits | Approx. bits/weight | Use case |
|---|---|---|---|
| oQ2 | 2 | ~2.9 | Extreme compression |
| oQ2.5 | 2 | ~3.2 | Code-preserving variant of oQ2 |
| oQ3 | 3 | ~3.5 | Balanced |
| oQ4 | 4 | ~4.6 | **Recommended default** |
| oQ5–oQ6 | 5–6 | — | Higher fidelity (this is the range your Ornith-1.5-9B 6-bit quant falls in) |
| oQ8 | 8 | ~8.6 | Near-lossless |

All levels use affine quantization with `group_size=64`, except oQ8 which uses `mxfp8`. oQ also supports VLM quantization (vision weights preserved in fp16) and can take native FP8 models as a quantization source.

### Calibration data
oQ ships a **built-in calibration set** — no external download needed: ~726 KB, 600 samples across code (200), English text (150), CJK Wikipedia text (170 combined), tool-calling (40), and reasoning traces (40). You can also point it at your own calibration data or an explicit `sensitivity_model_path` if the built-in set doesn't suit your domain (this is what the error `sensitivity measurement produced no scores` is telling you to do when it can't compute sensitivities from the default setup — see the GitHub issue in Sources).

### How to run it

**GUI (primary/documented path).** oMLX ships as a macOS menu-bar app with a web dashboard. Per a real bug report reproducing the flow (jundot/omlx#1400):

1. Dashboard → **Models → oQ Quantization**
2. Pick the **source model**
3. Set **sensitivity model** (leave `NONE` to use the built-in calibration set, or point at your own)
4. Choose the **oQ level** (e.g. `oQ4`, `oQ6`, `oQ8`)
5. Toggle options as needed: **Text Only**, **Preserve MTP weights**, **Non-quant weight dtype** (e.g. `float16`)
6. Click **Start**

**Programmatic (Python API).** Confirmed from a real model card's stated production recipe:

```python
from omlx.oq import quantize_oq_streaming

quantize_oq_streaming(
    model_path="/path/to/source-model",
    output_path="/path/to/output-model-oQ4",
    oq_level=4,
    group_size=64,
    dtype="bfloat16",
)
```

**Serving the result** is standard oMLX/mlx-lm:

```bash
omlx serve --model-dir ~/.omlx/models
# or, for a quick check outside the server:
mlx_lm.generate --model /path/to/output-model-oQ4 --prompt "..."
```

⚠️ **Not verified / not found in public docs:** a documented terminal subcommand for oQ itself (e.g. `omlx oq ...` or `omlx quantize ...`). The oMLX README documents `omlx serve|start|stop|restart|launch [tool]` for server management (and `omlx launch pi` for this team's stack specifically), but the oQ quantization flow as publicly documented is driven through the **dashboard GUI** or the **Python API** above — not a first-class CLI flag. If a CLI wrapper exists, it wasn't findable in the README, release notes, or the `docs/oQ_Quantization.md` spec as of this research.

## Sources
- [omlx/docs/oQ_Quantization.md](https://github.com/jundot/omlx/blob/main/docs/oQ_Quantization.md) — oQ methodology, bit-allocation, calibration set, benchmarks
- [jundot/omlx — main README](https://github.com/jundot/omlx) — CLI (`serve`/`start`/`stop`/`restart`/`launch`), supported model formats
- [oQ Quantization fails · Issue #1400 · jundot/omlx](https://github.com/jundot/omlx/issues/1400) — real GUI flow and error message
- [oQ quantization is way too memory hungry after v0.3.9 · Issue #1094 · jundot/omlx](https://github.com/jundot/omlx/issues/1094)
- [chevron7/MiniMax-M2.5-oQ4 (Hugging Face)](https://huggingface.co/chevron7/MiniMax-M2.5-oQ4) — real oQ4 production recipe (tool, calibration data, config)
- [vystartasv/AgenticQwen-8B-oQ4 (Hugging Face)](https://huggingface.co/vystartasv/AgenticQwen-8B-oQ4) — `quantize_oq_streaming` Python API usage
- [mlx-optiq.com — quantize, fine-tune and serve LLMs on Apple Silicon](https://mlx-optiq.com/docs/faq)
- [jacar.es — How to install and tune oMLX on M5 Max 128GB](https://jacar.es/en/how-to-install-and-tune-omlx-on-m5-max-128-gb/)
- [jacar.es — oMLX admin dashboard & CLI reference](https://jacar.es/en/omlx-admin-dashboard-cli/)
- [RapidMLX — MLX model catalog by Mac memory tier](https://rapidmlx.com/models/)
- [mlx-community (Hugging Face org)](https://huggingface.co/mlx-community)
- [mlx-community/Qwen3.5-9B-MLX-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit) and sibling Qwen3.5 size variants
- [mlx-community/Qwen3-Coder-Next-4bit](https://huggingface.co/mlx-community/Qwen3-Coder-Next-4bit)
- [Qwen3-Coder-Next: Running an 80B Coding Model Locally on 46GB RAM (Medium)](https://medium.com/coding-nexus/qwen3-coder-next-running-an-80b-coding-model-locally-on-46gb-ram-618cf1cba4be)
- [pipenetwork/GLM-5.3-Flash-MLX-4bit](https://huggingface.co/pipenetwork/GLM-5.3-Flash-MLX-4bit) and [shieldstackllc/GLM-4.7-Flash-PRISM-mlx-8bit](https://huggingface.co/shieldstackllc/GLM-4.7-Flash-PRISM-mlx-8bit)
- [Best Open-Weight Coding Models to Self-Host in 2026](https://www.digitalapplied.com/blog/best-open-weight-coding-models-self-host-hardware-match-2026)
- [The Best Open Source LLMs (2026), by Morph](https://www.morphllm.com/best-open-source-llm)
