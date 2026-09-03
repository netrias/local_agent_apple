# local_agent_apple

Fast local LLM inference on Apple Silicon for the Netrias team: **oMLX** (multi-model MLX inference server) + **Ornith-1.5-9B** (our default model) + **pi** (coding agent), wired together with web search/fetch tooling.

See [`docs/model-selection-and-quantization.md`](docs/model-selection-and-quantization.md) for model picks by RAM tier and a rundown of oMLX's `oQ` data-driven quantization.

## Requirements

- Apple Silicon Mac, macOS with admin/sudo access (Homebrew needs it)
- Enough unified memory for whatever model you plan to run — see the RAM-tier guide linked above (48GB comfortably runs Ornith-1.5-9B at 6-bit with plenty of headroom)

## Install

```bash
./install.sh
```

This sets up the whole stack in one pass:

1. Installs/updates **Homebrew**
2. Installs **nvm** via Homebrew and sources it properly into the shell (also appends the nvm init lines to `~/.zshrc` so new shells pick it up) — this is required because `pi`'s extensions need a current Node/npm, and a bare `brew install nvm` doesn't put `nvm` on your PATH by itself
3. Downloads and mounts the **oMLX** `.dmg`, copies into `/Applications`
4. Installs the latest LTS Node via nvm, updates npm, and installs the **pi** coding agent plus its `pi-smart-web-search` and `pi-smart-fetch` extensions

## Downloading a model into the oMLX cache

oMLX's server (`omlx serve`) discovers models from subdirectories of a model directory — **`~/.omlx/models` by default**. Each subdirectory name becomes the model's ID, and must contain a `config.json` plus its `*.safetensors` weight files:

```
~/.omlx/models/
├── Ornith-1.5-9B-MLX-oQ6/   → model_id: "Ornith-1.5-9B-MLX-oQ6"
│   ├── config.json
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   ├── model.safetensors.index.json
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── <other-model>/
```

To pull our team's default model — [`netrias/Ornith-1.5-9B-MLX-oQ6`](https://huggingface.co/netrias/Ornith-1.5-9B-MLX-oQ6), the 6-bit oQ quant — straight into the cache with the right layout:

```bash
hf download netrias/Ornith-1.5-9B-MLX-oQ6 --local-dir ~/.omlx/models/Ornith-1.5-9B-MLX-oQ6
```

(`hf` is the Hugging Face CLI — `install.sh` doesn't install it. If you don't have it: `brew install hf` or `pip install hf`. The repo is public, so no `hf auth login` is required to pull it.)

Any other MLX-format model works the same way — swap the repo ID and local dir name. See the [model selection guide](docs/model-selection-and-quantization.md) for picks sized to your machine's RAM.

## Running

```bash
omlx serve --model-dir ~/.omlx/models
pi   # start the Pi coding agent in your working dir
```

Point `pi` (or any OpenAI-compatible client) at oMLX's local server and select the model ID matching the subdirectory name above.
