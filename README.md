# Introduction

[Nvidia's parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) is an ASR model.
This repository contains the source code to host this on replicate.

# Testing locally

```bash
cog predict -i audio=@test.mp3
```

If you face issues for hugging face timeouts then use:

```bash
HF_HUB_DOWNLOAD_TIMEOUT=600 cog predict -i audio=@test.mp3
```
