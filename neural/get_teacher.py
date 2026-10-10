"""
Download a teacher model (GGUF) from Hugging Face.

    python3 neural/get_teacher.py phi-3.5-mini --out models/
prints the local path. Needs huggingface_hub and access to huggingface.co.
Check each model's licence allows training on its outputs before use.
"""

import argparse

TEACHERS = {
    # name: (repo, file, licence at time of writing)
    "phi-3.5-mini": ("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q4_K_M.gguf", "MIT"),
    "qwen2.5-7b": ("bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "Apache-2.0"),
    "smollm2-1.7b": ("bartowski/SmolLM2-1.7B-Instruct-GGUF", "SmolLM2-1.7B-Instruct-Q4_K_M.gguf", "Apache-2.0"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name", choices=sorted(TEACHERS))
    p.add_argument("--out", default="models")
    args = p.parse_args()
    from huggingface_hub import hf_hub_download
    repo, fname, _ = TEACHERS[args.name]
    print(hf_hub_download(repo, fname, local_dir=args.out))


if __name__ == "__main__":
    main()
