"""
Talk to a trained neural Elizilla: a GGUF model, or an .mdlo (v2) that wraps
one, run with llama.cpp. safety.py always runs before the model.

    m = ChatModel("elizilla-neural.mdlo")
    m.reply(["i feel lonely"])            -> "That sounds hard. When does it hit you the most?"

The eval and run_neural.py both use this, so they see the same model behaviour.
"""

import hashlib
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import safety  # noqa: E402
from mdlo import PERSONA  # noqa: E402

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "elizilla")


def unwrap(path):
    """For an .mdlo v2 GGUF file: check it and unwrap the GGUF into a cache
    (llama.cpp needs a file on disk). Returns (gguf_path, header or None)."""
    with open(path, "rb") as fh:
        head = fh.read(16)
        if head[:4] != b"MDLO":
            return path, None                          # a plain .gguf
        version = struct.unpack("<I", head[4:8])[0]
        if version != 2:
            raise ValueError(f"{path}: MDLO version {version} isn't a GGUF model")
        header = json.loads(fh.read(struct.unpack("<Q", head[8:16])[0]))
        if header.get("engine") != "gguf":
            raise ValueError(f"{path}: engine {header.get('engine')!r}, expected 'gguf'")
        if not header.get("safety", {}).get("crisis_lock"):
            raise ValueError(f"{path}: header doesn't declare the crisis lock; refusing to load")
        os.makedirs(CACHE, exist_ok=True)
        dest = os.path.join(CACHE, header["payload_sha256"][:16] + ".gguf")
        if not (os.path.exists(dest) and os.path.getsize(dest) == header["payload_bytes"]):
            h = hashlib.sha256()
            with open(dest + ".tmp", "wb") as out:
                while chunk := fh.read(1 << 20):
                    h.update(chunk)
                    out.write(chunk)
            if h.hexdigest() != header["payload_sha256"]:
                os.remove(dest + ".tmp")
                raise ValueError(f"{path}: payload checksum mismatch, the file is corrupt")
            os.replace(dest + ".tmp", dest)
        return dest, header


class ChatModel:
    def __init__(self, path, n_ctx=2048, threads=None, max_tokens=80, lock_path=safety.DEFAULT_LOCK):
        from llama_cpp import Llama
        gguf, header = unwrap(path)
        self.header = header
        self.system = (header or {}).get("chat", {}).get("system_prompt") or PERSONA
        self.max_tokens = max_tokens
        self.lock = safety.Lock(lock_path)
        self.llm = Llama(model_path=gguf, n_ctx=n_ctx, n_threads=threads or os.cpu_count(), verbose=False)

    def reply(self, user_turns, bot_turns=()):
        """user_turns: everything the person said, in order; bot_turns: the
        model's earlier replies. Returns the reply to the last user turn."""
        if self.lock.locked:
            return safety.LOCKED_MESSAGE
        kind = safety.check(user_turns[-1])
        if kind:
            self.lock.engage(kind)
            return safety.message_for(kind)
        msgs = [{"role": "system", "content": self.system}]
        for i, u in enumerate(user_turns):
            msgs.append({"role": "user", "content": u})
            if i < len(bot_turns):
                msgs.append({"role": "assistant", "content": bot_turns[i]})
        out = self.llm.create_chat_completion(messages=msgs, max_tokens=self.max_tokens, temperature=0.0)
        return out["choices"][0]["message"]["content"].strip()
