#!/usr/bin/env python3
"""
Chat with the neural Elizilla, a standalone .mdlo (GGUF inside) run with
llama.cpp. safety.py runs before the model on every message, exactly as in
the rule-based chat, and the crisis lock is the same one.

    pip install llama-cpp-python
    python3 neural/run_neural.py --model elizilla-neural.mdlo
    python3 neural/run_neural.py --model elizilla-neural.mdlo --selftest
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import safety  # noqa: E402
from chat_model import ChatModel  # noqa: E402
from checks import problems  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="elizilla-neural.mdlo")
    p.add_argument("--selftest", action="store_true", help="say three things, print the replies, exit")
    args = p.parse_args()

    if safety.Lock().locked and not args.selftest:
        print("NeoEliza:", safety.LOCKED_MESSAGE)
        return
    lock = safety.DEFAULT_LOCK if not args.selftest else os.path.join(HERE, ".selftest.lock")
    model = ChatModel(args.model, lock_path=lock)

    if args.selftest:
        users, bots, bad = ["hi", "my sourdough starter died", "i had it for two years"], [], 0
        for i in range(len(users)):
            r = model.reply(users[:i + 1], bots)
            bots.append(r)
            issues = problems(users[i], r, bots[:-1])
            bad += bool(issues)
            print(f"You: {users[i]}\nNeoEliza: {r}" + (f"   {issues}" if issues else ""))
        sys.exit(0 if bots and all(bots) else 1)

    print("(Elizilla is an experiment in whether a small language model can hold a conversation.")
    print(" It isn't a therapist or a source of advice.)\n")
    print("NeoEliza: Hello, I'm NeoEliza. How are you feeling today?\n")
    users, bots = [], []
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit", "bye", "goodbye"):
            print("NeoEliza: Goodbye. Take care of yourself.")
            break
        users.append(text)
        reply = model.reply(users, bots)
        bots.append(reply)
        print("NeoEliza:", reply)
        if model.lock.locked:
            break


if __name__ == "__main__":
    main()
