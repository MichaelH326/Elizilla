"""
Crisis safety: detect talk of suicide, self-harm, or harming others, answer
with the route crisis professionals advise, and lock the bot for good.

The reply follows the guidance crisis services give for these moments: say
you care, point to a real person right away (988 in the US, Crisis Text Line,
emergency services if anyone is in immediate danger), encourage reaching out
to someone trusted, and don't try to counsel. A small program isn't the right
support, so after a crisis message the bot stops: every later message, and
every later start, gets only the help resources. A person has to clear the
lock on purpose (`python3 neo_eliza.py --reset-lock`), ideally after checking
in with whoever was talking to it.

This module is plain Python with no other Elizilla imports, so any runner,
including a future neural one, can use the same check and the same lock.
"""

import json
import os
import re
from datetime import datetime, timezone

# Checked on the lowercased raw text and on the normalized text ("dont" ->
# "do not"). These err on the side of locking: a figure of speech like "I could
# kill him" locks the bot too.
SELF_HARM = re.compile(
    r"\bsuicid|\bkill(?:ing)? (?:my ?self|myself)\b|\bkms\b|\bunalive\b"
    r"|\bend (?:it all|my life|things|it)\b|\btake my (?:own )?life\b"
    r"|\b(?:want|wanna|wanted|going|gonna|planning|plan) (?:to )?die\b|\bwish i (?:was|were) dead\b"
    r"|\b(?:do not|dont|don't) want to (?:live|be alive|be here|exist|wake up)\b"
    r"|\bbetter off (?:dead|without me)\b|\bno reason to live\b|\bnot worth living\b"
    r"|\b(?:hurt|hurting|harm|harming|cut|cutting|burn|burning) (?:my ?self|myself)\b"
    r"|\bself[- ]?harm|\boverdos|\bhang (?:my ?self|myself)\b")
HARM_OTHERS = re.compile(
    r"\b(?:kill|murder|shoot|stab|hurt|harm|attack|beat up|strangle|choke) (?:him|her|them|someone"
    r"|somebody|people|everyone|everybody|(?:my|his|her|their|that|this) (?:mom|dad|mother|father"
    r"|brother|sister|wife|husband|boyfriend|girlfriend|partner|friend|boss|teacher|kids?|son"
    r"|daughter|baby|parents?|roommate|neighbou?r|coworker|classmate|ex|family|guy|girl|kid))\b"
    r"|\bshoot up\b|\bmake a bomb\b")

SELF_HARM_MESSAGE = (
    "I'm really glad you told me, and I'm so sorry you're hurting. You deserve to talk "
    "with a real person right now, not a program. In the US, call or text 988 (Suicide & "
    "Crisis Lifeline) any time, free and confidential, or text HOME to 741741 (Crisis Text "
    "Line). If you might act on these thoughts or are in danger right now, call 911 or go "
    "to the nearest emergency room. Outside the US, call your local emergency number or "
    "find a helpline at findahelpline.com. If you can, reach out to someone you trust, let "
    "them know how you're feeling, and try not to be alone right now. This chat is closed "
    "now so you can get real support."
)
HARM_OTHERS_MESSAGE = (
    "Thank you for saying it out loud. If anyone is in danger right now, call 911 (or your "
    "local emergency number) immediately. If you're having thoughts of hurting someone, "
    "you can call or text 988 in the US any time to talk with a trained counselor; outside "
    "the US, find a helpline at findahelpline.com. Please reach out to someone you trust "
    "and stay away from anything you could use to hurt someone. This chat is closed now "
    "so you can get real support."
)
LOCKED_MESSAGE = (
    "This chat is closed. If you're struggling, please talk to a real person: in the US, "
    "call or text 988, or text HOME to 741741. If anyone is in danger, call 911 or your "
    "local emergency number. Outside the US, find a helpline at findahelpline.com."
)

DEFAULT_LOCK = os.environ.get("ELIZILLA_LOCK") or os.path.join(os.path.expanduser("~"), ".elizilla", "crisis.lock")


def check(text, normalized=""):
    """Return "self", "others", or None."""
    for t in (text.lower(), normalized):
        if SELF_HARM.search(t):
            return "self"
    for t in (text.lower(), normalized):
        if HARM_OTHERS.search(t):
            return "others"
    return None


def message_for(kind):
    return SELF_HARM_MESSAGE if kind == "self" else HARM_OTHERS_MESSAGE


class Lock:
    """A lock file that keeps the bot closed after a crisis message. It stores
    only when and which kind of message triggered it, never what was said."""

    def __init__(self, path=DEFAULT_LOCK):
        self.path = path

    @property
    def locked(self):
        return bool(self.path) and os.path.exists(self.path)

    def engage(self, kind):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump({"locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "kind": kind}, fh)

    def reset(self):
        if self.locked:
            os.remove(self.path)
            return True
        return False
