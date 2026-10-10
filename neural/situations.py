"""
Load the situation map (situations.txt) and its fixed held-out split.

    from situations import load
    for s in load(): print(s["area"], s["id"], s["held_out"])
"""

import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
HELD_OUT_SHARE = 0.15
MOODS = {"anxious", "sad", "angry", "happy", "tired", "lonely", "bored", "proud", "guilty",
         "confused", "excited", "nervous", "frustrated", "calm", "mixed"}


def _rank(sid):
    return hashlib.sha256(("held-out:" + sid).encode()).hexdigest()


def _mark_held_out(situations):
    """In every area, hold out the 15% (at least one) whose hashed id ranks
    lowest. The split depends only on the ids, so it never shifts at random."""
    areas = {}
    for s in situations:
        areas.setdefault(s["area"], []).append(s)
    for group in areas.values():
        k = max(1, round(HELD_OUT_SHARE * len(group)))
        for s in sorted(group, key=lambda s: _rank(s["id"]))[:k]:
            s["held_out"] = True


def load(path=os.path.join(HERE, "situations.txt")):
    out, area = [], None
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if line.startswith("## "):
                area = line[3:].strip()
                continue
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) != 4 or parts[3] not in MOODS:
                raise ValueError(f"situations.txt line {n}: expected 'id | event | about | mood'")
            sid, event, about, mood = parts
            out.append({"id": f"{area}/{sid}", "area": area, "event": event, "about": about,
                        "mood": mood, "held_out": False})
    ids = [s["id"] for s in out]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate situation ids")
    _mark_held_out(out)
    return out


if __name__ == "__main__":
    from collections import Counter
    sits = load()
    held = [s for s in sits if s["held_out"]]
    print(f"{len(sits)} situations in {len({s['area'] for s in sits})} areas, {len(held)} held out")
    by_area = Counter(s["area"] for s in sits)
    held_area = Counter(s["area"] for s in held)
    for a in by_area:
        print(f"  {a:<20}{by_area[a]:>4}  held out {held_area[a]}")
