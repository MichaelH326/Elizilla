"""Checks that sentences are split into the right sections."""
from sentence import normalize, parse

CASES = {
    "its exciting but i dont have the time to do my homework": [
        dict(subj="@it", verb="be", acat="@good"),
        dict(conj="but", subj="@self", verb="have", neg=True, ocat="@timeamt", comp_type="to"),
    ],
    "I hate it when no one shows up": [dict(subj="@self", vcat="@dislike", comp_type="when")],
    "I want to quit school": [dict(subj="@self", vcat="@desire", comp_type="to")],
    "theyve been crazy": [dict(subj="@they", verb="be", acat="@busy")],
    "school and work": [dict(verb=None)],
    "I am really stressed about my exams": [dict(subj="@self", acat="@anxious", comp_type="about")],
    "i think my boss hates me": [dict(subj="@self", vcat="@belief", comp_type="that")],
    "i love playing guitar": [dict(subj="@self", vcat="@like", comp_type="ing")],
    "i cannot sleep": [dict(subj="@self", verb="sleep", modal="can", neg=True)],
    "my mom is always on my case": [dict(subj="@family", subj_owner="@self", verb="be")],
    "i dont know how to respond to that": [dict(subj="@self", vcat="@know", neg=True)],
}

for text, expected in CASES.items():
    frames, _ = parse(normalize(text), text)
    assert len(frames) == len(expected), (text, [f.describe() for f in frames])
    for f, exp in zip(frames, expected):
        for key, val in exp.items():
            assert getattr(f, key) == val, f"{text!r}: {key}={getattr(f, key)!r}, expected {val!r}\n  {f.describe()}"
print(f"ok: {len(CASES)} sentences parsed into the expected sections")
