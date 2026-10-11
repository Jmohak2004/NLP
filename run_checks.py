"""Project self-check: runs every notebook cell in one namespace (exactly like Colab's
"Run all"), then tests edge cases, the guardrails, human-in-the-loop review and model reloading.

Usage:  python run_checks.py        (takes ~3-5 minutes; exits with code 1 if anything fails)
"""
import builtins
import os
import contextlib
import io
import json
import sys

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heatwave_nlp.ipynb")
ns = {"__name__": "__main__"}
cells = [c for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
with contextlib.redirect_stdout(io.StringIO()):
    for i, cell in enumerate(cells):
        exec(compile("".join(cell["source"]), f"cell{i}", "exec"), ns)
print("all cells executed")

results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}  {name}")


g = ns
# --- IMD rule engine boundaries
check("plains 44.9 dep 4.4 -> WATCH", g["classify_heatwave"](44.9, 40.5)[0] == "WATCH")
check("plains 45.0 -> HEATWAVE (absolute)", g["classify_heatwave"](45.0, 43.0)[0] == "HEATWAVE")
check("plains 47.0 -> SEVERE (absolute)", g["classify_heatwave"](47.0, 43.0)[0] == "SEVERE")
check("plains dep 4.5 at 40 -> HEATWAVE", g["classify_heatwave"](40.0, 35.5)[0] == "HEATWAVE")
check("plains below 40 never a heatwave", g["classify_heatwave"](39.9, 30.0)[0] == "NORMAL")
check("coastal 37 dep 4.5 -> HEATWAVE", g["classify_heatwave"](37.0, 32.5, "coastal")[0] == "HEATWAVE")
check("coastal 46 dep 0 -> NORMAL (no absolute rule)", g["classify_heatwave"](46.0, 46.0, "coastal")[0] == "NORMAL")
check("hilly 30 dep 6.5 -> SEVERE", g["classify_heatwave"](30.0, 23.5, "hilly")[0] == "SEVERE")

# --- dates
pd = g["parse_date"]
check("parse '18th May'", pd("18th May") == "2025-05-18")
check("parse 'May 18 2024'", pd("May 18 2024") == "2024-05-18")
check("parse '18-05-25'", pd("18-05-25") == "2025-05-18")
check("invalid '31 June' -> None (no crash)", pd("31 June") is None)
check("US-style '05-31-2025' -> None (no crash)", pd("05-31-2025") is None)
check("no date -> None", pd("next 3 days") is None)

# --- segmentation / normalisation / morphology
check("segment heatwavealertnagpur", g["segment"]("heatwavealertnagpur") == ["heatwave", "alert", "nagpur"])
check("normalise empty post", g["normalize_post"]("") == [])
check("normalise emoji-only post", g["normalize_post"]("🥵🥵") == ["hot", "hot"])
check("find multiword place", g["find_places"]("so hot in Sri Ganganagar") == ["Sri Ganganagar"])
check("find place in hashtag", g["find_places"]("#nagpurheatwave") == ["Nagpur"])
check("find Marathi place", g["find_places"]("पुण्यात ऊन आहे") == ["Pune"])
check("no place -> []", g["find_places"]("hello world") == [])

# --- classifier robustness
labels = g["classify_any_script"](["", "🥵", "पुण्यात आज ऊन खूप आहे", "रुग्णालयात उष्माघात रुग्ण", "random text"])
check("classifier handles empty / emoji / Devanagari", len(labels) == 5)
check("Marathi heat post -> DISCOMFORT", labels[2] == 1)
check("Marathi emergency post -> EMERGENCY", labels[3] == 2)

# --- NER / pipeline robustness
check("NER on empty text", g["extract_entities"]("") == [])
check("bulletin without temperatures -> no records", g["records_from_bulletin"]("Heatwave over Vidarbha.") == [])
check("bulletin with unknown place -> no records", g["records_from_bulletin"]("Atlantis recorded 46.0 °C.") == [])
recs = g["records_from_bulletin"]("Sri Ganganagar recorded 47.3 °C on 2nd June.")
check("unseen multiword place extracted", len(recs) == 1 and recs[0]["place"] == "Sri Ganganagar"
      and recs[0]["severity"] == "SEVERE" and recs[0]["date"] == "2025-06-02")

# --- generation + guardrails for every severity x audience x language
import itertools
fails = []
for sev, tmax, normal in [("NORMAL", 38.0, 37.0), ("WATCH", 41.0, 38.0), ("HEATWAVE", 44.0, 39.0), ("SEVERE", 47.5, 42.0)]:
    rec = g["make_record"]("Akola", g["date"](2025, 5, 10), tmax, normal)
    assert rec["severity"] == sev, (sev, rec)
    for aud, lang in itertools.product(g["AUDIENCES"], g["LANGS"]):
        adv = g["generate_advisory"](rec, aud, lang)
        if not adv["checks"]["passed"]:
            fails.append((sev, aud, lang, adv["checks"]))
check(f"seq2seq passes all checks for 4x4x3 combos ({48 - len(fails)}/48)", not fails)
tmpl_ok = all(g["generate_advisory"](g["make_record"]("Mumbai", g["date"](2025, 5, 1), t, 33.0), a, l,
                                     method="template")["checks"]["passed"]
              for t in (34.0, 37.0, 38.0, 40.0) for a in g["AUDIENCES"] for l in g["LANGS"])
check("rule-based templates pass all checks (coastal place)", tmpl_ok)

# --- guardrail must catch corrupted outputs
rec = g["make_record"]("Nagpur", g["date"](2025, 5, 18), 47.6, 42.5)
good = g["generate_advisory"](rec, "CITIZEN", "hi")
v = g["verify_advisory"]
check("catches wrong temperature", not v(good["text"].replace("47.6", "45.6"), rec, "hi")["passed"])
check("catches wrong place", not v(good["text"].replace("नागपुर", "पुणे"), rec, "hi")["passed"])
check("catches Marathi text labelled Hindi",
      not v(g["generate_advisory"](rec, "CITIZEN", "mr")["text"], rec, "hi")["passed"])
toks = g["template_advisory"]("SEVERE", "CITIZEN", "en")
check("catches repeated sentence", not g["approved_content_checks"](toks + toks, rec, "CITIZEN", "en")["no_repetition"])
farmer = g["template_advisory"]("SEVERE", "FARMER", "en")
check("catches advice for the wrong audience",
      not g["approved_content_checks"](farmer, rec, "HEALTH", "en")["only_approved_sentences"])

# --- review / human in the loop
g["EXPERT_DECISIONS"].clear()
answers = iter(["y", "n"])
original_input = builtins.input
builtins.input = lambda prompt="": next(answers)
try:
    red = g["make_record"]("Nagpur", g["date"](2025, 5, 18), 47.6, 42.5)
    orange = g["make_record"]("Akola", g["date"](2025, 5, 18), 45.5, 42.5)
    s1 = [g["review_advisory"](g["generate_advisory"](red, a, l), red, mode="interactive")["status"]
          for a in g["AUDIENCES"] for l in g["LANGS"]]
    s2 = [g["review_advisory"](g["generate_advisory"](orange, a, l), orange, mode="interactive")["status"]
          for a in g["AUDIENCES"] for l in g["LANGS"]]
finally:
    builtins.input = original_input
check("interactive: one question per warning, approval applied to all 12", set(s1) == {"APPROVED_BY_EXPERT"})
check("interactive: rejection applied to all 12", set(s2) == {"REJECTED_BY_EXPERT"})
green = g["make_record"]("Mumbai", g["date"](2025, 5, 18), 34.0, 33.5)
check("NORMAL auto-approved", g["review_advisory"](g["generate_advisory"](green, "CITIZEN", "en"), green)["status"]
      == "AUTO_APPROVED")
broken = dict(g["generate_advisory"](red, "CITIZEN", "en"))
broken["checks"] = dict(broken["checks"], passed=False, tmax_correct=False)
fixed = g["review_advisory"](broken, red)
check("failed output replaced by template", fixed["method"] == "template" and "note" in fixed
      and fixed["checks"]["passed"])

# --- saved models reload
import pickle, torch
crf = pickle.load(open("outputs/ner_crf.pkl", "rb"))
check("CRF reloads and predicts", len(crf.predict_single(g["sent_features"](["Nagpur", "recorded", "46", "°C"]))) == 4)
ckpt = torch.load("outputs/seq2seq_advisory.pt", map_location="cpu")
m = g["Seq2Seq"](len(ckpt["src_itos"]), len(ckpt["tgt_itos"]))
m.load_state_dict(ckpt["model"])
check("Seq2Seq checkpoint reloads", True)

print(f"\n{sum(ok for _, ok in results)}/{len(results)} checks passed")
sys.exit(0 if all(ok for _, ok in results) else 1)
