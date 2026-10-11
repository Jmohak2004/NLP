"""Run the pipeline on your own bulletin and citizen posts.

Usage:
    python try_input.py                                   # built-in example below
    python try_input.py --bulletin bulletin.txt --posts posts.txt   # one post per line
    HEATWAVE_FAST=1 python try_input.py                    # quicker, fewer epochs

Trains the models exactly as heatwave_nlp.py does (output silenced), then runs
NER → IMD rules → Seq2Seq advisories → checks/review, and post classification.
"""
import argparse
import contextlib
import io
import json
import os
from collections import Counter, defaultdict

BULLETIN = (
    "Severe heatwave conditions are very likely over East Uttar Pradesh. "
    "Banda recorded the highest maximum temperature of 48.2 °C on 3rd June. "
    "Delhi recorded 45.4 °C, above normal by 5.9 °C. "
    "Shimla reported 31.0 °C on 3 June. "
    "Hyderabad : 41.0 °C ( 2.0 °C above normal ). "
    "Chennai recorded 38.0 °C."
)
POSTS = [
    "garmi bahut hai delhi mein yaarrr 🥵🥵 #delhiheatwave",
    "#bandaheatwave hospital full of heatstroke cases, 3 deaths reported",
    "two labourers collapsed from heat stroke in hyderabad today 🚑",
    "hot deals at the chennai mall this weekend 🔥",
    "शिमला में आज बहुत गर्मी है",
    "दिल्लीमध्ये उष्माघाताचे रुग्ण वाढले",
]

ap = argparse.ArgumentParser()
ap.add_argument("--bulletin", help="text file with an IMD-style bulletin")
ap.add_argument("--posts", help="text file, one citizen post per line")
args = ap.parse_args()
bulletin = open(args.bulletin, encoding="utf-8").read() if args.bulletin else BULLETIN
posts = [l.strip() for l in open(args.posts, encoding="utf-8") if l.strip()] if args.posts else POSTS

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
g = {"__name__": "__main__"}
print("Training models (output hidden, takes a few minutes)...")
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(open("heatwave_nlp.py", encoding="utf-8").read(), "heatwave_nlp.py", "exec"), g)

print("\nINPUT BULLETIN:\n  " + bulletin)
print("\nSTEP 1 · NER")
for sent in g["extract_entities"](bulletin):
    print("  ", [(k, t) for k, t, _, _ in sent["entities"]])

print("\nSTEP 2 · Records + IMD severity")
records = g["records_from_bulletin"](bulletin)
for r in records:
    print(f"   {r['place']:12s} {r['date']}  Tmax {r['tmax']} (normal {r['normal']}, "
          f"dep {r['departure']:+.1f}, {g['PLACES'][r['place']]['terrain']}) → {r['severity']} / {r['colour']}")

print("\nSTEP 3 · Advisories (CITIZEN audience shown; all 12 per place saved)")
released = [g["review_advisory"](g["generate_advisory"](r, a, l), r)
            for r in records for a in g["AUDIENCES"] for l in g["LANGS"]]
for adv in released:
    if adv["audience"] == "CITIZEN":
        print(f"   [{adv['status']:24s}] {adv['place']}/{adv['lang']} ({adv['method']}): {adv['text']}")
ok = sum(a["checks"]["passed"] for a in released)
print(f"   guardrail checks passed: {ok}/{len(released)}")

print("\nSTEP 4 · Citizen posts vs forecast")
signals = defaultdict(Counter)
for post, label in zip(posts, g["classify_any_script"](posts)):
    places = g["find_places"](post)
    print(f"   {g['POST_LABELS'][label]:16s} {places} ← {post}")
    for p in places:
        signals[p][g["POST_LABELS"][label]] += 1
forecast = {r["place"]: r["severity"] for r in records}
for place in sorted(set(signals) | set(forecast)):
    sev, sig = forecast.get(place, "no forecast"), signals.get(place, Counter())
    heat = sig["HEAT_EMERGENCY"] + sig["HEAT_DISCOMFORT"]
    if sev == "no forecast":
        verdict = "not in bulletin; citizen heat reports → check AWS data" if heat else "not in bulletin"
    elif sig["HEAT_EMERGENCY"] and sev in ("NORMAL", "WATCH"):
        verdict = "CHECK: citizens report emergencies the forecast does not show → verify with AWS"
    elif heat:
        verdict = "corroborated by citizen reports"
    else:
        verdict = "no citizen heat reports"
    print(f"   {place:12s} forecast {sev:12s} | {dict(sig)} → {verdict}")

out = os.path.join("outputs", "custom_advisories.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(released, f, ensure_ascii=False, indent=2)
print("\nSaved", out)
