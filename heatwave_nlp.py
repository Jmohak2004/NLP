# %% [markdown]
# # Heatwave NLP Toolkit — Use Case KJS-CES-01 (NLP course)
#
# **Multilingual heatwave advisory generation + citizen heat-report mining.**
#
# This notebook is the NLP part of *Climate Intelligence for Heatwave Monitoring, Prediction
# and Early Warning* (Phase V: "LLM-assisted advisory generation"), built only from
# techniques in the NLP syllabus:
#
# | Cell | What it builds | Syllabus |
# |---|---|---|
# | 2 | IMD heatwave criteria (rule engine) | – (domain knowledge) |
# | 4 | Rule-based multilingual advisory templates | 3.3 (hybrid rules + probabilistic, as a design idea) |
# | 5 | Hashtag word segmentation by argmax (Viterbi) | 1.2, 1.3 |
# | 6 | Noisy social-media text normalisation | 3.2 |
# | 7 | Marathi/Hindi finite-state morphological analyser | 2.1, 2.2 |
# | 8 | CRF Named Entity Recognition on IMD bulletins | 2.3 |
# | 9 | WordNet lexicon expansion, Lesk WSD, multilingual dictionary | 4.1–4.3 |
# | 10 | BiLSTM classifier for citizen posts | 5.1, 5.2 |
# | 11 | LSTM Seq2Seq + attention: forecast → advisory (en/hi/mr) | 5.3 |
# | 12 | Factual-consistency checks + human-in-the-loop review | Responsible AI |
# | 13 | End-to-end pipeline: bulletin + posts → approved advisories | all |
#
# **How to run:** Google Colab → *Runtime → Run all* (CPU is enough, GPU is faster).
# Locally: `python heatwave_nlp.py`. Run the cells in order — each cell uses names
# defined by the cells above it.

# %%
# ============================================================================
# CELL 0 · Setup — installs the only package Colab does not ship with
# ============================================================================
import importlib
import subprocess
import sys


def ensure_package(pip_name, import_name=None):
    """Import a package, pip-installing it first if it is missing."""
    try:
        return importlib.import_module(import_name or pip_name)
    except ImportError:
        print(f"Installing {pip_name} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=True)
        return importlib.import_module(import_name or pip_name)


ensure_package("sklearn-crfsuite", "sklearn_crfsuite")
ensure_package("nltk")
print("Setup complete.")

# %%
# ============================================================================
# CELL 1 · Imports and configuration
# ============================================================================
import json
import math
import os
import pickle
import random
import re
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from itertools import product

import numpy as np
import torch
import torch.nn as nn

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- switches you may want to change -------------------------------------
USE_REAL_IMD_DATA = False   # True → download IMD gridded Tmax with imdlib (needs internet)
APPROVAL_MODE = "auto"      # "auto" or "interactive" (asks you to approve warnings)
SEQ2SEQ_INCLUDE_SEVERITY = True  # False → the Seq2Seq must infer severity from numbers (harder lab)
FAST_MODE = os.environ.get("HEATWAVE_FAST", "0") == "1"  # fewer epochs, for a quick smoke test

print(f"Device: {DEVICE} | torch {torch.__version__} | fast mode: {FAST_MODE}")

# %%
# ============================================================================
# CELL 2 · Domain knowledge — IMD heatwave criteria, places, languages
# ============================================================================
# IMD heatwave criteria (as published by IMD):
#   * Only considered once Tmax ≥ 40 °C (plains), ≥ 37 °C (coastal), ≥ 30 °C (hilly).
#   * Departure-based: heatwave if Tmax is 4.5–6.4 °C above normal, severe if ≥ 6.5 °C.
#   * Actual-based (plains): heatwave if Tmax ≥ 45 °C, severe if Tmax ≥ 47 °C.
# "WATCH" (yellow) below is a project-defined pre-alert band, NOT an IMD criterion.
# The colour mapping is simplified for teaching; IMD's impact-based colours also
# depend on duration.

SEVERITY_LEVELS = ["NORMAL", "WATCH", "HEATWAVE", "SEVERE"]
COLOUR = {"NORMAL": "Green", "WATCH": "Yellow", "HEATWAVE": "Orange", "SEVERE": "Red"}
TERRAIN_THRESHOLD = {"plains": 40.0, "coastal": 37.0, "hilly": 30.0}


def classify_heatwave(tmax, normal, terrain="plains"):
    """Return (severity, departure) for one day at one place using IMD rules."""
    tmax, normal = float(tmax), float(normal)  # plain floats, also for numpy values from imdlib
    dep = round(tmax - normal, 1)
    if tmax >= TERRAIN_THRESHOLD[terrain]:
        if dep >= 6.5 or (terrain == "plains" and tmax >= 47.0):
            return "SEVERE", dep
        if dep >= 4.5 or (terrain == "plains" and tmax >= 45.0):
            return "HEATWAVE", dep
        if dep >= 2.5:
            return "WATCH", dep
    return "NORMAL", dep


# name: (state, IMD region*, terrain, lat, lon, normal May Tmax °C*, Hindi, Marathi)
# * Region mapping and normals are APPROXIMATE values for generating synthetic data.
#   Replace them with IMD's 1991–2020 normals and region shapefile for real work.
_PLACE_ROWS = {
    "Nagpur": ("Maharashtra", "Interior Peninsula", "plains", 21.15, 79.09, 42.5, "नागपुर", "नागपूर"),
    "Akola": ("Maharashtra", "Interior Peninsula", "plains", 20.70, 77.00, 42.5, "अकोला", "अकोला"),
    "Chandrapur": ("Maharashtra", "Interior Peninsula", "plains", 19.95, 79.30, 43.0, "चंद्रपुर", "चंद्रपूर"),
    "Jalgaon": ("Maharashtra", "Interior Peninsula", "plains", 21.00, 75.56, 42.0, "जलगांव", "जळगाव"),
    "Pune": ("Maharashtra", "Interior Peninsula", "plains", 18.52, 73.86, 37.0, "पुणे", "पुणे"),
    "Solapur": ("Maharashtra", "Interior Peninsula", "plains", 17.66, 75.91, 40.0, "सोलापुर", "सोलापूर"),
    "Mumbai": ("Maharashtra", "West Coast", "coastal", 19.08, 72.88, 33.5, "मुंबई", "मुंबई"),
    "Ratnagiri": ("Maharashtra", "West Coast", "coastal", 16.99, 73.31, 33.0, "रत्नागिरी", "रत्नागिरी"),
    "Delhi": ("Delhi", "North West", "plains", 28.61, 77.21, 39.5, "दिल्ली", "दिल्ली"),
    "Jaipur": ("Rajasthan", "North West", "plains", 26.91, 75.79, 40.5, "जयपुर", "जयपूर"),
    "Barmer": ("Rajasthan", "North West", "plains", 25.75, 71.39, 42.0, "बाड़मेर", "बाडमेर"),
    "Sri Ganganagar": ("Rajasthan", "North West", "plains", 29.92, 73.88, 41.5, "श्रीगंगानगर", "श्रीगंगानगर"),
    "Ahmedabad": ("Gujarat", "North West", "plains", 23.02, 72.57, 41.5, "अहमदाबाद", "अहमदाबाद"),
    "Banda": ("Uttar Pradesh", "North Central", "plains", 25.48, 80.33, 42.5, "बांदा", "बांदा"),
    "Prayagraj": ("Uttar Pradesh", "North Central", "plains", 25.44, 81.85, 42.0, "प्रयागराज", "प्रयागराज"),
    "Jhansi": ("Uttar Pradesh", "North Central", "plains", 25.45, 78.57, 42.5, "झांसी", "झाशी"),
    "Bhopal": ("Madhya Pradesh", "North Central", "plains", 23.26, 77.41, 40.5, "भोपाल", "भोपाळ"),
    "Hyderabad": ("Telangana", "Interior Peninsula", "plains", 17.39, 78.49, 39.0, "हैदराबाद", "हैदराबाद"),
    "Kurnool": ("Andhra Pradesh", "Interior Peninsula", "plains", 15.83, 78.04, 40.0, "कुर्नूल", "कुर्नूल"),
    "Chennai": ("Tamil Nadu", "East Coast", "coastal", 13.08, 80.27, 37.5, "चेन्नई", "चेन्नई"),
    "Visakhapatnam": ("Andhra Pradesh", "East Coast", "coastal", 17.69, 83.22, 34.5, "विशाखापत्तनम", "विशाखापट्टणम"),
    "Bhubaneswar": ("Odisha", "East Coast", "plains", 20.30, 85.82, 38.0, "भुवनेश्वर", "भुवनेश्वर"),
    "Kolkata": ("West Bengal", "East Coast", "plains", 22.57, 88.36, 36.0, "कोलकाता", "कोलकाता"),
    "Guwahati": ("Assam", "North East", "plains", 26.14, 91.74, 31.5, "गुवाहाटी", "गुवाहाटी"),
    "Shimla": ("Himachal Pradesh", "Western Himalaya", "hilly", 31.10, 77.17, 24.0, "शिमला", "शिमला"),
    "Dehradun": ("Uttarakhand", "Western Himalaya", "hilly", 30.32, 78.03, 35.0, "देहरादून", "डेहराडून"),
}
PLACES = {
    name: dict(zip(("state", "region", "terrain", "lat", "lon", "normal", "hi", "mr"), row))
    for name, row in _PLACE_ROWS.items()
}

# Meteorological subdivisions appear in bulletins as locations too (used by NER).
SUBDIVISIONS = [
    "Vidarbha", "Marathwada", "Madhya Maharashtra", "Konkan and Goa", "West Rajasthan",
    "East Rajasthan", "East Uttar Pradesh", "West Madhya Pradesh", "Gangetic West Bengal",
    "Telangana", "Rayalaseema", "Saurashtra and Kutch", "Odisha", "Haryana", "Punjab",
    "Coastal Andhra Pradesh", "Interior Odisha", "Jharkhand", "Bihar", "Chhattisgarh",
]

AUDIENCES = ["CITIZEN", "FARMER", "HEALTH", "AUTHORITY"]
LANGS = ["en", "hi", "mr"]
MONTHS = {
    "en": ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"],
    "hi": ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त",
           "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"],
    "mr": ["जानेवारी", "फेब्रुवारी", "मार्च", "एप्रिल", "मे", "जून", "जुलै", "ऑगस्ट",
           "सप्टेंबर", "ऑक्टोबर", "नोव्हेंबर", "डिसेंबर"],
}


def make_record(place, day, tmax, normal):
    """A forecast/observation record — the structured input of the NLP layer."""
    p = PLACES[place]
    severity, dep = classify_heatwave(tmax, normal, p["terrain"])
    return {
        "place": place, "state": p["state"], "region": p["region"], "terrain": p["terrain"],
        "date": day.isoformat(), "tmax": round(float(tmax), 1), "normal": round(float(normal), 1),
        "departure": dep, "severity": severity, "colour": COLOUR[severity],
    }


for t in (41.0, 44.6, 45.2, 47.5):
    print(f"Nagpur Tmax {t} °C (normal 42.5) →", classify_heatwave(t, 42.5, "plains"))

# %%
# ============================================================================
# CELL 3 · Datasets — forecast records (real IMD or synthetic) + citizen posts
# ============================================================================
# Real data path: IMD Pune gridded daily Tmax (the GRD files named in the use case),
# read with the `imdlib` package. Normals are approximated by the day-of-year mean
# over the downloaded years, so use as many years as you can.


def download_imd_tmax(imd, year, data_dir, retries=3):
    """Download one year of IMD 1° gridded Tmax, retrying (the IMD server is often slow).
    A complete file is 31 × 31 grid cells × 4 bytes × days; anything else is re-downloaded."""
    path = os.path.join(data_dir, "tmax", f"{year}.GRD")
    days = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
    complete = lambda: os.path.exists(path) and os.path.getsize(path) == 31 * 31 * 4 * days
    for attempt in range(1, retries + 1):
        if complete():
            return
        try:
            imd.get_data("tmax", year, year, fn_format="yearwise", file_dir=data_dir)
        except Exception as exc:
            print(f"  IMD download {year} failed (attempt {attempt}/{retries}): {exc}")
            time.sleep(5 * attempt)
    if not complete():
        raise RuntimeError(f"could not download IMD Tmax for {year}")


def load_imd_records(years=(2019, 2024), months=(4, 5, 6), data_dir="imd_data"):
    imd = ensure_package("imdlib")
    for year in range(years[0], years[1] + 1):  # files already on disk are reused
        download_imd_tmax(imd, year, data_dir)
    ds = imd.open_data("tmax", years[0], years[1], fn_format="yearwise", file_dir=data_dir).get_xarray()
    tmax_grid = ds["tmax"].where(ds["tmax"] < 99)  # 99.9 marks missing cells
    records = []
    for name, p in PLACES.items():
        series = tmax_grid.sel(lat=p["lat"], lon=p["lon"], method="nearest").to_series().dropna()
        if series.empty:
            continue  # e.g. a coastal city whose nearest 1° cell is sea
        climatology = series.groupby(series.index.dayofyear).mean()
        recent = series[(series.index.year == years[1]) & series.index.month.isin(months)]
        for ts, value in recent.items():
            records.append(make_record(name, ts.date(), value, climatology.loc[ts.dayofyear]))
    return records


DEP_RANGES = {"NORMAL": (-3.0, 2.4), "WATCH": (2.5, 4.4), "HEATWAVE": (4.5, 6.4), "SEVERE": (6.5, 9.0)}


def synthetic_records(n, seed=SEED):
    """Balanced synthetic records: each severity level gets n/4 examples."""
    rng = random.Random(seed)
    names = list(PLACES)
    records = []
    while len(records) < n:
        target = SEVERITY_LEVELS[len(records) % len(SEVERITY_LEVELS)]
        for _ in range(100):  # rejection sampling until IMD rules give the target level
            name = rng.choice(names)
            normal = round(PLACES[name]["normal"] + rng.uniform(-1.5, 1.5), 1)
            tmax = round(normal + rng.uniform(*DEP_RANGES[target]), 1)
            if classify_heatwave(tmax, normal, PLACES[name]["terrain"])[0] == target:
                day = date(2025, 4, 1) + timedelta(days=rng.randint(0, 75))
                records.append(make_record(name, day, tmax, normal))
                break
    return records


RECORDS = []
if USE_REAL_IMD_DATA:
    try:
        RECORDS = load_imd_records()
        print(f"Loaded {len(RECORDS)} real IMD records.")
    except Exception as exc:  # no internet, imdlib API change, ...
        print(f"Could not load IMD data ({exc}); falling back to synthetic records.")
if not RECORDS:
    RECORDS = synthetic_records(400 if FAST_MODE else 1000)
    print(f"Generated {len(RECORDS)} synthetic forecast records.")
print("Severity counts:", Counter(r["severity"] for r in RECORDS))
print("Example record:", RECORDS[2])

# ---- Citizen social-media posts (synthetic, with realistic noise) ----------
# Labels: 0 = not about heat (includes tricky senses of "heat"/"hot"/"fire"),
#         1 = heat discomfort, 2 = heat-health emergency.
POST_LABELS = ["NOT_HEAT", "HEAT_DISCOMFORT", "HEAT_EMERGENCY"]
POST_TEMPLATES = {
    0: [
        "feeling the heat before tomorrow's exam",
        "miami heat won again last night",
        "this new song is pure fire",
        "hot deals on smartphones today only",
        "the debate got really heated in class today",
        "traffic in {place} is crazy this evening",
        "spicy misal in {place} is too hot for me",
        "rain finally arrived in {place} , loving the weather",
        "my laptop keeps overheating while gaming",
        "new cafe opened near {place} station , must visit",
        # held out for testing ↓ (the model never sees these phrasings)
        "the team is under heat from fans after the loss",
        "{place} metro work is taking forever",
        "hot chocolate on a cold evening is the best",
    ],
    1: [
        "it is so hot in {place} today",
        "{place} is burning , {t} degrees already",
        "cannot step out in this garmi , {place} feels like an oven",
        "{t} degrees in {place} and the ac is not working",
        "bahut garmi hai aaj {place} mein",
        "the afternoon sun in {place} is brutal",
        "loo chal rahi hai {place} mein , stay inside",
        "power cut plus this heat , worst day in {place}",
        "sweating like crazy just walking to the bus stop in {place}",
        "even the fan is blowing hot air in {place}",
        # held out ↓
        "{place} weather is unbearable , {t} degrees at noon",
        "scorching day in {place} , roads are empty",
        "drank five bottles of water and still thirsty in this {place} heat",
    ],
    2: [
        "my grandfather collapsed due to heatstroke in {place} , need ambulance",
        "three labourers fainted in the heat near {place}",
        "hospital in {place} is full of heat stroke patients",
        "no drinking water in our area of {place} and it is {t} degrees , kids are falling sick",
        "{place} heatwave : two deaths reported today",
        "cattle dying due to extreme heat in {place} villages",
        "urgent : old woman unconscious in the sun at {place} bus stand",
        "heatstroke case at {place} construction site , worker taken to hospital",
        "please send water tanker to {place} , people fainting in this heat",
        "child hospitalised with severe dehydration in {place} after heat exposure",
        # held out ↓
        "dehydration cases rising fast in {place} hospitals",
        "farmer died of sunstroke in a field near {place}",
        "emergency : traffic police officer collapsed in the heat at {place} junction",
    ],
}
N_HELDOUT_POST_TEMPLATES = 3
POST_EMOJIS = {0: ["🔥", "😍", "🏀", "🌧️", ""], 1: ["🥵", "☀️", "😩", "😓", ""], 2: ["🚑", "🙏", "🥵", "💧", ""]}
POST_HASHTAGS = {
    0: ["#mondaymood", "#nba", "#exams", "#foodie", ""],
    1: ["#heatwave", "#garmi", "#{p}heat", "#stayhydrated", ""],
    2: ["#heatwavealert", "#emergency", "#{p}heatwave", "#help", ""],
}


def make_post(template, label, rng):
    place = rng.choice(list(PLACES))
    text = template.format(place=place if rng.random() < 0.5 else place.lower(), t=rng.randint(41, 48))
    words = []
    for w in text.split():  # inject noise: elongations and SHOUTING
        r = rng.random()
        if r < 0.08 and w.isalpha() and len(w) > 2:
            w = w + w[-1] * rng.randint(2, 3)
        elif r < 0.11:
            w = w.upper()
        words.append(w)
    tag = rng.choice(POST_HASHTAGS[label]).replace("{p}", place.lower().replace(" ", ""))
    return " ".join(words + [rng.choice(POST_EMOJIS[label]), tag]).strip()


def build_posts(per_template_train=60, per_template_test=40, seed=SEED):
    rng = random.Random(seed)
    train, test = [], []
    for label, templates in POST_TEMPLATES.items():
        for i, tpl in enumerate(templates):
            held_out = i >= len(templates) - N_HELDOUT_POST_TEMPLATES
            target, n = (test, per_template_test) if held_out else (train, per_template_train)
            target += [(make_post(tpl, label, rng), label) for _ in range(n)]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


POSTS_TRAIN, POSTS_TEST = build_posts()
print(f"\nPosts: {len(POSTS_TRAIN)} train / {len(POSTS_TEST)} test (test uses unseen phrasings)")
for text, label in POSTS_TRAIN[:4]:
    print(f"  [{POST_LABELS[label]}] {text}")

# %%
# ============================================================================
# CELL 4 · Rule-based multilingual advisory templates  (hybrid rules + probabilistic, idea of 3.3)
# ============================================================================
# Templates are written with slot placeholders (<place>, <date>, <tmax>, <dep>).
# This "delexicalised" form is what the Seq2Seq model learns to produce in Cell 11;
# real values are filled in afterwards (relexicalisation), so the model can never
# invent a temperature. The templates double as (a) the rule-based baseline,
# (b) training data for the Seq2Seq model and (c) the safe fallback in Cell 12.

HEADLINES = {
    "NORMAL": {
        "en": ["No heatwave expected in <place> on <date> . Maximum temperature around <tmax> °C , close to normal .",
               "Weather in <place> on <date> is near normal with a maximum of about <tmax> °C ."],
        "hi": ["<date> को <place> में लू की कोई संभावना नहीं है । अधिकतम तापमान लगभग <tmax> °C रहेगा , जो सामान्य के आसपास है ।",
               "<date> को <place> का मौसम सामान्य रहेगा , अधिकतम तापमान लगभग <tmax> °C ।"],
        "mr": ["<date> रोजी <place> येथे उष्णतेच्या लाटेची शक्यता नाही . कमाल तापमान सुमारे <tmax> °C , सामान्य पातळीच्या जवळ राहील .",
               "<date> रोजी <place> येथील हवामान सामान्य राहील , कमाल तापमान सुमारे <tmax> °C ."],
    },
    "WATCH": {
        "en": ["Yellow alert ( heat watch ) for <place> on <date> . Maximum temperature may reach <tmax> °C , <dep> °C above normal .",
               "Heat watch issued for <place> on <date> : temperatures up to <tmax> °C , about <dep> °C above normal ."],
        "hi": ["<date> को <place> के लिए पीला अलर्ट ( गर्मी की चेतावनी ) । अधिकतम तापमान <tmax> °C तक पहुँच सकता है , जो सामान्य से <dep> °C अधिक है ।",
               "<place> में <date> को तापमान <tmax> °C तक , सामान्य से लगभग <dep> °C अधिक रहने की संभावना है । पीला अलर्ट जारी ।"],
        "mr": ["<date> रोजी <place> साठी पिवळा इशारा ( उष्णतेची सूचना ) . कमाल तापमान <tmax> °C पर्यंत जाऊ शकते , जे सामान्यपेक्षा <dep> °C जास्त आहे .",
               "<place> येथे <date> रोजी तापमान <tmax> °C पर्यंत , सामान्यपेक्षा सुमारे <dep> °C जास्त राहण्याची शक्यता . पिवळा इशारा जारी ."],
    },
    "HEATWAVE": {
        "en": ["Orange alert : heatwave conditions likely in <place> on <date> . Maximum temperature <tmax> °C , <dep> °C above normal .",
               "Heatwave warning for <place> on <date> with maximum temperature of <tmax> °C , <dep> °C above normal ."],
        "hi": ["नारंगी अलर्ट : <date> को <place> में लू चलने की संभावना । अधिकतम तापमान <tmax> °C , सामान्य से <dep> °C अधिक ।",
               "<place> में <date> को लू की चेतावनी , अधिकतम तापमान <tmax> °C , जो सामान्य से <dep> °C अधिक है ।"],
        "mr": ["नारिंगी इशारा : <date> रोजी <place> येथे उष्णतेची लाट येण्याची शक्यता . कमाल तापमान <tmax> °C , सामान्यपेक्षा <dep> °C जास्त .",
               "<place> येथे <date> रोजी उष्णतेच्या लाटेचा इशारा , कमाल तापमान <tmax> °C , जे सामान्यपेक्षा <dep> °C जास्त आहे ."],
    },
    "SEVERE": {
        "en": ["Red alert : severe heatwave in <place> on <date> . Maximum temperature <tmax> °C , <dep> °C above normal .",
               "Severe heatwave warning for <place> on <date> : temperature may touch <tmax> °C , <dep> °C above normal ."],
        "hi": ["लाल अलर्ट : <date> को <place> में भीषण लू । अधिकतम तापमान <tmax> °C , सामान्य से <dep> °C अधिक ।",
               "<place> में <date> को भीषण लू की चेतावनी , तापमान <tmax> °C तक पहुँच सकता है , सामान्य से <dep> °C अधिक ।"],
        "mr": ["लाल इशारा : <date> रोजी <place> येथे तीव्र उष्णतेची लाट . कमाल तापमान <tmax> °C , सामान्यपेक्षा <dep> °C जास्त .",
               "<place> येथे <date> रोजी तीव्र उष्णतेच्या लाटेचा इशारा , तापमान <tmax> °C पर्यंत जाऊ शकते , सामान्यपेक्षा <dep> °C जास्त ."],
    },
}

# "low" actions are used for NORMAL/WATCH, "high" for HEATWAVE/SEVERE.
ACTIONS = {
    "CITIZEN": {
        "low": {
            "en": ["Drink water regularly and avoid long exposure to the afternoon sun .",
                   "Keep drinking water through the day and carry a water bottle when outside ."],
            "hi": ["नियमित रूप से पानी पिएँ और दोपहर की धूप में ज़्यादा देर न रहें ।",
                   "दिन भर पानी पीते रहें और बाहर जाते समय पानी की बोतल साथ रखें ।"],
            "mr": ["नियमितपणे पाणी प्या आणि दुपारच्या उन्हात जास्त वेळ थांबू नका .",
                   "दिवसभर पाणी पीत राहा आणि बाहेर जाताना पाण्याची बाटली सोबत ठेवा ."],
        },
        "high": {
            "en": ["Avoid going out between 12 noon and 3 pm , drink water even if not thirsty and use ORS , lassi or buttermilk .",
                   "Stay indoors during peak afternoon hours , wear light cotton clothes and cover your head when outside ."],
            "hi": ["दोपहर 12 बजे से 3 बजे के बीच बाहर निकलने से बचें , प्यास न लगने पर भी पानी पिएँ और ओआरएस , लस्सी या छाछ लें ।",
                   "दोपहर के समय घर के अंदर रहें , हल्के सूती कपड़े पहनें और बाहर जाते समय सिर ढकें ।"],
            "mr": ["दुपारी 12 ते 3 या वेळेत बाहेर जाणे टाळा , तहान नसली तरी पाणी प्या आणि ओआरएस , लस्सी किंवा ताक घ्या .",
                   "दुपारच्या वेळी घरातच राहा , हलके सुती कपडे घाला आणि बाहेर जाताना डोके झाका ."],
        },
    },
    "FARMER": {
        "low": {
            "en": ["Continue normal field operations and irrigate crops as per schedule .",
                   "Field work can continue as usual ; keep livestock supplied with clean water ."],
            "hi": ["खेती का काम सामान्य रूप से जारी रखें और समय पर सिंचाई करें ।",
                   "खेत का काम सामान्य रूप से चल सकता है , पशुओं को साफ पानी देते रहें ।"],
            "mr": ["शेतीची कामे नेहमीप्रमाणे सुरू ठेवा आणि वेळेवर पिकांना पाणी द्या .",
                   "शेतातील कामे नेहमीप्रमाणे करता येतील , जनावरांना स्वच्छ पाणी देत राहा ."],
        },
        "high": {
            "en": ["Irrigate crops in the early morning or evening , avoid spraying pesticides in the afternoon and keep livestock in shade .",
                   "Schedule field work before 11 am , apply mulch to retain soil moisture and give animals plenty of drinking water ."],
            "hi": ["फसलों की सिंचाई सुबह जल्दी या शाम को करें , दोपहर में कीटनाशक का छिड़काव न करें और पशुओं को छाया में रखें ।",
                   "खेत का काम सुबह 11 बजे से पहले करें , मिट्टी की नमी बनाए रखने के लिए मल्चिंग करें और पशुओं को भरपूर पानी दें ।"],
            "mr": ["पिकांना पहाटे किंवा संध्याकाळी पाणी द्या , दुपारी कीटकनाशक फवारणी टाळा आणि जनावरांना सावलीत ठेवा .",
                   "शेतीची कामे सकाळी 11 पूर्वी उरका , जमिनीतील ओलावा टिकवण्यासाठी आच्छादन करा आणि जनावरांना भरपूर पाणी द्या ."],
        },
    },
    "HEALTH": {
        "low": {
            "en": ["Continue routine surveillance and keep ORS stocks available at health centres .",
                   "Keep monitoring heat-related illness cases as part of routine surveillance ."],
            "hi": ["नियमित निगरानी जारी रखें और स्वास्थ्य केंद्रों पर ओआरएस का भंडार रखें ।",
                   "नियमित निगरानी के तहत गर्मी से जुड़ी बीमारियों के मामलों पर नज़र रखें ।"],
            "mr": ["नियमित देखरेख सुरू ठेवा आणि आरोग्य केंद्रांवर ओआरएसचा साठा ठेवा .",
                   "नियमित देखरेखीअंतर्गत उष्णतेशी संबंधित आजारांच्या रुग्णांवर लक्ष ठेवा ."],
        },
        "high": {
            "en": ["Activate heat-stroke wards , keep ice packs , IV fluids and ORS ready and report heat-related illness daily .",
                   "Prepare cooling rooms in hospitals , alert ambulance services and prioritise elderly , children and outdoor workers ."],
            "hi": ["हीट-स्ट्रोक वार्ड सक्रिय करें , आइस पैक , आईवी फ्लूइड और ओआरएस तैयार रखें तथा गर्मी से जुड़ी बीमारियों की रोज़ रिपोर्ट करें ।",
                   "अस्पतालों में कूलिंग रूम तैयार करें , एम्बुलेंस सेवाओं को सतर्क करें और बुज़ुर्गों , बच्चों और बाहर काम करने वालों को प्राथमिकता दें ।"],
            "mr": ["उष्माघात कक्ष सुरू करा , आईस पॅक , सलाईन आणि ओआरएस तयार ठेवा आणि उष्णतेशी संबंधित आजारांची रोज नोंद करा .",
                   "रुग्णालयांत शीतकक्ष तयार ठेवा , रुग्णवाहिका सेवांना सतर्क करा आणि वृद्ध , मुले व बाहेर काम करणाऱ्यांना प्राधान्य द्या ."],
        },
    },
    "AUTHORITY": {
        "low": {
            "en": ["Keep the heat action plan on standby and monitor forecasts .",
                   "Continue routine preparedness and ensure drinking water points are functional ."],
            "hi": ["हीट एक्शन प्लान तैयार रखें और पूर्वानुमान पर नज़र रखें ।",
                   "नियमित तैयारी जारी रखें और सुनिश्चित करें कि पेयजल केंद्र चालू हों ।"],
            "mr": ["उष्णता कृती आराखडा तयार ठेवा आणि हवामान अंदाजावर लक्ष ठेवा .",
                   "नेहमीची तयारी सुरू ठेवा आणि पिण्याच्या पाण्याची केंद्रे सुरू असल्याची खात्री करा ."],
        },
        "high": {
            "en": ["Open cooling centres and drinking water points , and shift outdoor work away from 12 noon to 3 pm .",
                   "Activate the heat action plan , deploy water tankers to vulnerable areas and issue public alerts ."],
            "hi": ["कूलिंग सेंटर और पेयजल केंद्र खोलें तथा बाहरी काम को दोपहर 12 से 3 बजे के समय से हटाएँ ।",
                   "हीट एक्शन प्लान सक्रिय करें , संवेदनशील क्षेत्रों में पानी के टैंकर भेजें और सार्वजनिक चेतावनी जारी करें ।"],
            "mr": ["शीतकेंद्रे आणि पिण्याच्या पाण्याची केंद्रे सुरू करा , आणि बाहेरील कामे दुपारी 12 ते 3 या वेळेतून हलवा .",
                   "उष्णता कृती आराखडा सक्रिय करा , संवेदनशील भागांत पाण्याचे टँकर पाठवा आणि सार्वजनिक इशारा जारी करा ."],
        },
    },
}

EXTRA_LINE = {
    "WATCH": {"en": "Stay alert for further updates as temperatures are rising .",
              "hi": "तापमान बढ़ रहा है , आगे की जानकारी पर नज़र रखें ।",
              "mr": "तापमान वाढत आहे , पुढील सूचनांकडे लक्ष ठेवा ."},
    "SEVERE": {"en": "Call 108 for heat-stroke emergencies .",
               "hi": "लू लगने की आपात स्थिति में 108 पर कॉल करें ।",
               "mr": "उष्माघाताच्या आपत्कालीन स्थितीत 108 वर संपर्क करा ."},
}


def action_band(severity):
    return "high" if severity in ("HEATWAVE", "SEVERE") else "low"


def _assemble(head, action, severity, lang):
    extra = [EXTRA_LINE[severity][lang]] if severity in EXTRA_LINE else []
    return " ".join([head, action] + extra).split()


def template_advisory(severity, audience, lang, rng=random):
    """Rule-based baseline: pick one headline and one action sentence (delexicalised tokens)."""
    head = rng.choice(HEADLINES[severity][lang])
    action = rng.choice(ACTIONS[audience][action_band(severity)][lang])
    return _assemble(head, action, severity, lang)


def all_references(severity, audience, lang):
    """Every valid advisory for this input — used as multi-reference for BLEU."""
    return [_assemble(h, a, severity, lang)
            for h, a in product(HEADLINES[severity][lang], ACTIONS[audience][action_band(severity)][lang])]


def format_date(iso, lang):
    d = date.fromisoformat(iso)
    return f"{d.day} {MONTHS[lang][d.month - 1]}"


def relexicalize(tokens, record, lang):
    """Fill the slots with the record's real values and join tokens into readable text."""
    place = record["place"] if lang == "en" else PLACES[record["place"]][lang]
    values = {"<place>": place, "<date>": format_date(record["date"], lang),
              "<tmax>": f"{record['tmax']:.1f}", "<dep>": f"{record['departure']:.1f}"}
    text = " ".join(values.get(t, t) for t in tokens)
    text = re.sub(r"\s+([.,:;!?।)])", r"\1", text)
    return re.sub(r"\(\s+", "(", text)


demo = RECORDS[2]
for lang in LANGS:
    print(f"[{lang}] {relexicalize(template_advisory(demo['severity'], 'FARMER', lang), demo, lang)}")

# %%
# ============================================================================
# CELL 5 · Word boundary detection by argmax  (Syllabus 1.2, 1.3)
# ============================================================================
# Hashtags have no spaces: "#heatwavealertnagpur". We choose the segmentation
#     argmax_{w1..wk}  Σ log P(wi)        (unigram language model)
# with dynamic programming (Viterbi): best[i] = max_j best[j] + log P(text[j:i]).
# The same argmax-over-paths idea is what HMM speech recognisers use to find word
# boundaries in continuous speech.

def _corpus_words():
    texts = [tpl for tpls in POST_TEMPLATES.values() for tpl in tpls]
    texts += [s for sev in HEADLINES.values() for s in sev["en"]]
    texts += [s for aud in ACTIONS.values() for band in aud.values() for s in band["en"]]
    words = re.findall(r"[a-z]+", " ".join(texts).lower())
    for name in PLACES:  # place names, with and without spaces
        words += name.lower().split() * 3 + [name.lower().replace(" ", "")] * 3
    words += ("heat wave heatwave alert stay hydrated help emergency weather mood monday "
              "foodie exams nba garmi water save summer temperature india today").split() * 3
    return words


UNIGRAMS = Counter(_corpus_words())
UNI_TOTAL = sum(UNIGRAMS.values())


def word_logp(word):
    count = UNIGRAMS.get(word)
    if count:
        return math.log(count / UNI_TOTAL)
    return math.log(10.0 / (UNI_TOTAL * 10 ** len(word)))  # unknown words: longer → less likely


def segment(text, max_word_len=20):
    text = text.lower()
    n = len(text)
    best = [0.0] + [-math.inf] * n
    back = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - max_word_len), i):
            score = best[j] + word_logp(text[j:i])
            if score > best[i]:
                best[i], back[i] = score, j
    words, i = [], n
    while i > 0:
        words.append(text[back[i]:i])
        i = back[i]
    return words[::-1]


for tag in ["heatwavealertnagpur", "stayhydrated", "jaipurheat", "sriganganagarheatwave"]:
    print(f"#{tag:24s} → {segment(tag)}")

# %%
# ============================================================================
# CELL 6 · Robust processing of noisy web text  (Syllabus 3.2)
# ============================================================================
EMOJI_MAP = {"🥵": "hot", "🔥": "fire", "☀️": "sun", "☀": "sun", "😩": "tired", "😓": "sweat",
             "🚑": "ambulance", "💧": "water", "🌧️": "rain", "🌧": "rain", "😍": "love",
             "🏀": "basketball", "🙏": "please"}
HINGLISH = {"garmi": "heat", "garam": "hot", "loo": "heatwave", "dhoop": "sun", "pani": "water",
            "paani": "water", "bahut": "very", "bohot": "very", "aaj": "today", "hai": "is",
            "mein": "in", "tapman": "temperature", "behosh": "unconscious"}
_place_forms = {}
for _name in PLACES:
    _place_forms[_name.lower()] = _name
    _place_forms[_name.lower().replace(" ", "")] = _name
PLACE_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _place_forms), key=len, reverse=True)) + r")\b")


def normalize_post(text, mask_places=True):
    """Noisy post → clean token list. Order of steps matters (see README)."""
    t = text
    for emoji, word in EMOJI_MAP.items():
        t = t.replace(emoji, f" {word} ")
    t = t.lower()
    t = re.sub(r"https?://\S+|@\w+", " ", t)
    t = re.sub(r"#(\w+)", lambda m: " " + " ".join(segment(m.group(1))) + " ", t)
    t = re.sub(r"°\s*c\b|°", " degrees ", t)
    t = re.sub(r"(\d)\s*c\b", r"\1 degrees", t)
    t = re.sub(r"\bdeg\b", "degrees", t)
    t = re.sub(r"([a-z])\1{2,}", r"\1", t)          # hottt → hot, sooo → so
    t = re.sub(r"\d+(?:\.\d+)?", " <num> ", t)
    t = re.sub(r"[^a-z<>\s']", " ", t)               # drop punctuation / other scripts
    t = " ".join(HINGLISH.get(w, w) for w in t.split())
    if mask_places:
        t = PLACE_RE.sub("<place>", t)               # helps generalise to unseen places
    return t.split()


for raw in ["NAGPUR is burninggg 🥵 47°C already!! #heatwavealert @imd",
            "bahut garmi hai aaj Jaipur mein #jaipurheat"]:
    print(f"{raw}\n   → {normalize_post(raw)}")

# %%
# ============================================================================
# CELL 7 · Morphology of Indian languages — finite-state analyser  (Syllabus 2.1, 2.2)
# ============================================================================
# Marathi attaches case markers to the noun: नागपूरमध्ये = नागपूर + मध्ये ("in Nagpur").
# Many stems also change before a suffix (oblique form): पुणे → पुण्य + ात = पुण्यात.
# A two-layer finite-state transducer models this:
#   layer 1  strip a suffix from the suffix lexicon        पुण्यात → पुण्य + ात
#   layer 2  undo the oblique stem change                   पुण्य   → पुणे
# Candidates whose stem is in the lexicon are preferred (lexicon-driven disambiguation).

MR_SUFFIXES = [("ामध्ये", "LOC:in"), ("मध्ये", "LOC:in"), ("ातील", "LOC:of"), ("तील", "LOC:of"),
               ("च्या", "GEN:obl"), ("साठी", "BEN:for"), ("पर्यंत", "TERM:until"), ("ाहून", "ABL:from"),
               ("हून", "ABL:from"), ("कडे", "ALL:towards"), ("ाला", "DAT"), ("ला", "DAT"), ("ाने", "INS"),
               ("ने", "INS"), ("चा", "GEN:m"), ("ची", "GEN:f"), ("चे", "GEN:n"), ("ात", "LOC:in"), ("त", "LOC:in")]
HI_SUFFIXES = [("ियों", "PL:obl"), ("ाओं", "PL:obl"), ("ों", "PL:obl"), ("ियाँ", "PL"), ("ाएँ", "PL"), ("ें", "PL")]
OBLIQUE_RULES = [("्य", "े"), ("े", "ा"), ("", "ा"), ("", "")]  # (stem ending, restored ending)

DEV_TO_PLACE = {}
for _name, _p in PLACES.items():
    DEV_TO_PLACE[_p["hi"]] = _name
    DEV_TO_PLACE[_p["mr"]] = _name
MORPH_LEXICON = set(DEV_TO_PLACE) | {"विदर्भ", "मराठवाडा", "कोकण", "उष्णता", "किसान", "जिला", "गाँव",
                                     "शेतकरी", "पाणी", "तापमान", "रुग्णालय", "शहर"}


def morph_analyze(word):
    """All (stem, suffix, gloss, in_lexicon) analyses, best first."""
    analyses = []
    for suffix, gloss in MR_SUFFIXES + HI_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            raw_stem = word[: -len(suffix)]
            for ending, restored in OBLIQUE_RULES:
                if raw_stem.endswith(ending):
                    stem = raw_stem[: len(raw_stem) - len(ending)] + restored
                    analyses.append((stem, suffix, gloss, stem in MORPH_LEXICON))
    analyses.append((word, "", "BARE", word in MORPH_LEXICON))
    analyses.sort(key=lambda a: (not a[3], -len(a[1])))  # lexicon hits first, then longest suffix
    return analyses


def find_places(text):
    """Place names in a post: Latin script via the gazetteer, Devanagari via morphology."""
    found = {_place_forms[m] for m in PLACE_RE.findall(" ".join(normalize_post(text, mask_places=False)))}
    for word in re.findall(r"[\u0900-\u097F]+", text):
        stem = morph_analyze(word)[0][0]
        if stem in DEV_TO_PLACE:
            found.add(DEV_TO_PLACE[stem])
    return sorted(found)


for w in ["नागपूरमध्ये", "पुण्यात", "जळगावात", "मुंबईत", "विदर्भातील", "उष्णतेची", "किसानों", "जिलों"]:
    stem, suffix, gloss, known = morph_analyze(w)[0]
    print(f"{w:14s} → stem {stem:10s} suffix {suffix or '-':6s} {gloss:10s} {'(lexicon)' if known else ''}")
print("Places in a Marathi post:", find_places("पुण्यात आणि नागपूरमध्ये खूप उष्णता आहे"))

# %%
# ============================================================================
# CELL 8 · Named Entity Recognition with a CRF  (Syllabus 2.3: NER, MaxEnt, CRF)
# ============================================================================
# Entities in IMD-style bulletins:
#   LOC (place/subdivision) · DATE · TEMP (observed temperature) · DEP (departure
#   from normal) · SEV (heatwave / severe heatwave)
# A linear-chain CRF = a MaxEnt model over the whole tag sequence: hand-made
# features per token + learned tag-to-tag transition weights.
# Evaluation is honest: test sentences use places the model never saw, and some
# test templates are new phrasings.
import sklearn_crfsuite

TOKEN_RE = re.compile(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d+(?:st|nd|rd|th)\b|\d+(?:\.\d+)?|°C|°"
                      r"|[A-Za-z]+(?:[-'][A-Za-z]+)*|[^\sA-Za-z\d]")


def tokenize(text):
    return TOKEN_RE.findall(text)


NER_TEMPLATES_TRAIN = [
    "{SEV} conditions very likely in isolated pockets over {LOC} on {DATE} .",
    "The highest maximum temperature of {TEMP} was recorded at {LOC} on {DATE} .",
    "{SEV} conditions prevailed in many parts of {LOC} and {LOC} .",
    "Maximum temperatures were in the range of {TEMP} to {TEMP} over {LOC} .",
    "{LOC} recorded {TEMP} , which is above normal by {DEP} .",
    "Yellow alert issued for {LOC} for {DATE} due to {SEV} .",
    "Orange alert : {SEV} likely over {LOC} from {DATE} to {DATE} .",
    "No significant change in maximum temperatures over {LOC} during the next 3 days .",
    "Departure of maximum temperature at {LOC} was {DEP} above normal on {DATE} .",
    "{LOC} : {TEMP} ( {DEP} above normal ) .",
    "Residents of {LOC} are advised to stay indoors as {SEV} is expected on {DATE} .",
    "Temperatures likely to rise by 2 - 3 °C over {LOC} during the next 48 hours .",
    "{SEV} warning for {LOC} , {LOC} and {LOC} from {DATE} .",
    "Day temperatures of {TEMP} or more were reported at most places in {LOC} .",
    "{LOC} was the hottest place in the country at {TEMP} on {DATE} .",
    "Red alert : {SEV} over {LOC} on {DATE} and {DATE} .",
]
NER_TEMPLATES_TEST_ONLY = [
    "On {DATE} , {LOC} sizzled at {TEMP} , {DEP} above the seasonal normal .",
    "{SEV} is likely to continue over {LOC} till {DATE} .",
    "Hot and humid weather with {SEV} in {LOC} expected on {DATE} .",
    "Night temperatures at {LOC} stayed high while the day touched {TEMP} .",
]
NER_HELDOUT_LOCS = {"Chandrapur", "Sri Ganganagar", "Kurnool", "Visakhapatnam", "Banda", "Ratnagiri",
                    "Rayalaseema", "Saurashtra and Kutch", "Chhattisgarh", "Interior Odisha"}
ALL_LOCS = list(PLACES) + SUBDIVISIONS
TRAIN_LOCS = [l for l in ALL_LOCS if l not in NER_HELDOUT_LOCS]
TEST_LOCS = sorted(NER_HELDOUT_LOCS)
GAZETTEER = {w.lower() for loc in TRAIN_LOCS for w in loc.split()} - {"and"}  # train places only


def _ordinal(d):
    return "th" if 11 <= d <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")


def sample_entity(kind, rng, locs):
    if kind == "LOC":
        return rng.choice(locs)
    if kind == "DATE":
        m, d = rng.choice([4, 5, 6]), rng.randint(1, 30)
        month = MONTHS["en"][m - 1]
        return rng.choice([f"{d} {month}", f"{d}{_ordinal(d)} {month}", f"{month} {d}",
                           f"{d:02d}-{m:02d}-2025", f"{d:02d}/{m:02d}/2025", f"{d} {month} 2025"])
    if kind == "TEMP":
        value = rng.choice([f"{rng.uniform(38, 48.5):.1f}", str(rng.randint(38, 48))])
        return value + rng.choice([" °C", "°C", " degrees Celsius", " deg C"])
    if kind == "DEP":
        return f"{rng.uniform(3, 9):.1f}" + rng.choice([" °C", "°C", " degrees"])
    if kind == "SEV":
        return rng.choice(["heatwave", "heat wave", "severe heatwave", "severe heat wave"])
    raise ValueError(kind)


def fill_ner_template(template, rng, locs):
    tokens, tags = [], []
    for piece in template.split():
        slot = re.fullmatch(r"\{(\w+)\}", piece)
        if slot:
            kind = slot.group(1)
            toks = tokenize(sample_entity(kind, rng, locs))
            tags += [f"B-{kind}"] + [f"I-{kind}"] * (len(toks) - 1)
        else:
            toks = tokenize(piece)
            tags += ["O"] * len(toks)
        tokens += toks
    tokens[0] = tokens[0][0].upper() + tokens[0][1:]
    return tokens, tags


def build_ner_data(seed=SEED):
    rng = random.Random(seed)
    train = [fill_ner_template(rng.choice(NER_TEMPLATES_TRAIN), rng, TRAIN_LOCS) for _ in range(1500)]
    test = [fill_ner_template(rng.choice(NER_TEMPLATES_TRAIN), rng, TEST_LOCS) for _ in range(150)]
    test += [fill_ner_template(rng.choice(NER_TEMPLATES_TEST_ONLY), rng, TEST_LOCS) for _ in range(150)]
    return train, test


def word_shape(w):
    shape = re.sub(r"[A-Z]", "X", w)
    shape = re.sub(r"[a-z]", "x", shape)
    shape = re.sub(r"\d", "d", shape)
    return re.sub(r"(.)\1+", r"\1", shape)  # Xxxxx → Xx, 46.8 → d.d


def _is_number(w):
    return re.fullmatch(r"\d+(?:\.\d+)?", w) is not None


def token_features(tokens, i, use_gazetteer=True):
    w = tokens[i]
    feats = {
        "bias": 1.0, "w.lower": w.lower(), "w.suffix3": w[-3:].lower(), "w.prefix3": w[:3].lower(),
        "w.istitle": w.istitle(), "w.isupper": w.isupper(), "w.shape": word_shape(w),
        "w.isnum": _is_number(w), "w.hot_range": _is_number(w) and 25 <= float(w) <= 52,
        "w.small_num": _is_number(w) and float(w) < 15, "w.month": w[:3].lower() in
        {m[:3].lower() for m in MONTHS["en"]}, "w.gazetteer": use_gazetteer and w.lower() in GAZETTEER,
        "w.degree": w.lower() in {"°c", "°", "degrees", "deg", "celsius", "c"},
    }
    for offset in (-2, -1, 1, 2):
        j = i + offset
        if 0 <= j < len(tokens):
            n = tokens[j]
            feats.update({f"{offset}:w.lower": n.lower(), f"{offset}:w.istitle": n.istitle(),
                          f"{offset}:w.isnum": _is_number(n)})
        else:
            feats[f"{offset}:BOUNDARY"] = True
    return feats


def sent_features(tokens, use_gazetteer=True):
    return [token_features(tokens, i, use_gazetteer) for i in range(len(tokens))]


def bio_spans(tags):
    """[(type, start, end)] from a BIO tag list (an I- without B- starts a span)."""
    spans, start, kind = [], None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        continues = tag.startswith("I-") and tag[2:] == kind
        if not continues:
            if kind is not None:
                spans.append((kind, start, i))
                kind = None
            if tag != "O":
                kind, start = tag[2:], i
    return spans


def entity_report(gold_seqs, pred_seqs):
    stats = defaultdict(lambda: [0, 0, 0])  # tp, n_pred, n_gold
    for gold, pred in zip(gold_seqs, pred_seqs):
        g, p = set(bio_spans(gold)), set(bio_spans(pred))
        for kind, *_ in g:
            stats[kind][2] += 1
        for span in p:
            stats[span[0]][1] += 1
            stats[span[0]][0] += span in g
    print(f"{'entity':8s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'n':>5s}")
    totals = [0, 0, 0]
    for kind in sorted(stats):
        tp, n_pred, n_gold = stats[kind]
        totals = [a + b for a, b in zip(totals, stats[kind])]
        p, r = tp / max(n_pred, 1), tp / max(n_gold, 1)
        print(f"{kind:8s} {p:6.3f} {r:6.3f} {2 * p * r / max(p + r, 1e-9):6.3f} {n_gold:5d}")
    p, r = totals[0] / max(totals[1], 1), totals[0] / max(totals[2], 1)
    print(f"{'MICRO':8s} {p:6.3f} {r:6.3f} {2 * p * r / max(p + r, 1e-9):6.3f} {totals[2]:5d}")


NER_TRAIN, NER_TEST = build_ner_data()
print("Example:", list(zip(*NER_TRAIN[0])))

crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.05, max_iterations=150,
                           all_possible_transitions=True)
t0 = time.time()
# Gazetteer dropout: hide the gazetteer feature in half the training sentences, otherwise
# the CRF memorises the list and cannot recognise places it has never seen.
gaz_rng = random.Random(SEED)
crf.fit([sent_features(t, use_gazetteer=gaz_rng.random() < 0.5) for t, _ in NER_TRAIN],
        [tags for _, tags in NER_TRAIN])
print(f"CRF trained in {time.time() - t0:.1f}s")

pred = crf.predict([sent_features(t) for t, _ in NER_TEST])
print("\nEntity-level scores on UNSEEN places (+ unseen phrasings):")
entity_report([tags for _, tags in NER_TEST], pred)

print("\nStrongest learned features (what the CRF relies on):")
for (attr, label), weight in Counter(crf.state_features_).most_common(8):
    print(f"  {weight:6.2f}  {label:7s} {attr}")
print("Strongest transitions:")
for (a, b), weight in Counter(crf.transition_features_).most_common(5):
    print(f"  {weight:6.2f}  {a} → {b}")


def extract_entities(text):
    """Text → list of sentences, each with its tokens and (type, text, start, end) entities."""
    results = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        tokens = tokenize(sentence)
        if not tokens:
            continue
        tags = crf.predict_single(sent_features(tokens))
        entities = [(k, " ".join(tokens[s:e]), s, e) for k, s, e in bio_spans(tags)]
        results.append({"tokens": tokens, "entities": entities})
    return results


for sent in extract_entities("Severe heatwave conditions prevailed over Vidarbha. "
                             "Chandrapur recorded 47.2 °C on 18th May, which is above normal by 5.1 °C."):
    print([(k, s) for k, s, _, _ in sent["entities"]])

# %%
# ============================================================================
# CELL 9 · Lexical knowledge — WordNet, WSD, multilingual dictionary  (Syllabus 4)
# ============================================================================
import nltk

NLTK_DIR = os.path.abspath("nltk_data")
nltk.data.path.insert(0, NLTK_DIR)
try:
    for resource in ("wordnet", "omw-1.4"):
        nltk.download(resource, download_dir=NLTK_DIR, quiet=True)
    from nltk.corpus import wordnet as wn
    from nltk.wsd import lesk
    wn.synsets("heat")
    WORDNET_OK = True
except Exception as exc:
    print(f"WordNet unavailable ({exc}); continuing with seed lexicons only.")
    WORDNET_OK = False

HEAT_SEEDS = {"heat", "hot", "heatwave", "scorching", "burning", "sweltering", "sun", "sweat", "sweating",
              "temperature", "degrees", "thirsty", "oven", "unbearable"}
EMERGENCY_SEEDS = {"heatstroke", "sunstroke", "collapsed", "fainted", "fainting", "unconscious", "hospital",
                   "hospitalised", "ambulance", "dehydration", "died", "deaths", "dying", "emergency", "urgent"}
HEAT_LEXICON, EMERGENCY_LEXICON = set(HEAT_SEEDS), set(EMERGENCY_SEEDS)

if WORDNET_OK:
    WEATHER_CUES = ("temperature", "heat", "warm", "weather", "sun", "hot")
    naive, filtered = set(), set()
    for seed in ["heat", "hot", "heat_wave", "sweltering", "scorching"]:
        for syn in wn.synsets(seed):
            lemmas = {l.replace("_", " ") for l in syn.lemma_names() if "_" not in l}
            naive |= lemmas
            if any(cue in syn.definition() for cue in WEATHER_CUES):
                filtered |= lemmas
    print(f"Naive WordNet expansion adds {len(naive - HEAT_SEEDS)} words, e.g. {sorted(naive - filtered)[:8]}")
    print("  ↑ unrelated senses of heat/hot creep in → keep only synsets whose gloss is about weather")
    print(f"Sense-filtered expansion adds: {sorted(filtered - HEAT_SEEDS)}")
    HEAT_LEXICON |= filtered
    for seed in ["heatstroke", "dehydration", "faint"]:
        for syn in wn.synsets(seed):
            EMERGENCY_LEXICON |= {l for l in syn.lemma_names() if "_" not in l}

    print("\nHypernym path of heat wave:",
          " → ".join(s.name() for s in wn.synset("heat_wave.n.01").hypernym_paths()[0][-4:]))
    print("\nLesk word-sense disambiguation for 'heat':")
    for sentence in ["miami heat won the basketball game last night",
                     "the heat wave raised the temperature of the city to 47 degrees"]:
        sense = lesk(sentence.split(), "heat", pos="n")
        print(f"  '{sentence}'\n     → {sense.name() if sense else None}: {sense.definition() if sense else ''}")
    print("  (Lesk only counts word overlap with dictionary glosses, so it is often wrong —"
          " a good motivation for the learned classifier in Cell 10.)")

# A small English → Hindi → Marathi dictionary used to explain advisory vocabulary.
# For a full lexicon, use IndoWordNet (pip install pyiwn).
MULTI_LEXICON = {
    "heatwave": {"hi": "लू", "mr": "उष्णतेची लाट"},
    "severe": {"hi": "भीषण", "mr": "तीव्र"},
    "heatstroke": {"hi": "लू लगना", "mr": "उष्माघात"},
    "warning": {"hi": "चेतावनी", "mr": "इशारा"},
    "temperature": {"hi": "तापमान", "mr": "तापमान"},
    "water": {"hi": "पानी", "mr": "पाणी"},
    "shade": {"hi": "छाया", "mr": "सावली"},
    "livestock": {"hi": "पशु", "mr": "जनावरे"},
    "irrigation": {"hi": "सिंचाई", "mr": "पाणी देणे"},
}
print("\nMultilingual dictionary:")
for en, tr in MULTI_LEXICON.items():
    print(f"  {en:12s} hi: {tr['hi']:10s} mr: {tr['mr']}")

# %%
# ============================================================================
# CELL 10 · BiLSTM classifier for citizen posts  (Syllabus 5.1, 5.2)
# ============================================================================
from sklearn.metrics import classification_report, confusion_matrix

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


class Vocab:
    SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def __init__(self, token_lists, min_freq=1):
        counts = Counter(t for toks in token_lists for t in toks)
        self.itos = self.SPECIALS + sorted(t for t, c in counts.items() if c >= min_freq and t not in self.SPECIALS)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, tokens):
        return [self.stoi.get(t, UNK_ID) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

    def __len__(self):
        return len(self.itos)


def pad_batch(seqs):
    width = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD_ID] * (width - len(s)) for s in seqs], dtype=torch.long, device=DEVICE)


class PostClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden=64, n_classes=3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        out, _ = self.lstm(self.emb(x))                        # (B, T, 2H): one hidden state per word
        out = out.masked_fill((x == PAD_ID).unsqueeze(-1), -1e9)
        return self.fc(self.drop(out.max(dim=1).values))       # max-pool over time


post_train_tok = [normalize_post(t) for t, _ in POSTS_TRAIN]
post_test_tok = [normalize_post(t) for t, _ in POSTS_TEST]
post_vocab = Vocab(post_train_tok)
y_train = [l for _, l in POSTS_TRAIN]
y_test = [l for _, l in POSTS_TEST]

clf = PostClassifier(len(post_vocab)).to(DEVICE)
optimizer = torch.optim.Adam(clf.parameters(), lr=2e-3)
loss_fn = nn.CrossEntropyLoss()
rng = random.Random(SEED)
WORD_DROPOUT = 0.15  # randomly hide words as <unk> so the model copes with unseen vocabulary
for epoch in range(1, (3 if FAST_MODE else 8) + 1):
    clf.train()
    order = list(range(len(post_train_tok)))
    rng.shuffle(order)
    total = 0.0
    for b in range(0, len(order), 32):
        idx = order[b:b + 32]
        seqs = [[UNK_ID if rng.random() < WORD_DROPOUT else i for i in post_vocab.encode(post_train_tok[k])]
                for k in idx]
        logits = clf(pad_batch(seqs))
        loss = loss_fn(logits, torch.tensor([y_train[k] for k in idx], device=DEVICE))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(clf.parameters(), 1.0)  # guards against exploding gradients
        optimizer.step()
        total += loss.item() * len(idx)
    print(f"epoch {epoch}: train loss {total / len(order):.4f}")


@torch.no_grad()
def classify_posts(texts):
    clf.eval()
    seqs = [post_vocab.encode(normalize_post(t)) or [UNK_ID] for t in texts]
    probs = torch.softmax(clf(pad_batch(seqs)), dim=-1).cpu().numpy()
    return probs.argmax(1), probs


def lexicon_baseline(tokens):
    if any(t in EMERGENCY_LEXICON for t in tokens):
        return 2
    return 1 if any(t in HEAT_LEXICON for t in tokens) else 0


# The BiLSTM only knows English/Hinglish. Devanagari posts go through a keyword
# lexicon instead, matched on morphological stems from Cell 7 (उष्णतेची → उष्णता).
DEV_HEAT_WORDS = {"गर्मी", "उष्णता", "ऊन", "लू", "गरम", "तापमान", "उकाडा"}
DEV_EMERGENCY_WORDS = {"उष्माघात", "बेहोश", "अस्पताल", "रुग्णालय", "मौत", "मृत्यू", "एम्बुलेंस", "रुग्णवाहिका"}


def classify_any_script(texts):
    labels, _ = classify_posts(texts)
    labels = list(labels)
    for i, text in enumerate(texts):
        dev_words = re.findall(r"[\u0900-\u097F]+", text)
        if len(dev_words) > len(re.findall(r"[A-Za-z]+", text)):
            forms = {w for word in dev_words for w in (word, morph_analyze(word)[0][0])}
            labels[i] = 2 if forms & DEV_EMERGENCY_WORDS else 1 if forms & DEV_HEAT_WORDS else 0
    return labels


lstm_pred, _ = classify_posts([t for t, _ in POSTS_TEST])
lex_pred = [lexicon_baseline(t) for t in post_test_tok]
print("\nLexicon baseline (WordNet-expanded keywords) on unseen phrasings:")
print(classification_report(y_test, lex_pred, target_names=POST_LABELS, digits=3, zero_division=0))
print("BiLSTM on unseen phrasings:")
print(classification_report(y_test, lstm_pred, target_names=POST_LABELS, digits=3, zero_division=0))
print("BiLSTM confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(y_test, lstm_pred))


@torch.no_grad()
def hidden_state_probe(text):
    """'Meta meaning' of the hidden state: how the belief changes word by word."""
    tokens = normalize_post(text)
    print(f"\n{text}")
    for k in range(1, len(tokens) + 1):
        _, probs = classify_posts([" ".join(tokens[:k])])
        bar = "█" * int(probs[0][2] * 20)
        print(f"  {' '.join(tokens[:k])[-45:]:45s} P(emergency)={probs[0][2]:.2f} {bar}")


hidden_state_probe("my uncle fainted in the heat near pune bus stand")

# %%
# ============================================================================
# CELL 11 · Seq2Seq with attention: forecast → advisory  (Syllabus 5.3)
# ============================================================================
# Encoder input  : SEV_HEATWAVE AUD_FARMER LANG_hi TER_plains TMAX_46 DEP_5.0
# Decoder output : delexicalised advisory, e.g. "नारंगी अलर्ट : <date> को <place> में लू ..."
# Encoder = BiLSTM; decoder = LSTM with Luong (general) attention; trained with
# teacher forcing. Decoding: greedy (argmax per step) and beam search.


def src_tokens(record, audience, lang):
    tokens = [f"SEV_{record['severity']}"] if SEQ2SEQ_INCLUDE_SEVERITY else []
    dep_bin = max(-5.0, min(10.0, math.floor(record["departure"] * 2) / 2))
    return tokens + [f"AUD_{audience}", f"LANG_{lang}", f"TER_{record['terrain']}",
                     f"TMAX_{int(math.floor(record['tmax']))}", f"DEP_{dep_bin:.1f}"]


def make_pairs(records, seed):
    rng = random.Random(seed)
    return [(src_tokens(r, a, l), template_advisory(r["severity"], a, l, rng), r, a, l)
            for r in records for a in AUDIENCES for l in LANGS]


shuffled = RECORDS[:]
random.Random(SEED).shuffle(shuffled)
n_val = n_test = len(shuffled) // 10
S2S_TEST_RECORDS, S2S_VAL_RECORDS = shuffled[:n_test], shuffled[n_test:n_test + n_val]
S2S_TRAIN_RECORDS = shuffled[n_test + n_val:]
train_pairs = make_pairs(S2S_TRAIN_RECORDS, 1)
val_pairs = make_pairs(S2S_VAL_RECORDS, 2)
test_pairs = make_pairs(S2S_TEST_RECORDS, 3)

all_src = [p[0] for p in train_pairs]
all_tgt = [p[1] for p in train_pairs] + [ref for s, a, l in product(SEVERITY_LEVELS, AUDIENCES, LANGS)
                                         for ref in all_references(s, a, l)]
src_vocab, tgt_vocab = Vocab(all_src + [p[0] for p in val_pairs + test_pairs]), Vocab(all_tgt)
print(f"{len(train_pairs)} train pairs | src vocab {len(src_vocab)} | tgt vocab {len(tgt_vocab)}")


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.bridge_h = nn.Linear(2 * hidden, hidden)
        self.bridge_c = nn.Linear(2 * hidden, hidden)

    def forward(self, src):
        out, (h, c) = self.lstm(self.emb(src))                      # out: (B, S, 2H)
        h = torch.tanh(self.bridge_h(torch.cat([h[0], h[1]], -1))).unsqueeze(0)
        c = torch.tanh(self.bridge_c(torch.cat([c[0], c[1]], -1))).unsqueeze(0)
        return out, (h, c)


class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden, enc_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.key = nn.Linear(enc_dim, hidden, bias=False)          # Luong "general" score
        self.combine = nn.Linear(hidden + enc_dim, hidden)
        self.out = nn.Linear(hidden, vocab_size)
        self.drop = nn.Dropout(0.2)

    def forward(self, tgt_in, state, enc_out, src_mask):
        dec_out, state = self.lstm(self.drop(self.emb(tgt_in)), state)        # (B, T, H)
        scores = torch.bmm(dec_out, self.key(enc_out).transpose(1, 2))       # (B, T, S)
        attn = torch.softmax(scores.masked_fill(~src_mask.unsqueeze(1), -1e9), dim=-1)
        context = torch.bmm(attn, enc_out)                                   # (B, T, 2H)
        hidden = torch.tanh(self.combine(torch.cat([dec_out, context], -1)))
        return self.out(self.drop(hidden)), state, attn


class Seq2Seq(nn.Module):
    def __init__(self, src_size, tgt_size, emb_dim=128, hidden=256):
        super().__init__()
        self.encoder = Encoder(src_size, emb_dim, hidden)
        self.decoder = AttnDecoder(tgt_size, emb_dim, hidden, 2 * hidden)

    def forward(self, src, tgt_in):
        enc_out, state = self.encoder(src)
        logits, _, _ = self.decoder(tgt_in, state, enc_out, src != PAD_ID)
        return logits


def tensorize(pairs):
    src = [src_vocab.encode(p[0]) for p in pairs]
    tgt = [[BOS_ID] + tgt_vocab.encode(p[1]) + [EOS_ID] for p in pairs]
    return src, tgt


def run_epoch(model, src, tgt, optimizer=None, batch_size=128):
    training = optimizer is not None
    model.train(training)
    order = list(range(len(src)))
    if training:
        random.shuffle(order)
    total_loss, total_tokens = 0.0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    with torch.set_grad_enabled(training):
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            s, t = pad_batch([src[i] for i in idx]), pad_batch([tgt[i] for i in idx])
            logits = model(s, t[:, :-1])                       # teacher forcing: feed the gold prefix
            loss = criterion(logits.reshape(-1, logits.size(-1)), t[:, 1:].reshape(-1))
            if training:
                optimizer.zero_grad()
                (loss / len(idx)).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            total_tokens += (t[:, 1:] != PAD_ID).sum().item()
    return total_loss / total_tokens


s2s = Seq2Seq(len(src_vocab), len(tgt_vocab)).to(DEVICE)
s2s_opt = torch.optim.Adam(s2s.parameters(), lr=1e-3)
train_src, train_tgt = tensorize(train_pairs)
val_src, val_tgt = tensorize(val_pairs)
for epoch in range(1, (3 if FAST_MODE else 10) + 1):
    t0 = time.time()
    train_loss = run_epoch(s2s, train_src, train_tgt, s2s_opt)
    val_loss = run_epoch(s2s, val_src, val_tgt)
    print(f"epoch {epoch:2d}: train loss {train_loss:.3f} | val loss {val_loss:.3f} "
          f"(perplexity {math.exp(val_loss):.2f}) | {time.time() - t0:.1f}s")


@torch.no_grad()
def greedy_decode(model, src_ids, max_len=90, return_attention=False):
    model.eval()
    src = torch.tensor([src_ids], device=DEVICE)
    enc_out, state = model.encoder(src)
    token, output, attentions = torch.tensor([[BOS_ID]], device=DEVICE), [], []
    for _ in range(max_len):
        logits, state, attn = model.decoder(token, state, enc_out, src != PAD_ID)
        token = logits[:, -1].argmax(-1, keepdim=True)          # argmax at every step
        if token.item() == EOS_ID:
            break
        output.append(token.item())
        attentions.append(attn[0, -1].cpu().numpy())
    tokens = tgt_vocab.decode(output)
    return (tokens, attentions) if return_attention else tokens


@torch.no_grad()
def beam_search(model, src_ids, beam_size=3, max_len=90):
    model.eval()
    src = torch.tensor([src_ids], device=DEVICE)
    enc_out, state = model.encoder(src)
    mask = src != PAD_ID
    beams, finished = [(0.0, [BOS_ID], state)], []
    for _ in range(max_len):
        candidates = []
        for score, seq, st in beams:
            logits, new_state, _ = model.decoder(torch.tensor([[seq[-1]]], device=DEVICE), st, enc_out, mask)
            log_probs = torch.log_softmax(logits[0, -1], -1)
            top = torch.topk(log_probs, beam_size)
            for lp, idx in zip(top.values.tolist(), top.indices.tolist()):
                candidates.append((score + lp, seq + [idx], new_state))
        beams = []
        for cand in sorted(candidates, key=lambda c: c[0] / len(c[1]), reverse=True):
            (finished if cand[1][-1] == EOS_ID else beams).append(cand)
            if len(beams) == beam_size:
                break
        if not beams or len(finished) >= beam_size:
            break
    best = max(finished or beams, key=lambda c: c[0] / len(c[1]))[1]
    return tgt_vocab.decode([i for i in best[1:] if i != EOS_ID])


# ---- Evaluation: multi-reference BLEU + exact match ---------------------------
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

unique_tests = {tuple(p[0]): p for p in test_pairs}.values()
hyps, refs, exact = [], [], 0
for src_tok, _, record, aud, lang in unique_tests:
    hyp = greedy_decode(s2s, src_vocab.encode(src_tok))
    ref_set = all_references(record["severity"], aud, lang)
    hyps.append(hyp)
    refs.append(ref_set)
    exact += hyp in ref_set
bleu = corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)
print(f"\nTest BLEU (multi-reference): {bleu * 100:.1f} | exact match with a valid advisory: "
      f"{exact}/{len(hyps)} ({exact / len(hyps):.1%})")

sample = S2S_TEST_RECORDS[0]
print("\nSample record:", {k: sample[k] for k in ("place", "date", "tmax", "departure", "severity")})
for lang in LANGS:
    tokens = beam_search(s2s, src_vocab.encode(src_tokens(sample, "CITIZEN", lang)))
    print(f"[{lang}] {relexicalize(tokens, sample, lang)}")

tokens, attn = greedy_decode(s2s, src_vocab.encode(src_tokens(sample, "HEALTH", "en")), return_attention=True)
src_tok = src_tokens(sample, "HEALTH", "en")
print("\nAttention — which input token each output word looked at most:")
for tok, weights in list(zip(tokens, attn))[:12]:
    print(f"  {tok:12s} ← {src_tok[int(np.argmax(weights))]:14s} ({weights.max():.2f})")

# %%
# ============================================================================
# CELL 12 · Responsible AI — factual checks + human-in-the-loop review
# ============================================================================
# Every generated advisory is checked against the structured record BEFORE release:
#   * the place, Tmax (and departure) of the record appear in the text
#   * the severity wording matches the IMD rule engine (e.g. no "orange" for a red day)
#   * correct script and language, no leftover slots or <unk>
#   * every sentence comes from the approved advisory library, nothing repeated
# Failed outputs are replaced by the deterministic rule-based template (safe fallback).
# HEATWAVE/SEVERE warnings always need a meteorologist's approval (use case §13).

SEVERITY_MARKERS = {  # checked in priority order: most severe first
    "en": {"SEVERE": ["red alert", "severe heatwave"], "HEATWAVE": ["orange alert", "heatwave warning"],
           "WATCH": ["yellow alert", "heat watch"], "NORMAL": ["no heatwave", "near normal"]},
    "hi": {"SEVERE": ["लाल अलर्ट", "भीषण लू"], "HEATWAVE": ["नारंगी अलर्ट", "लू की चेतावनी"],
           "WATCH": ["पीला अलर्ट"], "NORMAL": ["लू की कोई संभावना नहीं", "मौसम सामान्य"]},
    "mr": {"SEVERE": ["लाल इशारा", "तीव्र उष्णतेच"], "HEATWAVE": ["नारिंगी इशारा", "उष्णतेच्या लाटेचा इशारा"],
           "WATCH": ["पिवळा इशारा"], "NORMAL": ["शक्यता नाही", "हवामान सामान्य"]},
}
LANG_HINTS = {"hi": {"को", "में"}, "mr": {"रोजी", "येथे"}}


def detect_severity(text, lang):
    lowered = text.lower()
    for severity in ("SEVERE", "HEATWAVE", "WATCH", "NORMAL"):
        if any(marker in lowered for marker in SEVERITY_MARKERS[lang][severity]):
            return severity
    return None


SENTENCE_END = {".", "।"}


def split_sentences(tokens):
    sentences, current = [], []
    for tok in tokens:
        current.append(tok)
        if tok in SENTENCE_END:
            sentences.append(tuple(current))
            current = []
    return sentences + ([tuple(current)] if current else [])


def approved_content_checks(tokens, record, audience, lang):
    """Sentence-level grounding: every generated sentence must come from the approved
    advisory library for this severity/audience/language (catches loops, truncation and
    advice meant for another audience — all classic Seq2Seq failures)."""
    sev = record["severity"]
    headline_sents = {s for h in HEADLINES[sev][lang] for s in split_sentences(h.split())}
    action_sents = {tuple(a.split()) for a in ACTIONS[audience][action_band(sev)][lang]}
    extra_sents = {tuple(EXTRA_LINE[sev][lang].split())} if sev in EXTRA_LINE else set()
    sentences = split_sentences(tokens)
    return {
        "only_approved_sentences": all(s in headline_sents | action_sents | extra_sents for s in sentences),
        "has_headline_and_action": bool(set(sentences) & headline_sents) and bool(set(sentences) & action_sents),
        "no_repetition": len(sentences) == len(set(sentences)),
    }


def verify_advisory(text, record, lang, tokens=None, audience=None):
    place = record["place"] if lang == "en" else PLACES[record["place"]][lang]
    letters = re.findall(r"[A-Za-z\u0900-\u097F]", text)
    devanagari_share = sum("\u0900" <= ch <= "\u097F" for ch in letters) / max(len(letters), 1)
    words = set(text.split())
    checks = {
        "place_mentioned": place in text,
        "tmax_correct": f"{record['tmax']:.1f}" in text,
        "departure_correct": record["severity"] == "NORMAL" or f"{record['departure']:.1f}" in text,
        "severity_matches_rules": detect_severity(text, lang) == record["severity"],
        "no_unfilled_slots": not re.search(r"<\w+>", text),
        "correct_script": devanagari_share < 0.2 if lang == "en" else devanagari_share > 0.6,
        "correct_language": lang == "en" or bool(words & LANG_HINTS[lang]),
    }
    if tokens is not None:
        checks.update(approved_content_checks(tokens, record, audience, lang))
    checks["passed"] = all(checks.values())
    return checks


def generate_advisory(record, audience, lang, method="seq2seq"):
    if method == "seq2seq":
        tokens = beam_search(s2s, src_vocab.encode(src_tokens(record, audience, lang)))
    else:
        tokens = template_advisory(record["severity"], audience, lang)
    text = relexicalize(tokens, record, lang)
    return {"place": record["place"], "date": record["date"], "severity": record["severity"],
            "colour": record["colour"], "audience": audience, "lang": lang, "method": method,
            "text": text, "checks": verify_advisory(text, record, lang, tokens, audience)}


EXPERT_DECISIONS = {}  # (place, date, severity) → True/False, so the expert is asked once per warning


def review_advisory(advisory, record, mode=None):
    mode = mode or APPROVAL_MODE
    if not advisory["checks"]["passed"]:
        failed = [k for k, ok in advisory["checks"].items() if not ok and k != "passed"]
        advisory = generate_advisory(record, advisory["audience"], advisory["lang"], method="template")
        advisory["note"] = f"seq2seq output failed {failed}; replaced by rule-based template"
    if advisory["severity"] in ("HEATWAVE", "SEVERE"):
        if mode == "interactive":
            key = (record["place"], record["date"], record["severity"])
            if key not in EXPERT_DECISIONS:
                answer = input(f"\n{advisory['text']}\nApprove this {advisory['colour']} warning for "
                               f"{record['place']} (all audiences and languages)? [y/n] ")
                EXPERT_DECISIONS[key] = answer.strip().lower() == "y"
            advisory["status"] = "APPROVED_BY_EXPERT" if EXPERT_DECISIONS[key] else "REJECTED_BY_EXPERT"
        else:
            advisory["status"] = "PENDING_EXPERT_APPROVAL"
    else:
        advisory["status"] = "AUTO_APPROVED"
    return advisory


passed = total = 0
failures = Counter()
for record in S2S_TEST_RECORDS[:40]:
    for aud, lang in product(AUDIENCES, LANGS):
        checks = generate_advisory(record, aud, lang)["checks"]
        total += 1
        passed += checks["passed"]
        failures.update(k for k, ok in checks.items() if not ok and k != "passed")
print(f"Seq2Seq advisories passing all checks: {passed}/{total} ({passed / total:.1%})")
print("Most common failed checks:", failures.most_common(4) or "none")

normal_day = next(r for r in S2S_TEST_RECORDS if r["severity"] == "NORMAL")
broken = dict(normal_day, severity="SEVERE")  # pretend the rules said SEVERE but the text says normal
bad = generate_advisory(normal_day, "CITIZEN", "en")
print("\nCheck catches a severity mismatch:", verify_advisory(bad["text"], broken, "en"))

# %%
# ============================================================================
# CELL 13 · End-to-end pipeline
# ============================================================================
# bulletin text ──NER──▶ (place, Tmax, date) ──IMD rules──▶ severity
#                 ──Seq2Seq──▶ advisories (4 audiences × 3 languages) ──checks + review──▶ release
# citizen posts ──normalise + classify + find places──▶ ground signals per place
# Ground signals are compared with the forecast, like the AWS validation in Phase IV.


def parse_date(text, default_year=2025):
    """'18th May', 'May 18', '18-05-2025' → '2025-05-18'; None if it is not a valid date."""
    try:
        m = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})", text)
        if m:  # Indian format: day first
            d, mo, y = map(int, m.groups())
            return date(y + 2000 if y < 100 else y, mo, d).isoformat()
        month_idx = {m[:3].lower(): i + 1 for i, m in enumerate(MONTHS["en"])}
        day = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", text)
        month = next((month_idx[w[:3].lower()] for w in re.findall(r"[A-Za-z]+", text)
                      if w[:3].lower() in month_idx), None)
        year = re.search(r"\b(20\d\d)\b", text)
        if day and month:
            return date(int(year.group(1)) if year else default_year, month, int(day.group(1))).isoformat()
    except ValueError:  # e.g. "31 June" or a month > 12
        pass
    return None


def records_from_bulletin(text, default_date="2025-05-18"):
    records, last_date = [], default_date
    canonical = {name.lower(): name for name in PLACES}
    for sent in extract_entities(text):
        ents = sent["entities"]
        dates = [parse_date(t) for k, t, _, _ in ents if k == "DATE"]
        last_date = next((d for d in dates if d), last_date)
        temps = [(s, float(num.group())) for k, t, s, _ in ents
                 if k == "TEMP" and (num := re.search(r"\d+(?:\.\d+)?", t))]
        for kind, loc, start, _ in ents:
            if kind != "LOC" or loc.lower() not in canonical or not temps:
                continue
            nearest_temp = min(temps, key=lambda st: abs(st[0] - start))[1]
            place = canonical[loc.lower()]
            records.append(make_record(place, date.fromisoformat(last_date), nearest_temp, PLACES[place]["normal"]))
    return records


BULLETIN = (
    "Heatwave to severe heatwave conditions prevailed over parts of Vidarbha and West Rajasthan. "
    "The highest maximum temperature of 47.6 °C was recorded at Chandrapur on 18th May. "
    "Nagpur recorded 46.1 °C, which is above normal by 3.6 °C. "
    "Jaipur reported 44.2 °C on 18 May. "
    "Mumbai : 34.0 °C ( 0.5 °C above normal )."
)
CITIZEN_POSTS = [
    "my grandmother fainted in the heat at chandrapur market, ambulance took 40 min 🚑",
    "Chandrapur is literally an oven today 47°C!!! #heatwave",
    "nagpur roads are empty, way too hot to go out 🥵",
    "heat stroke patients filling the hospital in Nagpur #nagpurheatwave",
    "lovely sea breeze in mumbai this evening",
    "miami heat fans in mumbai celebrating tonight 🏀",
    "जयपुर में बहुत गर्मी है, दोपहर में बाहर मत निकलो",
    "पुण्यात आज ऊन खूप आहे",
]

print("STEP 1 · NER on the bulletin")
for sent in extract_entities(BULLETIN):
    print("  ", [(k, t) for k, t, _, _ in sent["entities"]])

print("\nSTEP 2 · Records + IMD severity")
pipeline_records = records_from_bulletin(BULLETIN)
for r in pipeline_records:
    print(f"   {r['place']:12s} {r['date']}  Tmax {r['tmax']} (normal {r['normal']}, "
          f"dep {r['departure']:+.1f}) → {r['severity']} / {r['colour']}")

print("\nSTEP 3 · Advisories (Seq2Seq → checks → review)")
released = []
for r in pipeline_records:
    for aud, lang in product(AUDIENCES, LANGS):
        released.append(review_advisory(generate_advisory(r, aud, lang), r))
for adv in released:
    if adv["audience"] in ("CITIZEN", "FARMER"):
        print(f"   [{adv['status']:24s}] {adv['place']}/{adv['audience']}/{adv['lang']}: {adv['text']}")

print("\nSTEP 4 · Citizen signals vs forecast")
labels = classify_any_script(CITIZEN_POSTS)
signals = defaultdict(Counter)
for post, label in zip(CITIZEN_POSTS, labels):
    places = find_places(post)
    print(f"   {POST_LABELS[label]:16s} {places} ← {post}")
    for place in places:
        signals[place][POST_LABELS[label]] += 1
forecast = {r["place"]: r["severity"] for r in pipeline_records}
for place in sorted(set(signals) | set(forecast)):
    sev, sig = forecast.get(place, "no forecast"), signals.get(place, Counter())
    heat_reports = sig["HEAT_EMERGENCY"] + sig["HEAT_DISCOMFORT"]
    if sev == "no forecast":
        verdict = "not in bulletin; citizen heat reports → check AWS data" if heat_reports else "not in bulletin"
    elif sig["HEAT_EMERGENCY"] and sev in ("NORMAL", "WATCH"):
        verdict = "CHECK: citizens report emergencies the forecast does not show → verify with AWS"
    elif heat_reports:
        verdict = "corroborated by citizen reports"
    else:
        verdict = "no citizen heat reports"
    print(f"   {place:12s} forecast {sev:12s} | {dict(sig)} → {verdict}")

# %%
# ============================================================================
# CELL 14 · Save outputs and models
# ============================================================================
with open(os.path.join(OUT_DIR, "advisories.json"), "w", encoding="utf-8") as f:
    json.dump(released, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUT_DIR, "citizen_signals.json"), "w", encoding="utf-8") as f:
    json.dump({p: dict(c) for p, c in signals.items()}, f, ensure_ascii=False, indent=2)
with open(os.path.join(OUT_DIR, "ner_crf.pkl"), "wb") as f:
    pickle.dump(crf, f)
torch.save({"model": s2s.state_dict(), "src_itos": src_vocab.itos, "tgt_itos": tgt_vocab.itos},
           os.path.join(OUT_DIR, "seq2seq_advisory.pt"))
torch.save({"model": clf.state_dict(), "itos": post_vocab.itos}, os.path.join(OUT_DIR, "post_classifier.pt"))
print("Saved to", os.path.abspath(OUT_DIR), "→", sorted(os.listdir(OUT_DIR)))
# In Colab, download with:  from google.colab import files; files.download("outputs/advisories.json")
