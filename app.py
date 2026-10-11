"""Web demo: place + date → real Tmax → IMD severity → advisories, citizen posts, region map, relief points.

Usage:  python app.py      then open http://localhost:8000
Uses Cells 0–7 of heatwave_nlp.py plus the trained BiLSTM saved in outputs/post_classifier.pt
(run heatwave_nlp.py once first). Starts in seconds — no model training.

Live data: Open-Meteo (Tmax + 1991–2020 normal), OpenStreetMap via Nominatim (relief points),
OpenStreetMap tiles via Leaflet (map).
"""
import contextlib
import io
import json
import math
import os
import random
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from collections import Counter
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.chdir(os.path.dirname(os.path.abspath(__file__)))
src = open("heatwave_nlp.py", encoding="utf-8").read()


def cell(n):
    start = src.rfind("# %%", 0, src.index(f"# CELL {n} ·"))
    return src[start:src.find("# %%", start + 4)]


g = {"__name__": "heatwave_demo"}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src[:src.rfind("# %%", 0, src.index("# CELL 8 ·"))], "heatwave_nlp.py", "exec"), g)  # cells 0–7

# Citizen-post classifier: class definitions from Cell 10 + the weights saved by the full run.
CLASSIFIER_READY = os.path.exists("outputs/post_classifier.pt")
if CLASSIFIER_READY:
    c10 = cell(10)
    exec(c10[:c10.index("post_train_tok =")], g)
    ckpt = g["torch"].load("outputs/post_classifier.pt", map_location="cpu")
    vocab = g["Vocab"]([])
    vocab.itos, vocab.stoi = ckpt["itos"], {t: i for i, t in enumerate(ckpt["itos"])}
    g["post_vocab"] = vocab
    g["clf"] = g["PostClassifier"](len(vocab)).to(g["DEVICE"])
    g["clf"].load_state_dict(ckpt["model"])
    g["clf"].eval()
    exec(c10[c10.index("@torch.no_grad()\ndef classify_posts"):c10.index("lstm_pred, _ =")], g)

THRESHOLD = {"plains": 40, "coastal": 37, "hilly": 30}
AUD_LABEL = {"CITIZEN": "Citizens", "FARMER": "Farmers", "HEALTH": "Health dept.", "AUTHORITY": "Authorities"}
LANG_LABEL = {"en": "English", "hi": "हिन्दी", "mr": "मराठी"}
UA = {"User-Agent": "HeatwaveNLP-demo/1.0 (KJS-CES-01 student project)"}

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
_CLIMATE = {}  # place → (dates, tmax list) for 1991–2020


def _daily_tmax(url, p, start, end):
    q = urllib.parse.urlencode({"latitude": p["lat"], "longitude": p["lon"], "start_date": start, "end_date": end,
                                "daily": "temperature_2m_max", "timezone": "Asia/Kolkata"})
    d = json.load(urllib.request.urlopen(f"{url}?{q}", timeout=25))["daily"]
    return d["time"], d["temperature_2m_max"]


def real_tmax(place, day):
    """Observed (ERA5 reanalysis) for past dates, Open-Meteo forecast for recent / next 15 days."""
    p, d = g["PLACES"][place], date.fromisoformat(day)
    if d > date.today() + timedelta(days=15):
        raise ValueError("forecasts are only available up to 15 days ahead")
    if d <= date.today() - timedelta(days=6):
        _, t = _daily_tmax(ARCHIVE, p, day, day)
        src_name = "Open-Meteo Historical Weather API (ECMWF ERA5 reanalysis, Copernicus)"
    else:
        _, t = _daily_tmax(FORECAST, p, day, day)
        src_name = "Open-Meteo Forecast API (national weather-model forecasts)"
    if t[0] is None:
        raise ValueError("no temperature available for that date yet")
    return round(t[0], 1), src_name


def real_normal(place, day):
    """1991–2020 mean daily Tmax within ±7 days of this calendar date (same idea as IMD's 30-year normal)."""
    if place not in _CLIMATE:
        _CLIMATE[place] = _daily_tmax(ARCHIVE, g["PLACES"][place], "1991-01-01", "2020-12-31")
    target = date.fromisoformat(day).timetuple().tm_yday
    vals = []
    for t, v in zip(*_CLIMATE[place]):
        diff = abs(date.fromisoformat(t).timetuple().tm_yday - target)
        if v is not None and min(diff, 365 - diff) <= 7:
            vals.append(v)
    return round(sum(vals) / len(vals), 1)


def citizen_signals(posts, place, severity):
    """Same logic as Cell 13 step 4: classify posts, find places, compare with the forecast."""
    if not posts:
        return None
    if not CLASSIFIER_READY:
        return {"error": "Trained classifier not found — run heatwave_nlp.py once to create outputs/post_classifier.pt"}
    rows, sig = [], Counter()
    for post, label in zip(posts, g["classify_any_script"](posts)):
        places = g["find_places"](post)
        rows.append({"post": post, "label": g["POST_LABELS"][label], "places": places})
        if place in places:
            sig[g["POST_LABELS"][label]] += 1
    heat = sig["HEAT_EMERGENCY"] + sig["HEAT_DISCOMFORT"]
    if sig["HEAT_EMERGENCY"] and severity in ("NORMAL", "WATCH"):
        verdict, kind = "CHECK: citizens report heat emergencies the forecast does not show → verify with AWS / station data", "warn"
    elif heat:
        verdict, kind = "Forecast corroborated by citizen reports", "ok"
    elif severity in ("HEATWAVE", "SEVERE"):
        verdict, kind = "No citizen heat reports yet for a forecast heatwave → keep monitoring", "info"
    else:
        verdict, kind = "No citizen heat reports", "info"
    return {"rows": rows, "counts": dict(sig), "verdict": verdict, "kind": kind}


def analyse(place, tmax, day, normal=None, posts=()):
    p = g["PLACES"][place]
    sources = []
    if tmax in (None, ""):
        tmax, src_name = real_tmax(place, day)
        sources.append(f"Max temperature: {src_name}, grid point {p['lat']}, {p['lon']}")
    else:
        sources.append("Max temperature: entered by user")
    if normal in (None, ""):
        normal = real_normal(place, day)
        sources.append("Normal: 1991–2020 mean of daily Tmax within ±7 days of this date, ERA5 reanalysis via Open-Meteo")
    else:
        normal = float(normal)
        sources.append("Normal: entered by user")
    sources.append("Severity: IMD heatwave criteria (Tmax threshold by terrain; departure ≥ 4.5 / 6.5 °C; "
                   "Tmax ≥ 45 / 47 °C on plains)")
    rec = g["make_record"](place, date.fromisoformat(day), float(tmax), normal)
    sev = rec["severity"]
    rule = (f"{p['terrain'].title()} station: heatwave only considered once Tmax ≥ {THRESHOLD[p['terrain']]} °C. "
            f"Departure from normal = {rec['departure']:+.1f} °C.")
    advisories = {aud: {lang: g["relexicalize"](g["template_advisory"](sev, aud, lang, random.Random(7)), rec, lang)
                        for lang in g["LANGS"]} for aud in g["AUDIENCES"]}
    return {"record": rec, "rule": rule, "needs_review": sev in ("HEATWAVE", "SEVERE"), "advisories": advisories,
            "sources": sources, "aud_label": AUD_LABEL, "citizen": citizen_signals(list(posts), place, sev),
            "lat": p["lat"], "lon": p["lon"]}


# ---- relief / cooling points from OpenStreetMap (Nominatim; max 1 request per second) ----
RELIEF_QUERIES = [("hospital", "Hospital"), ("clinic", "Clinic"), ("drinking water", "Drinking water"),
                  ("community centre", "Community centre"), ("library", "Library")]
_RELIEF, _NOMINATIM_LOCK = {}, threading.Lock()


def _km(a_lat, a_lon, b_lat, b_lon):
    dlat, dlon = math.radians(b_lat - a_lat), math.radians(b_lon - a_lon)
    h = math.sin(dlat / 2) ** 2 + math.cos(math.radians(a_lat)) * math.cos(math.radians(b_lat)) * math.sin(dlon / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def relief_points(place, radius_deg=0.045):
    if place in _RELIEF:
        return _RELIEF[place]
    p, seen, out = g["PLACES"][place], set(), []
    box = f"{p['lon'] - radius_deg},{p['lat'] + radius_deg},{p['lon'] + radius_deg},{p['lat'] - radius_deg}"
    with _NOMINATIM_LOCK:
        for term, label in RELIEF_QUERIES:
            q = urllib.parse.urlencode({"q": term, "format": "jsonv2", "limit": 15, "bounded": 1, "viewbox": box})
            try:
                res = json.load(urllib.request.urlopen(urllib.request.Request(
                    "https://nominatim.openstreetmap.org/search?" + q, headers=UA), timeout=15))
            except Exception:
                res = []
            for r in res:
                lat, lon = float(r["lat"]), float(r["lon"])
                key = (round(lat, 4), round(lon, 4))
                if key in seen:
                    continue
                seen.add(key)
                out.append({"name": r.get("name") or f"{label} point", "type": label, "lat": lat, "lon": lon,
                            "km": round(_km(p["lat"], p["lon"], lat, lon), 1)})
            time.sleep(1.1)
    out.sort(key=lambda x: x["km"])
    per_type = Counter()  # nearest 6 of each type, so hospitals are not crowded out by water points
    _RELIEF[place] = [x for x in out if (per_type.update([x["type"]]) or per_type[x["type"]] <= 6)]
    return _RELIEF[place]


DEFAULT_PLACE = "Delhi"
DEFAULT_POSTS = """delhi is literally burning today 46 degrees 🥵 #delhiheatwave
two labourers collapsed from heat stroke near delhi station, ambulance took 30 min 🚑
garmi bahut hai yaar delhi mein, ac bhi kaam nahi kar raha
दिल्ली में बहुत गर्मी है, दोपहर में बाहर मत निकलो
hot deals at the delhi mall this weekend 🔥
lovely evening breeze at india gate today"""

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Heatwave Early Warning</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{--bg:#f7f5f2;--card:#fff;--ink:#1d1b19;--mute:#6b655e;--line:#e4dfd8;--acc:#c2410c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 -apple-system,system-ui,"Segoe UI",sans-serif}
.wrap{max-width:1150px;margin:0 auto;padding:28px 16px 60px}h1{margin:0 0 4px;font-size:28px}h2{font-size:18px;margin:0 0 12px}
.sub{color:var(--mute);margin:0 0 22px}.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:18px}
form .row{display:grid;grid-template-columns:2fr 1fr 1.3fr 1fr auto;gap:12px;align-items:end}
label{font-size:13px;color:var(--mute);display:block;margin-bottom:4px}
input,select,textarea{width:100%;padding:10px 12px;border:1px solid var(--line);border-radius:9px;font:inherit;background:#fff}
textarea{font-size:14px;min-height:130px;margin-top:4px}
button{padding:11px 22px;border:0;border-radius:9px;background:var(--acc);color:#fff;font:600 16px inherit;cursor:pointer}
button.ghost{background:#fff;color:var(--ink);border:1px solid var(--line);padding:7px 14px;font-size:14px}
details{margin-top:14px}summary{cursor:pointer;color:var(--mute);font-size:14px}
.sev{display:flex;gap:20px;align-items:center;flex-wrap:wrap}.badge{padding:14px 22px;border-radius:12px;color:#fff;font-weight:700;font-size:22px}
.Green{background:#15803d}.Yellow{background:#ca8a04}.Orange{background:#ea580c}.Red{background:#b91c1c}
.stats{display:flex;gap:26px;flex-wrap:wrap}.stat b{display:block;font-size:22px}.stat span{font-size:13px;color:var(--mute)}
.rule{color:var(--mute);margin:12px 0 0;font-size:14px}.src{margin:10px 0 0;padding-left:18px;font-size:13px;color:var(--mute)}
.review{margin-top:14px;padding:12px 14px;border-radius:10px;background:#fff7ed;border:1px solid #fed7aa;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.review.ok{background:#f0fdf4;border-color:#bbf7d0}.review.no{background:#fef2f2;border-color:#fecaca}
.review .ap{background:#15803d}.review .rj{background:#b91c1c}.review button{padding:7px 14px;font-size:14px}
table{width:100%;border-collapse:collapse}th,td{text-align:left;vertical-align:top;padding:10px;border-top:1px solid var(--line);font-size:14px}
th{font-size:13px;color:var(--mute);font-weight:600}td:first-child{font-weight:600;white-space:nowrap}
.withheld{opacity:.35}
.lab{display:inline-block;padding:2px 8px;border-radius:6px;font-size:12px;font-weight:700;color:#fff;white-space:nowrap}
.HEAT_EMERGENCY{background:#b91c1c}.HEAT_DISCOMFORT{background:#ea580c}.NOT_HEAT{background:#78716c}
.verdict{margin-top:12px;padding:10px 14px;border-radius:10px;font-weight:600}.verdict.ok{background:#f0fdf4}.verdict.warn{background:#fef2f2;color:#991b1b}.verdict.info{background:#f5f5f4}
.mapgrid{display:grid;grid-template-columns:1.6fr 1fr;gap:16px}#map{height:440px;border-radius:12px;border:1px solid var(--line)}
.list{max-height:440px;overflow:auto;font-size:14px}.list div{padding:8px 4px;border-top:1px solid var(--line)}.list small{color:var(--mute)}
.note{font-size:13px;color:var(--mute);margin-top:10px}.btns{display:flex;gap:8px;margin-bottom:10px}
@media(max-width:800px){form .row{grid-template-columns:1fr 1fr}.mapgrid{grid-template-columns:1fr}table,tbody,tr,td,th{display:block}thead{display:none}td:first-child{padding-top:16px}}
</style></head><body><div class="wrap">
<h1>Heatwave Early-Warning · Advisory Generator</h1>
<p class="sub">Pick a station and date. Leave the temperature blank to fetch real observed / forecast data (Open-Meteo), or type a value from an IMD bulletin. IMD rules decide the severity; advisories are generated for 4 audiences in 3 languages, checked against citizen reports, and Orange/Red warnings wait for a meteorologist.</p>
<div class="card"><form id="f"><div class="row">
<div><label>Place</label><select id="place">__OPTIONS__</select></div>
<div><label>Max temperature (°C)</label><input id="tmax" type="number" step="0.1" placeholder="blank = real data"></div>
<div><label>Date</label><input id="day" type="date" value="2024-05-28" required></div>
<div><label>Normal Tmax (optional)</label><input id="normal" type="number" step="0.1" placeholder="blank = real 30-yr normal"></div>
<button id="go">Analyse</button></div>
<details open><summary>Citizen posts (one per line — paste from X / WhatsApp / news comments)</summary>
<textarea id="posts">__POSTS__</textarea></details></form></div>
<div id="out"></div>
<div class="card" id="mapcard" style="display:none"><h2 id="maptitle">Region map</h2>
<div class="btns"><button type="button" class="ghost" onclick="viewRegion()">Region view</button><button type="button" class="ghost" onclick="viewCity()">Relief points near station</button></div>
<div class="mapgrid"><div id="map"></div><div><b>Cooling / relief points within ~5 km</b><div class="list" id="relief">Loading…</div></div></div>
<p class="note">Relief points are real places from OpenStreetMap (hospitals, clinics, drinking-water points, community centres, libraries) that can serve as cooling / heat-relief spots. India has no open dataset of officially designated cooling centres — city Heat Action Plans list those separately.</p></div>
</div>
<script>
const STATIONS=__STATIONS__;const COL={Green:'#15803d',Yellow:'#ca8a04',Orange:'#ea580c',Red:'#b91c1c'};
const TYPECOL={'Hospital':'#7c3aed','Clinic':'#a855f7','Drinking water':'#0284c7','Community centre':'#0d9488','Library':'#4f46e5'};
const f=document.getElementById('f'),out=document.getElementById('out');let map,layer,regionBounds,cityBounds,cur;
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function viewRegion(){if(regionBounds)map.fitBounds(regionBounds,{padding:[30,30]})}
function viewCity(){if(cityBounds)map.fitBounds(cityBounds,{padding:[20,20]})}
function review(ok){const r=document.getElementById('rv');r.className='review '+(ok?'ok':'no');
 r.innerHTML=ok?'<b>✔ APPROVED_BY_EXPERT</b> — warning released to all 4 audiences in 3 languages':'<b>✖ REJECTED_BY_EXPERT</b> — warning withheld; advisories not released';
 document.getElementById('advtable').classList.toggle('withheld',!ok)}
function drawMap(d){const x=d.record,card=document.getElementById('mapcard');card.style.display='block';
 if(!map){map=L.map('map');L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:18,attribution:'© OpenStreetMap contributors'}).addTo(map)}
 if(layer)layer.remove();layer=L.layerGroup().addTo(map);
 document.getElementById('maptitle').textContent='Region map · '+x.region+' (IMD region) · '+x.place;
 const reg=STATIONS.filter(s=>s.region===x.region);
 reg.forEach(s=>{if(s.name===x.place)return;L.circleMarker([s.lat,s.lon],{radius:6,color:'#78716c',fillOpacity:.6}).addTo(layer)
  .bindTooltip(s.name).on('click',()=>{place.value=s.name;tmax.value='';normal.value='';f.requestSubmit()})});
 L.circle([d.lat,d.lon],{radius:5000,color:COL[x.colour],weight:1,fillOpacity:.06}).addTo(layer);
 L.circleMarker([d.lat,d.lon],{radius:12,color:'#fff',weight:2,fillColor:COL[x.colour],fillOpacity:1}).addTo(layer)
  .bindPopup('<b>'+x.place+'</b><br>'+x.colour+' · '+x.severity+'<br>Tmax '+x.tmax.toFixed(1)+' °C ('+(x.departure>=0?'+':'')+x.departure.toFixed(1)+')').openPopup();
 regionBounds=L.latLngBounds(reg.map(s=>[s.lat,s.lon]).concat([[d.lat,d.lon]])).pad(.3);
 cityBounds=L.latLng(d.lat,d.lon).toBounds(11000);viewRegion();
 const list=document.getElementById('relief');list.textContent='Loading real places from OpenStreetMap…';const want=x.place;
 fetch('/api/relief?place='+encodeURIComponent(x.place)).then(r=>r.json()).then(pts=>{if(cur!==want)return;
  if(!pts.length){list.textContent='No mapped relief points found (OpenStreetMap coverage is thin here or the service is busy).';return}
  list.innerHTML=pts.map(p=>'<div><b>'+esc(p.name)+'</b><br><small><span style="color:'+(TYPECOL[p.type]||'#444')+'">●</span> '+p.type+' · '+p.km+' km · <a target="_blank" href="https://www.google.com/maps/dir/?api=1&destination='+p.lat+','+p.lon+'">directions</a></small></div>').join('');
  pts.forEach(p=>L.circleMarker([p.lat,p.lon],{radius:5,color:TYPECOL[p.type]||'#444',fillOpacity:.9}).addTo(layer).bindTooltip(p.name+' ('+p.type+', '+p.km+' km)'));
  viewCity()}).catch(()=>{list.textContent='Could not reach OpenStreetMap right now.'})}
f.onsubmit=async e=>{e.preventDefault();go.disabled=true;go.textContent='Fetching…';
 const body={place:place.value,tmax:tmax.value,day:day.value,normal:normal.value,posts:posts.value.split('\\n').map(s=>s.trim()).filter(Boolean)};
 let d;try{const r=await fetch('/api/analyse',{method:'POST',body:JSON.stringify(body)});d=await r.json()}catch(err){d={error:'server not reachable'}}
 go.disabled=false;go.textContent='Analyse';
 if(d.error){out.innerHTML='<div class="card">Error: '+esc(d.error)+'</div>';return}
 const x=d.record;cur=x.place;let rows='';
 for(const a in d.advisories){rows+='<tr><td>'+d.aud_label[a]+'</td>';for(const l in d.advisories[a])rows+='<td>'+esc(d.advisories[a][l])+'</td>';rows+='</tr>'}
 const rv=d.needs_review?'<div class="review" id="rv"><b>⏳ PENDING_EXPERT_APPROVAL</b> — '+x.colour+' warning: a meteorologist must approve before release.<button type="button" class="ap" onclick="review(true)">Approve</button><button type="button" class="rj" onclick="review(false)">Reject</button></div>'
  :'<div class="review ok" id="rv"><b>✔ AUTO_APPROVED</b> — no heatwave warning, released automatically</div>';
 let cz='';const c=d.citizen;
 if(c&&c.error)cz='<div class="card"><h2>Citizen reports</h2>'+esc(c.error)+'</div>';
 else if(c){cz='<div class="card"><h2>Citizen reports vs forecast · BiLSTM classifier + place finder</h2><table><thead><tr><th>Label</th><th>Places found</th><th>Post</th></tr></thead><tbody>'+
  c.rows.map(r=>'<tr><td><span class="lab '+r.label+'">'+r.label+'</span></td><td>'+(r.places.join(', ')||'—')+'</td><td>'+esc(r.post)+'</td></tr>').join('')+
  '</tbody></table><div class="verdict '+c.kind+'">'+x.place+': forecast '+x.severity+' | reports '+esc(JSON.stringify(c.counts))+' → '+esc(c.verdict)+'</div></div>'}
 out.innerHTML='<div class="card"><div class="sev"><div class="badge '+x.colour+'">'+x.colour+' · '+x.severity+'</div><div class="stats">'+
 '<div class="stat"><b>'+x.tmax.toFixed(1)+' °C</b><span>Max temp</span></div><div class="stat"><b>'+x.normal.toFixed(1)+' °C</b><span>Normal</span></div>'+
 '<div class="stat"><b>'+(x.departure>=0?'+':'')+x.departure.toFixed(1)+' °C</b><span>Departure</span></div>'+
 '<div class="stat"><b>'+x.place+'</b><span>'+x.state+' · '+x.terrain+'</span></div></div></div>'+
 '<p class="rule">'+d.rule+'</p>'+rv+'<ul class="src">'+d.sources.map(s=>'<li>'+esc(s)+'</li>').join('')+'</ul></div>'+
 '<div class="card"><h2>Advisories</h2><table id="advtable"><thead><tr><th>Audience</th><th>English</th><th>हिन्दी</th><th>मराठी</th></tr></thead><tbody>'+rows+'</tbody></table></div>'+cz;
 drawMap(d)};
f.requestSubmit();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/api/relief":
            place = urllib.parse.parse_qs(url.query).get("place", [""])[0]
            pts = relief_points(place) if place in g["PLACES"] else []
            return self._send(200, json.dumps(pts, ensure_ascii=False), "application/json")
        opts = "".join(f'<option{" selected" if n == DEFAULT_PLACE else ""}>{n}</option>' for n in g["PLACES"])
        stations = [{"name": n, "lat": p["lat"], "lon": p["lon"], "region": p["region"]} for n, p in g["PLACES"].items()]
        page = (PAGE.replace("__OPTIONS__", opts).replace("__POSTS__", DEFAULT_POSTS)
                .replace("__STATIONS__", json.dumps(stations)))
        self._send(200, page, "text/html")

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            res = analyse(req["place"], req["tmax"], req["day"], req.get("normal"), req.get("posts", []))
        except Exception as e:  # show the error on the page instead of crashing
            res = {"error": str(e)}
        self._send(200, json.dumps(res, ensure_ascii=False), "application/json")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=relief_points, args=(DEFAULT_PLACE,), daemon=True).start()  # warm the cache
    print("Heatwave demo running at http://localhost:8000  (Ctrl+C to stop)")
    webbrowser.open("http://localhost:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
