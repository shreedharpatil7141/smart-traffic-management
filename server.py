"""
Smart Traffic Control Centre — Dashboard Server
Run:   python server.py
Open:  http://localhost:5000
"""
from flask import Flask, jsonify, abort, make_response
import os, json, time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
STATE_LIVE  = os.path.join(DATA_DIR, "state_live.json")
STATE_A     = os.path.join(DATA_DIR, "state_a.json")
STATE_B     = os.path.join(DATA_DIR, "state_b.json")
STATE_FILE  = os.path.join(DATA_DIR, "state.json")
STATE_INDEX = os.path.join(DATA_DIR, "state_index.json")

app = Flask(__name__, static_folder=None)

def _read(path):
    for _ in range(3):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            time.sleep(0.01)
    return None

def get_state():
    priority = []
    idx = _read(STATE_INDEX)
    if idx and isinstance(idx.get("path"), str):
        priority.append(idx["path"])
    priority += [STATE_LIVE, STATE_A, STATE_B, STATE_FILE]
    best_ts, best = 0.0, {}
    seen = set()
    for p in priority:
        if not p or p in seen or not os.path.exists(p):
            continue
        seen.add(p)
        d = _read(p)
        if not d:
            continue
        ts = float(d.get("City", {}).get("last_update_epoch", 0) or 0)
        if ts > best_ts:
            best_ts, best = ts, d
    return best

def get_frame(prefix):
    if not os.path.isdir(FRAMES_DIR):
        return None
    safe = "".join(c for c in prefix if c.isalnum() or c in "_-")
    try:
        files = [f for f in os.listdir(FRAMES_DIR)
                 if f.startswith(safe + "_") and f.endswith(".jpg")]
        if not files:
            return None
        files.sort(key=lambda x: os.path.getmtime(os.path.join(FRAMES_DIR, x)), reverse=True)
        path = os.path.join(FRAMES_DIR, files[0])
        s1 = os.path.getsize(path)
        time.sleep(0.008)
        if os.path.getsize(path) != s1 or s1 == 0:
            path = os.path.join(FRAMES_DIR, files[1]) if len(files) > 1 else None
        if not path:
            return None
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None

@app.after_request
def headers(r):
    r.headers["Cache-Control"] = "no-store"
    r.headers["Access-Control-Allow-Origin"] = "*"
    return r

@app.route("/api/state")
def api_state():
    return jsonify(get_state())

@app.route("/api/frame/<path:prefix>")
def api_frame(prefix):
    data = get_frame(prefix)
    if not data:
        abort(404)
    r = make_response(data)
    r.headers["Content-Type"] = "image/jpeg"
    return r

@app.route("/")
def root():
    p = os.path.join(BASE_DIR, "dashboard_live.html")
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}
    except FileNotFoundError:
        return "<h1>dashboard_live.html missing</h1>", 404

if __name__ == "__main__":
    print("=" * 55)
    print("  Smart Traffic Control Centre — Dashboard Server")
    print("  http://localhost:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
