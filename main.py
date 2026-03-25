import cv2
import json
import time
import os
import serial
import shutil
import torch
import threading
import queue
from collections import deque
import numpy as np
from ultralytics import YOLO
import io as _io
try:
    from flask import Flask as _Flask, jsonify as _jsonify, make_response as _mr, abort as _abort
    _FLASK_OK = True
except ImportError:
    _FLASK_OK = False
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STATE_SNAP_DIR = os.path.join(DATA_DIR, "state_snapshots")
CAMERA_URLS = {
    "Temple": "http://192.168.137.21:8080/video",
    "Market": "http://192.168.137.87:8080/video",
    "Ground": "http://192.168.137.73:8080/video"
}
USE_LIVE_CAMERAS = False
STATE_FILE = os.path.join(DATA_DIR, "state.json")
STATE_A_FILE = os.path.join(DATA_DIR, "state_a.json")
STATE_B_FILE = os.path.join(DATA_DIR, "state_b.json")
STATE_INDEX_FILE = os.path.join(DATA_DIR, "state_index.json")
MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = os.getenv("VIDEO_PATH", "").strip()
VIDEO_LIST = os.getenv("VIDEO_LIST", "").strip()
VIDEO_LIST_FILE = os.getenv("VIDEO_LIST_FILE", "").strip()
SOURCE_CONFIG_FILE = os.path.join(DATA_DIR, "source_config.json")
MANUAL_OVERRIDE_FILE = os.path.join(DATA_DIR, "manual_override.json")
MANUAL_COUNTS_FILE = os.path.join(DATA_DIR, "manual_lane_counts.json")
ESP32_TARGET_FILE = os.path.join(DATA_DIR, "esp32_target.txt")
CROWD_VIDEO_DIRS = {
    "Temple": os.path.join(DATA_DIR, "videos", "crowd", "temple"),
    "Market": os.path.join(DATA_DIR, "videos", "crowd", "market"),
    "Ground": os.path.join(DATA_DIR, "videos", "crowd", "ground")
}
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]
PERSON_CLASS_ID = 0
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]
PEDESTRIAN_CLASSES = ["person"]
FRAME_WIDTH = 480
FRAME_HEIGHT = 270
CROWD_FRAME_WIDTH = 416
CROWD_FRAME_HEIGHT = 234
CROWD_DETECTION_STRIDE = 1
CROWD_CONF_THRESHOLD = 0.03
CROWD_IMG_SIZE = 416
DETECTION_STRIDE = 1
CONF_THRESHOLD = 0.1
JUNCTION_CONF_THRESHOLD = 0.08
JUNCTION_IMG_SIZE = 480
JUNCTION_MIN_GREEN_SEC = 12
JUNCTION_SWITCH_MARGIN = 2
ANNOTATE_INTERVAL = 1
PERSON_AREA = 8000
VEHICLE_AREA = 12000
HIGH_DENSITY_THRESHOLD = 0.7
MEDIUM_DENSITY_THRESHOLD = 0.4
HIGH_CONGESTION_THRESHOLD = 0.7
MEDIUM_CONGESTION_THRESHOLD = 0.4
MIN_CRITICAL_PEOPLE = 20
MIN_WARNING_PEOPLE = 10
SIGNAL_GREEN_LOW = 15
SIGNAL_GREEN_MEDIUM = 25
SIGNAL_GREEN_HIGH = 40
SIGNAL_MIN_SWITCH_SEC = 8
EMERGENCY_LOCK_SEC = 20
CITY_PLACES = ["Temple", "Market", "Ground"]
PRAYER_HOURS = [6, 12, 18, 20]
FESTIVAL_HOURS_FILE = os.path.join(DATA_DIR, "festival_hours.json")
MAX_CAPACITY = {"Temple": 120, "Market": 150, "Ground": 200}
GATE_OVERCROWD_THRESHOLD = 6
PLACE_SIGNAL_LINKS = {
    "Temple": ["Junction-1", "Junction-2"],
    "Market": ["Junction-2", "Junction-3"],
    "Ground": ["Junction-4", "Junction-5"]
}
JUNCTION_WAYS = ["way-a", "way-b", "way-c", "way-d"]
JUNCTION_WAY_TO_LANE = {
    "way-a": "lane-1",
    "way-b": "lane-2",
    "way-c": "lane-3",
    "way-d": "lane-4"
}
JUNCTION_COMMON_DIR = os.path.join(DATA_DIR, "videos", "junctions", "common")
JUNCTION_PRIMARY_DIR = os.path.join(DATA_DIR, "videos", "junctions", "junction-1")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
FRAME_INDEX_DIR = os.path.join(FRAMES_DIR, "index")
JUNCTION_REOPEN_SEC = 10
VIDEO_SKIP_FRAMES = 0
FRAME_SKIP_CROWD = 3
FRAME_SKIP_JUNCTION = 2
JUNCTION_SAMPLE_SEC = 5
LOST_TIME_SEC = 4
SAT_FLOW_VPH = 1800
PEDESTRIAN_PHASE_SEC = 15
PEDESTRIAN_CYCLES = 2

# ── In-memory shared state for built-in API server ────────────────
_state_lock = threading.Lock()
_shared_state = {"state": {}, "frames": {}}

if _FLASK_OK:
    _flask_app = _Flask(__name__, static_folder=None)

    @_flask_app.after_request
    def _nc(r):
        r.headers["Cache-Control"] = "no-store"
        r.headers["Access-Control-Allow-Origin"] = "*"
        return r

    @_flask_app.route("/api/state")
    def _route_state():
        with _state_lock:
            data = _shared_state["state"]
        import numpy as _np
        def _np_clean(obj):
            if isinstance(obj, dict):
                return {k: _np_clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_np_clean(v) for v in obj]
            if isinstance(obj, _np.generic):
                return obj.item()
            return obj
        r = _mr(json.dumps(_np_clean(data)) + "\n")
        r.headers["Content-Type"] = "application/json"
        return r

    @_flask_app.route("/api/frame/<path:prefix>")
    def _route_frame(prefix):
        safe = "".join(c for c in prefix if c.isalnum() or c in "_-")
        with _state_lock:
            fb = _shared_state["frames"].get(safe)
        if not fb:
            _abort(404)
        r = _mr(fb)
        r.headers["Content-Type"] = "image/jpeg"
        return r

    @_flask_app.route("/")
    def _route_dash():
        p = os.path.join(BASE_DIR, "dashboard_live.html")
        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}
        except Exception:
            return "<h1>dashboard_live.html not found</h1>", 404

def compute_signal(vehicles, density, risk):
    if density == "HIGH" or risk == "CRITICAL":
        return "RED", 0, "RESTRICT", "Crowd control"
    if vehicles >= 8:
        return "GREEN", SIGNAL_GREEN_HIGH, "ALLOW", "High traffic"
    if vehicles >= 4:
        return "GREEN", SIGNAL_GREEN_MEDIUM, "ALLOW", "Moderate traffic"
    return "GREEN", SIGNAL_GREEN_LOW, "ALLOW", "Light traffic"

def compute_route(risk, congestion):
    if risk == "CRITICAL":
        return "DIVERT IMMEDIATELY", "Use Road B", "DIVERT BEFORE AREA"
    if congestion == "HIGH":
        return "DIVERT SOON", "Use Road C", "DIVERT BEFORE AREA"
    if congestion == "MEDIUM":
        return "Monitor", "Use Main Road", "ALLOW"
    return "Normal", "Use Main Road", "ALLOW"

def load_festival_hours():
    if os.path.exists(FESTIVAL_HOURS_FILE):
        try:
            with open(FESTIVAL_HOURS_FILE, "r") as f:
                data = json.load(f)
                hours = data.get("festival_hours", [])
                return [int(h) for h in hours if isinstance(h, int)]
        except Exception:
            return []
    return []

def predictive_surge(people_window, critical_people):
    if len(people_window) < 5:
        return False, 0.0, None
    y = np.array(people_window, dtype=float)
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    surge = slope > 0.8
    if slope > 0 and critical_people is not None:
        eta = max(0.0, (critical_people - y[-1]) / slope)
    else:
        eta = None
    return surge, float(slope), eta

def stampede_risk_score(density_val, congestion_val, flow, zone_left, zone_right, surge_rate):
    flow_weight = 1.0 if flow in ("INCREASING", "RAPID INCREASE") else 0.2
    total = max(1, zone_left + zone_right)
    imbalance = abs(zone_left - zone_right) / total
    surge_norm = min(1.0, max(0.0, surge_rate / 3.0))
    score = (
        0.35 * density_val +
        0.25 * congestion_val +
        0.15 * imbalance +
        0.15 * flow_weight +
        0.10 * surge_norm
    ) * 100.0
    score = int(max(0, min(100, score)))
    if score <= 30:
        level = "SAFE"
    elif score <= 60:
        level = "WARNING"
    elif score <= 80:
        level = "HIGH"
    else:
        level = "CRITICAL"
    return score, level

def flow_direction(zone_left_hist, zone_right_hist):
    if len(zone_left_hist) < 3 or len(zone_right_hist) < 3:
        return "STABLE"
    left_trend = zone_left_hist[-1] - zone_left_hist[0]
    right_trend = zone_right_hist[-1] - zone_right_hist[0]
    if left_trend > 0 and right_trend > 0:
        return "CONVERGING"
    if left_trend < 0 and right_trend < 0:
        return "DISPERSING"
    if left_trend > right_trend:
        return "LEFT_TO_RIGHT"
    if right_trend > left_trend:
        return "RIGHT_TO_LEFT"
    return "STABLE"

def hotspot_zones(person_boxes, frame_w, frame_h):
    grid_counts = np.zeros((2, 3), dtype=int)
    for (x1, y1, x2, y2) in person_boxes:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        col = min(2, max(0, int(cx / (frame_w / 3))))
        row = min(1, max(0, int(cy / (frame_h / 2))))
        grid_counts[row, col] += 1
    avg = grid_counts.mean() if grid_counts.size else 0
    hotspot = []
    max_zone = "Z0"
    max_val = -1
    for r in range(2):
        for c in range(3):
            zone_id = f"Z{r * 3 + c + 1}"
            val = grid_counts[r, c]
            if val > 2 * avg and val > 0:
                hotspot.append(zone_id)
            if val > max_val:
                max_val = val
                max_zone = zone_id
    heatmap = grid_counts.flatten().astype(float)
    max_h = max(1.0, heatmap.max() if heatmap.size else 1.0)
    heatmap_norm = (heatmap / max_h).tolist()
    return hotspot, max_zone, heatmap_norm

def time_adjusted_risk(score, festival_hours):
    now = time.localtime()
    active_event = "NORMAL"
    multiplier = 1.0
    if now.tm_hour in PRAYER_HOURS:
        multiplier = 1.5
        active_event = "PRAYER"
    if now.tm_hour in festival_hours:
        multiplier = 2.0
        active_event = "FESTIVAL"
    return score * multiplier, active_event

def automated_actions(risk_score, base_green_time):
    if risk_score <= 30:
        return {"action": "NONE", "green_time": base_green_time, "alert_level": 0}
    if risk_score <= 60:
        return {"action": "ADVISORY", "green_time": base_green_time, "alert_level": 1}
    if risk_score <= 80:
        return {"action": "POLICE_ALERT", "green_time": max(5, int(base_green_time * 0.7)), "alert_level": 2}
    return {"action": "EMERGENCY", "green_time": 0, "alert_level": 3}

def refresh_hourly_history(cache, path="data/history_places.csv"):
    now_ts = time.time()
    if now_ts - cache.get("last_refresh", 0.0) < 600:
        return cache
    if not os.path.exists(path):
        cache["avg"] = {}
        cache["last_refresh"] = now_ts
        return cache
    cutoff = now_ts - (7 * 24 * 3600)
    hourly = {}
    counts = {}
    try:
        with open(path, "r") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                ts_str = parts[0]
                place = parts[1]
                people = int(float(parts[2])) if parts[2] else 0
                try:
                    ts = time.mktime(time.strptime(ts_str, "%Y-%m-%d %H:%M:%S"))
                except Exception:
                    continue
                if ts < cutoff:
                    continue
                hour = time.localtime(ts).tm_hour
                key = (place, hour)
                hourly[key] = hourly.get(key, 0) + people
                counts[key] = counts.get(key, 0) + 1
    except Exception:
        cache["avg"] = {}
        cache["last_refresh"] = now_ts
        return cache
    avg = {}
    for key, total in hourly.items():
        avg[key] = total / max(1, counts.get(key, 1))
    cache["avg"] = avg
    cache["last_refresh"] = now_ts
    return cache

def historical_anomaly(cache, place, current_people):
    now = time.localtime()
    key = (place, now.tm_hour)
    avg = cache.get("avg", {}).get(key)
    if not avg or avg <= 0:
        return False, 0.0
    ratio = current_people / avg
    return ratio > 1.5, float(ratio)

def gate_control_by_capacity(place_name, total_people):
    max_cap = MAX_CAPACITY.get(place_name, 150)
    ratio = total_people / max_cap if max_cap > 0 else 0
    if ratio >= 1.2:
        return "CLOSE"
    if ratio >= 1.0:
        return "HOLD"
    if ratio >= 0.8:
        return "SLOW"
    return "OPEN"

def webster_cycle_time(counts, lost_time=LOST_TIME_SEC):
    if not counts:
        return 30.0
    max_count = max(counts.values()) if counts else 0
    if max_count <= 0:
        return 30.0
    ratios = [min(0.9, c / max_count) for c in counts.values()]
    Y = min(0.95, sum(ratios) / max(1, len(ratios)))
    C = (1.5 * lost_time + 5) / max(0.05, (1 - Y))
    return float(max(30.0, min(120.0, C)))

def green_time_allocation(counts, cycle_time, all_red_time=4):
    total = sum(counts.values()) if counts else 0
    effective = max(10.0, cycle_time - all_red_time)
    greens = {}
    for way, count in counts.items():
        if total > 0:
            g = (count / total) * effective
        else:
            g = effective / max(1, len(counts))
        greens[way] = int(max(10, min(90, g)))
    return greens, effective

def queue_length_estimate(count):
    return float(count * 6)

class FrameSkipper:
    def __init__(self):
        self.counter = {}
    def should_run(self, key, stride):
        val = self.counter.get(key, 0) + 1
        self.counter[key] = val
        return val % stride == 0

def batch_inference(model, frames, imgsz, conf, classes, device, half=True):
    if not frames:
        return []
    results = model(
        frames,
        imgsz=imgsz,
        conf=conf,
        classes=classes,
        device=device,
        half=half,
        verbose=False
    )
    return results

class ThreadedCamera:
    def __init__(self, source, name="cam"):
        self.source = source
        self.name = name
        self.cap = cv2.VideoCapture(source) if source else None
        self.lock = threading.Lock()
        self.latest = None
        self.latest_ts = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    def _loop(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.05)
                continue
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest = frame
                    self.latest_ts = time.time()
    def read(self):
        with self.lock:
            return self.latest, self.latest_ts
    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()

class AsyncStateWriter:
    def __init__(self, path):
        self.path = path
        self.q = queue.Queue(maxsize=5)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
    def _loop(self):
        while True:
            state = self.q.get()
            try:
                temp_path = self.path + ".tmp"
                with open(temp_path, "w") as f:
                    json.dump(state, f)
                safe_replace_file(temp_path, self.path)
            except Exception:
                pass
    def write(self, state):
        try:
            if self.q.full():
                _ = self.q.get_nowait()
            self.q.put_nowait(state)
        except Exception:
            pass

def write_state_sync(state, path=STATE_FILE):
    try:
        temp_path = path + ".tmp.json"
        with open(temp_path, "w") as f:
            json.dump(state, f)
        safe_replace_file(temp_path, path)
    except Exception:
        pass

def write_state_double_buffer(state):
    try:
        active = "a"
        if os.path.exists(STATE_INDEX_FILE):
            try:
                with open(STATE_INDEX_FILE, "r") as f:
                    idx = json.load(f)
                if idx.get("active") in ("a", "b"):
                    active = idx.get("active")
            except Exception:
                pass
        next_slot = "b" if active == "a" else "a"
        target = STATE_A_FILE if next_slot == "a" else STATE_B_FILE
        temp_path = target + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(state, f)
        safe_replace_file(temp_path, target)
        idx_payload = {
            "active": next_slot,
            "path": target,
            "last_update_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_update_epoch": time.time()
        }
        idx_temp = STATE_INDEX_FILE + ".tmp"
        with open(idx_temp, "w") as f:
            json.dump(idx_payload, f)
        safe_replace_file(idx_temp, STATE_INDEX_FILE)
    except Exception:
        pass

STATE_LIVE_FILE = os.path.join(DATA_DIR, "state_live.json")

def write_state_roll(state):
    """Write state to a fixed state_live.json (atomic). Updates the index pointer."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        fname = STATE_LIVE_FILE
        temp_path = fname + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(state, f)
        safe_replace_file(temp_path, fname)
        idx_payload = {
            "active": "roll",
            "path": fname,
            "last_update_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_update_epoch": time.time()
        }
        idx_temp = STATE_INDEX_FILE + ".tmp"
        with open(idx_temp, "w") as f:
            json.dump(idx_payload, f)
        safe_replace_file(idx_temp, STATE_INDEX_FILE)
    except Exception:
        pass

def write_state_snapshot(state, keep=5):
    try:
        os.makedirs(STATE_SNAP_DIR, exist_ok=True)
        ts = int(time.time() * 1000)
        fname = os.path.join(STATE_SNAP_DIR, f"state_{ts}.json")
        temp_path = fname + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(state, f)
        safe_replace_file(temp_path, fname)
        files = [f for f in os.listdir(STATE_SNAP_DIR) if f.startswith("state_") and f.endswith(".json")]
        if len(files) > keep:
            files_sorted = sorted(files, key=lambda x: os.path.getmtime(os.path.join(STATE_SNAP_DIR, x)))
            for old in files_sorted[:-keep]:
                try:
                    os.remove(os.path.join(STATE_SNAP_DIR, old))
                except Exception:
                    pass
    except Exception:
        pass

def write_frame_latest(prefix, img, keep=5):
    try:
        os.makedirs(FRAMES_DIR, exist_ok=True)
        ts = int(time.time() * 1000)
        fname = os.path.join(FRAMES_DIR, f"{prefix}_{ts}.jpg")
        if not cv2.imwrite(fname, img):
            return
        # Store in shared memory for API server (instant, no disk read needed)
        if _FLASK_OK:
            try:
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    with _state_lock:
                        _shared_state["frames"][prefix] = bytes(buf)
            except Exception:
                pass
        # cleanup old frames
        pattern = f"{prefix}_"
        files = [f for f in os.listdir(FRAMES_DIR) if f.startswith(pattern) and f.endswith(".jpg")]
        if len(files) > keep:
            files_sorted = sorted(files, key=lambda x: os.path.getmtime(os.path.join(FRAMES_DIR, x)))
            for old in files_sorted[:-keep]:
                try:
                    os.remove(os.path.join(FRAMES_DIR, old))
                except Exception:
                    pass
    except Exception:
        pass

def compute_zone_action(zone_left, zone_right, density):
    if zone_left == 0 and zone_right == 0:
        return "NO CROWD", "NONE"
    if density == "HIGH":
        if zone_left > zone_right + 1:
            return "CONTROL LEFT", "LEFT"
        if zone_right > zone_left + 1:
            return "CONTROL RIGHT", "RIGHT"
    return "BALANCED", "BALANCED"

def compute_gate_control(place_name, gate_name, gate_count, alt_gate):
    if gate_count >= GATE_OVERCROWD_THRESHOLD:
        road_action = "ONE_WAY_OUT"
        signal_action = "RED"
        vehicle_divert = "DIVERT"
        police_action = f"DIVERT TO {alt_gate}" if alt_gate else "CLEAR EXIT"
        status = "OVERLOADED"
    else:
        road_action = "TWO_WAY"
        signal_action = "GREEN"
        vehicle_divert = "ALLOW"
        police_action = "MONITOR"
        status = "NORMAL"
    return {
        "place": place_name,
        "gate": gate_name,
        "count": gate_count,
        "status": status,
        "road_action": road_action,
        "signal_action": signal_action,
        "vehicle_divert": vehicle_divert,
        "police_action": police_action
    }

def _load_video_list():
    paths = []
    if VIDEO_LIST:
        paths.extend([p.strip() for p in VIDEO_LIST.split(";") if p.strip()])
    if VIDEO_LIST_FILE and os.path.exists(VIDEO_LIST_FILE):
        with open(VIDEO_LIST_FILE, "r") as f:
            for line in f:
                p = line.strip()
                if p:
                    paths.append(p)
    return paths

def list_videos_in_dir(path):
    if not os.path.isdir(path):
        return []
    exts = (".mp4", ".avi", ".mkv", ".mov")
    files = []
    for name in os.listdir(path):
        if name.lower().endswith(exts):
            files.append(os.path.join(path, name))
    return sorted(files)

def skip_video_frames(cap, skip):
    if skip <= 0 or cap is None:
        return
    try:
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if pos >= 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos + skip)
            return
    except Exception:
        pass
    for _ in range(skip):
        cap.read()

def seek_cap_random(cap):
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            offset = int(time.time() * 1000) % total
            cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
    except Exception:
        pass

def safe_replace_file(temp_file, final_file):
    for _ in range(3):
        try:
            os.replace(temp_file, final_file)
            return True
        except PermissionError:
            try:
                shutil.copyfile(temp_file, final_file)
                os.remove(temp_file)
                return True
            except PermissionError:
                time.sleep(0.02)
            except Exception:
                return False
        except Exception:
            return False
    return False

def init_junction_caps(junction_name):
    caps = {}
    primary_base = os.path.join(DATA_DIR, "videos", "junctions", junction_name.lower())
    if not os.path.isdir(primary_base):
        primary_base = JUNCTION_PRIMARY_DIR
    for way in JUNCTION_WAYS:
        primary = os.path.join(primary_base, way)
        primary_lane = os.path.join(primary_base, JUNCTION_WAY_TO_LANE[way])
        common = os.path.join(JUNCTION_COMMON_DIR, way)
        common_lane = os.path.join(JUNCTION_COMMON_DIR, JUNCTION_WAY_TO_LANE[way])
        vlist = (
            list_videos_in_dir(primary) or
            list_videos_in_dir(primary_lane) or
            list_videos_in_dir(common) or
            list_videos_in_dir(common_lane)
        )
        cap = cv2.VideoCapture(vlist[0]) if vlist else None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap is not None else 0
        fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else 0
        caps[way] = {
            "cap": cap,
            "list": vlist,
            "index": 0,
            "pos": 0,
            "total": total,
            "fps": fps if fps and fps > 0 else 25.0,
            "last_ts": 0.0,
            "start_ts": time.time()
        }
    return caps

def read_junction_frame(caps, way):
    entry = caps.get(way)
    if not entry:
        return False, None
    cap = entry.get("cap")
    if cap is None or not cap.isOpened():
        return False, None
    # Sample junction frames based on real video time, not loop speed.
    fps = entry.get("fps", 25.0)
    now_ts = time.time()
    sample_interval = JUNCTION_SAMPLE_SEC
    if now_ts - entry.get("last_ts", 0.0) < sample_interval:
        return False, None
    total = entry.get("total", 0)
    if total <= 0:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        entry["total"] = total
    if total > 0:
        elapsed = max(0.0, now_ts - entry.get("start_ts", now_ts))
        sample_index = int(elapsed / sample_interval)
        frame_index = int((sample_index * sample_interval * fps) % total)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    if ret:
        entry["last_ts"] = now_ts
    if not ret:
        vlist = entry.get("list", [])
        if vlist:
            entry["index"] = (entry["index"] + 1) % len(vlist)
            cap.release()
            cap = cv2.VideoCapture(vlist[entry["index"]])
            entry["cap"] = cap
            entry["total"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            entry["fps"] = cap.get(cv2.CAP_PROP_FPS) if cap is not None else entry.get("fps", 25.0)
            entry["pos"] = 0
            entry["start_ts"] = now_ts
            ret, frame = cap.read()
            if ret:
                entry["last_ts"] = now_ts
    return ret, frame

def load_source_config():
    ensure_data_dir()
    if os.path.exists(SOURCE_CONFIG_FILE):
        try:
            with open(SOURCE_CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "Temple": "live",
        "Market": "live",
        "Ground": "live"
    }

def load_manual_override():
    try:
        if os.path.exists(MANUAL_OVERRIDE_FILE):
            with open(MANUAL_OVERRIDE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {
        "enabled": False,
        "junction": "Junction-1",
        "mode": "auto",
        "force_way": "way-a"
    }

def load_manual_lane_counts():
    try:
        if os.path.exists(MANUAL_COUNTS_FILE):
            with open(MANUAL_COUNTS_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {"enabled": False, "junction": "Junction-1", "counts": {"way-a": 0, "way-b": 0, "way-c": 0, "way-d": 0}}

def init_video_capture():
    video_list = _load_video_list()
    if video_list:
        return cv2.VideoCapture(video_list[0]), True, video_list, 0
    if VIDEO_PATH:
        return cv2.VideoCapture(VIDEO_PATH), True, [], 0
    cap = cv2.VideoCapture(CAMERA_URLS["Temple"])
    return cap, False, [], 0

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)

def append_csv(path, header, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        if not exists:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(x) for x in row) + "\n")

def log_history(ts, places_state, city_signals, city_places):
    ensure_data_dir()
    for place_name, data in places_state.items():
        append_csv(
            os.path.join(DATA_DIR, "history_places.csv"),
            [
                "ts", "place", "people", "vehicles", "density", "congestion",
                "flow", "risk", "signal_state", "green_time", "zone_hotspot"
            ],
            [
                ts, place_name, data.get("people", 0), data.get("vehicles", 0),
                data.get("density", ""), data.get("congestion", ""),
                data.get("flow", ""), data.get("risk", ""),
                data.get("signal_state", ""), data.get("green_time", 0),
                data.get("zone_hotspot", "")
            ],
        )
    for signal_name, s in city_signals.items():
        append_csv(
            os.path.join(DATA_DIR, "history_signals.csv"),
            ["ts", "signal", "linked_gate", "state", "action"],
            [ts, signal_name, s.get("linked_gate", ""), s.get("state", ""), s.get("action", "")]
        )
    for place_name, p in city_places.items():
        for g in p.get("gates", []):
            append_csv(
                os.path.join(DATA_DIR, "history_gates.csv"),
                ["ts", "place", "gate", "count", "status", "road_action", "vehicle_divert", "police_action"],
                [
                    ts, place_name, g.get("gate", ""), g.get("count", 0),
                    g.get("status", ""), g.get("road_action", ""),
                    g.get("vehicle_divert", ""), g.get("police_action", "")
                ]
            )

def init_serial():
    port = os.getenv("ESP32_PORT", "").strip()
    if not port:
        return None
    try:
        ser = serial.Serial(port, 9600, timeout=0.1)
        return ser
    except Exception:
        return None

def send_esp32(ser, payload):
    if not ser:
        return
    try:
        msg = json.dumps(payload) + "\n"
        ser.write(msg.encode("utf-8"))
    except Exception:
        pass

def read_esp32_target():
    try:
        with open(ESP32_TARGET_FILE, "r") as f:
            target = f.read().strip()
            return target if target else "Temple"
    except Exception:
        return "Temple"

def find_gate_status(city_places, linked_gate):
    parts = linked_gate.split(" ", 1)
    if len(parts) != 2:
        return "NORMAL"
    place = parts[0].strip()
    gate_name = parts[1].strip()
    place_data = city_places.get(place, {})
    for g in place_data.get("gates", []):
        if g.get("gate") == gate_name:
            return g.get("status", "NORMAL")
    return "NORMAL"

def compute_esp32_output(target, full_state):
    if target in full_state:
        data = full_state.get(target, {})
        divert = ""
        if data.get("risk") == "CRITICAL" or data.get("congestion") == "HIGH":
            divert = f"DIVERT {target}"
        return {
            "signal_state": data.get("signal_state", "RED"),
            "active_lane": data.get("active_way", ""),
            "green_time_remaining": data.get("green_time", 0),
            "alert_level": data.get("alert_level", 0),
            "diversion_message": (divert or "")[:80],
            "pedestrian_phase": False
        }
    city = full_state.get("City", {})
    signals = city.get("signals", {})
    places = city.get("places", {})
    diversions = city.get("diversions", [])
    if target in signals:
        s = signals.get(target, {})
        signal_state = s.get("state", "RED")
        green_time = s.get("green_remaining_sec", 0)
        gate_status = find_gate_status(places, s.get("linked_gate", ""))
        alert_on = gate_status == "OVERLOADED"
        divert = ""
        for d in diversions:
            if d.get("signal") == target:
                divert = f"DIVERT {d.get('place', '')}".strip()
                break
        return {
            "signal_state": signal_state,
            "active_lane": s.get("active_way", ""),
            "green_time_remaining": green_time,
            "alert_level": 3 if alert_on else 0,
            "diversion_message": (divert or "")[:80],
            "pedestrian_phase": s.get("pedestrian_phase_active", False)
        }
    data = full_state.get("Temple", {})
    return {
        "signal_state": data.get("signal_state", "RED"),
        "active_lane": "",
        "green_time_remaining": data.get("green_time", 0),
        "alert_level": data.get("alert_level", 0),
        "diversion_message": "",
        "pedestrian_phase": False
    }

def default_place_state():
    return {
        "density": "LOW",
        "congestion": "LOW",
        "flow": "STABLE",
        "risk": "SAFE",
        "people": 0,
        "vehicles": 0,
        "zone_left": 0,
        "zone_right": 0,
        "zone_hotspot": "BALANCED",
        "zone_action": "BALANCED",
        "police": "None",
        "route": "Normal",
        "route_alt": "Use Main Road",
        "pre_signal": "ALLOW",
        "signal_state": "GREEN",
        "green_time": SIGNAL_GREEN_LOW,
        "traffic_action": "ALLOW",
        "signal_reason": "No feed",
        "alert": ""
    }

def main():
    print("Loading YOLO...")
    model = YOLO(MODEL_PATH)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        device = 0
        use_half = True
        print("YOLO device: CUDA")
    else:
        device = "cpu"
        use_half = False
        print("WARNING: CUDA not available. Falling back to CPU (slower).")
    print(f"State file: {STATE_FILE}")
    ensure_data_dir()
    config = load_source_config()
    festival_hours = load_festival_hours()
    manual = load_manual_override()
    manual_counts = load_manual_lane_counts()
    config_mtime = os.path.getmtime(SOURCE_CONFIG_FILE) if os.path.exists(SOURCE_CONFIG_FILE) else 0
    last_config_check = 0.0
    global_video_list = _load_video_list()
    if VIDEO_PATH and not global_video_list:
        global_video_list = [VIDEO_PATH]
    place_live_caps = {}
    place_video_caps = {}
    place_video_lists = {}
    place_video_index = {}
    place_mode = {}
    crowd_dir_mtime = {}
    for loc in CITY_PLACES:
        mode = str(config.get(loc, "live")).lower()
        place_mode[loc] = mode
        vlist = global_video_list[:] if global_video_list else list_videos_in_dir(CROWD_VIDEO_DIRS[loc])
        place_video_lists[loc] = vlist
        place_video_index[loc] = 0
        try:
            crowd_dir_mtime[loc] = os.path.getmtime(CROWD_VIDEO_DIRS[loc])
        except Exception:
            crowd_dir_mtime[loc] = 0.0
        if USE_LIVE_CAMERAS and mode in ("live", "hybrid"):
            place_live_caps[loc] = cv2.VideoCapture(CAMERA_URLS[loc])
            try:
                place_live_caps[loc].set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        else:
            place_live_caps[loc] = None
        if mode in ("video", "hybrid") and vlist:
            place_video_caps[loc] = cv2.VideoCapture(vlist[0])
        else:
            place_video_caps[loc] = None
    prev_people = {loc: 0 for loc in CITY_PLACES}
    people_hist = {loc: deque(maxlen=10) for loc in CITY_PLACES}
    left_hist = {loc: deque(maxlen=5) for loc in CITY_PLACES}
    right_hist = {loc: deque(maxlen=5) for loc in CITY_PLACES}
    hist_cache = {"last_refresh": 0.0, "avg": {}}
    frame_count = {loc: 0 for loc in CITY_PLACES}
    last_state = {}
    signal_hold_until = {loc: 0.0 for loc in CITY_PLACES}
    last_signal_change = {loc: time.time() for loc in CITY_PLACES}
    ser = init_serial()
    last_serial_push = 0.0
    junction_caps = {"Junction-1": init_junction_caps("Junction-1")}
    junction_counts = {f"Junction-{i}": {w: 0 for w in JUNCTION_WAYS} for i in range(1, 6)}
    junction_active_way = {f"Junction-{i}": "way-a" for i in range(1, 6)}
    junction_last_switch = {f"Junction-{i}": time.time() for i in range(1, 6)}
    junction_frame_count = {"Junction-1": {w: 0 for w in JUNCTION_WAYS}}
    junction_emergency = {f"Junction-{i}": {"detected": False, "lane": None, "hold_until": 0.0} for i in range(1, 6)}
    pedestrian_until = {f"Junction-{i}": 0.0 for i in range(1, 6)}
    pedestrian_cycle_count = {f"Junction-{i}": 0 for i in range(1, 6)}
    junction_warned = False
    last_junction_log = 0.0
    junction_dir_mtime = {w: 0.0 for w in JUNCTION_WAYS}
    junction_last_reopen = time.time()
    junction_video_source = {f"Junction-{i}": ("LIVE" if i == 1 else "SIMULATED") for i in range(1, 6)}
    print("System Running...")
    if _FLASK_OK:
        _ft = threading.Thread(
            target=lambda: _flask_app.run(
                host="0.0.0.0", port=5000, debug=False,
                threaded=True, use_reloader=False
            ), daemon=True
        )
        _ft.start()
        print("Dashboard API running at: http://localhost:5000")
    else:
        print("Flask not available — run: pip install flask")
    ensure_data_dir()
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(STATE_SNAP_DIR, exist_ok=True)
    # Initialize state file to avoid empty dashboard
    init_state = {
        "Temple": {"density": "LOW", "congestion": "LOW", "flow": "STABLE", "risk": "SAFE", "people": 0, "vehicles": 0,
                   "zone_left": 0, "zone_right": 0, "zone_hotspot": "BALANCED", "zone_action": "BALANCED",
                   "police": "None", "route": "Normal", "route_alt": "Use Main Road", "pre_signal": "ALLOW",
                   "signal_state": "GREEN", "green_time": SIGNAL_GREEN_LOW, "traffic_action": "ALLOW",
                   "signal_reason": "Init", "alert": ""},
        "Market": {"density": "LOW", "congestion": "LOW", "flow": "STABLE", "risk": "SAFE", "people": 0, "vehicles": 0,
                   "zone_left": 0, "zone_right": 0, "zone_hotspot": "BALANCED", "zone_action": "BALANCED",
                   "police": "None", "route": "Normal", "route_alt": "Use Main Road", "pre_signal": "ALLOW",
                   "signal_state": "GREEN", "green_time": SIGNAL_GREEN_LOW, "traffic_action": "ALLOW",
                   "signal_reason": "Init", "alert": ""},
        "Ground": {"density": "LOW", "congestion": "LOW", "flow": "STABLE", "risk": "SAFE", "people": 0, "vehicles": 0,
                   "zone_left": 0, "zone_right": 0, "zone_hotspot": "BALANCED", "zone_action": "BALANCED",
                   "police": "None", "route": "Normal", "route_alt": "Use Main Road", "pre_signal": "ALLOW",
                   "signal_state": "GREEN", "green_time": SIGNAL_GREEN_LOW, "traffic_action": "ALLOW",
                   "signal_reason": "Init", "alert": ""},
        "City": {"places": {"Temple": {"gates": []}, "Market": {"gates": []}, "Ground": {"gates": []}}, "signals": {}, "diversions": []}
    }
    write_state_sync(init_state)
    write_state_double_buffer(init_state)
    write_state_roll(init_state)
    last_full_write_ts = 0.0
    last_snapshot_ts = 0.0
    last_log_ts = 0.0
    _loop_count = 0
    STATE_WRITE_INTERVAL = 2.0   # seconds between full state flushes
    SNAPSHOT_INTERVAL = 30.0     # seconds between snapshots
    LOG_INTERVAL = 10.0          # seconds between CSV log writes
    while True:
        hist_cache = refresh_hourly_history(hist_cache)
        if time.time() - last_config_check > 2.0:
            last_config_check = time.time()
            if os.path.exists(SOURCE_CONFIG_FILE):
                new_mtime = os.path.getmtime(SOURCE_CONFIG_FILE)
                if new_mtime != config_mtime:
                    config_mtime = new_mtime
                    config = load_source_config()
                    for loc in CITY_PLACES:
                        new_mode = str(config.get(loc, "live")).lower()
                        if new_mode != place_mode.get(loc):
                            place_mode[loc] = new_mode
                            if USE_LIVE_CAMERAS and new_mode in ("live", "hybrid"):
                                if place_live_caps.get(loc) is None:
                                    place_live_caps[loc] = cv2.VideoCapture(CAMERA_URLS[loc])
                                    try:
                                        place_live_caps[loc].set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                    except Exception:
                                        pass
                            else:
                                if place_live_caps.get(loc) is not None:
                                    place_live_caps[loc].release()
                                    place_live_caps[loc] = None
                            if new_mode in ("video", "hybrid"):
                                vlist = global_video_list[:] if global_video_list else list_videos_in_dir(CROWD_VIDEO_DIRS[loc])
                                place_video_lists[loc] = vlist
                                place_video_index[loc] = 0
                                if vlist:
                                    if place_video_caps.get(loc) is not None:
                                        place_video_caps[loc].release()
                                    place_video_caps[loc] = cv2.VideoCapture(vlist[0])
                            else:
                                if place_video_caps.get(loc) is not None:
                                    place_video_caps[loc].release()
                                    place_video_caps[loc] = None
            manual = load_manual_override()
            manual_counts = load_manual_lane_counts()
            # Refresh crowd video lists if files changed
            if not global_video_list:
                for loc in CITY_PLACES:
                    try:
                        mtime = os.path.getmtime(CROWD_VIDEO_DIRS[loc])
                    except Exception:
                        mtime = 0.0
                    if mtime != crowd_dir_mtime.get(loc, 0.0):
                        crowd_dir_mtime[loc] = mtime
                        vlist = list_videos_in_dir(CROWD_VIDEO_DIRS[loc])
                        place_video_lists[loc] = vlist
                        place_video_index[loc] = 0
                        if vlist:
                            if place_video_caps.get(loc) is not None:
                                place_video_caps[loc].release()
                            place_video_caps[loc] = cv2.VideoCapture(vlist[0])
            # Refresh junction lane lists if files changed (Junction-1 lanes)
            for way in JUNCTION_WAYS:
                try:
                    primary_lane = os.path.join(JUNCTION_PRIMARY_DIR, JUNCTION_WAY_TO_LANE[way])
                    mtime = os.path.getmtime(primary_lane)
                except Exception:
                    mtime = 0.0
                if mtime != junction_dir_mtime.get(way, 0.0):
                    junction_dir_mtime[way] = mtime
                    vlist = list_videos_in_dir(primary_lane)
                    for jname, jc in junction_caps.items():
                        entry = jc.get(way)
                        if entry is not None:
                            if entry.get("cap") is not None:
                                entry["cap"].release()
                            entry["list"] = vlist
                            entry["index"] = 0
                            entry["cap"] = cv2.VideoCapture(vlist[0]) if vlist else None
            # Periodic reopen to catch replaced files with same name
            if time.time() - junction_last_reopen > JUNCTION_REOPEN_SEC:
                junction_last_reopen = time.time()
                for way in JUNCTION_WAYS:
                    primary_lane = os.path.join(JUNCTION_PRIMARY_DIR, JUNCTION_WAY_TO_LANE[way])
                    vlist = list_videos_in_dir(primary_lane)
                    for jname, jc in junction_caps.items():
                        entry = jc.get(way)
                        if entry is not None:
                            if entry.get("cap") is not None:
                                entry["cap"].release()
                            entry["list"] = vlist
                            entry["index"] = 0
                            entry["cap"] = cv2.VideoCapture(vlist[0]) if vlist else None
        for loc in CITY_PLACES:
            ret, frame = False, None
            live_cap = place_live_caps.get(loc)
            video_cap = place_video_caps.get(loc)
            mode = place_mode.get(loc, "live")
            if live_cap is not None:
                if live_cap.isOpened():
                    ret, frame = live_cap.read()
                else:
                    ret, frame = False, None
            if not ret and video_cap is not None:
                if video_cap.isOpened():
                    ret, frame = video_cap.read()
                    if ret:
                        skip_video_frames(video_cap, VIDEO_SKIP_FRAMES)
                else:
                    ret, frame = False, None
                if not ret:
                    vlist = place_video_lists.get(loc, [])
                    if vlist:
                        place_video_index[loc] = (place_video_index[loc] + 1) % len(vlist)
                        video_cap.release()
                        video_cap = cv2.VideoCapture(vlist[place_video_index[loc]])
                        seek_cap_random(video_cap)
                        place_video_caps[loc] = video_cap
                        ret, frame = video_cap.read()
            if not ret and mode == "video" and live_cap is not None:
                ret, frame = live_cap.read()
            if not ret and mode == "live":
                if live_cap is None:
                    if USE_LIVE_CAMERAS:
                        place_live_caps[loc] = cv2.VideoCapture(CAMERA_URLS[loc])
                continue
            if not ret and mode in ("video", "hybrid"):
                continue
            frame_count[loc] += 1
            frame = cv2.resize(frame, (CROWD_FRAME_WIDTH, CROWD_FRAME_HEIGHT))
            if frame_count[loc] % CROWD_DETECTION_STRIDE != 0:
                continue
            results = model(
                frame,
                imgsz=CROWD_IMG_SIZE,
                conf=CROWD_CONF_THRESHOLD,
                classes=[PERSON_CLASS_ID] + VEHICLE_CLASS_IDS,
                device=device,
                half=use_half,
                verbose=False
            )[0]
            vehicle_count = 0
            pedestrian_count = 0
            zone_left = 0
            zone_right = 0
            person_boxes = []
            for box in results.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label in VEHICLE_CLASSES:
                    vehicle_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                elif label in PEDESTRIAN_CLASSES:
                    pedestrian_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    if center_x < frame.shape[1] // 2:
                        zone_left += 1
                    else:
                        zone_right += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    person_boxes.append((x1, y1, x2, y2))
            cv2.putText(
                frame,
                f"People: {pedestrian_count} | Vehicles: {vehicle_count}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            frame_area = frame.shape[0] * frame.shape[1]
            density_val = (pedestrian_count * PERSON_AREA) / frame_area
            if density_val > HIGH_DENSITY_THRESHOLD:
                density = "HIGH"
            elif density_val > MEDIUM_DENSITY_THRESHOLD:
                density = "MEDIUM"
            else:
                density = "LOW"
            vehicle_density_val = (vehicle_count * VEHICLE_AREA) / frame_area
            congestion_val = min(1.0, density_val + (vehicle_density_val * 0.3))
            if congestion_val > HIGH_CONGESTION_THRESHOLD:
                congestion = "HIGH"
            elif congestion_val > MEDIUM_CONGESTION_THRESHOLD:
                congestion = "MEDIUM"
            else:
                congestion = "LOW"
            delta = pedestrian_count - prev_people[loc]
            if delta > 2:
                flow = "RAPID INCREASE"
            elif delta > 0:
                flow = "INCREASING"
            elif delta < 0:
                flow = "DECREASING"
            else:
                flow = "STABLE"
            prev_people[loc] = pedestrian_count
            people_hist[loc].append(pedestrian_count)
            left_hist[loc].append(zone_left)
            right_hist[loc].append(zone_right)
            critical_people = int((HIGH_DENSITY_THRESHOLD * frame_area) / max(1, PERSON_AREA))
            surge_detected, surge_rate, eta_critical = predictive_surge(people_hist[loc], critical_people)
            flow_dir = flow_direction(left_hist[loc], right_hist[loc])
            hotspot_list, max_zone, heatmap_data = hotspot_zones(person_boxes, frame.shape[1], frame.shape[0])
            base_score, base_level = stampede_risk_score(
                density_val, congestion_val, flow, zone_left, zone_right, surge_rate
            )
            adjusted_score, active_event = time_adjusted_risk(base_score, festival_hours)
            adjusted_score = min(100.0, adjusted_score)
            if adjusted_score <= 30:
                risk = "SAFE"
            elif adjusted_score <= 60:
                risk = "WARNING"
            elif adjusted_score <= 80:
                risk = "HIGH"
            else:
                risk = "CRITICAL"
            if pedestrian_count < MIN_WARNING_PEOPLE:
                risk = "SAFE"
            elif pedestrian_count < MIN_CRITICAL_PEOPLE and risk == "CRITICAL":
                risk = "WARNING"
            zone_action, zone_hotspot = compute_zone_action(zone_left, zone_right, density)
            police = "None"
            if risk == "CRITICAL":
                police = "Unit A"
            elif congestion == "MEDIUM":
                police = "Unit B"
            route, route_alt, pre_signal = compute_route(risk, congestion)
            desired_signal_state, desired_green_time, desired_traffic_action, desired_signal_reason = compute_signal(
                vehicle_count, density, risk
            )
            auto_act = automated_actions(int(adjusted_score), desired_green_time)
            anomaly_detected, anomaly_ratio = historical_anomaly(hist_cache, loc, pedestrian_count)
            alert = ""
            if risk == "CRITICAL":
                alert = "STAMPede risk"
            now = time.time()
            if risk == "CRITICAL" or auto_act["action"] == "EMERGENCY":
                signal_state = "RED"
                green_time = 0
                traffic_action = "RESTRICT"
                signal_reason = "Emergency lock"
                signal_hold_until[loc] = now + EMERGENCY_LOCK_SEC
            elif now < signal_hold_until[loc]:
                signal_state = last_state.get(loc, {}).get("signal_state", "GREEN")
                green_time = last_state.get(loc, {}).get("green_time", SIGNAL_GREEN_LOW)
                traffic_action = last_state.get(loc, {}).get("traffic_action", "ALLOW")
                signal_reason = f"Holding {signal_state}"
            elif desired_signal_state != last_state.get(loc, {}).get("signal_state", "GREEN") and (now - last_signal_change[loc]) < SIGNAL_MIN_SWITCH_SEC:
                signal_state = last_state.get(loc, {}).get("signal_state", "GREEN")
                green_time = last_state.get(loc, {}).get("green_time", SIGNAL_GREEN_LOW)
                traffic_action = last_state.get(loc, {}).get("traffic_action", "ALLOW")
                signal_reason = "Hold to avoid flicker"
            else:
                if desired_signal_state != last_state.get(loc, {}).get("signal_state", "GREEN"):
                    last_signal_change[loc] = now
                signal_state = desired_signal_state
                green_time = auto_act["green_time"]
                traffic_action = desired_traffic_action
                signal_reason = desired_signal_reason
                if signal_state == "GREEN":
                    signal_hold_until[loc] = now + SIGNAL_MIN_SWITCH_SEC
            last_state[loc] = {
                "density": density,
                "congestion": congestion,
                "flow": flow,
                "risk": risk,
                "risk_score": int(adjusted_score),
                "risk_level": risk,
                "surge_detected": surge_detected,
                "surge_rate": float(surge_rate),
                "eta_to_critical_sec": eta_critical,
                "flow_direction": flow_dir,
                "hotspot_zones": hotspot_list,
                "max_density_zone": max_zone,
                "time_adjusted_risk": float(adjusted_score),
                "active_event": active_event,
                "heatmap_data": heatmap_data,
                "anomaly_detected": anomaly_detected,
                "anomaly_ratio": anomaly_ratio,
                "people": pedestrian_count,
                "vehicles": vehicle_count,
                "zone_left": zone_left,
                "zone_right": zone_right,
                "zone_hotspot": zone_hotspot,
                "zone_action": zone_action,
                "police": police,
                "route": route,
                "route_alt": route_alt,
                "pre_signal": pre_signal,
                "signal_state": signal_state,
                "green_time": green_time,
                "traffic_action": traffic_action,
                "signal_reason": signal_reason,
                "alert": alert,
                "alert_level": auto_act["alert_level"],
                "gate_status": gate_control_by_capacity(loc, pedestrian_count)
            }
            if frame_count[loc] % ANNOTATE_INTERVAL == 0:
                write_frame_latest(f"crowd_{loc.lower()}", frame)
        for loc in CITY_PLACES:
            if loc not in last_state:
                last_state[loc] = default_place_state()
        # Junction lane processing (common videos)
        if not junction_warned:
            any_video = any(
                entry.get("cap") is not None
                for jc in junction_caps.values()
                for entry in jc.values()
            )
            if not any_video:
                print("No junction lane videos found. Place videos in data/videos/junctions/junction-1/lane-1..lane-4 or common/way-a..way-d")
                junction_warned = True
        for jname in junction_caps.keys():
            junction_emergency[jname]["detected"] = False
            junction_emergency[jname]["lane"] = None
            for way in JUNCTION_WAYS:
                ret, jframe = read_junction_frame(junction_caps[jname], way)
                if not ret:
                    continue
                junction_frame_count[jname][way] += 1
                if junction_frame_count[jname][way] % DETECTION_STRIDE != 0:
                    continue
                    continue
                junction_frame_count[jname][way] += 1
                if junction_frame_count[jname][way] % DETECTION_STRIDE != 0:
                    continue
                jframe = cv2.resize(jframe, (JUNCTION_IMG_SIZE, int(JUNCTION_IMG_SIZE * (FRAME_HEIGHT / FRAME_WIDTH))))
                jres = model(
                    jframe,
                    imgsz=JUNCTION_IMG_SIZE,
                    conf=JUNCTION_CONF_THRESHOLD,
                    classes=VEHICLE_CLASS_IDS,
                    device=device,
                    half=use_half,
                    verbose=False
                )[0]
                vcount = 0
                emergency_hit = False
                for box in jres.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    if label in VEHICLE_CLASSES:
                        vcount += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = max(1, (x2 - x1) * (y2 - y1))
                        if area > 0.4 * (jframe.shape[0] * jframe.shape[1]):
                            emergency_hit = True
                        cv2.rectangle(jframe, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            jframe,
                            label,
                            (x1, max(10, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1
                        )
                junction_counts[jname][way] = vcount
                cv2.putText(
                    jframe,
                    f"Count: {vcount}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                if emergency_hit:
                    junction_emergency[jname]["detected"] = True
                    junction_emergency[jname]["lane"] = way
                if jname == "Junction-1" and junction_frame_count[jname][way] % ANNOTATE_INTERVAL == 0:
                    write_frame_latest(f"frame_j1_{way}", jframe)
        # If other junctions have no dedicated videos, mirror Junction-1 counts for demo stability
        for jname in junction_counts.keys():
            if jname != "Junction-1" and junction_video_source.get(jname) == "SIMULATED":
                junction_counts[jname] = dict(junction_counts.get("Junction-1", {}))
        if manual_counts.get("enabled"):
            jname = manual_counts.get("junction", "Junction-1")
            if jname in junction_counts:
                for way in JUNCTION_WAYS:
                    junction_counts[jname][way] = int(manual_counts.get("counts", {}).get(way, 0))
        if time.time() - last_junction_log > 5.0:
            last_junction_log = time.time()
            j1 = junction_counts.get("Junction-1", {})
            print(f"Junction-1 lane counts: {j1}")
        temple_gate_a = compute_gate_control("Temple", "Zone A", last_state["Temple"]["zone_left"], "Zone B")
        temple_gate_b = compute_gate_control("Temple", "Zone B", last_state["Temple"]["zone_right"], "Zone A")
        market_gate_a = compute_gate_control("Market", "Zone A", last_state["Market"]["zone_left"], "Zone B")
        market_gate_b = compute_gate_control("Market", "Zone B", last_state["Market"]["zone_right"], "Zone A")
        ground_gate_a = compute_gate_control("Ground", "Zone A", last_state["Ground"]["zone_left"], "Zone B")
        ground_gate_b = compute_gate_control("Ground", "Zone B", last_state["Ground"]["zone_right"], "Zone A")
        city_signals = {
            "Junction-1": {
                "linked_gate": "Temple Zone A",
                "state": "RED" if temple_gate_a["signal_action"] == "RED" else "GREEN",
                "action": temple_gate_a["vehicle_divert"]
            },
            "Junction-2": {
                "linked_gate": "Temple Zone B",
                "state": "RED" if temple_gate_b["signal_action"] == "RED" else "GREEN",
                "action": temple_gate_b["vehicle_divert"]
            },
            "Junction-3": {
                "linked_gate": "Market Zone A",
                "state": "RED" if market_gate_a["signal_action"] == "RED" else "GREEN",
                "action": market_gate_a["vehicle_divert"]
            },
            "Junction-4": {
                "linked_gate": "Ground Zone A",
                "state": "RED" if ground_gate_a["signal_action"] == "RED" else "GREEN",
                "action": ground_gate_a["vehicle_divert"]
            },
            "Junction-5": {
                "linked_gate": "Ground Zone B",
                "state": "RED" if ground_gate_b["signal_action"] == "RED" else "GREEN",
                "action": ground_gate_b["vehicle_divert"]
            }
        }
        # Override signal state based on junction lane videos (real-world timing)
        for jname in city_signals.keys():
            counts = junction_counts.get(jname, {})
            total = sum(counts.values()) if counts else 0
            current = junction_active_way.get(jname, "way-a")
            now = time.time()
            cycle_time = webster_cycle_time(counts)
            green_plan, effective_green = green_time_allocation(counts, cycle_time)
            if junction_emergency[jname]["detected"]:
                junction_emergency[jname]["hold_until"] = max(junction_emergency[jname]["hold_until"], now + 15)
                if junction_emergency[jname]["lane"]:
                    junction_active_way[jname] = junction_emergency[jname]["lane"]
                    junction_last_switch[jname] = now
            if now < pedestrian_until[jname]:
                remaining = max(0, int(pedestrian_until[jname] - now))
                city_signals[jname]["state"] = "RED"
                city_signals[jname]["action"] = "PEDESTRIAN PHASE"
                city_signals[jname]["active_way"] = "WALK"
                city_signals[jname]["green_remaining_sec"] = remaining
                city_signals[jname]["pedestrian_phase_active"] = True
            else:
                if junction_emergency[jname]["hold_until"] > now:
                    active = junction_active_way[jname]
                else:
                    if counts:
                        max_way = max(counts, key=lambda k: counts[k])
                    else:
                        max_way = current
                    if now - junction_last_switch[jname] >= JUNCTION_MIN_GREEN_SEC:
                        if counts.get(max_way, 0) >= counts.get(current, 0) + JUNCTION_SWITCH_MARGIN:
                            junction_active_way[jname] = max_way
                            junction_last_switch[jname] = now
                            pedestrian_cycle_count[jname] += 1
                            if pedestrian_cycle_count[jname] >= PEDESTRIAN_CYCLES:
                                pedestrian_until[jname] = now + PEDESTRIAN_PHASE_SEC
                                pedestrian_cycle_count[jname] = 0
                    active = junction_active_way[jname]
                remaining = max(0, int(green_plan.get(active, JUNCTION_MIN_GREEN_SEC) - (now - junction_last_switch[jname])))
                city_signals[jname]["state"] = "GREEN" if total > 0 else "RED"
                city_signals[jname]["action"] = f"GREEN {active} / RED others"
                city_signals[jname]["active_way"] = active
                city_signals[jname]["green_remaining_sec"] = remaining
                city_signals[jname]["pedestrian_phase_active"] = False
            queue_lengths = {w: queue_length_estimate(counts.get(w, 0)) for w in JUNCTION_WAYS}
            queue_overflow = {w: queue_lengths[w] > 50 for w in JUNCTION_WAYS}
            next_way = max(counts, key=lambda k: counts[k]) if counts else current
            city_signals[jname]["vehicles"] = total
            city_signals[jname]["lanes"] = counts
            city_signals[jname]["min_green_sec"] = JUNCTION_MIN_GREEN_SEC
            city_signals[jname]["switch_margin"] = JUNCTION_SWITCH_MARGIN
            city_signals[jname]["cycle_time_sec"] = cycle_time
            city_signals[jname]["green_plan_sec"] = green_plan
            city_signals[jname]["effective_green_sec"] = effective_green
            city_signals[jname]["emergency_detected"] = junction_emergency[jname]["detected"]
            city_signals[jname]["priority_lane"] = junction_emergency[jname]["lane"]
            city_signals[jname]["queue_length_m"] = queue_lengths
            city_signals[jname]["queue_overflow"] = queue_overflow
            city_signals[jname]["next_way"] = next_way
            city_signals[jname]["active_count"] = counts.get(city_signals[jname].get("active_way", ""), 0) if counts else 0
            city_signals[jname]["next_count"] = counts.get(next_way, 0) if counts else 0
            city_signals[jname]["video_source"] = junction_video_source.get(jname, "LIVE")
        for jname in city_signals.keys():
            city_signals[jname]["green_wave_active"] = False
            city_signals[jname]["coordinated_junctions"] = []
        if city_signals.get("Junction-1", {}).get("active_way") == "way-a":
            travel_time = int((0.5 / 30.0) * 3600)
            city_signals["Junction-2"]["green_wave_active"] = True
            city_signals["Junction-2"]["coordinated_junctions"] = ["Junction-1"]
            city_signals["Junction-2"]["green_wave_delay_sec"] = travel_time
        if manual.get("enabled"):
            target = manual.get("junction", "Junction-1")
            mode = manual.get("mode", "auto")
            force_way = manual.get("force_way", "way-a")
            if target == "ALL" and mode == "all_red":
                for jname in city_signals.keys():
                    city_signals[jname]["state"] = "RED"
                    city_signals[jname]["action"] = "MANUAL ALL RED"
            elif target in city_signals:
                if mode == "all_red":
                    city_signals[target]["state"] = "RED"
                    city_signals[target]["action"] = "MANUAL ALL RED"
                elif mode == "force_way":
                    city_signals[target]["state"] = "GREEN"
                    city_signals[target]["active_way"] = force_way
                    city_signals[target]["action"] = f"MANUAL GREEN {force_way}"
                    city_signals[target]["next_way"] = force_way
                    city_signals[target]["green_remaining_sec"] = JUNCTION_MIN_GREEN_SEC
        diversions = []
        for place, linked in PLACE_SIGNAL_LINKS.items():
            pdata = last_state.get(place, {})
            if pdata.get("risk") == "CRITICAL" or pdata.get("congestion") == "HIGH":
                for s in linked:
                    diversions.append({
                        "place": place,
                        "signal": s,
                        "message": f"Divert traffic away from {place} via {s}"
                    })
        state = {
            "Temple": last_state["Temple"],
            "Market": last_state["Market"],
            "Ground": last_state["Ground"],
            "City": {
                "places": {
                    "Temple": {"gates": [temple_gate_a, temple_gate_b]},
                    "Market": {"gates": [market_gate_a, market_gate_b]},
                    "Ground": {"gates": [ground_gate_a, ground_gate_b]}
                },
                "signals": city_signals,
                "diversions": diversions,
                "manual_override": manual,
                "last_update_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "last_update_epoch": time.time(),
                "loop_count": _loop_count
            }
        }
        # Update in-memory shared state immediately (no file I/O, no race conditions)
        _loop_count += 1
        if _FLASK_OK:
            with _state_lock:
                _shared_state["state"] = state
        # Throttle disk writes: full flush every STATE_WRITE_INTERVAL seconds
        _now = time.time()
        if _now - last_full_write_ts >= STATE_WRITE_INTERVAL:
            write_state_sync(state, STATE_FILE)
            write_state_double_buffer(state)
            write_state_roll(state)
            last_full_write_ts = _now
        if _now - last_snapshot_ts >= SNAPSHOT_INTERVAL:
            write_state_snapshot(state)
            last_snapshot_ts = _now
        # Throttle CSV history logging (files already 60MB+) to avoid blocking
        if _now - last_log_ts >= LOG_INTERVAL:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            log_history(
                ts,
                {"Temple": state["Temple"], "Market": state["Market"], "Ground": state["Ground"]},
                state["City"]["signals"],
                state["City"]["places"]
            )
            last_log_ts = _now
        now = time.time()
        if now - last_serial_push > 1.0:
            target = read_esp32_target()
            payload = compute_esp32_output(target, state)
            send_esp32(ser, payload)
            last_serial_push = now
if __name__ == "__main__":
    main()
