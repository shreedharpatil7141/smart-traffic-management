import os
import json
import time
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
STATE_FILE = os.path.join(DATA_DIR, "state.json")
STATE_A_FILE = os.path.join(DATA_DIR, "state_a.json")
STATE_B_FILE = os.path.join(DATA_DIR, "state_b.json")
STATE_INDEX_FILE = os.path.join(DATA_DIR, "state_index.json")
STATE_SNAP_DIR = os.path.join(DATA_DIR, "state_snapshots")
STATE_LIVE_FILE = os.path.join(DATA_DIR, "state_live.json")
MANUAL_OVERRIDE_FILE = os.path.join(DATA_DIR, "manual_override.json")
ESP32_TARGET_FILE = os.path.join(DATA_DIR, "esp32_target.txt")
DEMO_MODE = os.environ.get("DEMO_MODE", "0").strip() == "1"


def _read_json(path, retries=3, delay=0.02):
    for _ in range(retries):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            time.sleep(delay)
    return None


def load_state():
    # Read candidates and choose the freshest by last_update_epoch
    candidates = []
    index_path = None
    if os.path.exists(STATE_INDEX_FILE):
        try:
            idx = _read_json(STATE_INDEX_FILE, retries=2)
            if isinstance(idx, dict) and isinstance(idx.get("path"), str):
                index_path = idx.get("path")
        except Exception:
            index_path = None
    for path in [index_path, STATE_LIVE_FILE, STATE_A_FILE, STATE_B_FILE, STATE_FILE]:
        if path and os.path.exists(path):
            data = _read_json(path, retries=3)
            if data:
                city = data.get("City", {})
                ts = city.get("last_update_epoch", 0) or 0
                candidates.append((float(ts), path, data))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_ts, best_path, best_data = candidates[0]
        # Reject state that is more than 15 seconds old
        age = time.time() - best_ts
        if age > 15:
            st.session_state["state_stale"] = True
        else:
            st.session_state["state_stale"] = False
        st.session_state["state_cache"] = best_data
        st.session_state["state_source"] = best_path
        st.session_state["state_age_sec"] = age
        return best_data

    # Fallback to latest snapshot
    if os.path.isdir(STATE_SNAP_DIR):
        files = [f for f in os.listdir(STATE_SNAP_DIR) if f.startswith("state_") and f.endswith(".json")]
        files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(STATE_SNAP_DIR, x)), reverse=True)
        for fname in files[:5]:
            data = _read_json(os.path.join(STATE_SNAP_DIR, fname), retries=2)
            if data:
                st.session_state["state_cache"] = data
                st.session_state["state_source"] = os.path.join(STATE_SNAP_DIR, fname)
                st.session_state["state_stale"] = True
                return data
    st.session_state["state_stale"] = True
    return st.session_state.get("state_cache", {})


def latest_frame(prefix):
    try:
        files = [f for f in os.listdir(FRAMES_DIR) if f.startswith(f"{prefix}_") and f.endswith(".jpg")]
        if not files:
            return None
        def ts_from_name(name):
            try:
                base = name.replace(f"{prefix}_", "").replace(".jpg", "")
                return int(base)
            except Exception:
                return 0
        files_sorted = sorted(files, key=lambda x: (ts_from_name(x), os.path.getmtime(os.path.join(FRAMES_DIR, x))), reverse=True)
        # Return a few newest candidates for stability checks
        return [os.path.join(FRAMES_DIR, f) for f in files_sorted[:3]]
    except Exception:
        return None


def _read_stable_image(path):
    try:
        size1 = os.path.getsize(path)
        time.sleep(0.02)
        size2 = os.path.getsize(path)
        if size1 != size2 or size2 == 0:
            return None
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


def show_frame(prefix, caption):
    candidates = latest_frame(prefix)
    if not candidates:
        st.info("Waiting for YOLO frame...")
        return
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        data = _read_stable_image(path)
        if data:
            st.image(data, use_container_width=True, caption=caption)
            return
    st.info("Waiting for YOLO frame...")


def load_manual_override():
    try:
        if os.path.exists(MANUAL_OVERRIDE_FILE):
            with open(MANUAL_OVERRIDE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {"enabled": False, "junction": "Junction-1", "mode": "auto", "force_way": "way-a"}


def save_manual_override(cfg):
    try:
        with open(MANUAL_OVERRIDE_FILE, "w") as f:
            json.dump(cfg, f)
    except Exception:
        st.warning("Unable to save manual override.")


st.set_page_config(page_title="Smart City Control Center", layout="wide")
st.title("Smart City Control Center")

if DEMO_MODE:
    st.warning("Demo mode enabled. Disable DEMO_MODE to show live counts from the engine.")
    state = {}
else:
    state = load_state()
city = state.get("City", {})
signals = city.get("signals", {})
diversions = city.get("diversions", [])

tabs = st.tabs(["Overview", "Places", "Junctions", "Map", "Controls", "History"])

with tabs[0]:
    st.subheader("Overview")
    last_update = city.get("last_update_ts", "-")
    age_sec = st.session_state.get("state_age_sec", None)
    is_stale = st.session_state.get("state_stale", False)
    if is_stale:
        st.error("⚠️ Engine may be stopped — state is older than 15 seconds.")
    else:
        age_str = f"{age_sec:.1f}s ago" if age_sec is not None else ""
        st.success(f"🟢 LIVE — last update: **{last_update}** ({age_str})")
    source = st.session_state.get("state_source")
    if source:
        st.caption(f"State source: {source}")
    cols = st.columns(3)
    for col, place in zip(cols, ["Temple", "Market", "Ground"]):
        data = state.get(place, {})
        with col:
            st.markdown(f"**{place}**")
            st.metric("People", data.get("people", 0))
            st.metric("Vehicles", data.get("vehicles", 0))
            st.write("Risk:", data.get("risk", "-"))
            if data.get("alert"):
                st.error(data.get("alert"))

with tabs[1]:
    st.subheader("Places (Live YOLO)")
    pcols = st.columns(3)
    for col, place, prefix in zip(
        pcols,
        ["Temple", "Market", "Ground"],
        ["crowd_temple", "crowd_market", "crowd_ground"],
    ):
        data = state.get(place, {})
        with col:
            st.markdown(f"**{place}**")
            show_frame(prefix, f"{place} YOLO")
            st.write("People:", data.get("people", 0))
            st.write("Vehicles:", data.get("vehicles", 0))
            st.write("Density:", data.get("density", "-"))
            st.write("Congestion:", data.get("congestion", "-"))
            st.write("Flow:", data.get("flow", "-"))
            st.write("Risk:", data.get("risk", "-"))
            st.write("Gate Status:", data.get("gate_status", "-"))

with tabs[2]:
    st.subheader("Junctions (Live YOLO)")
    jnames = ["Junction-1", "Junction-2", "Junction-3", "Junction-4", "Junction-5"]
    sel = st.selectbox("Choose junction", jnames, index=0)
    jdata = signals.get(sel, {})
    st.write("Active Way:", jdata.get("active_way", "-"))
    st.write("Green Remaining (s):", jdata.get("green_remaining_sec", "-"))
    st.write("Total Vehicles:", jdata.get("vehicles", "-"))
    st.write("Next Way:", jdata.get("next_way", "-"))
    if jdata.get("video_source") == "SIMULATED":
        st.caption("Note: This junction uses simulated counts (no lane video source detected).")

    if sel == "Junction-1":
        lane_cols = st.columns(4)
        lane_keys = ["frame_j1_way-a", "frame_j1_way-b", "frame_j1_way-c", "frame_j1_way-d"]
        lane_labels = ["Way-A", "Way-B", "Way-C", "Way-D"]
        lanes = jdata.get("lanes", {})
        for col, key, label in zip(lane_cols, lane_keys, lane_labels):
            with col:
                st.markdown(f"**{label}**")
                show_frame(key, label)
                st.write("Count:", lanes.get(label.lower().replace("-", ""), lanes.get(label.replace("Way-", "way-").lower(), "-")))
                is_active = jdata.get("active_way") == label.replace("Way-", "way-").lower()
                if is_active:
                    st.success("GREEN")
                else:
                    st.error("RED")

with tabs[3]:
    st.subheader("City Map & Diversions")
    st.code(
        "                (Bypass Road)\n"
        "J1 ===== Temple ===== J2 ===== Market ===== J3\n"
        " |         |  |         |         |  |\n"
        " |       ZoneA/B      ZoneA/B    ZoneA/B\n"
        " |                       |         |\n"
        " J4 ===== Ground ======= J5\n"
        " |         |  |          |\n"
        " |       ZoneA/B      ZoneA/B\n",
        language="text",
    )
    st.markdown("### Diversions")
    if diversions:
        for d in diversions:
            st.warning(d.get("message", ""))
    else:
        st.success("No diversions required.")

with tabs[4]:
    st.subheader("Manual Control")
    manual = load_manual_override()
    enabled = st.checkbox("Enable Manual Override", value=manual.get("enabled", False))
    junction_options = ["Junction-1", "Junction-2", "Junction-3", "Junction-4", "Junction-5", "ALL"]
    junction = st.selectbox("Target Junction", junction_options, index=0)
    mode = st.selectbox("Mode", ["auto", "all_red", "force_way"], index=["auto", "all_red", "force_way"].index(manual.get("mode", "auto")))
    way = st.selectbox("Force Way", ["way-a", "way-b", "way-c", "way-d"], index=["way-a", "way-b", "way-c", "way-d"].index(manual.get("force_way", "way-a")))
    if st.button("Apply Manual Override"):
        save_manual_override({
            "enabled": enabled,
            "junction": junction,
            "mode": mode,
            "force_way": way
        })
        st.success("Manual override updated.")

    st.subheader("ESP32 Target")
    target_options = ["Temple", "Market", "Ground", "Junction-1"]
    current_target = "Temple"
    try:
        if os.path.exists(ESP32_TARGET_FILE):
            with open(ESP32_TARGET_FILE, "r") as f:
                t = f.read().strip()
                if t in target_options:
                    current_target = t
    except Exception:
        pass
    selected = st.selectbox("Send ESP32 output for", target_options, index=target_options.index(current_target))
    try:
        with open(ESP32_TARGET_FILE, "w") as f:
            f.write(selected)
    except Exception:
        st.warning("Unable to write ESP32 target file.")

with tabs[5]:
    st.subheader("History (Last 200 Samples)")
    places_path = os.path.join(DATA_DIR, "history_places.csv")
    if not os.path.exists(places_path):
        st.info("No history data yet. Run the engine to generate logs.")
    else:
        from collections import deque

        lines = deque(maxlen=200)
        try:
            with open(places_path, "r") as f:
                header = f.readline()
                for line in f:
                    lines.append(line.strip())
        except Exception:
            lines = deque(maxlen=0)

        people_series = {"Temple": [], "Market": [], "Ground": []}
        vehicles_series = {"Temple": [], "Market": [], "Ground": []}

        for line in lines:
            parts = line.split(",")
            if len(parts) < 4:
                continue
            place = parts[1]
            try:
                people = int(float(parts[2])) if parts[2] else 0
                vehicles = int(float(parts[3])) if parts[3] else 0
            except Exception:
                people = 0
                vehicles = 0
            if place in people_series:
                people_series[place].append(people)
                vehicles_series[place].append(vehicles)

        st.markdown("**People Over Time**")
        st.line_chart(people_series)
        st.markdown("**Vehicles Over Time**")
        st.line_chart(vehicles_series)

time.sleep(2.0)
st.rerun()
