# app.py
import os
import json
import calendar
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# -----------------------------
# Config
# -----------------------------
load_dotenv()

STATIONS = {
    "Station 1": ["B51", "T51", "A51", "A151"],
    "Station 2": ["E52", "A52"],
}

MIN_CREW = {
    "B51": 1,
    "T51": 2,
    "A51": 2,
    "A151": 2,
    "E52": 2,
    "A52": 2,
}

API_URL = os.getenv("API_URL", "").strip()
API_KEY = os.getenv("API_KEY", "").strip()  # optional

# session state
if "override_df" not in st.session_state:
    st.session_state.override_df = None
if "last_fetch_range" not in st.session_state:
    st.session_state.last_fetch_range = (None, None)
if "people_pool" not in st.session_state:
    # starting pool for quick-add; grows as you import/edit
    st.session_state.people_pool = set(
        ["J. Smith","A. Johnson","C. Davis","M. Brown","L. Garcia","R. Lee",
         "K. Wilson","P. Miller","B. Clark","S. Lewis","D. Young","E. Walker"]
    )

# -----------------------------
# Demo data
# -----------------------------
def generate_demo_data(start: date, end: date) -> pd.DataFrame:
    names = sorted(list(st.session_state.people_pool))
    if not names:
        names = ["J. Smith","A. Johnson"]
    rows, i = [], 0
    day = start
    while day <= end:
        for station, vehicles in STATIONS.items():
            for v in vehicles:
                need = MIN_CREW.get(v, 2)
                assigned = [names[(i + k) % len(names)] for k in range(need)]
                i += 1
                rows.append({
                    "date": day.isoformat(),
                    "station": station,
                    "vehicle": v,
                    "start_time": "07:00",
                    "end_time": "07:00 (+1)",
                    "crew": assigned,
                    "count": len(assigned),
                })
        day += timedelta(days=1)
    return pd.DataFrame(rows)

# -----------------------------
# Fetchers / Normalizers
# -----------------------------
REQUIRED_COLS = ["date","station","vehicle","start_time","end_time","crew"]

def ensure_list(x):
    if isinstance(x, list): return x
    if pd.isna(x) or x is None: return []
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try: return json.loads(s)
            except Exception: pass
        parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
        return parts
    return [x]

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = "" if c != "crew" else []
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df["crew"] = df["crew"].apply(ensure_list)
    df["count"] = df["crew"].apply(lambda L: len(L))
    # learn names into pool
    for names in df["crew"]:
        for n in names:
            st.session_state.people_pool.add(n)
    return df

def fetch_roster(from_date: date, to_date: date) -> pd.DataFrame:
    if st.session_state.override_df is not None:
        df = st.session_state.override_df.copy()
        mask = (pd.to_datetime(df["date"]) >= pd.to_datetime(from_date)) & (
               pd.to_datetime(df["date"]) <= pd.to_datetime(to_date))
        return normalize_df(df.loc[mask].reset_index(drop=True))

    if not API_URL:
        return normalize_df(generate_demo_data(from_date, to_date))

    try:
        headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
        params = {"start": from_date.isoformat(), "end": to_date.isoformat()}
        r = requests.get(API_URL, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        rows = []
        for item in data:
            rows.append({
                "date": item.get("date"),
                "station": item.get("station"),
                "vehicle": item.get("vehicle"),
                "start_time": item.get("start_time", "07:00"),
                "end_time": item.get("end_time", "07:00 (+1)"),
                "crew": item.get("crew", []),
            })
        df = pd.DataFrame(rows)
        return normalize_df(df) if not df.empty else pd.DataFrame(columns=REQUIRED_COLS+["count"])
    except Exception as e:
        st.warning(f"API error: {e}. Falling back to demo data.")
        return normalize_df(generate_demo_data(from_date, to_date))

# -----------------------------
# View helpers
# -----------------------------
def day_df(df: pd.DataFrame, d: date) -> pd.DataFrame:
    dstr = d.isoformat()
    out = df[df["date"] == dstr].copy()
    filled = set(zip(out["station"], out["vehicle"]))
    for station, vehicles in STATIONS.items():
        for v in vehicles:
            if (station, v) not in filled:
                out = pd.concat([out, pd.DataFrame([{
                    "date": dstr, "station": station, "vehicle": v,
                    "start_time": "—", "end_time": "—", "crew": [], "count": 0
                }])], ignore_index=True)
    out["station_order"] = out["station"].map(lambda s: list(STATIONS.keys()).index(s))
    out.sort_values(by=["station_order", "vehicle"], inplace=True)
    out.drop(columns=["station_order"], inplace=True)
    return out

def render_daily(df: pd.DataFrame, d: date, collapse_empty: bool):
    st.subheader(f"Daily Roster — {d.strftime('%A, %b %d, %Y')}")
    for station in STATIONS:
        st.markdown(f"### {station}")
        station_rows = day_df(df, d)
        station_rows = station_rows[station_rows["station"] == station].copy()
        for _, row in station_rows.iterrows():
            vehicle = row["vehicle"]
            crew_list = row["crew"] if isinstance(row["crew"], list) else []
            filled = len(crew_list) > 0
            if collapse_empty and not filled:
                exp = st.expander(f"{vehicle} — (empty)")
                with exp:
                    st.write("No assignments.")
            else:
                st.markdown(
                    f"**{vehicle}**  \n"
                    f"Time: {row['start_time']} → {row['end_time']}  \n"
                    f"Crew ({len(crew_list)}): {', '.join(crew_list) if crew_list else '—'}"
                )

def render_week(df: pd.DataFrame, start_day: date):
    st.subheader(f"7-Day View — week of {start_day.strftime('%b %d, %Y')}")
    days = [start_day + timedelta(days=i) for i in range(7)]
    for station in STATIONS:
        st.markdown(f"### {station}")
        vehicles = STATIONS[station]
        data = []
        for v in vehicles:
            row = {"Vehicle": v}
            for d in days:
                sub = df[(df["date"] == d.isoformat()) & (df["station"] == station) & (df["vehicle"] == v)]
                if len(sub) == 0:
                    row[d.strftime("%a %m/%d")] = "—"
                else:
                    c = int(sub.iloc[0]["count"])
                    need = MIN_CREW.get(v, 2)
                    row[d.strftime("%a %m/%d")] = f"{c}/{need}"
            data.append(row)
        st.dataframe(pd.DataFrame(data), use_container_width=True)

def render_month(df: pd.DataFrame, anchor_day: date):
    first = anchor_day.replace(day=1)
    month_days = calendar.monthrange(first.year, first.month)[1]
    st.subheader(f"Month View — {first.strftime('%B %Y')}")
    counts = {}
    for i in range(month_days):
        d = first + timedelta(days=i)
        sub = df[df["date"] == d.isoformat()]
        counts[d.day] = int(sub["count"].sum()) if not sub.empty else 0
    week_day_of_first = first.weekday()  # Mon=0
    cols = st.columns(7)
    for idx, name in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]):
        cols[idx].markdown(f"**{name}**")
    day_cursor, row_offset = 1, 0
    while day_cursor <= month_days:
        cols = st.columns(7)
        for wd in range(7):
            if row_offset == 0 and wd < week_day_of_first:
                cols[wd].markdown("&nbsp;", unsafe_allow_html=True); continue
            if day_cursor > month_days:
                cols[wd].markdown("&nbsp;", unsafe_allow_html=True); continue
            total = counts.get(day_cursor, 0)
            cols[wd].markdown(f"**{day_cursor}**  \nAssigned crew (total): **{total}**")
            day_cursor += 1
        row_offset += 1

# -----------------------------
# CSV utils
# -----------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    out = out[["date","station","vehicle","start_time","end_time","crew"]]
    out["crew"] = out["crew"].apply(lambda L: ", ".join(L) if isinstance(L, list) else str(L))
    return out.to_csv(index=False).encode("utf-8")

def csv_template_bytes() -> bytes:
    sample = pd.DataFrame([
        {"date": date.today().isoformat(), "station": "Station 1", "vehicle": "A51",
         "start_time": "07:00", "end_time": "07:00 (+1)", "crew": "J. Smith, A. Johnson"},
        {"date": (date.today()+timedelta(days=1)).isoformat(), "station": "Station 2", "vehicle": "E52",
         "start_time": "07:00", "end_time": "07:00 (+1)", "crew": "[\"K. Wilson\",\"P. Miller\"]"},
    ], columns=REQUIRED_COLS)
    return sample.to_csv(index=False).encode("utf-8")

# -----------------------------
# Admin editing helpers
# -----------------------------
def upsert_rows(base_df: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    """Insert or replace rows by (date,station,vehicle)."""
    if base_df is None or base_df.empty:
        merged = pd.DataFrame(rows)
    else:
        key_cols = ["date","station","vehicle"]
        incoming = pd.DataFrame(rows)
        if incoming.empty:
            return normalize_df(base_df)
        # drop clashes from base
        keys = set(map(tuple, incoming[key_cols].values.tolist()))
        mask = ~base_df[key_cols].apply(tuple, axis=1).isin(keys)
        merged = pd.concat([base_df.loc[mask], incoming], ignore_index=True)
    return normalize_df(merged)

def parse_crew_text(text: str) -> list[str]:
    if not text: return []
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            L = json.loads(text); return [str(x).strip() for x in L if str(x).strip()]
        except Exception:
            pass
    parts = [p.strip() for p in text.replace(";", ",").split(",")]
    return [p for p in parts if p]

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="FD Roster", layout="wide")
st.title("Fire Department Roster")

with st.sidebar:
    st.markdown("### Data Source")
    mode = "CSV/Edits (session)" if st.session_state.override_df is not None else ("API" if API_URL else "Demo")
    st.write(f"Source: **{mode}**")
    if mode == "API":
        st.caption(f"API_URL: {API_URL or '(not set)'}")
    st.markdown("---")
    today = date.today()
    picked_date = st.date_input("Pick a date", value=today)
    collapse_empty = st.toggle("Collapse empty vehicles (Daily)", value=True)

tabs = st.tabs(["Daily", "7-Day", "Month", "CSV Import/Export", "Admin"])

daily_from = date(picked_date.year, picked_date.month, picked_date.day)
week_start = picked_date - timedelta(days=picked_date.weekday())
month_first = picked_date.replace(day=1)
month_last = (month_first + relativedelta(months=1)) - timedelta(days=1)

fetch_start = min(daily_from, week_start, month_first)
fetch_end = max(daily_from, week_start + timedelta(days=6), month_last)

df = fetch_roster(fetch_start, fetch_end)
df = normalize_df(df) if not df.empty else pd.DataFrame(columns=REQUIRED_COLS+["count"])

# ----- Daily / Week / Month
with tabs[0]: render_daily(df, picked_date, collapse_empty)
with tabs[1]: render_week(df, week_start)
with tabs[2]: render_month(df, picked_date)

# ----- CSV Import/Export
with tabs[3]:
    st.markdown("#### Export current data")
    st.download_button("Download CSV",
                       data=df_to_csv_bytes(df),
                       file_name=f"roster_{fetch_start}_{fetch_end}.csv",
                       mime="text/csv", use_container_width=True)
    st.markdown("---")
    st.markdown("#### CSV Template")
    st.download_button("Download CSV template",
                       data=csv_template_bytes(),
                       file_name="roster_template.csv",
                       mime="text/csv", use_container_width=True)
    st.markdown("---")
    st.markdown("#### Import CSV")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_upload")
    import_mode = st.radio("Import mode",
                           ["Replace all (session)", "Merge by (date,station,vehicle)"],
                           horizontal=True)
    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
            missing = [c for c in REQUIRED_COLS if c not in incoming.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                incoming = normalize_df(incoming)
                base_start, base_end = st.session_state.last_fetch_range
                if base_start is None or base_end is None:
                    base_start, base_end = fetch_start, fetch_end
                base_df = fetch_roster(base_start, base_end)
                base_df = normalize_df(base_df) if not base_df.empty else pd.DataFrame(columns=REQUIRED_COLS+["count"])
                if import_mode.startswith("Replace"):
                    st.session_state.override_df = incoming
                else:
                    st.session_state.override_df = upsert_rows(base_df, incoming.to_dict("records"))
                st.session_state.last_fetch_range = (fetch_start, fetch_end)
                st.success("CSV imported. Refresh tabs or change date to view.")
        except Exception as e:
            st.error(f"Import failed: {e}")

# ----- Admin Editor
with tabs[4]:
    st.markdown("### Admin — Edit Assignments")
    colA, colB, colC = st.columns(3)
    with colA:
        admin_date = st.date_input("Date to edit", value=picked_date, key="admin_date")
    with colB:
        admin_station = st.selectbox("Station", list(STATIONS.keys()), key="admin_station")
    with colC:
        admin_vehicle = st.selectbox("Vehicle", STATIONS[admin_station], key="admin_vehicle")

    # pull current row (or create blank)
    ddf = day_df(df, admin_date)
    row = ddf[(ddf["station"] == admin_station) & (ddf["vehicle"] == admin_vehicle)]
    if row.empty:
        current_start, current_end, current_crew = "07:00", "07:00 (+1)", []
    else:
        r0 = row.iloc[0]
        current_start = r0["start_time"]
        current_end = r0["end_time"]
        current_crew = r0["crew"] if isinstance(r0["crew"], list) else []

    st.markdown("#### Times")
    t1, t2 = st.columns(2)
    with t1:
        start_time = st.text_input("Start time", value=current_start, help="e.g., 07:00")
    with t2:
        end_time = st.text_input("End time", value=current_end, help="e.g., 07:00 (+1)")

    st.markdown("#### Crew")
    crew_text = st.text_input(
        "Crew (comma-separated or JSON list)",
        value=", ".join(current_crew),
        key="admin_crew_text",
        help='Examples: `J. Smith, A. Johnson`  or  `["J. Smith","A. Johnson"]`'
    )

    # quick add/remove chips
    st.caption("Quick add/remove from known names")
    names_sorted = sorted(list(st.session_state.people_pool))
    chip_cols = st.columns(6)
    current_set = set(parse_crew_text(crew_text))
    for i, name in enumerate(names_sorted[:30]):  # show up to 30 chips
        col = chip_cols[i % 6]
        if name in current_set:
            if col.button(f"➖ {name}", key=f"rm_{i}"):
                current_set.remove(name)
                crew_text = ", ".join(sorted(current_set))
                st.session_state.admin_crew_text = crew_text
        else:
            if col.button(f"➕ {name}", key=f"add_{i}"):
                current_set.add(name)
                crew_text = ", ".join(sorted(current_set))
                st.session_state.admin_crew_text = crew_text

    st.markdown("---")
    apply_col, add_col, clear_col = st.columns(3)
    with add_col:
        new_name = st.text_input("Add a new name to pool", placeholder="Type name, press Enter")
        if new_name:
            st.session_state.people_pool.add(new_name.strip())
            st.success(f"Added '{new_name}' to quick-add list. (Reopen tab to refresh chips)")

    with clear_col:
        if st.button("Clear crew for this unit"):
            crew_text = ""
            st.session_state.admin_crew_text = ""

    with apply_col:
        if st.button("Apply changes"):
            crew_list = parse_crew_text(st.session_state.get("admin_crew_text", crew_text))
            for n in crew_list:
                st.session_state.people_pool.add(n)
            updated_row = {
                "date": admin_date.isoformat(),
                "station": admin_station,
                "vehicle": admin_vehicle,
                "start_time": start_time.strip() or "07:00",
                "end_time": end_time.strip() or "07:00 (+1)",
                "crew": crew_list,
            }

            # base to merge into: current override if set, else the fetched window
            base_start, base_end = st.session_state.last_fetch_range
            if base_start is None or base_end is None:
                base_start, base_end = fetch_start, fetch_end
            base_df = fetch_roster(base_start, base_end)
            base_df = normalize_df(base_df) if not base_df.empty else pd.DataFrame(columns=REQUIRED_COLS+["count"])

            st.session_state.override_df = upsert_rows(base_df, [updated_row])
            st.session_state.last_fetch_range = (base_start, base_end)
            st.success("Saved. Switch tabs or change date to see it reflected.")
