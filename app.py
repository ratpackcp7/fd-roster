# app.py
import os
import json
import io
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

# Persist imported/edited data during the session
if "override_df" not in st.session_state:
    st.session_state.override_df = None
if "last_fetch_range" not in st.session_state:
    st.session_state.last_fetch_range = (None, None)


# -----------------------------
# Demo data (used if no API_URL and no import)
# -----------------------------
def generate_demo_data(start: date, end: date) -> pd.DataFrame:
    names = [
        "J. Smith", "A. Johnson", "C. Davis", "M. Brown",
        "L. Garcia", "R. Lee", "K. Wilson", "P. Miller",
        "B. Clark", "S. Lewis", "D. Young", "E. Walker",
    ]
    rows = []
    day = start
    i = 0
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
# API fetcher
# -----------------------------
def fetch_roster(from_date: date, to_date: date) -> pd.DataFrame:
    """Return DataFrame with columns:
    date, station, vehicle, start_time, end_time, crew(list), count(int)
    """
    # If we already imported CSV this session, prefer that
    if st.session_state.override_df is not None:
        df = st.session_state.override_df.copy()
        # filter to requested range
        mask = (pd.to_datetime(df["date"]) >= pd.to_datetime(from_date)) & (
            pd.to_datetime(df["date"]) <= pd.to_datetime(to_date)
        )
        df = df.loc[mask].reset_index(drop=True)
        return df

    if not API_URL:
        return generate_demo_data(from_date, to_date)

    try:
        headers = {}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"
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
                "count": len(item.get("crew", [])),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["date","station","vehicle","start_time","end_time","crew","count"])
        return df
    except Exception as e:
        st.warning(f"API error: {e}. Falling back to demo data.")
        return generate_demo_data(from_date, to_date)


# -----------------------------
# Helpers
# -----------------------------
REQUIRED_COLS = ["date","station","vehicle","start_time","end_time","crew"]

def ensure_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x) or x is None:
        return []
    if isinstance(x, str):
        # accept JSON list or comma/semicolon separated
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(s)
            except Exception:
                pass
        # split by comma
        parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
        return parts
    return [x]

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # ensure required columns exist
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = "" if c != "crew" else []
    # type conversions
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df["crew"] = df["crew"].apply(ensure_list)
    df["count"] = df["crew"].apply(lambda L: len(L))
    return df

def day_df(df: pd.DataFrame, d: date) -> pd.DataFrame:
    dstr = d.isoformat()
    out = df[df["date"] == dstr].copy()
    # Ensure all vehicles show even if unassigned
    filled = set(zip(out["station"], out["vehicle"]))
    for station, vehicles in STATIONS.items():
        for v in vehicles:
            if (station, v) not in filled:
                out = pd.concat([out, pd.DataFrame([{
                    "date": dstr,
                    "station": station,
                    "vehicle": v,
                    "start_time": "—",
                    "end_time": "—",
                    "crew": [],
                    "count": 0,
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

    day_cursor = 1
    row_offset = 0
    while day_cursor <= month_days:
        cols = st.columns(7)
        for wd in range(7):
            if row_offset == 0 and wd < week_day_of_first:
                cols[wd].markdown("&nbsp;", unsafe_allow_html=True)
                continue
            if day_cursor > month_days:
                cols[wd].markdown("&nbsp;", unsafe_allow_html=True)
                continue
            total = counts.get(day_cursor, 0)
            cols[wd].markdown(f"**{day_cursor}**  \nAssigned crew (total): **{total}**")
            day_cursor += 1
        row_offset += 1


# -----------------------------
# CSV Utilities
# -----------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    # For export, write crew as comma-separated string
    out = df.copy()
    out = out[["date","station","vehicle","start_time","end_time","crew"]]  # keep core columns
    out["crew"] = out["crew"].apply(lambda L: ", ".join(L) if isinstance(L, list) else str(L))
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    return csv_bytes

def csv_template_bytes() -> bytes:
    sample = pd.DataFrame([
        {
            "date": date.today().isoformat(),
            "station": "Station 1",
            "vehicle": "A51",
            "start_time": "07:00",
            "end_time": "07:00 (+1)",
            "crew": "J. Smith, A. Johnson"
        },
        {
            "date": (date.today() + timedelta(days=1)).isoformat(),
            "station": "Station 2",
            "vehicle": "E52",
            "start_time": "07:00",
            "end_time": "07:00 (+1)",
            "crew": "[\"K. Wilson\",\"P. Miller\"]"
        },
    ], columns=REQUIRED_COLS)
    return sample.to_csv(index=False).encode("utf-8")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="FD Roster", layout="wide")
st.title("Fire Department Roster")

with st.sidebar:
    st.markdown("### Data Source")
    mode = "CSV Import (session)" if st.session_state.override_df is not None else ("API" if API_URL else "Demo")
    st.write(f"Source: **{mode}**")
    if mode == "API":
        st.caption(f"API_URL: {API_URL or '(not set)'}")

    st.markdown("---")
    st.markdown("### Filters")
    today = date.today()
    picked_date = st.date_input("Pick a date", value=today)
    collapse_empty = st.toggle("Collapse empty vehicles (Daily)", value=True)

tabs = st.tabs(["Daily", "7-Day", "Month", "CSV Import/Export"])

daily_from = date(picked_date.year, picked_date.month, picked_date.day)
week_start = picked_date - timedelta(days=picked_date.weekday())  # Monday
month_first = picked_date.replace(day=1)
month_last = (month_first + relativedelta(months=1)) - timedelta(days=1)

fetch_start = min(daily_from, week_start, month_first)
fetch_end = max(daily_from, week_start + timedelta(days=6), month_last)

df = fetch_roster(fetch_start, fetch_end)
if not df.empty:
    df = normalize_df(df)

with tabs[0]:
    render_daily(df, picked_date, collapse_empty)

with tabs[1]:
    render_week(df, week_start)

with tabs[2]:
    render_month(df, picked_date)

with tabs[3]:
    st.markdown("#### Export current data")
    st.caption("Exports the rows currently loaded for the selected date range.")
    csv_bytes = df_to_csv_bytes(df if not df.empty else pd.DataFrame(columns=REQUIRED_COLS))
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"roster_{fetch_start}_{fetch_end}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("#### CSV Template")
    st.caption("Download a ready-to-edit CSV with correct headers and example rows.")
    st.download_button(
        "Download CSV template",
        data=csv_template_bytes(),
        file_name="roster_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("#### Import CSV")
    st.caption("Upload a CSV to replace or merge. Accepted crew formats: JSON list or comma-separated names.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_upload")

    import_mode = st.radio(
        "Import mode",
        ["Replace all (session)", "Merge by (date,station,vehicle)"],
        horizontal=True,
    )

    if uploaded is not None:
        try:
            incoming = pd.read_csv(uploaded)
            missing = [c for c in REQUIRED_COLS if c not in incoming.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                incoming = normalize_df(incoming)

                if import_mode == "Replace all (session)":
                    st.session_state.override_df = incoming
                    st.success("Imported and replaced in current session.")
                else:
                    # Merge with existing (either current override, API, or demo)
                    # Merge key: (date, station, vehicle). New rows overwrite existing matches.
                    base_start, base_end = st.session_state.last_fetch_range
                    if base_start is None or base_end is None:
                        base_start, base_end = fetch_start, fetch_end
                    base_df = fetch_roster(base_start, base_end)
                    base_df = normalize_df(base_df) if not base_df.empty else pd.DataFrame(columns=REQUIRED_COLS + ["count"])

                    # Remove duplicates in base matching incoming keys
                    key_cols = ["date","station","vehicle"]
                    keys = set(map(tuple, incoming[key_cols].values.tolist()))
                    mask = ~base_df[key_cols].apply(tuple, axis=1).isin(keys)
                    merged = pd.concat([base_df.loc[mask], incoming], ignore_index=True)
                    merged = normalize_df(merged)
                    st.session_state.override_df = merged
                    st.success("Imported and merged in current session.")

                # Remember last fetch window for consistent merging
                st.session_state.last_fetch_range = (fetch_start, fetch_end)
                st.info("Refresh the tabs (or change the date) to see updates.")
        except Exception as e:
            st.error(f"Import failed: {e}")
