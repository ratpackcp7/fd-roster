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

# Vehicles by station (your config)
STATIONS = {
    "Station 1": ["B51", "T51", "A51", "A151"],
    "Station 2": ["E52", "A52"],
}

# Number of minimum crew per vehicle (customize if you like)
MIN_CREW = {
    "B51": 1,
    "T51": 2,
    "A51": 2,
    "A151": 2,
    "E52": 2,
    "A52": 2,
}

# API settings (set API_URL in a .env file later)
API_URL = os.getenv("API_URL", "").strip()
API_KEY = os.getenv("API_KEY", "").strip()  # if your API needs a key; optional


# -----------------------------
# Demo data (used if no API_URL)
# -----------------------------
def generate_demo_data(start: date, end: date) -> pd.DataFrame:
    """Generate a simple demo roster between start and end (inclusive)."""
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
                assigned = []
                for k in range(need):
                    assigned.append(names[(i + k) % len(names)])
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
    """
    Fetch roster rows with columns:
    date (YYYY-MM-DD), station, vehicle, start_time, end_time, crew (list), count (int)

    If API_URL is not set, returns demo data.
    """
    if not API_URL:
        return generate_demo_data(from_date, to_date)

    try:
        headers = {}
        if API_KEY:
            headers["Authorization"] = f"Bearer {API_KEY}"

        params = {
            "start": from_date.isoformat(),
            "end": to_date.isoformat(),
        }
        r = requests.get(API_URL, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        # Expecting a list of records; transform to DataFrame
        # You can adapt the mapping below to your real API schema
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
    # Sort: Station then vehicle
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

        # Build an expandable per-vehicle card
        for _, row in station_rows.iterrows():
            vehicle = row["vehicle"]
            crew_list = row["crew"] if isinstance(row["crew"], list) else []
            filled = len(crew_list) > 0

            if collapse_empty and not filled:
                exp = st.expander(f"{vehicle} — (empty)")
                with exp:
                    st.write("No assignments.")
            else:
                st.markdown(f"**{vehicle}**  \n"
                            f"Time: {row['start_time']} → {row['end_time']}  \n"
                            f"Crew ({len(crew_list)}): {', '.join(crew_list) if crew_list else '—'}")


def render_week(df: pd.DataFrame, start_day: date):
    st.subheader(f"7-Day View — week of {start_day.strftime('%b %d, %Y')}")
    days = [start_day + timedelta(days=i) for i in range(7)]
    for station in STATIONS:
        st.markdown(f"### {station}")
        # Build a table: rows=vehicles, cols=days, cells=crew counts
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
    # Build a simple calendar grid with count of filled slots per day
    first = anchor_day.replace(day=1)
    month_days = calendar.monthrange(first.year, first.month)[1]
    st.subheader(f"Month View — {first.strftime('%B %Y')}")

    # Totals per day
    counts = {}
    for i in range(month_days):
        d = first + timedelta(days=i)
        sub = df[df["date"] == d.isoformat()]
        counts[d.day] = int(sub["count"].sum()) if not sub.empty else 0

    # Render as 7-column grid
    week_day_of_first = first.weekday()  # Mon=0
    cols = st.columns(7)
    # Headers
    for idx, name in enumerate(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]):
        cols[idx].markdown(f"**{name}**")

    # Start rows
    day_cursor = 1
    row_offset = 0
    while day_cursor <= month_days:
        cols = st.columns(7)
        for wd in range(7):
            cell_idx = wd
            if row_offset == 0 and wd < week_day_of_first:
                cols[cell_idx].markdown("&nbsp;", unsafe_allow_html=True)
                continue
            if day_cursor > month_days:
                cols[cell_idx].markdown("&nbsp;", unsafe_allow_html=True)
                continue

            d = first.replace(day=day_cursor)
            total = counts.get(day_cursor, 0)
            cols[cell_idx].markdown(
                f"**{day_cursor}**  \nAssigned crew (total): **{total}**"
            )
            day_cursor += 1
        row_offset += 1


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="FD Roster", layout="wide")

st.title("Fire Department Roster")

with st.sidebar:
    st.markdown("### Data Source")
    mode = "API" if API_URL else "Demo"
    st.write(f"Source: **{mode}**")
    if mode == "API":
        st.caption(f"API_URL: {API_URL}")

    st.markdown("---")
    st.markdown("### Filters")
    today = date.today()
    picked_date = st.date_input("Pick a date", value=today)
    collapse_empty = st.toggle("Collapse empty vehicles (Daily)", value=True)

tabs = st.tabs(["Daily", "7-Day", "Month"])

# Decide fetch window based on view (to limit network calls)
daily_from = date(picked_date.year, picked_date.month, picked_date.day)
week_start = picked_date - timedelta(days=picked_date.weekday())  # Monday start
month_first = picked_date.replace(day=1)
month_last = (month_first + relativedelta(months=1)) - timedelta(days=1)

# Fetch once for the largest range we need
fetch_start = min(daily_from, week_start, month_first)
fetch_end = max(daily_from, week_start + timedelta(days=6), month_last)

df = fetch_roster(fetch_start, fetch_end)
# Normalize types
if not df.empty:
    # Ensure 'crew' is list
    def ensure_list(x):
        if isinstance(x, list):
            return x
        if pd.isna(x) or x is None:
            return []
        if isinstance(x, str):
            try:
                return json.loads(x)
            except Exception:
                return [x]
        return [x]

    df["crew"] = df["crew"].apply(ensure_list)
    # counts might be missing on API data; recompute to be safe
    df["count"] = df["crew"].apply(lambda L: len(L))

with tabs[0]:
    render_daily(df, picked_date, collapse_empty)

with tabs[1]:
    render_week(df, week_start)

with tabs[2]:
    render_month(df, picked_date)
