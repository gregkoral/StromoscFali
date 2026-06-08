"""
Prognoza Fal Bałtyku — Streamlit + Copernicus Marine Toolbox v2.0 (STABLE FIX)
"""

# ── 0. AUTO-INSTALL ────────────────────────────────────────────────────────────
import importlib, subprocess, sys

_WYMAGANE = {
    "streamlit": "streamlit>=1.35",
    "copernicusmarine": "copernicusmarine>=2.0.0",
    "xarray": "xarray",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
}

for mod, pkg in _WYMAGANE.items():
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ── 1. IMPORTY ────────────────────────────────────────────────────────────────
import streamlit as st
import copernicusmarine
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta, timezone

UTC = timezone.utc

# ── 2. UI ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Prognoza Fal Bałtyku", layout="centered")

# ── 3. STAŁE ───────────────────────────────────────────────────────────────────
MIN_LON, MAX_LON = 11.0, 15.0
MIN_LAT, MAX_LAT = 53.5, 56.5
DATASET_ID = "cmems_mod_bal_wav_anfc_PT1H-i"

# ── 4. SAFE LOGIN ──────────────────────────────────────────────────────────────
def bezpieczne_logowanie():
    if "COPERNICUS_USERNAME" not in st.secrets or "COPERNICUS_PASSWORD" not in st.secrets:
        st.error("Brak Copernicus secrets")
        st.stop()

    copernicusmarine.login(
        username=st.secrets["COPERNICUS_USERNAME"],
        password=st.secrets["COPERNICUS_PASSWORD"],
        force_overwrite=True,
    )

# ── 5. DATA FETCH (STABLE CACHE) ──────────────────────────────────────────────
@st.cache_resource
def pobierz_dane(base_time: datetime):

    bezpieczne_logowanie()

    tz_ref = base_time.replace(tzinfo=UTC) if base_time.tzinfo is None else base_time.astimezone(UTC)

    start_dt = tz_ref - timedelta(hours=12)
    end_dt   = tz_ref + timedelta(hours=48)

    ds = copernicusmarine.open_dataset(
        dataset_id=DATASET_ID,
        start_datetime=start_dt,
        end_datetime=end_dt,
        minimum_longitude=MIN_LON,
        maximum_longitude=MAX_LON,
        minimum_latitude=MIN_LAT,
        maximum_latitude=MAX_LAT,
        variables=["VHM0", "VTM02", "VMDR_WW"],
        coordinates_selection_method="inside",
    )

    return ds.load().chunk()

# ── 6. TIME HELPERS ───────────────────────────────────────────────────────────
def now_utc_hour():
    return datetime.now(UTC).replace(minute=0, second=0, microsecond=0, tzinfo=None)

def to_ns(dt):
    return np.datetime64(pd.Timestamp(dt).tz_convert("UTC").tz_localize(None), "ns")

# ── 7. SESSION STATE ──────────────────────────────────────────────────────────
if "base_time" not in st.session_state:
    st.session_state.base_time = now_utc_hour()

if "current_time" not in st.session_state:
    st.session_state.current_time = st.session_state.base_time

if "filter" not in st.session_state:
    st.session_state.filter = 0.5

# ── 8. LOAD DATA ───────────────────────────────────────────────────────────────
ds = pobierz_dane(st.session_state.base_time)

# ── 9. TIME SLICE ──────────────────────────────────────────────────────────────
tmin = to_ns(ds.time.values[0])
tmax = to_ns(ds.time.values[-1])
tsel = to_ns(st.session_state.current_time)

if not (tmin <= tsel <= tmax):
    st.error("Wybrany czas poza zakresem danych")
    st.stop()

slice = ds.sel(time=tsel, method="nearest")

lons = slice["longitude"].values
lats = slice["latitude"].values
h = slice["VHM0"].values
t = slice["VTM02"].values
dir = slice["VMDR_WW"].values

# ── 10. MATH ──────────────────────────────────────────────────────────────────
with np.errstate(divide="ignore", invalid="ignore"):
    wave_length = (9.81 * (t ** 2)) / (2 * np.pi)
    steep = h / wave_length

mask = np.where(np.isnan(h), 1, np.nan)
filtered = np.where(h >= st.session_state.filter, steep, np.nan)

# ── 11. COLORS ────────────────────────────────────────────────────────────────
cmap = LinearSegmentedColormap.from_list(
    "wave",
    [(0.0,"#8ecae6"),(0.4,"#219ebc"),(0.7,"#ffb703"),(1.0,"#fb8500")]
)

# ── 12. MAIN MAP ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,7))
ax.set_facecolor("#404040")
ax.set_xlim(MIN_LON, MAX_LON)
ax.set_ylim(MIN_LAT, MAX_LAT)
ax.axis("off")

ax.pcolormesh(lons, lats, mask, cmap="Greys", alpha=0.3)
im = ax.pcolormesh(lons, lats, filtered, cmap=cmap, vmin=0, vmax=0.1)

fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.02)

st.pyplot(fig, clear_figure=True)

# ── 13. CONTROLS ──────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

def shift(h):
    st.session_state.current_time += timedelta(hours=h)

col1.button("-6h", on_click=shift, args=(-6,))
col2.button("-1h", on_click=shift, args=(-1,))
col3.button("Teraz", on_click=lambda: st.session_state.update(current_time=now_utc_hour()))
col4.button("+1h", on_click=shift, args=(1,))
col5.button("+6h", on_click=shift, args=(6,))

colf1, colf2, colf3 = st.columns(3)

colf1.button("-0.1", on_click=lambda: st.session_state.update(filter=max(0, st.session_state.filter-0.1)))
colf2.button("0.5", on_click=lambda: st.session_state.update(filter=0.5))
colf3.button("+0.1", on_click=lambda: st.session_state.update(filter=min(5, st.session_state.filter+0.1)))

# ── 14. MAPS 2 ────────────────────────────────────────────────────────────────
fig2, (ax1, ax2) = plt.subplots(1,2, figsize=(6,3.5))

ax1.set_title("VHM0")
ax2.set_title("VTM02")

ax1.pcolormesh(lons, lats, h)
ax2.pcolormesh(lons, lats, t)

st.pyplot(fig2, clear_figure=True)

# ── 15. INFO ──────────────────────────────────────────────────────────────────
st.write("Time:", st.session_state.current_time)
st.write("Filter:", st.session_state.filter)