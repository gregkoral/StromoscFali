import sys
import subprocess

# ==============================================================================
# AUTOMATYCZNA INSTALACJA / WERYFIKACJA BIBLIOTEK
# ==============================================================================
required_libraries = ["streamlit", "copernicusmarine", "xarray", "matplotlib", "numpy"]
installed_any = False

for lib in required_libraries:
    try:
        __import__(lib)
    except ImportError:
        print(f"Brak biblioteki '{lib}'. Instalacja w toku...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        installed_any = True

# ==============================================================================
# WŁAŚCIWY PROGRAM
# ==============================================================================
import streamlit as st
import copernicusmarine
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta, UTC
import time

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="Prognoza Fal Bałtyku", layout="centered")

# =========================
# CONSTANTS
# =========================
MIN_LON, MAX_LON = 11.0, 15.0
MIN_LAT, MAX_LAT = 53.5, 56.5
DATASET_ID = "cmems_mod_bal_wav_anfc_PT1H-i"

# =========================
# CACHE DATASET (ZAKTUALIZOWANE API COPERNICUS)
# =========================
@st.cache_resource(show_spinner=False)
def get_dataset():
    # Nowe API wymaga otwarcia pełnego datasetu. Xarray obsługuje to leniwie (lazy-loading),
    # więc pobierane są tylko metadane – operacja jest natychmiastowa.
    ds = copernicusmarine.open_dataset(
        dataset_id=DATASET_ID,
        username=st.secrets["COPERNICUS_USERNAME"],
        password=st.secrets["COPERNICUS_PASSWORD"]
    )
    return ds

# =========================
# SESSION STATE
# =========================
if "base_time" not in st.session_state:
    st.session_state.base_time = datetime.now(UTC).replace(
        minute=0, second=0, microsecond=0, tzinfo=None
    )

if "current_time" not in st.session_state:
    st.session_state.current_time = st.session_state.base_time

if "prog_filtra" not in st.session_state:
    st.session_state.prog_filtra = 0.5

# =========================
# LOAD DATASET
# =========================
with st.spinner("Ładowanie metadanych Copernicus (lazy)..."):
    ds_full = get_dataset()

# =========================
# TIME & SPACE SLICE (PRZENIESIENIE FILTROWANIA DO XARRAY)
# =========================
t0 = time.time()

# Nowe API: Filtrowanie współrzędnych, czasu i zmiennych odbywa się za pomocą xarray.
# Pobieramy mały wycinek danych (.load()) dopiero w tym momencie, co gwarantuje szybkość.
wave_slice = ds_full[[ "VHM0", "VTM02", "VMDR_WW" ]].sel(
    longitude=slice(MIN_LON, MAX_LON),
    latitude=slice(MIN_LAT, MAX_LAT),
    time=st.session_state.current_time,
    method="nearest"
).load()

slice_load_time = time.time() - t0
st.sidebar.write(f"slice load: {slice_load_time:.2f}s")

# =========================
# DATA
# =========================
lons_raw = wave_slice["longitude"].values
lats_raw = wave_slice["latitude"].values
h_signif = wave_slice["VHM0"].values
t_mean = wave_slice["VTM02"].values
vmdr_ww = wave_slice["VMDR_WW"].values

# =========================
# MATH
# =========================
with np.errstate(divide="ignore", invalid="ignore"):
    wave_length = (9.81 * (t_mean ** 2)) / (2 * np.pi)
    wave_steepness = h_signif / wave_length

land_mask = np.where(np.isnan(h_signif), 1, np.nan)
wave_filtered = np.where(h_signif >= st.session_state.prog_filtra, wave_steepness, np.nan)

# =========================
# MAP
# =========================
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_facecolor("#404040")
ax.set_xlim(MIN_LON, MAX_LON)
ax.set_ylim(MIN_LAT, MAX_LAT)

ax.pcolormesh(
    lons_raw, lats_raw, land_mask,
    cmap=LinearSegmentedColormap.from_list("land", ["#6b798d", "#6b798d"])
)

im = ax.pcolormesh(
    lons_raw, lats_raw, wave_filtered,
    cmap=plt.cm.viridis,
    vmin=0.0,
    vmax=0.1
)

fig.colorbar(im, ax=ax, orientation="horizontal")

st.pyplot(fig)
plt.close(fig)

# =========================
# UI
# =========================
st.write(st.session_state.current_time)

col1, col2, col3, col4, col5 = st.columns(5)

if col1.button("-6h"):
    st.session_state.current_time -= timedelta(hours=6)
    st.rerun()

if col2.button("-1h"):
    st.session_state.current_time -= timedelta(hours=1)
    st.rerun()

if col3.button("Teraz"):
    st.session_state.current_time = datetime.now(UTC).replace(
        minute=0, second=0, microsecond=0, tzinfo=None
    )
    st.rerun()

if col4.button("+1h"):
    st.session_state.current_time += timedelta(hours=1)
    st.rerun()

if col5.button("+6h"):
    st.session_state.current_time += timedelta(hours=6)
    st.rerun()

# ==============================================================================
# PODSUMOWANIE REZULTATÓW I ZAMKNIĘCIE (KONSOLA)
# ==============================================================================
print("\n" + "="*50)
print("PODSUMOWANIE DZIAŁANIA APLIKACJI")
print("="*50)
print(f"Status instalacji bibliotek: {'Zaktualizowano brakujące' if installed_any else 'Wszystkie obecne'}")
print(f"Wybrany punkt czasowy (UTC): {st.session_state.current_time}")
print(f"Czas pobierania i cięcia danych: {slice_load_time:.2f} sekundy")
print(f"Rozmiar wygenerowanej siatki danych: {wave_filtered.shape}")
print("="*50)
input("Naciśnij [ENTER], aby zakończyć działanie skryptu podsumowującego...")