import subprocess
import sys

# --- AUTOMATYCZNA INSTALACJA BIBLIOTEK ---
wymagane_biblioteki = {
    "streamlit": "streamlit",
    "copernicusmarine": "copernicusmarine",
    "xarray": "xarray",
    "netcdf4": "netcdf4",
    "matplotlib": "matplotlib",
    "numpy": "numpy"
}

for modul, pakiet in wymagane_biblioteki.items():
    try:
        __import__(modul)
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pakiet])
        except Exception as e:
            sys.exit(1)
# ------------------------------------------

import streamlit as st
import copernicusmarine
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta

# Konfiguracja strony Streamlit (musi być na samym początku)
st.set_page_config(page_title="Prognoza Stromości Fal", layout="wide")

# --- USTAWIENIA DLA MORZA BAŁTYCKIEGO ---
MIN_LON, MAX_LON = 10.0, 16.0
MIN_LAT, MAX_LAT = 53.0, 56.0
DATASET_ID = "cmems_mod_bal_wav_anfc_PT1H-i"

# --- CACHOWANIE DANYCH W RAMIE SERWERA ---
# Dzięki temu serwer pobiera dane z Copernicus tylko wtedy, gdy zmienimy zakres czasu.
@st.cache_data(show_spinner=False)
def pobierz_paczke_danych(start_str, end_str):
    try:
        # Próba automatycznego logowania (wykorzystuje plik konfiguracyjny lub zmienne środowiskowe)
        copernicusmarine.login()
    except Exception:
        pass
    
    ds = copernicusmarine.open_dataset(
        dataset_id=DATASET_ID, start_datetime=start_str, end_datetime=end_str,
        minimum_longitude=MIN_LON, maximum_longitude=MAX_LON,
        minimum_latitude=MIN_LAT, maximum_latitude=MAX_LAT
    ).load()
    return ds

# --- PASEK BOCZNY / INTERFEJS STEROWANIA ---
st.sidebar.title("Sterowanie prognozą")

# Inicjalizacja stanu sesji dla czasu (żeby przyciski mogły go zmieniać)
if 'current_time' not in st.session_state:
    st.session_state.current_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

# Przyciski do szybkiej zmiany czasu
col_prev, col_now, col_next = st.sidebar.columns(3)
if col_prev.button("⬅️ -1h"):
    st.session_state.current_time -= timedelta(hours=1)
if col_now.button("Teraz UTC"):
    st.session_state.current_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
if col_next.button("+1h ➡️"):
    st.session_state.current_time += timedelta(hours=1)

# Dokładny wybór dnia i godziny
wybrana_data = st.sidebar.date_input("Wybierz dzień", st.session_state.current_time.date())
wybrana_godzina = st.sidebar.slider("Wybierz godzinę (UTC)", 0, 23, int(st.session_state.current_time.hour))

# Aktualizacja czasu na podstawie widgetów
st.session_state.current_time = datetime.combine(wybrana_data, datetime.min.time()) + timedelta(hours=wybrana_godzina)

# Suwak do progu filtra
prog_filtra = st.sidebar.slider("Próg filtra wysokości fali (m)", 0.0, 5.0, 0.5, step=0.1)

# --- LOGIKA POBIERANIA I PRZETWARZANIA ---
wybrany_czas = st.session_state.current_time

# Definiujemy szeroki bufor cache (12h wstecz, 48h w przód), aby przełączanie godzin było błyskawiczne
cache_start = wybrany_czas - timedelta(hours=12)
cache_end = wybrany_czas + timedelta(hours=48)
start_str = cache_start.strftime("%Y-%m-%d %H:%M:%S")
end_str = cache_end.strftime("%Y-%m-%d %H:%M:%S")

with st.spinner("Pobieranie/Ładowanie danych z Copernicus Marine..."):
    try:
        cached_dataset = pobierz_paczke_danych(start_str, end_str)
        wave_slice = cached_dataset.sel(time=wybrany_czas, method='nearest')
        
        h_signif_raw = wave_slice['VHM0']
        t_mean_raw = wave_slice['VTM02']
        vmdr_ww_raw = wave_slice['VMDR_WW']
        
        with np.errstate(divide='ignore', invalid='ignore'):
            wave_length = (9.81 * (t_mean_raw ** 2)) / (2 * np.pi)
            wave_steepness = h_signif_raw / wave_length
            
        land_mask = xr.where(np.isnan(h_signif_raw), 1, np.nan)
        wave_filtered = wave_steepness.where(h_signif_raw >= prog_filtra)
        
    except Exception as e:
        st.error(f"Błąd podczas ładowania danych: {e}")
        st.stop()

# --- GENEROWANIE WYKRESU MATPLOTLIB ---
kolor_ladu = "#6b798d"  
kolor_strzalek = "#ffffff"

kolory_stromość = [(0.0, "#8ecae6"), (0.3, "#8ecae6"), (0.45, "#219ebc"), (0.6, "#023047"), (0.7, "#ffb703"), (0.8, "#fb8500"), (1.0, "#fb8500")]
cmap_stromość = LinearSegmentedColormap.from_list("stromość_custom", kolory_stromość)
cmap_vhm0 = plt.cm.viridis
cmap_vtm02 = plt.cm.plasma

# Tworzenie figury dopasowanej do wyświetlania na stronie
fig = plt.figure(figsize=(12, 7))
gs = GridSpec(2, 3, figure=fig)

ax1 = fig.add_subplot(gs[:, :2])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 2])

# Rysowanie głównej mapy (AX1)
ax1.set_facecolor('#404040')
land_mask.plot(ax=ax1, cmap=LinearSegmentedColormap.from_list("lc", [kolor_ladu, kolor_ladu]), add_colorbar=False, add_labels=False, zorder=1)
im1 = wave_filtered.plot(ax=ax1, cmap=cmap_stromość, add_colorbar=False, vmin=0.0, vmax=0.1, zorder=2, add_labels=False)
fig.colorbar(im1, ax=ax1, label='', pad=0.02, fraction=0.03)

# Strzałki kierunku fal
try:
    lon_dim = 'longitude' if 'longitude' in vmdr_ww_raw.dims else 'lon'
    lat_dim = 'latitude' if 'latitude' in vmdr_ww_raw.dims else 'lat'
    s_lat, s_lon = 10, 12
    lons, lats = vmdr_ww_raw[lon_dim].values[::s_lon], vmdr_ww_raw[lat_dim].values[::s_lat]
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    angles_deg = vmdr_ww_raw.values[::s_lat, ::s_lon]
    angles_rad = np.deg2rad(90.0 - (np.round(angles_deg / 30.0) * 30.0) + 180)
    u, v = np.cos(angles_rad), np.sin(angles_rad)
    u, v = np.where(np.isnan(angles_deg), np.nan, u), np.where(np.isnan(angles_deg), np.nan, v)
    ax1.quiver(lon_grid, lat_grid, u, v, color=kolor_strzalek, scale=25, width=0.0020, headwidth=6, headlength=5, pivot='middle', zorder=3)
except:
    pass
ax1.set_title("STROMOŚĆ FALI", fontsize=10, fontweight='bold')

# Rysowanie mapy VHM0 (AX2)
ax2.set_facecolor('#202020')
land_mask.plot(ax=ax2, cmap=LinearSegmentedColormap.from_list("lc", [kolor_ladu, kolor_ladu]), add_colorbar=False, add_labels=False)
im2 = h_signif_raw.plot(ax=ax2, cmap=cmap_vhm0, add_colorbar=False, vmin=0.25, vmax=1.0, add_labels=False)
fig.colorbar(im2, ax=ax2, label='VHM0 [m]', pad=0.02)
ax2.set_title("Wysokość fali (VHM0)", fontsize=9)

# Rysowanie mapy VTM02 (AX3)
ax3.set_facecolor('#202020')
land_mask.plot(ax=ax3, cmap=LinearSegmentedColormap.from_list("lc", [kolor_ladu, kolor_ladu]), add_colorbar=False, add_labels=False)
im3 = t_mean_raw.plot(ax=ax3, cmap=cmap_vtm02, add_colorbar=False, vmin=1.0, vmax=3.5, add_labels=False)
fig.colorbar(im3, ax=ax3, label='VTM02 [s]', pad=0.02)
ax3.set_title("Okres fali (VTM02)", fontsize=9)

# Tytuł główny wykresu
fig.suptitle(f"Czas: {wybrany_czas.strftime('%Y-%m-%d %H:%M')} UTC | Filtr stromości: >{prog_filtra:.1f}m", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# --- WYŚWIETLENIE NA STRONIE ---
st.title("🌊 Interaktywna Prognoza Fal Bałtyku")
st.subheader(f"Wybrany czas: {wybrany_czas.strftime('%Y-%m-%d %H:%M')} UTC")

# Renderowanie wykresu matplotlib jako responsywnego elementu HTML
st.pyplot(fig)