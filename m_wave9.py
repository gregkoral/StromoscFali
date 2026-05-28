import streamlit as st
import copernicusmarine
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta, UTC

# Konfiguracja strony Streamlit (musi być na samym początku skryptu)
st.set_page_config(page_title="Prognoza Stromości Fal", layout="wide")

# --- USTAWIENIA DLA MORZA BAŁTYCKIEGO ---
MIN_LON, MAX_LON = 10.0, 16.0
MIN_LAT, MAX_LAT = 53.0, 56.0
DATASET_ID = "cmems_mod_bal_wav_anfc_PT1H-i"

# Funkcja logowania - wywoływana przy każdym zapytaniu do sieci, bez dodatkowych parametrów
def bezpieczne_logowanie():
    try:
        username = st.secrets["COPERNICUS_USERNAME"]
        password = st.secrets["COPERNICUS_PASSWORD"]
        copernicusmarine.login(username=username, password=password)
    except Exception as e:
        st.error(f"Błąd autoryzacji Copernicus: {e}. Sprawdź konfigurację Secrets w panelu Streamlit.")
        st.stop()

# Keszujemy tylko bezpieczne, surowe tablice NumPy opakowane w słownik
@st.cache_data(show_spinner=False)
def pobierz_dane_godzinowe(wybrany_czas):
    bezpieczne_logowanie()
    
    # Tworzymy wąskie okno czasowe (1 godzina) wokół wybranego punktu
    start_str = (wybrany_czas - timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
    end_str = (wybrany_czas + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        ds = copernicusmarine.open_dataset(
            dataset_id=DATASET_ID, start_datetime=start_str, end_datetime=end_str,
            minimum_longitude=MIN_LON, maximum_longitude=MAX_LON,
            minimum_latitude=MIN_LAT, maximum_latitude=MAX_LAT
        )
        
        # Precyzyjne wycięcie najbliższej pełnej godziny
        wave_slice = ds.sel(time=wybrany_czas, method='nearest').load()
        
        # Ekstrakcja czystych danych numerycznych (odpornych na błędy klonowania w cache)
        dane = {
            "lon": wave_slice['longitude'].values,
            "lat": wave_slice['latitude'].values,
            "VHM0": wave_slice['VHM0'].values,
            "VTM02": wave_slice['VTM02'].values,
            "VMDR_WW": wave_slice['VMDR_WW'].values
        }
        return dane
    except Exception as e:
        st.error(f"Błąd pobierania danych z Copernicus Marine Service: {e}")
        st.stop()

# --- PASEK BOCZNY / INTERFEJS STEROWANIA ---
st.sidebar.title("Sterowanie prognozą")

# Inicjalizacja czasu w stanie sesji (z zaokrągleniem do pełnej godziny)
if 'current_time' not in st.session_state:
    st.session_state.current_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0, tzinfo=None)

# Przyciski szybkiego skoku czasu
col_prev, col_now, col_next = st.sidebar.columns(3)
if col_prev.button("⬅️ -1h"):
    st.session_state.current_time -= timedelta(hours=1)
if col_now.button("Teraz UTC"):
    st.session_state.current_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0, tzinfo=None)
if col_next.button("+1h ➡️"):
    st.session_state.current_time += timedelta(hours=1)

# Interaktywny kalendarz i suwak godzinowy
wybrana_data = st.sidebar.date_input("Wybierz dzień", st.session_state.current_time.date())
wybrana_godzina = st.sidebar.slider("Wybierz godzinę (UTC)", 0, 23, int(st.session_state.current_time.hour))

# Konsolidacja czasu na podstawie wprowadzonych widgetów
st.session_state.current_time = datetime.combine(wybrana_data, datetime.min.time()) + timedelta(hours=wybrana_godzina)
wybrany_czas = st.session_state.current_time

# Filtr wysokości fali
prog_filtra = st.sidebar.slider("Próg filtra wysokości fali (m)", 0.0, 5.0, 0.5, step=0.1)

# --- PRZETWARZANIE I MATEMATYKA (NUMPY) ---
with st.spinner("Pobieranie i przetwarzanie danych..."):
    data_dict = pobierz_dane_godzinowe(wybrany_czas)
    
    lons_raw = data_dict["lon"]
    lats_raw = data_dict["lat"]
    h_signif = data_dict["VHM0"]
    t_mean = data_dict["VTM02"]
    vmdr_ww = data_dict["VMDR_WW"]
    
    # Wyliczanie stromości fali
    with np.errstate(divide='ignore', invalid='ignore'):
        wave_length = (9.81 * (t_mean ** 2)) / (2 * np.pi)
        wave_steepness = h_signif / wave_length
        
    land_mask = np.where(np.isnan(h_signif), 1, np.nan)
    wave_filtered = np.where(h_signif >= prog_filtra, wave_steepness, np.nan)

# --- GENEROWANIE WYKRESÓW (MATPLOTLIB) ---
kolor_ladu = "#6b798d"  
kolor_strzalek = "#ffffff"

kolory_stromość = [(0.0, "#8ecae6"), (0.3, "#8ecae6"), (0.45, "#219ebc"), (0.6, "#023047"), (0.7, "#ffb703"), (0.8, "#fb8500"), (1.0, "#fb8500")]
cmap_stromość = LinearSegmentedColormap.from_list("stromość_custom", kolory_stromość)
cmap_vhm0 = plt.cm.viridis
cmap_vtm02 = plt.cm.plasma

fig = plt.figure(figsize=(12, 7))
gs = GridSpec(2, 3, figure=fig)

ax1 = fig.add_subplot(gs[:, :2])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 2])

# 1. Główna mapa: Stromość fali
ax1.set_facecolor('#404040')
ax1.pcolormesh(lons_raw, lats_raw, land_mask, cmap=LinearSegmentedColormap.from_list("lc", [kolor_ladu, kolor_ladu]), zorder=1)
im1 = ax1.pcolormesh(lons_raw, lats_raw, wave_filtered, cmap=cmap_stromość, vmin=0.0, vmax=0.1, zorder=2)
fig.colorbar(im1, ax=ax1, label='', pad=0.02, fraction=0.03)

# Nakładanie strzałek kierunku wiatru/fal (VMDR_WW)
try:
    s_lat, s_lon = 10, 12
    lons_sub, lats_sub = lons_raw[::s_lon], lats_raw[::s_lat]
    lon_grid, lat_grid = np.meshgrid(lons_sub, lats_sub)
    angles_deg = vmdr_ww[::s_lat, ::s_lon]
    angles_rad = np.deg2rad(90.0 - (np.round(angles_deg / 30.0) * 30.0) + 180)
    u, v = np.cos(angles_rad), np.sin(angles_rad)
    u, v = np.where(np.isnan(angles_deg), np.nan, u), np.where(np.isnan(angles_deg), np.nan, v)
    ax1.quiver(lon_grid, lat_grid, u, v, color=kolor_strzalek, scale=25, width=0.0020, headwidth=6, headlength=5, pivot='middle', zorder=3)
except:
    pass
ax1.set_title("STROMOŚĆ FALI", fontsize=10, fontweight='bold')

# 2. Mała mapa góra: Wysokość fali (VHM0)
ax2.set_facecolor('#202020')
ax2.pcolormesh(lons_raw, lats_raw, land_mask, cmap=LinearSegmentedColormap.from_list("lc", [kolor_ladu, kolor_ladu]))
im2 = ax2.pcolormesh(lons_raw, lats_raw, h_signif, cmap=cmap_vhm0, vmin=0.25, vmax=1.0)
fig.colorbar(im2, ax=ax2, label='VHM0 [m]', pad=0.02)
ax2.set_title("Wysokość fali (VHM0)", fontsize=9)

# 3. Mała mapa dół: Okres fali (VTM02)
ax3.set_facecolor('#202020')
ax3.pcolormesh(lons_raw, lats_raw, land_mask, cmap=LinearSegmentedColormap.from_list("lc", [kolor_ladu, kolor_ladu]))
im3 = ax3.pcolormesh(lons_raw, lats_raw, t_mean, cmap=cmap_vtm02, vmin=1.0, vmax=3.5)
fig.colorbar(im3, ax=ax3, label='VTM02 [s]', pad=0.02)
ax3.set_title("Okres fali (VTM02)", fontsize=9)

# Nagłówek wykresu i wyrównanie układu
fig.suptitle(f"Czas: {wybrany_czas.strftime('%Y-%m-%d %H:%M')} UTC | Filtr stromości: >{prog_filtra:.1f}m", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# --- RENDEROWANIE INTERFEJSU WWW ---
st.title("🌊 Interaktywna Prognoza Fal Bałtyku")
st.subheader(f"Wybrany czas: {wybrany_czas.strftime('%Y-%m-%d %H:%M')} UTC")
st.pyplot(fig)