import streamlit as st
import copernicusmarine
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta, UTC
import sys





# Konfiguracja strony Streamlit - układ dopasowany do telefonów
st.set_page_config(page_title="Prognoza Fal Bałtyku", layout="centered")

st.markdown("""
    <style>
    .block-container,
    div.stMainBlockContainer,
    section.main > div {
        padding-top: 0rem !important;
        padding-bottom: 1rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        max-width: 100% !important;
    }
    [data-testid="stHeader"] {
        display: none !important;
    }
    /* Domyślny gap między wszystkimi elementami */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockSeparator"],
    [data-testid="stVerticalBlock"] > div {
        margin-bottom: 0rem !important;
    }
    /* Odstęp PRZED panelem przycisków czasu */
    [data-testid="stVerticalBlock"] > div:has([data-testid="stHorizontalBlock"]:has(button[kind="secondary"])) {
        margin-top: 14px !important;
        margin-bottom: 14px !important;
    }
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        min-width: 0 !important;
        flex: 1 1 0 !important;
    }
    </style>
""", unsafe_allow_html=True)


# --- USTAWIENIA DLA MORZA BAŁTYCKIEGO ---
MIN_LON, MAX_LON = 11.0, 15.0
MIN_LAT, MAX_LAT = 53.5, 56.5
DATASET_ID = "cmems_mod_bal_wav_anfc_PT1H-i"
#DATASET_ID = "cmems_mod_glo_wav_anfc_0.083deg_PT1H-i"


# --- NOWE, ULTRA-SZYBKIE KESZOWANIE BLOKU DANYCH (-12h do +48h) ---
@st.cache_data(show_spinner=False)
def pobierz_pelny_blok_danych(odniesienie_czasu):
    status = st.empty()
    
    # Pobieranie danych logowania z Secrets
    try:
        username = st.secrets["COPERNICUS_USERNAME"]
        password = st.secrets["COPERNICUS_PASSWORD"]
    except Exception as e:
        st.error(f"Błąd odczytu st.secrets: {e}")
        st.stop()
    
    # Definiujemy ramy czasowe: 12 godzin w tył, 48 godzin w przód
    start_time = odniesienie_czasu - timedelta(hours=12)
    end_time = odniesienie_czasu + timedelta(hours=48)
    
    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        status.markdown(f"⏳ **Krok 1/3:** Otwieranie stabilnego połączenia NetCDF...<br><small>{DATASET_ID}</small>", unsafe_allow_html=True)
        print(f"LOG: Wywoływanie open_dataset (NetCDF) dla {DATASET_ID}...", flush=True)
        
        # Wymuszamy service="netcdf" aby ominąć błędy asynchronicznego pobierania zarr3
        ds = copernicusmarine.open_dataset(
            dataset_id=DATASET_ID, 
            username=username,
            password=password,
            start_datetime=start_str, 
            end_datetime=end_str,
            minimum_longitude=MIN_LON, 
            maximum_longitude=MAX_LON,
            minimum_latitude=MIN_LAT, 
            maximum_latitude=MAX_LAT,
            variables=["VHM0", "VTM02", "VMDR_WW"],
            service="netcdf"
        )
        
        status.markdown("⏳ **Krok 2/3:** Pobieranie i ładowanie struktur do pamięci RAM...")
        print("LOG: open_dataset zakończone. Rozpoczynanie ds.load()...", flush=True)
        
        # Ładujemy cały przefiltrowany dataset bezpośrednio do RAMu serwera
        ds_loaded = ds.load()
        
        status.markdown("⏳ **Krok 3/3:** Zamykanie strumieni sieciowych...")
        print("LOG: ds.load() zakończone pomyślnie. Zamykanie obiektu...", flush=True)
        ds.close()
        
        status.empty() # Czyszczenie komunikatu po sukcesie
        print("LOG: Pobieranie danych zakończone sukcesem.", flush=True)
        return ds_loaded
    except Exception as e:
        print(f"LOG BŁĄD POBIERANIA: {e}", flush=True)
        status.markdown(f"❌ **Błąd:** {e}")
        st.error(f"Błąd pobierania bloku danych z Copernicus: {e}")
        st.stop()

# --- INICJALIZACJA STANU SESJI (STATE) ---
# Zaokrąglamy aktualny czas bazowy do pełnej godziny
if 'base_time' not in st.session_state:
    st.session_state.base_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0, tzinfo=None)

if 'current_time' not in st.session_state:
    st.session_state.current_time = st.session_state.base_time

if 'prog_filtra' not in st.session_state:
    st.session_state.prog_filtra = 0.5

# --- JEDNORAZOWE POBRANIE DUŻEGO BLOKU (idzie do cache na podstawie base_time) ---
pelny_dataset = pobierz_pelny_blok_danych(st.session_state.base_time)

# --- BŁYSKAWICZNE WYCIĘCIE AKTUALNEY GODZINY Z RAMU ---
wybrany_czas = st.session_state.current_time

try:
    # Wycinamy tylko jedną klatkę czasową z gotowego obiektu w pamięci
    wave_slice = pelny_dataset.sel(time=wybrany_czas, method='nearest')
    
    lons_raw = wave_slice['longitude'].values
    lats_raw = wave_slice['latitude'].values
    h_signif = wave_slice['VHM0'].values
    t_mean = wave_slice['VTM02'].values
    vmdr_ww = wave_slice['VMDR_WW'].values
except Exception as e:
    st.error(f"Wybrana godzina ({wybrany_czas.strftime('%Y-%m-%d %H:%M')}) wykracza poza zakres pobranej pamięci podręcznej.")
    st.button("Zresetuj do teraz", on_click=lambda: st.session_state.update(current_time=st.session_state.base_time))
    st.stop()

# --- MATEMATYKA (Wykonuje się natychmiast na tablicach NumPy) ---
with np.errstate(divide='ignore', invalid='ignore'):
    wave_length = (9.81 * (t_mean ** 2)) / (2 * np.pi)
    wave_steepness = h_signif / wave_length
    
land_mask = np.where(np.isnan(h_signif), 1, np.nan)
wave_filtered = np.where(h_signif >= st.session_state.prog_filtra, wave_steepness, np.nan)

# --- PALETY I KOLORY ---
kolor_ladu = "#6b798d"  
kolor_strzalek = "#ffffff"
kolory_stromość = [(0.0, "#8ecae6"), (0.3, "#8ecae6"), (0.45, "#219ebc"), (0.6, "#023047"), (0.7, "#ffb703"), (0.8, "#fb8500"), (1.0, "#fb8500")]
cmap_stromość = LinearSegmentedColormap.from_list("stromość_custom", kolory_stromość)
cmap_vhm0 = plt.cm.viridis
cmap_vtm02 = plt.cm.plasma

# --- GLOBALNY STYL CSS ---


# 1. GŁÓWNA MAPA (STROMOŚĆ FALI)
fig1, ax1 = plt.subplots(figsize=(7, 7))
fig1.subplots_adjust(top=1, bottom=0, left=0, right=1)
ax1.set_facecolor('#404040')
ax1.set_xlim(MIN_LON, MAX_LON)
ax1.set_ylim(MIN_LAT, MAX_LAT)

ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

ax1.pcolormesh(lons_raw, lats_raw, land_mask, cmap=LinearSegmentedColormap.from_list("lc", [kolor_ladu, kolor_ladu]), zorder=1)
im1 = ax1.pcolormesh(lons_raw, lats_raw, wave_filtered, cmap=cmap_stromość, vmin=0.0, vmax=0.1, zorder=2)

fig1.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.02, fraction=0.046, aspect=35)

try:
    s_lat, s_lon = 10, 12
    lons_sub, lats_sub = lons_raw[::s_lon], lats_raw[::s_lat]
    lon_grid, lat_grid = np.meshgrid(lons_sub, lats_sub)
    angles_deg = vmdr_ww[::s_lat, ::s_lon]
    angles_rad = np.deg2rad(90.0 - (np.round(angles_deg / 30.0) * 30.0) + 180)
    u, v = np.cos(angles_rad), np.sin(angles_rad)
    u, v = np.where(np.isnan(angles_deg), np.nan, u), np.where(np.isnan(angles_deg), np.nan, v)
    ax1.quiver(lon_grid, lat_grid, u, v, color=kolor_strzalek, scale=25, width=0.0025, headwidth=6, headlength=5, pivot='middle', zorder=3)
except:
    pass

plt.tight_layout()
st.pyplot(fig1)

st.markdown(
    f"<p style='text-align: center; color: #888888; font-size: 20px; margin-top: 4px; margin-bottom: 8px; font-family: sans-serif;'>"
    f"{wybrany_czas.strftime('%Y-%m-%d %H:%M')} UTC | Filtr: > {st.session_state.prog_filtra:.1f}m"
    f"</p>", 
    unsafe_allow_html=True
)


# --- PANEL STEROWANIA ---

st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)

if col_t1.button("-6h", use_container_width=True, key="time_btn_prev6"):
    st.session_state.current_time -= timedelta(hours=6)
    st.rerun()
    
if col_t2.button("-1h", use_container_width=True, key="time_btn_prev"):
    st.session_state.current_time -= timedelta(hours=1)
    st.rerun()

if col_t3.button("Teraz", use_container_width=True, key="time_btn_now"):
    st.session_state.current_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    st.rerun()

if col_t4.button("+1h", use_container_width=True, key="time_btn_next"):
    st.session_state.current_time += timedelta(hours=1)
    st.rerun()
    
if col_t5.button("+6h", use_container_width=True, key="time_btn_next6"):
    st.session_state.current_time += timedelta(hours=6)
    st.rerun()



col_f1, col_f2, col_f3 = st.columns(3)

if col_f1.button("-0.1m", use_container_width=True):
    st.session_state.prog_filtra = max(0.0, st.session_state.prog_filtra - 0.1)
    st.rerun()

if col_f2.button("0.5m", use_container_width=True):
    st.session_state.prog_filtra = 0.5
    st.rerun()

if col_f3.button("+0.1m", use_container_width=True):
    st.session_state.prog_filtra = min(5.0, st.session_state.prog_filtra + 0.1)
    st.rerun()

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)


# --- 3. ZESTAW WIELOWYKRESOWY NA DOLE ---
fig_desktop, (ax2_d, ax3_d) = plt.subplots(1, 2, figsize=(6, 3.75))

ax2_d.set_facecolor('#202020')
ax2_d.axis('off')
ax2_d.set_xlim(MIN_LON, MAX_LON)
ax2_d.set_ylim(MIN_LAT, MAX_LAT)
ax2_d.pcolormesh(lons_raw, lats_raw, land_mask, cmap=LinearSegmentedColormap.from_list("lc", [kolor_ladu, kolor_ladu]))
im2_d = ax2_d.pcolormesh(lons_raw, lats_raw, h_signif, cmap=cmap_vhm0, vmin=0.25, vmax=1.0)
fig_desktop.colorbar(im2_d, ax=ax2_d, orientation='horizontal', pad=0.08, fraction=0.046, aspect=20)
ax2_d.set_title("Wysokość fali (VHM0)", fontsize=11, color="black", pad=8)

ax3_d.set_facecolor('#202020')
ax3_d.axis('off')
ax3_d.set_xlim(MIN_LON, MAX_LON)
ax3_d.set_ylim(MIN_LAT, MAX_LAT)
ax3_d.pcolormesh(lons_raw, lats_raw, land_mask, cmap=LinearSegmentedColormap.from_list("lc", [kolor_ladu, kolor_ladu]))
im3_d = ax3_d.pcolormesh(lons_raw, lats_raw, t_mean, cmap=cmap_vtm02, vmin=1.0, vmax=3.5)
fig_desktop.colorbar(im3_d, ax=ax3_d, orientation='horizontal', pad=0.08, fraction=0.046, aspect=20)
ax3_d.set_title("Okres fali (VTM02)", fontsize=11, color="black", pad=8)

fig_desktop.tight_layout()
st.pyplot(fig_desktop)