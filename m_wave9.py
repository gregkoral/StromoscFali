import streamlit as st
import copernicusmarine
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta, UTC

# Konfiguracja strony Streamlit - układ dopasowany do telefonów
st.set_page_config(page_title="Prognoza Fal Bałtyku", layout="centered")

# --- USTAWIENIA DLA MORZA BAŁTYCKIEGO ---
MIN_LON, MAX_LON = 11.0, 15.0
MIN_LAT, MAX_LAT = 53.5, 56.5
DATASET_ID = "cmems_mod_bal_wav_anfc_PT1H-i"

# Funkcja logowania
def bezpieczne_logowanie():
    try:
        username = st.secrets["COPERNICUS_USERNAME"]
        password = st.secrets["COPERNICUS_PASSWORD"]
        copernicusmarine.login(username=username, password=password)
    except Exception as e:
        st.error(f"Błąd autoryzacji Copernicus: {e}. Sprawdź konfigurację Secrets.")
        st.stop()

# Keszowanie danych
@st.cache_data(show_spinner=False, ttl=3600)  # ttl=3600 odświeży cache automatycznie po godzinie
def pobierz_paczke_danych():
    bezpieczne_logowanie()
    
    # Wyznaczamy punkt odniesienia (teraz) i zakres -12h do +48h
    teraz = datetime.now(UTC).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    start_forecast = teraz - timedelta(hours=12)
    end_forecast = teraz + timedelta(hours=48)
    
    start_str = start_forecast.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_forecast.strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Pobieramy cały trójwymiarowy blok (Lon, Lat, Time) za jednym razem
        ds = copernicusmarine.open_dataset(
            dataset_id=DATASET_ID, 
            start_datetime=start_str, 
            end_datetime=end_str,
            minimum_longitude=MIN_LON, maximum_longitude=MAX_LON,
            minimum_latitude=MIN_LAT, maximum_latitude=MAX_LAT
        )
        
        # Bardzo ważne: .load() wymusza pobranie z chmury do pamięci RAM całego wyciętego chunk'a
        return ds.load()
    except Exception as e:
        st.error(f"Błąd pobierania zbiorczej paczki danych: {e}")
        st.stop()

# --- INICJALIZACJA STANU SESJI (STATE) ---
if 'current_time' not in st.session_state:
    st.session_state.current_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0, tzinfo=None)

if 'prog_filtra' not in st.session_state:
    st.session_state.prog_filtra = 0.5

wybrany_czas = st.session_state.current_time

# --- POBRANIE I MATEMATYKA ---
with st.spinner("Inicjalizacja i pobieranie prognozy (-12h do +48h)..."):
    # To wywoła się RAZ na godzinę (lub przy pierwszym wejściu użytkownika)
    pelny_zbior_danych = pobierz_paczke_danych()

# Wycięcie konkretnej godziny dzieje się już lokalnie w pamięci RAM (błyskawicznie)
try:
    wave_slice = pelny_zbior_danych.sel(time=wybrany_czas, method='nearest')
    
    lons_raw = wave_slice['longitude'].values
    lats_raw = wave_slice['latitude'].values
    h_signif = wave_slice['VHM0'].values
    t_mean = wave_slice['VTM02'].values
    vmdr_ww = wave_slice['VMDR_WW'].values
except Exception as e:
    st.error(f"Wybrana godzina ({wybrany_czas}) znajduje się poza zakresem załadowanej prognozy.")
    st.stop()

# Dalej idzie Twoja czysta matematyka (wyliczanie stromości, maskowanie lądu):
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

# --- CSS ROZCIĄGAJĄCE STRONĘ NA 100% (BEZ MARGINESÓW I PADDINGÓW) ---
st.markdown(
    """
    <style>
    /* Zerowanie marginesów głównego kontenera Streamit dla efektu Edge-to-Edge */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 1rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        max-width: 100% !important;
    }
    /* Ukrycie systemowego paska nagłówka Streamlita */
    [data-testid="stHeader"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 1. GŁÓWNA MAPA (STROMOŚĆ FALI)
fig1, ax1 = plt.subplots(figsize=(10, 10))
ax1.set_facecolor('#404040')

# Sztywne wymuszenie granic kadrowania na wykresie
ax1.set_xlim(MIN_LON, MAX_LON)
ax1.set_ylim(MIN_LAT, MAX_LAT)

# --- UKRYCIE ETYKIET OSI (SZEROKOŚĆ I DŁUGOŚĆ GEO) ---
ax1.tick_params(
    axis='both',       # Dotyczy obu osi
    which='both',      
    bottom=False,      # Usuwa kreski podziałki na dole
    top=False,         # Usuwa kreski podziałki na górze
    left=False,        # Usuwa kreski podziałki po lewej
    right=False,       # Usuwa kreski podziałki po prawej
    labelbottom=False, # Ukrywa numery długości geo (X)
    labelleft=False    # Ukrywa numery szerokości geo (Y)
)

ax1.pcolormesh(lons_raw, lats_raw, land_mask, cmap=LinearSegmentedColormap.from_list("lc", [kolor_ladu, kolor_ladu]), zorder=1)
im1 = ax1.pcolormesh(lons_raw, lats_raw, wave_filtered, cmap=cmap_stromość, vmin=0.0, vmax=0.1, zorder=2)

# Pozioma skala na 100% szerokości pod wykresem
fig1.colorbar(
    im1, 
    ax=ax1, 
    orientation='horizontal',  
    pad=0.02,                  
    fraction=0.046,            
    aspect=35                  
)

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
#ax1.set_title("Stromość fali + Kierunek fali wiatrowej", fontsize=12, pad=10)
plt.tight_layout()
st.pyplot(fig1)

st.markdown(
    f"<p style='text-align: center; color: #888888; font-size: 20px; margin-top: 4px; margin-bottom: 4px; font-family: sans-serif;'>"
    f"{wybrany_czas.strftime('%Y-%m-%d %H:%M')} UTC | Filtr: > {st.session_state.prog_filtra:.1f}m"
    f"</p>", 
    unsafe_allow_html=True
)


# --- NATYWNY I ELASTYCZNY PANEL STEROWANIA ---
col_t1, col_t2, col_t3 = st.columns(3)

if col_t1.button("-1h", use_container_width=True):
    st.session_state.current_time -= timedelta(hours=1)
    st.rerun()

if col_t2.button("Teraz", use_container_width=True):
    st.session_state.current_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0, tzinfo=None)
    st.rerun()

if col_t3.button("+1h", use_container_width=True):
    st.session_state.current_time += timedelta(hours=1)
    st.rerun()


col_f1, col_f2, col_f3 = st.columns(3)

if col_f1.button("-0.1m", use_container_width=True):
    st.session_state.prog_filtra = max(0.0, st.session_state.prog_filtra - 0.1)
    st.rerun()

if col_f2.button("Reset", use_container_width=True):
    st.session_state.prog_filtra = 0.5
    st.rerun()

if col_f3.button("+0.1m", use_container_width=True):
    st.session_state.prog_filtra = min(5.0, st.session_state.prog_filtra + 0.1)
    st.rerun()



# 3. DWIE MAŁE MAPKI NA SAMYM DOLE (ZOPTYMALIZOWANA GĘSTOŚĆ)
col_map1, col_map2 = st.columns(2)

# Definiujemy krok próbkowania danych dla małych mapek (np. co 3 lub co 4 piksel)
# Im większa liczba, tym szybsze renderowanie dolnych wykresów
skok_dol = 3  

# Przycinamy tablice dwuwymiarowe (oraz osie) dla dolnych wykresów
# Jeśli Twoje lons_raw/lats_raw są 1D, używamy pcolormesh na odchudzonych danych:
lons_sub_dol = lons_raw[::skok_dol]
lats_sub_dol = lats_raw[::skok_dol]
h_signif_sub = h_signif[::skok_dol, ::skok_dol]
t_mean_sub = t_mean[::skok_dol, ::skok_dol]
land_mask_sub = land_mask[::skok_dol, ::skok_dol]

with col_map1:
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.set_facecolor('#202020')
    ax2.axis('off')
    
    ax2.set_xlim(MIN_LON, MAX_LON)
    ax2.set_ylim(MIN_LAT, MAX_LAT)
    
    # Rysujemy na odchudzonych danych (_sub)
    ax2.pcolormesh(lons_sub_dol, lats_sub_dol, land_mask_sub, cmap=LinearSegmentedColormap.from_list("lc", [kolor_ladu, kolor_ladu]))
    im2 = ax2.pcolormesh(lons_sub_dol, lats_sub_dol, h_signif_sub, cmap=cmap_vhm0, vmin=0.25, vmax=1.0)
    fig2.colorbar(im2, ax=ax2, label='', pad=0.02)
    ax2.set_title("Wysokość fali (VHM0)", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2)

with col_map2:
    fig3, ax3 = plt.subplots(figsize=(4, 4))
    ax3.set_facecolor('#202020')
    ax3.axis('off')
    
    ax3.set_xlim(MIN_LON, MAX_LON)
    ax3.set_ylim(MIN_LAT, MAX_LAT)
    
    # Rysujemy na odchudzonych danych (_sub)
    ax3.pcolormesh(lons_sub_dol, lats_sub_dol, land_mask_sub, cmap=LinearSegmentedColormap.from_list("lc", [kolor_ladu, kolor_ladu]))
    im3 = ax3.pcolormesh(lons_sub_dol, lats_sub_dol, t_mean_sub, cmap=cmap_vtm02, vmin=1.0, vmax=3.5)
    fig3.colorbar(im3, ax=ax3, label='', pad=0.02)
    ax3.set_title("Okres fali (VTM02)", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig3)