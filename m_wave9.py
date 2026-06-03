import streamlit as st
import copernicusmarine
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta, UTC

# Konfiguracja strony Streamlit - układ dopasowany do telefonów
st.set_page_config(page_title="Prognoza Fal Bałtyku", layout="centered")
st.markdown("""
<style>
    /* Miejsce na Twój globalny styl CSS, jeśli z niego korzystasz */
</style>
""", unsafe_allow_html=True)

# --- USTAWIENIA DLA MORZA BAŁTYCKIEGO ---
MIN_LON, MAX_LON = 11.0, 15.0
MIN_LAT, MAX_LAT = 53.5, 56.5
DATASET_ID = "cmems_mod_bal_wav_anfc_PT1H-i"

# --- LOGOWANIE ---
def bezpieczne_logowanie():
    # Poświadczenia powinny być zapisane w środowisku (COPERNICUSMARINE_SERVICE_USERNAME i PASSWORD)
    copernicusmarine.login()

# --- NOWE, ULTRA-SZYBKIE KESZOWANIE BLOKU DANYCH (-12h do +48h) ---
@st.cache_data(show_spinner=False)
def pobierz_pelny_blok_danych(odniesienie_czasu):
    bezpieczne_logowanie()
    
    start_str = (odniesienie_czasu - timedelta(hours=12)).strftime('%Y-%m-%d %H:%M:%S')
    end_str = (odniesienie_czasu + timedelta(hours=48)).strftime('%Y-%m-%d %H:%M:%S')
    
    # Wykorzystanie nowego API open_dataset zintegrowanego z parametrami przycinania
    ds = copernicusmarine.open_dataset(
        dataset_id=DATASET_ID,
        longitude=[MIN_LON, MAX_LON],
        latitude=[MIN_LAT, MAX_LAT],
        time=[start_str, end_str],
        variables=["VHM0", "VTM02"]
    )
    
    # Wczytujemy do pamięci lokalnej przed zbuforowaniem
    return ds.load()

# --- INICJALIZACJA STANU SESJI (STATE) ---
# Zaokrąglamy aktualny czas bazowy do pełnej godziny
if 'base_time' not in st.session_state:
    st.session_state.base_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
if 'current_time' not in st.session_state:
    st.session_state.current_time = st.session_state.base_time
if 'prog_filtra' not in st.session_state:
    st.session_state.prog_filtra = 0.5

# --- JEDNORAZOWE POBRANIE DUŻEGO BLOKU ---
with st.spinner("Pobieranie pakietu danych (-12h / +48h)..."):
    dane = pobierz_pelny_blok_danych(st.session_state.base_time)

# --- BŁYSKAWICZNE WYCIĘCIE AKTUALNEJ GODZINY Z RAMU ---
wybrany_czas = st.session_state.current_time

# Metoda "nearest" zapobiega błędom w przypadku minimalnych przesunięć czasowych
aktualne_dane = dane.sel(time=wybrany_czas, method="nearest")

h_signif = aktualne_dane["VHM0"].values
t_mean = aktualne_dane["VTM02"].values
lons_raw = aktualne_dane["longitude"].values
lats_raw = aktualne_dane["latitude"].values

# --- MATEMATYKA (Wykonuje się natychmiast na tablicach NumPy) ---
with np.errstate(divide='ignore', invalid='ignore'):
    wave_steepness = h_signif / (1.56 * (t_mean**2))
    land_mask = np.where(np.isnan(h_signif), 1, np.nan)
    wave_filtered = np.where(h_signif >= st.session_state.prog_filtra, wave_steepness, np.nan)

# --- PALETY I KOLORY ---
kolor_ladu = "#6b798d"
kolor_strzalek = "#ffffff"
kolory_stromość = [(0.0, "#8ecae6"), (0.3, "#8ecae6"), (0.45, "#219ebc"), (0.6, "#023047"), (0.7, "#ffb703"), (0.8, "#fb8500"), (1.0, "#fb8500")]
cmap_stromość = LinearSegmentedColormap.from_list("stromość_custom", kolory_stromość)
cmap_vhm0 = plt.cm.viridis
cmap_vtm02 = plt.cm.plasma

# --- 1. GŁÓWNA MAPA (STROMOŚĆ FALI) ---
fig1, ax1 = plt.subplots(figsize=(7, 7))
fig1.subplots_adjust(top=1, bottom=0, left=0, right=1)
ax1.set_facecolor('#404040')
ax1.set_xlim(MIN_LON, MAX_LON)
ax1.set_ylim(MIN_LAT, MAX_LAT)
ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

ax1.pcolormesh(lons_raw, lats_raw, land_mask, cmap=LinearSegmentedColormap.from_list("lc", [kolor_ladu, kolor_ladu]), zorder=1)
im1 = ax1.pcolormesh(lons_raw, lats_raw, wave_filtered, cmap=cmap_stromość, vmin=0.0, vmax=0.1, zorder=2)
fig1.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.02, fraction=0.046, aspect=35)

plt.tight_layout()
st.pyplot(fig1)

# Wyświetlanie aktualnie sprawdzanego czasu
st.markdown(f"<h4 style='text-align: center;'>Wybrany czas: {wybrany_czas.strftime('%Y-%m-%d %H:%M')} UTC</h4>", unsafe_allow_html=True)

# --- PANEL STEROWANIA ---
st.markdown("</p><div></div>", unsafe_allow_html=True)
col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
if col_t1.button("-6h", use_container_width=True, key="time_btn_prev6"):
    st.session_state.current_time -= timedelta(hours=6)
    st.rerun()
if col_t2.button("-1h", use_container_width=True, key="time_btn_prev"):
    st.session_state.current_time -= timedelta(hours=1)
    st.rerun()
if col_t3.button("Teraz", use_container_width=True, key="time_btn_now"):
    st.session_state.current_time = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
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
    st.session_state.prog_filtra += 0.1
    st.rerun()

st.markdown("<div></div>", unsafe_allow_html=True)

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