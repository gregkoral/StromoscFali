import streamlit as st
import copernicusmarine
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta, UTC
import time

# Konfiguracja strony Streamlit
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
[data-testid="stHeader"] { display: none !important; }
[data-testid="stVerticalBlock"] > div { margin-bottom: 0rem !important; }
</style>
""", unsafe_allow_html=True)

# --- USTAWIENIA ---
MIN_LON, MAX_LON = 11.0, 15.0
MIN_LAT, MAX_LAT = 53.5, 56.5
DATASET_ID = "cmems_mod_bal_wav_anfc_PT1H-i"

# --- LOGOWANIE ---
def bezpieczne_logowanie():
    try:
        username = st.secrets["COPERNICUS_USERNAME"]
        password = st.secrets["COPERNICUS_PASSWORD"]
        copernicusmarine.login(username=username, password=password)
    except Exception as e:
        st.error(f"Błąd logowania: {e}")
        st.stop()

# --- CACHE (ZMIANA KLUCZOWA) ---
@st.cache_resource(show_spinner=False)
def pobierz_pelny_blok_danych(odniesienie_czasu):

    bezpieczne_logowanie()

    start_time = odniesienie_czasu - timedelta(hours=12)
    end_time = odniesienie_czasu + timedelta(hours=48)

    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        t0 = time.time()
        st.sidebar.write("1. open_dataset...")

        ds = copernicusmarine.open_dataset(
            dataset_id=DATASET_ID,
            start_datetime=start_str,
            end_datetime=end_str,
            minimum_longitude=MIN_LON,
            maximum_longitude=MAX_LON,
            minimum_latitude=MIN_LAT,
            maximum_latitude=MAX_LAT,
            variables=["VHM0", "VTM02", "VMDR_WW"]
        )

        st.sidebar.write(f"2. open OK ({time.time()-t0:.1f}s)")
        st.sidebar.write(ds.sizes)

        st.sidebar.write("3. load...")

        t1 = time.time()
        ds_loaded = ds.load()

        st.sidebar.write(f"4. load OK ({time.time()-t1:.1f}s)")

        try:
            st.sidebar.write(
                f"RAM: {ds_loaded.nbytes / 1024 / 1024:.1f} MB"
            )
        except:
            pass

        ds.close()
        return ds_loaded

    except Exception as e:
        st.error(f"Błąd Copernicus: {e}")
        st.stop()


# --- SESSION STATE ---
if 'base_time' not in st.session_state:
    st.session_state.base_time = datetime.now(UTC).replace(
        minute=0, second=0, microsecond=0, tzinfo=None
    )

if 'current_time' not in st.session_state:
    st.session_state.current_time = st.session_state.base_time

if 'prog_filtra' not in st.session_state:
    st.session_state.prog_filtra = 0.5


# --- CACHE CLEAR ---
if st.sidebar.button("🧹 Clear cache"):
    st.cache_resource.clear()
    st.rerun()


# --- POBIERANIE DANYCH ---
with st.spinner("Pobieranie pakietu danych..."):
    pelny_dataset = pobierz_pelny_blok_danych(st.session_state.base_time)

st.sidebar.write("Dataset OK")


# --- WYBÓR CZASU ---
wybrany_czas = st.session_state.current_time

try:
    wave_slice = pelny_dataset.sel(time=wybrany_czas, method='nearest')

    lons_raw = wave_slice['longitude'].values
    lats_raw = wave_slice['latitude'].values
    h_signif = wave_slice['VHM0'].values
    t_mean = wave_slice['VTM02'].values
    vmdr_ww = wave_slice['VMDR_WW'].values

except Exception as e:
    st.error("Błąd czasu w danych")
    st.stop()


# --- MATH ---
with np.errstate(divide='ignore', invalid='ignore'):
    wave_length = (9.81 * (t_mean ** 2)) / (2 * np.pi)
    wave_steepness = h_signif / wave_length

land_mask = np.where(np.isnan(h_signif), 1, np.nan)
wave_filtered = np.where(h_signif >= st.session_state.prog_filtra, wave_steepness, np.nan)


# --- MAPA ---
fig1, ax1 = plt.subplots(figsize=(7, 7))
ax1.set_facecolor('#404040')
ax1.set_xlim(MIN_LON, MAX_LON)
ax1.set_ylim(MIN_LAT, MAX_LAT)

ax1.pcolormesh(lons_raw, lats_raw, land_mask,
               cmap=LinearSegmentedColormap.from_list("lc", ["#6b798d", "#6b798d"]))

im1 = ax1.pcolormesh(lons_raw, lats_raw, wave_filtered,
                     cmap=plt.cm.viridis, vmin=0.0, vmax=0.1)

fig1.colorbar(im1, ax=ax1, orientation='horizontal')

st.pyplot(fig1)
plt.close(fig1)


# --- PANEL ---
st.write(f"{wybrany_czas} | filtr {st.session_state.prog_filtra}")


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


# --- DRUGI WYKRES ---
fig2, ax2 = plt.subplots(1, 2, figsize=(6, 3.75))

ax2[0].pcolormesh(lons_raw, lats_raw, h_signif, cmap=plt.cm.viridis)
ax2[1].pcolormesh(lons_raw, lats_raw, t_mean, cmap=plt.cm.plasma)

st.pyplot(fig2)
plt.close(fig2)