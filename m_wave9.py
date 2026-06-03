import streamlit as st
import copernicusmarine
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="Prognoza Fal Bałtyku", layout="centered")

# --- USTAWIENIA DLA MORZA BAŁTYCKIEGO ---
MIN_LON, MAX_LON = 11.0, 15.0
MIN_LAT, MAX_LAT = 53.5, 56.5

DATASET_ID = "cmems_mod_bal_wav_anfc_PT1H-i"


# --- POBIERANIE DANYCH ---
@st.cache_data(show_spinner=False)
def pobierz_pelny_blok_danych(odniesienie_czasu):

    start_time = odniesienie_czasu - timedelta(hours=12)
    end_time = odniesienie_czasu + timedelta(hours=48)

    try:
        ds = copernicusmarine.open_dataset(
            dataset_id=DATASET_ID,
            username=st.secrets["COPERNICUS_USERNAME"],
            password=st.secrets["COPERNICUS_PASSWORD"],
        )

        ds = ds.sel(
            time=slice(start_time, end_time),
            longitude=slice(MIN_LON, MAX_LON),
            latitude=slice(MIN_LAT, MAX_LAT),
        )

        ds = ds.load()
        return ds

    except Exception as e:
        st.error(f"Błąd Copernicus: {e}")
        st.stop()


# --- SESSION STATE ---
if "base_time" not in st.session_state:
    st.session_state.base_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

if "current_time" not in st.session_state:
    st.session_state.current_time = st.session_state.base_time

if "prog_filtra" not in st.session_state:
    st.session_state.prog_filtra = 0.5


# --- POBRANIE DANYCH ---
with st.spinner("Pobieranie pakietu danych (-12h / +48h)..."):
    pelny_dataset = pobierz_pelny_blok_danych(st.session_state.base_time)


# --- WYBÓR CZASU ---
wybrany_czas = st.session_state.current_time

try:
    wave_slice = pelny_dataset.sel(time=wybrany_czas, method="nearest")

    # 🔥 COPERNICUS RÓŻNE SCHEMATY OSI
    if "longitude" in wave_slice.coords:
        lons_raw = wave_slice["longitude"].values
    else:
        lons_raw = wave_slice["lon"].values

    if "latitude" in wave_slice.coords:
        lats_raw = wave_slice["latitude"].values
    else:
        lats_raw = wave_slice["lat"].values

    h_signif = wave_slice["VHM0"].values
    t_mean = wave_slice["VTM02"].values
    vmdr_ww = wave_slice["VMDR_WW"].values

except Exception as e:
    st.error(f"Brak danych dla wybranej godziny: {e}")
    st.stop()


# --- MATEMATYKA ---
with np.errstate(divide="ignore", invalid="ignore"):
    wave_length = (9.81 * (t_mean ** 2)) / (2 * np.pi)
    wave_steepness = h_signif / wave_length

land_mask = np.where(np.isnan(h_signif), 1, np.nan)
wave_filtered = np.where(h_signif >= st.session_state.prog_filtra, wave_steepness, np.nan)


# --- KOLORY ---
kolor_ladu = "#6b798d"
kolor_strzalek = "#ffffff"

kolory_stromość = [
    (0.0, "#8ecae6"),
    (0.3, "#8ecae6"),
    (0.45, "#219ebc"),
    (0.6, "#023047"),
    (0.7, "#ffb703"),
    (0.8, "#fb8500"),
    (1.0, "#fb8500"),
]

cmap_stromość = LinearSegmentedColormap.from_list("stromość_custom", kolory_stromość)
cmap_vhm0 = plt.cm.viridis
cmap_vtm02 = plt.cm.plasma


# --- MAPA GŁÓWNA ---
fig1, ax1 = plt.subplots(figsize=(7, 7))
fig1.subplots_adjust(top=1, bottom=0, left=0, right=1)

ax1.set_facecolor("#404040")
ax1.set_xlim(MIN_LON, MAX_LON)
ax1.set_ylim(MIN_LAT, MAX_LAT)

ax1.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

ax1.pcolormesh(lons_raw, lats_raw, land_mask,
               cmap=LinearSegmentedColormap.from_list("lc", [kolor_ladu, kolor_ladu]))

im1 = ax1.pcolormesh(lons_raw, lats_raw, wave_filtered,
                     cmap=cmap_stromość, vmin=0.0, vmax=0.1)

fig1.colorbar(im1, ax=ax1, orientation="horizontal", pad=0.02, fraction=0.046, aspect=35)


# --- STRZAŁKI ---
try:
    s_lat, s_lon = 10, 12

    lons_sub = lons_raw[::s_lon]
    lats_sub = lats_raw[::s_lat]

    lon_grid, lat_grid = np.meshgrid(lons_sub, lats_sub)

    angles_deg = vmdr_ww[::s_lat, ::s_lon]
    angles_rad = np.deg2rad(90.0 - (np.round(angles_deg / 30.0) * 30.0) + 180)

    u = np.cos(angles_rad)
    v = np.sin(angles_rad)

    u = np.where(np.isnan(angles_deg), np.nan, u)
    v = np.where(np.isnan(angles_deg), np.nan, v)

    ax1.quiver(lon_grid, lat_grid, u, v,
               color=kolor_strzalek,
               scale=25, width=0.0025,
               headwidth=6, headlength=5)

except:
    pass


st.pyplot(fig1)


# --- INFO ---
st.markdown(
    f"<div style='text-align:center;color:gray;'>"
    f"{wybrany_czas.strftime('%Y-%m-%d %H:%M')} UTC | "
    f"Filtr: {st.session_state.prog_filtra:.1f} m"
    f"</div>",
    unsafe_allow_html=True
)


# --- STEROWANIE ---
col1, col2, col3, col4, col5 = st.columns(5)

if col1.button("-6h"):
    st.session_state.current_time -= timedelta(hours=6)
    st.rerun()

if col2.button("-1h"):
    st.session_state.current_time -= timedelta(hours=1)
    st.rerun()

if col3.button("Teraz"):
    st.session_state.current_time = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    st.rerun()

if col4.button("+1h"):
    st.session_state.current_time += timedelta(hours=1)
    st.rerun()

if col5.button("+6h"):
    st.session_state.current_time += timedelta(hours=6)
    st.rerun()


colf1, colf2, colf3 = st.columns(3)

if colf1.button("-0.1m"):
    st.session_state.prog_filtra = max(0.0, st.session_state.prog_filtra - 0.1)
    st.rerun()

if colf2.button("0.5m"):
    st.session_state.prog_filtra = 0.5
    st.rerun()

if colf3.button("+0.1m"):
    st.session_state.prog_filtra = min(5.0, st.session_state.prog_filtra + 0.1)
    st.rerun()


# --- DRUGI WYKRES ---
fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(6, 3.75))

ax2.pcolormesh(lons_raw, lats_raw, h_signif, cmap=cmap_vhm0)
ax2.set_title("VHM0")
ax2.axis("off")

ax3.pcolormesh(lons_raw, lats_raw, t_mean, cmap=cmap_vtm02)
ax3.set_title("VTM02")
ax3.axis("off")

st.pyplot(fig2)