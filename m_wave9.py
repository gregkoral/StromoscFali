"""
Prognoza Fal Bałtyku — Streamlit + Copernicus Marine Toolbox v2.0
=================================================================
Zmiany względem wersji v1.x:
  • overwrite_metadata_cache usunięty (cache nie istnieje w v2)
  • Daty przekazywane jako obiekty timezone-aware (UTC) zamiast stringów
  • subset_method usunięty → coordinates_selection_method (domyślnie "inside")
  • copernicusmarine.login() z force_overwrite=True zamiast overwrite=True
  • Automatyczna instalacja brakujących bibliotek
  • Pasek postępu podczas ładowania danych
  • Podsumowanie na końcu sesji
  • Skrypt czeka na SPACJĘ z zamknięciem (tryb standalone)
"""

# ── 0. AUTOMATYCZNA INSTALACJA BRAKUJĄCYCH BIBLIOTEK ──────────────────────────
import importlib, subprocess, sys

_WYMAGANE = {
    "streamlit":          "streamlit>=1.35",
    "copernicusmarine":   "copernicusmarine>=2.0.0",
    "xarray":             "xarray",
    "matplotlib":         "matplotlib",
    "numpy":              "numpy",
}

_brakujace = []
for _mod, _pkg in _WYMAGANE.items():
    try:
        importlib.import_module(_mod)
    except ImportError:
        _brakujace.append(_pkg)

if _brakujace:
    print(f"[SETUP] Instaluję brakujące pakiety: {', '.join(_brakujace)}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet"] + _brakujace
    )
    print("[SETUP] Instalacja zakończona. Uruchom skrypt ponownie jeśli coś się nie powiodło.")

# ── 1. IMPORTY ─────────────────────────────────────────────────────────────────
import streamlit as st
import copernicusmarine
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta, timezone

# Alias UTC (timezone-aware)
UTC = timezone.utc

# ── 2. KONFIGURACJA STRONY ─────────────────────────────────────────────────────
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
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockSeparator"],
    [data-testid="stVerticalBlock"] > div { margin-bottom: 0rem !important; }
    [data-testid="stVerticalBlock"] > div:has([data-testid="stHorizontalBlock"]:has(button[kind="secondary"])) {
        margin-top: 14px !important;
        margin-bottom: 14px !important;
    }
    div[data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
        min-width: 0 !important;
        flex: 1 1 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ── 3. STAŁE ───────────────────────────────────────────────────────────────────
MIN_LON, MAX_LON = 11.0, 15.0
MIN_LAT, MAX_LAT = 53.5, 56.5
DATASET_ID       = "cmems_mod_bal_wav_anfc_PT1H-i"

# ── 4. LOGOWANIE DO COPERNICUS (API 2.0) ───────────────────────────────────────
def bezpieczne_logowanie() -> None:
    """
    Loguje do Copernicus Marine używając Streamlit Secrets.
    API 2.0: parametr force_overwrite zamiast overwrite.
    """
    try:
        username = st.secrets["COPERNICUS_USERNAME"]
        password = st.secrets["COPERNICUS_PASSWORD"]
        # force_overwrite=True — nadpisuje plik konfiguracyjny przy każdym uruchomieniu
        copernicusmarine.login(
            username=username,
            password=password,
            force_overwrite=True,
        )
    except Exception as e:
        st.error(f"Błąd autoryzacji Copernicus: {e}. Sprawdź konfigurację Secrets.")
        st.stop()

# ── 5. POBIERANIE DANYCH (API 2.0) ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def pobierz_pelny_blok_danych(odniesienie_czasu: datetime) -> xr.Dataset:
    """
    Pobiera blok danych (-12h … +48h) z Copernicus Marine API 2.0.

    Zmiany względem v1:
      • Daty jako obiekty datetime timezone-aware (UTC) — API 2.0 jest TZ-aware
      • Usunięto overwrite_metadata_cache (parametr nie istnieje w v2)
      • Dodano coordinates_selection_method="inside" (zastępuje subset_method)
    """
    bezpieczne_logowanie()

    # Copernicus Marine v2 wymaga dat timezone-aware (UTC)
    tz_ref = odniesienie_czasu.replace(tzinfo=UTC) if odniesienie_czasu.tzinfo is None \
             else odniesienie_czasu.astimezone(UTC)

    start_dt = tz_ref - timedelta(hours=12)
    end_dt   = tz_ref + timedelta(hours=48)

    progress = st.progress(0, text="Łączenie z Copernicus Marine…")

    try:
        progress.progress(15, text="Otwieranie zbioru danych…")

        ds = copernicusmarine.open_dataset(
            dataset_id=DATASET_ID,
            # ▶ API 2.0: datetime objects (timezone-aware) lub ISO-8601 string z offset
            start_datetime=start_dt,
            end_datetime=end_dt,
            minimum_longitude=MIN_LON,
            maximum_longitude=MAX_LON,
            minimum_latitude=MIN_LAT,
            maximum_latitude=MAX_LAT,
            variables=["VHM0", "VTM02", "VMDR_WW"],
            # ▶ API 2.0: coordinates_selection_method zamiast usuniętego subset_method
            coordinates_selection_method="inside",
            # ▶ overwrite_metadata_cache USUNIĘTE — cache nie istnieje w v2
        )

        progress.progress(55, text="Ładowanie danych do pamięci RAM…")
        ds_loaded = ds.load()
        ds.close()

        progress.progress(100, text="Dane załadowane ✓")
        progress.empty()

        return ds_loaded

    except Exception as e:
        progress.empty()
        st.error(f"Błąd pobierania danych z Copernicus (API 2.0): {e}")
        st.stop()

# ── 6. STAN SESJI ──────────────────────────────────────────────────────────────
import pandas as pd

def _teraz_utc_bez_minut() -> datetime:
    """Aktualny czas UTC zaokrąglony do pełnej godziny (timezone-naive — dla cache key)."""
    return datetime.now(UTC).replace(minute=0, second=0, microsecond=0, tzinfo=None)

def _do_numpy_ts(dt: datetime) -> np.datetime64:
    """
    Konwertuje datetime na numpy.datetime64[ns] BEZ strefy czasowej.

    Problem: po copernicusmarine.open_dataset().load() oś czasu datasetu
    ma dtype datetime64[ns] (tz-naive). Jeśli do sel() przekażemy obiekt
    tz-aware (pd.Timestamp z UTC lub datetime64[us, UTC]), xarray rzuca:
      'Cannot compare dtypes datetime64[ns] and datetime64[us, UTC]'
    Rozwiązanie: zawsze konwertujemy wybrany czas do UTC, a następnie
    usuwamy informację o strefie, by typy były identyczne.
    """
    if dt.tzinfo is None:
        ts = pd.Timestamp(dt, tz="UTC")
    else:
        ts = pd.Timestamp(dt).tz_convert("UTC")
    return np.datetime64(ts.tz_localize(None), "ns")

if "base_time"    not in st.session_state:
    st.session_state.base_time    = _teraz_utc_bez_minut()
if "current_time" not in st.session_state:
    st.session_state.current_time = st.session_state.base_time
if "prog_filtra"  not in st.session_state:
    st.session_state.prog_filtra  = 0.5

# ── 7. POBRANIE DANYCH ─────────────────────────────────────────────────────────
pelny_dataset: xr.Dataset = pobierz_pelny_blok_danych(st.session_state.base_time)

# ── 8. WYCIĘCIE WYBRANEGO KROKU CZASOWEGO ──────────────────────────────────────
wybrany_czas: datetime = st.session_state.current_time

# Oś czasu datasetu jako numpy.datetime64[ns] naive — jednolity typ do porównań
_t_min = np.datetime64(pd.Timestamp(pelny_dataset.time.values[0]).tz_localize(None)
                       if pd.Timestamp(pelny_dataset.time.values[0]).tzinfo is not None
                       else pd.Timestamp(pelny_dataset.time.values[0]), "ns")
_t_max = np.datetime64(pd.Timestamp(pelny_dataset.time.values[-1]).tz_localize(None)
                       if pd.Timestamp(pelny_dataset.time.values[-1]).tzinfo is not None
                       else pd.Timestamp(pelny_dataset.time.values[-1]), "ns")

# Wybrany czas → ten sam typ (datetime64[ns] naive)
_czas_sel = _do_numpy_ts(wybrany_czas)

# Jeśli wybrany czas wykracza poza cache — pokaż diagnostykę i przycisk
if not (_t_min <= _czas_sel <= _t_max):
    _fmt = lambda t: pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")
    st.error(
        f"⚠️ Wybrana godzina **{wybrany_czas.strftime('%Y-%m-%d %H:%M')} UTC** "
        f"wykracza poza zakres pobranej pamięci podręcznej.\n\n"
        f"Dostępny zakres: `{_fmt(_t_min)}` → `{_fmt(_t_max)}` UTC"
    )
    st.button(
        "🔄 Zresetuj do teraz",
        on_click=lambda: st.session_state.update(current_time=_teraz_utc_bez_minut()),
    )
    st.stop()

try:
    wave_slice = pelny_dataset.sel(time=_czas_sel, method="nearest")

    lons_raw  = wave_slice["longitude"].values
    lats_raw  = wave_slice["latitude"].values
    h_signif  = wave_slice["VHM0"].values
    t_mean    = wave_slice["VTM02"].values
    vmdr_ww   = wave_slice["VMDR_WW"].values

except Exception as e:
    st.error(f"Błąd przy wyborze kroku czasowego: {e}")
    st.button(
        "🔄 Zresetuj do teraz",
        on_click=lambda: st.session_state.update(current_time=_teraz_utc_bez_minut()),
    )
    st.stop()

# ── 9. MATEMATYKA ──────────────────────────────────────────────────────────────
with np.errstate(divide="ignore", invalid="ignore"):
    wave_length    = (9.81 * (t_mean ** 2)) / (2 * np.pi)
    wave_steepness = h_signif / wave_length

land_mask     = np.where(np.isnan(h_signif), 1, np.nan)
wave_filtered = np.where(h_signif >= st.session_state.prog_filtra, wave_steepness, np.nan)

# ── 10. PALETY KOLORÓW ─────────────────────────────────────────────────────────
kolor_ladu    = "#6b798d"
kolor_strzalek = "#ffffff"

_kolory_stromość = [
    (0.00, "#8ecae6"), (0.30, "#8ecae6"), (0.45, "#219ebc"),
    (0.60, "#023047"), (0.70, "#ffb703"), (0.80, "#fb8500"),
    (1.00, "#fb8500"),
]
cmap_stromość  = LinearSegmentedColormap.from_list("stromość_custom", _kolory_stromość)
cmap_vhm0      = plt.cm.viridis
cmap_vtm02     = plt.cm.plasma

# ── 11. MAPA GŁÓWNA (STROMOŚĆ FALI) ───────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(7, 7))
fig1.subplots_adjust(top=1, bottom=0, left=0, right=1)
ax1.set_facecolor("#404040")
ax1.set_xlim(MIN_LON, MAX_LON)
ax1.set_ylim(MIN_LAT, MAX_LAT)
ax1.tick_params(
    axis="both", which="both",
    bottom=False, top=False, left=False, right=False,
    labelbottom=False, labelleft=False,
)

_cmap_ląd = LinearSegmentedColormap.from_list("lc", [kolor_ladu, kolor_ladu])
ax1.pcolormesh(lons_raw, lats_raw, land_mask,     cmap=_cmap_ląd,    zorder=1)
im1 = ax1.pcolormesh(lons_raw, lats_raw, wave_filtered, cmap=cmap_stromość,
                     vmin=0.0, vmax=0.1, zorder=2)
fig1.colorbar(im1, ax=ax1, orientation="horizontal", pad=0.02, fraction=0.046, aspect=35)

try:
    s_lat, s_lon     = 10, 12
    lons_sub         = lons_raw[::s_lon]
    lats_sub         = lats_raw[::s_lat]
    lon_grid, lat_grid = np.meshgrid(lons_sub, lats_sub)
    angles_deg       = vmdr_ww[::s_lat, ::s_lon]
    angles_rad       = np.deg2rad(90.0 - (np.round(angles_deg / 30.0) * 30.0) + 180)
    u = np.where(np.isnan(angles_deg), np.nan, np.cos(angles_rad))
    v = np.where(np.isnan(angles_deg), np.nan, np.sin(angles_rad))
    ax1.quiver(
        lon_grid, lat_grid, u, v,
        color=kolor_strzalek, scale=25, width=0.0025,
        headwidth=6, headlength=5, pivot="middle", zorder=3,
    )
except Exception:
    pass

plt.tight_layout()
st.pyplot(fig1)

st.markdown(
    f"<p style='text-align:center; color:#888888; font-size:20px; "
    f"margin-top:4px; margin-bottom:8px; font-family:sans-serif;'>"
    f"{wybrany_czas.strftime('%Y-%m-%d %H:%M')} UTC | "
    f"Filtr: > {st.session_state.prog_filtra:.1f} m"
    f"</p>",
    unsafe_allow_html=True,
)

# ── 12. PANEL STEROWANIA — CZAS ────────────────────────────────────────────────
st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)

if col_t1.button("-6h",  use_container_width=True, key="btn_prev6"):
    st.session_state.current_time -= timedelta(hours=6);  st.rerun()
if col_t2.button("-1h",  use_container_width=True, key="btn_prev1"):
    st.session_state.current_time -= timedelta(hours=1);  st.rerun()
if col_t3.button("Teraz", use_container_width=True, key="btn_now"):
    st.session_state.current_time = _teraz_utc_bez_minut(); st.rerun()
if col_t4.button("+1h",  use_container_width=True, key="btn_next1"):
    st.session_state.current_time += timedelta(hours=1);  st.rerun()
if col_t5.button("+6h",  use_container_width=True, key="btn_next6"):
    st.session_state.current_time += timedelta(hours=6);  st.rerun()

# ── 13. PANEL STEROWANIA — FILTR WYSOKOŚCI ─────────────────────────────────────
col_f1, col_f2, col_f3 = st.columns(3)

if col_f1.button("-0.1m", use_container_width=True, key="btn_filtm"):
    st.session_state.prog_filtra = round(max(0.0, st.session_state.prog_filtra - 0.1), 1)
    st.rerun()

if col_f2.button("0.0m",  use_container_width=True, key="btn_filt00"):
    st.session_state.prog_filtra = 0.0; st.rerun()

if col_f3.button("0.5m",  use_container_width=True, key="btn_filt05"):
    st.session_state.prog_filtra = 0.5; st.rerun()
    
if col_f4.button("1.0m",  use_container_width=True, key="btn_filt10"):
    st.session_state.prog_filtra = 1.0; st.rerun()

    
if col_f5.button("+0.1m", use_container_width=True, key="btn_filtp"):
    st.session_state.prog_filtra = round(min(5.0, st.session_state.prog_filtra + 0.1), 1)
    st.rerun()

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ── 14. MAPY POMOCNICZE (VHM0 + VTM02) ────────────────────────────────────────
fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(6, 3.75))

for ax, data, cmap, vmin, vmax, tytul in [
    (ax2, h_signif, cmap_vhm0,  0.25, 1.0, "Wysokość fali (VHM0)"),
    (ax3, t_mean,   cmap_vtm02, 1.0,  3.5, "Okres fali (VTM02)"),
]:
    ax.set_facecolor("#202020")
    ax.axis("off")
    ax.set_xlim(MIN_LON, MAX_LON)
    ax.set_ylim(MIN_LAT, MAX_LAT)
    ax.pcolormesh(lons_raw, lats_raw, land_mask, cmap=_cmap_ląd)
    im = ax.pcolormesh(lons_raw, lats_raw, data, cmap=cmap, vmin=vmin, vmax=vmax)
    fig2.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, fraction=0.046, aspect=20)
    ax.set_title(tytul, fontsize=11, color="black", pad=8)

fig2.tight_layout()
st.pyplot(fig2)


# ── 15. PODSUMOWANIE SESJI ─────────────────────────────────────────────────────
with st.expander("📊 Podsumowanie sesji", expanded=False):
    vhm0_valid  = h_signif[~np.isnan(h_signif)]
    vtm02_valid = t_mean[~np.isnan(t_mean)]

    def _stat(arr, jednostka):
        if arr.size:
            return (f"{arr.max():.2f} {jednostka}",
                    f"{arr.mean():.2f} {jednostka}",
                    f"{arr.min():.2f} {jednostka}")
        return ("—", "—", "—")

    v_max, v_avg, v_min = _stat(vhm0_valid, "m")
    t_max, t_avg, t_min = _stat(vtm02_valid, "s")

    zakres_od = (st.session_state.base_time - timedelta(hours=12)).strftime("%Y-%m-%d %H:%M")
    zakres_do = (st.session_state.base_time + timedelta(hours=48)).strftime("%Y-%m-%d %H:%M")

    st.markdown(f"""
<style>
.podsum td, .podsum th {{
    padding: 2px 12px 2px 0;
    font-size: 0.875rem;
    vertical-align: top;
}}
.podsum th {{
    font-weight: 600;
    color: var(--text-color);
    padding-bottom: 4px;
}}
.podsum td:first-child {{
    color: #888;
    white-space: nowrap;
}}
</style>
<table class="podsum">
  <tr>
    <th></th>
    <th>VHM0 — wysokość fali</th>
    <th>VTM02 — okres fali</th>
  </tr>
  <tr><td>maks.</td> <td>{v_max}</td><td>{t_max}</td></tr>
  <tr><td>śred.</td> <td>{v_avg}</td><td>{t_avg}</td></tr>
  <tr><td>min.</td>  <td>{v_min}</td><td>{t_min}</td></tr>
</table>
<br>
<span style="font-size:0.875rem">
  <b>Krok czasowy:</b> {wybrany_czas.strftime('%Y-%m-%d %H:%M')} UTC &nbsp;|&nbsp;
  <b>Filtr:</b> ≥ {st.session_state.prog_filtra:.1f} m &nbsp;|&nbsp;
  <b>Cache:</b> {zakres_od} → {zakres_do} UTC
</span>
""", unsafe_allow_html=True)
