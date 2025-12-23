import math
import numpy as np
import streamlit as st

from src.io_xml import load_xml_bytes, parse_race_xml
from src.geo import make_context_from_boundary, to_xy_marks_and_polys, compute_PI_xy
from src.model import compute_all_geometry_and_times
from src.viz import build_deck


st.set_page_config(page_title="SailGP F50 - Start Aid", layout="wide")
st.title("SailGP F50 – Aide au départ (trajectoires PI → M → StartLine)")


def heading_from_xy(A_xy: np.ndarray, B_xy: np.ndarray) -> float:
    """
    Heading (deg) from A to B with convention: 0°=North, 90°=East.
    A_xy, B_xy in meters (x=East, y=North).
    """
    dx = float(B_xy[0] - A_xy[0])
    dy = float(B_xy[1] - A_xy[1])
    ang = math.degrees(math.atan2(dx, dy))  # atan2(E, N)
    return ang % 360.0


# ---------
# 1) Load XML first (so we can compute TWD default from SL2->SL1)
# ---------
with st.sidebar:
    st.header("1) Données")
    uploaded = st.file_uploader("Charger le fichier XML", type=["xml"])
    use_default = st.checkbox(
        "Utiliser sample.xml (fallback si aucun upload)",
        value=True,
        help="Si aucun fichier n’est uploadé, l’app tente de charger sample.xml à la racine du projet.",
    )

xml_bytes = load_xml_bytes(uploaded, use_default)
if xml_bytes is None:
    st.info("Charge un fichier XML via la barre latérale (ou place sample.xml à côté de app.py).")
    st.stop()

marks_ll, boundary_latlon = parse_race_xml(xml_bytes)

ctx = make_context_from_boundary(boundary_latlon)

# defaults requested: size_buffer=15, BSP=40, PI=60
DEFAULT_SIZE_BUFFER = 15.0
DEFAULT_BSP = 40.0
DEFAULT_PI = 60.0

geom = to_xy_marks_and_polys(ctx, marks_ll, boundary_latlon, DEFAULT_SIZE_BUFFER)

# Default TWD = heading SL2 -> SL1
TWD_default = heading_from_xy(geom["SL2_xy"], geom["SL1_xy"])

# Initialize session state once
if "TWD" not in st.session_state:
    st.session_state["TWD"] = float(TWD_default)


# ---------
# 2) Sidebar controls (now we can show proper defaults)
# ---------
with st.sidebar:
    st.header("2) Vent & paramètres")
    TWD = st.number_input("TWD (°)", min_value=0.0, max_value=360.0, value=float(st.session_state["TWD"]), step=1.0)
    # keep state in sync
    st.session_state["TWD"] = float(TWD)

    PI_m = st.number_input("PI (m)", min_value=-500.0, max_value=500.0, value=float(DEFAULT_PI), step=1.0)

    st.header("3) Angles")
    TWA_port = st.number_input("TWA_port (°)", min_value=40.0, max_value=160.0, value=60.0, step=1.0)
    TWA_UW = st.number_input("TWA_UW (°)", min_value=40.0, max_value=60.0, value=45.0, step=1.0)

    st.header("4) Polygones")
    size_buffer_BDY = st.number_input("size_buffer_BDY (m)", min_value=0.0, max_value=500.0, value=float(DEFAULT_SIZE_BUFFER), step=1.0)

    st.header("5) Performance")
    BSP_kmh = st.number_input("BSP (km/h)", min_value=0.0, max_value=200.0, value=float(DEFAULT_BSP), step=1.0)
    M_lost = st.number_input("M_lost (s)", min_value=0.0, max_value=120.0, value=12.0, step=1.0)

    st.caption("Convention: 0°=Nord, 90°=Est. La carte est orientée pour que le vent vienne du haut (bearing = 360 - TWD).")


# Rebuild geometry with chosen buffer size (since it affects poly_buffer)
geom = to_xy_marks_and_polys(ctx, marks_ll, boundary_latlon, size_buffer_BDY)

PI_xy = compute_PI_xy(geom["SL1_xy"], geom["SL2_xy"], PI_m)

out = compute_all_geometry_and_times(
    ctx=ctx,
    geom=geom,
    PI_xy=PI_xy,
    TWD=TWD,
    TWA_port=TWA_port,
    TWA_UW=TWA_UW,
    BSP_kmh=BSP_kmh,
    M_lost=M_lost,
)

deck = build_deck(ctx, geom, PI_xy, out)

col1, col2 = st.columns([2, 1], vertical_alignment="top")

with col1:
    # Streamlit deprec: use width="stretch"
    st.pydeck_chart(deck, width="stretch")

with col2:
    st.subheader("Paramètres")
    st.write(f"TWD (défaut SL2→SL1) = **{float(TWD):.1f}°**")
    st.write(f"ANG_port = **{out['ANG_port']:.1f}°**")
    st.write(f"ANG_UW = **{out['ANG_UW']:.1f}°**")

    st.subheader("Trajectoires (6)")
    if not out["results"]:
        st.warning(
            "Impossible de calculer les 6 trajectoires (intersections manquantes). "
            "Vérifie que les intersections existent (PI→buffer et/ou PI→layline SL1)."
        )
    else:
        st.markdown(out["results_html"], unsafe_allow_html=True)
