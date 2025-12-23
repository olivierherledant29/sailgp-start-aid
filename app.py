import streamlit as st

from src.io_xml import load_xml_bytes, parse_race_xml
from src.geo import make_context_from_boundary, to_xy_marks_and_polys, compute_PI_xy
from src.model import compute_all_geometry_and_times
from src.viz import build_deck


st.set_page_config(page_title="SailGP F50 - Start Aid", layout="wide")
st.title("SailGP F50 – Aide au départ (trajectoires PI → M → StartLine)")


def sidebar_controls():
    with st.sidebar:
        st.header("1) Données")
        uploaded = st.file_uploader("Charger le fichier XML", type=["xml"])
        use_default = st.checkbox(
            "Utiliser le XML sample.xml AD25",
            value=(uploaded is None),
            help="En local: pose le XML dans le même dossier que app.py (ex: 25113006_10-40-01.xml)",
        )

        st.header("2) Vent & paramètres")
        TWD = st.number_input("TWD (°)", min_value=0.0, max_value=360.0, value=0.0, step=1.0)
        PI_m = st.number_input("PI (m)", min_value=-500.0, max_value=500.0, value=-40.0, step=1.0)

        st.header("3) Angles")
        TWA_port = st.number_input("TWA_port (°)", min_value=40.0, max_value=160.0, value=60.0, step=1.0)
        TWA_UW = st.number_input("TWA_UW (°)", min_value=40.0, max_value=60.0, value=45.0, step=1.0)

        st.header("4) Polygones")
        size_buffer_BDY = st.number_input(
            "size_buffer_BDY (m)", min_value=0.0, max_value=500.0, value=50.0, step=5.0
        )

        st.header("5) Performance")
        BSP_kmh = st.number_input("BSP (km/h)", min_value=0.0, max_value=200.0, value=45.0, step=1.0)
        M_lost = st.number_input("M_lost (s)", min_value=0.0, max_value=120.0, value=12.0, step=1.0)

        return dict(
            uploaded=uploaded,
            use_default=use_default,
            TWD=TWD,
            PI_m=PI_m,
            TWA_port=TWA_port,
            TWA_UW=TWA_UW,
            size_buffer_BDY=size_buffer_BDY,
            BSP_kmh=BSP_kmh,
            M_lost=M_lost,
        )


params = sidebar_controls()

xml_bytes = load_xml_bytes(params["uploaded"], params["use_default"])
if xml_bytes is None:
    st.info("Charge un fichier XML via la barre latérale (ou mets-le à côté de app.py et coche l’option).")
    st.stop()

marks_ll, boundary_latlon = parse_race_xml(xml_bytes)

ctx = make_context_from_boundary(boundary_latlon)
geom = to_xy_marks_and_polys(ctx, marks_ll, boundary_latlon, params["size_buffer_BDY"])
PI_xy = compute_PI_xy(geom["SL1_xy"], geom["SL2_xy"], params["PI_m"])

out = compute_all_geometry_and_times(
    ctx=ctx,
    geom=geom,
    PI_xy=PI_xy,
    TWD=params["TWD"],
    TWA_port=params["TWA_port"],
    TWA_UW=params["TWA_UW"],
    BSP_kmh=params["BSP_kmh"],
    M_lost=params["M_lost"],
)

deck = build_deck(ctx, geom, PI_xy, out)

col1, col2 = st.columns([2, 1], vertical_alignment="top")

with col1:
    # FIX Streamlit warning: width="stretch" instead of use_container_width=True
    st.pydeck_chart(deck, width="stretch")

with col2:
    st.subheader("Paramètres")
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

