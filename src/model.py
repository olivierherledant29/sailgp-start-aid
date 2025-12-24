import numpy as np
from shapely.geometry import LineString

from .geo import (
    heading_to_unit_vector,
    intersection_ray_with_polygon_boundary,
    line_infinite_through_points,
    xy_to_ll,
)


def kmh_to_mps(v_kmh: float) -> float:
    return v_kmh / 3.6


def meters_to_seconds(dist_m: float, bsp_kmh: float) -> float:
    v = kmh_to_mps(bsp_kmh)
    return dist_m / v if v > 1e-6 else float("inf")


def compute_forward_intersection_between_lines(
    ray_PI: LineString,
    PI_xy: np.ndarray,
    dir_PI: np.ndarray,
    layline: LineString,
    SL1_xy: np.ndarray,
    dir_lay: np.ndarray,
):
    """
    Intersection ray_PI ∩ layline, filtered so that intersection is forward along both rays.
    """
    inter = ray_PI.intersection(layline)
    if inter.is_empty:
        return None

    pts = []
    if inter.geom_type == "Point":
        pts = [inter]
    elif inter.geom_type in ("MultiPoint", "GeometryCollection"):
        pts = [g for g in inter.geoms if g.geom_type == "Point"]
    else:
        return None

    candidates = []
    for p in pts:
        pxy = np.array([p.x, p.y], dtype=float)
        v1 = pxy - PI_xy
        v2 = pxy - SL1_xy

        if float(np.dot(v1, dir_PI)) <= 1e-6:
            continue
        if float(np.dot(v2, dir_lay)) <= 1e-6:
            continue

        candidates.append((float(np.linalg.norm(v1)), p))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


def rgb_to_css(rgb):
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


# 6 couleurs pour les 2e segments (évite: rose startline, jaune marques, rouge, vert, bleu vent, orange laylines)
SECOND_COLORS = {
    ("buffer_BDY", "to_SL1"): [160, 32, 240],   # purple
    ("buffer_BDY", "to_SL2"): [0, 255, 255],    # cyan
    ("buffer_BDY", "to_SP"):  [165, 42, 42],    # brown
    ("LL_SL1",     "to_SL1"): [112, 128, 144],  # slate
    ("LL_SL1",     "to_SL2"): [0, 128, 128],    # teal
    ("LL_SL1",     "to_SP"):  [138, 43, 226],   # blueviolet
}


def compute_startline_special_points(SL1_xy: np.ndarray, SL2_xy: np.ndarray, X_percent: float, offset_m: float = 7.0):
    """
    Returns:
      SL1_7m: point on segment SL1-SL2, at 7m from SL1 toward SL2
      SL2_7m: point on segment SL1-SL2, at 7m from SL2 toward SL1
      SP:     point on segment, at X% from SL2 to SL1 (0=SL2, 100=SL1)
    """
    v = SL1_xy - SL2_xy
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        return SL1_xy.copy(), SL2_xy.copy(), SL2_xy.copy()

    u = v / L  # unit from SL2 -> SL1

    # Clamp offset to avoid crossing if line is very small
    off = min(float(offset_m), max(0.0, 0.49 * L))

    SL1_7m = SL1_xy - u * off
    SL2_7m = SL2_xy + u * off

    x = max(0.0, min(100.0, float(X_percent)))
    SP = SL2_xy + u * (L * (x / 100.0))

    return SL1_7m, SL2_7m, SP


def compute_all_geometry_and_times(
    ctx,
    geom,
    PI_xy,
    TWD,
    TWA_port,
    TWA_UW,
    BSP_approche_BAB,
    BSP_retour_TRIB,
    M_lost,
    X_percent,
    TTS_intersection,
):
    """
    Nouvelles règles:
      - 2 vitesses: BSP_approche_BAB (PI->M) et BSP_retour_TRIB (M->target)
      - Targets:
          * vers SL1 => SL1_7m (sur start line, à 7m de SL1)
          * vers SL2 => SL2_7m (sur start line, à 7m de SL2)
          * vers M1  => SP (sur start line, X% depuis SL2 vers SL1)
      - TTK = TTS_intersection - t_total, affiché même couleur que t_total
    """
    to_wgs = ctx["to_wgs"]
    SL1_xy, SL2_xy, M1_xy = geom["SL1_xy"], geom["SL2_xy"], geom["M1_xy"]
    poly_BDY = geom["poly_BDY"]
    poly_buffer = geom["poly_buffer"]

    # Angles
    ANG_port = (float(TWD) + float(TWA_port)) % 360.0
    ANG_UW = (float(TWD) - float(TWA_UW) + 180.0) % 360.0

    # Start line infinite (still used for layline visuals / general geometry)
    _ = line_infinite_through_points(SL1_xy, SL2_xy, scale=80000.0)

    # Common first-leg ray from PI (ANG_port)
    dir_port = heading_to_unit_vector(ANG_port)
    ray_PI = LineString([tuple(PI_xy), tuple(PI_xy + dir_port * 60000.0)])

    # Group 1: buffer intersection => M_buffer_BDY
    M_buffer_BDY_xy = None
    if poly_buffer is not None:
        p = intersection_ray_with_polygon_boundary(ray_PI, PI_xy, poly_buffer, dir_port)
        if p is not None:
            M_buffer_BDY_xy = np.array([p.x, p.y], dtype=float)

    # Group 2: PI ray ∩ SL1 layline => M_LL_SL1
    dir_UW = heading_to_unit_vector(ANG_UW)
    layline_SL1 = LineString([tuple(SL1_xy), tuple(SL1_xy + dir_UW * 80000.0)])

    p_ll = compute_forward_intersection_between_lines(
        ray_PI=ray_PI,
        PI_xy=PI_xy,
        dir_PI=dir_port,
        layline=layline_SL1,
        SL1_xy=SL1_xy,
        dir_lay=dir_UW,
    )
    M_LL_SL1_xy = None
    if p_ll is not None:
        M_LL_SL1_xy = np.array([p_ll.x, p_ll.y], dtype=float)

    # Laylines visual (UW) from SL1 & SL2 to BDY boundary
    def layline_to_boundary(start_xy):
        r = LineString([tuple(start_xy), tuple(start_xy + dir_UW * 120000.0)])
        p = intersection_ray_with_polygon_boundary(r, start_xy, poly_BDY, dir_UW)
        if p is None:
            return None
        return LineString([tuple(start_xy), (p.x, p.y)])

    lay_vis_SL1 = layline_to_boundary(SL1_xy)
    lay_vis_SL2 = layline_to_boundary(SL2_xy)

    # New targets on start line
    SL1_7m_xy, SL2_7m_xy, SP_xy = compute_startline_special_points(SL1_xy, SL2_xy, X_percent, offset_m=7.0)

    DESTS = {
        "to_SL1": SL1_7m_xy,
        "to_SL2": SL2_7m_xy,
        "to_SP": SP_xy,   # replaces previous "to_M1" target on the line
    }
    GROUPS = {"buffer_BDY": M_buffer_BDY_xy, "LL_SL1": M_LL_SL1_xy}

    # Collect trajectories and results
    results = []
    traj_second_segments = []
    first_leg_paths = []

    PI_ll = xy_to_ll(to_wgs, PI_xy[0], PI_xy[1])

    # First-leg paths in red (PI -> each M found)
    if M_buffer_BDY_xy is not None:
        M_ll = xy_to_ll(to_wgs, M_buffer_BDY_xy[0], M_buffer_BDY_xy[1])
        first_leg_paths.append({"path": [[PI_ll[1], PI_ll[0]], [M_ll[1], M_ll[0]]], "name": "PI->M_buffer_BDY"})
    if M_LL_SL1_xy is not None:
        M_ll = xy_to_ll(to_wgs, M_LL_SL1_xy[0], M_LL_SL1_xy[1])
        first_leg_paths.append({"path": [[PI_ll[1], PI_ll[0]], [M_ll[1], M_ll[0]]], "name": "PI->M_LL_SL1"})

    # Build 6 trajectories: 2 groups * 3 targets (all targets already on start line)
    for gname, M_xy in GROUPS.items():
        if M_xy is None:
            continue

        d1 = float(np.linalg.norm(M_xy - PI_xy))
        t1 = meters_to_seconds(d1, float(BSP_approche_BAB))

        for dname, target_xy in DESTS.items():
            d2 = float(np.linalg.norm(target_xy - M_xy))
            t2 = meters_to_seconds(d2, float(BSP_retour_TRIB))

            t_total = t1 + t2 + float(M_lost)
            ttk = float(TTS_intersection) - t_total

            color = SECOND_COLORS[(gname, dname)]

            M_ll = xy_to_ll(to_wgs, M_xy[0], M_xy[1])
            end_ll = xy_to_ll(to_wgs, target_xy[0], target_xy[1])

            traj_second_segments.append({
                "group": gname,
                "dest": dname,
                "color": color,
                "path": [[M_ll[1], M_ll[0]], [end_ll[1], end_ll[0]]],
            })

            results.append({
                "group": gname,
                "dest": dname,
                "t1": t1,
                "t2": t2,
                "t_total": t_total,
                "ttk": ttk,
                "color": color,
            })

    # Condensed HTML table, t_total and TTK colored like the second segment
    rows = []
    for r in results:
        c = rgb_to_css(r["color"])
        rows.append(
            f"<tr>"
            f"<td>{r['group']}</td>"
            f"<td>{r['dest']}</td>"
            f"<td style='text-align:right;'>{r['t1']:.1f}</td>"
            f"<td style='text-align:right;'>{r['t2']:.1f}</td>"
            f"<td style='text-align:right; color:{c}; font-weight:700;'>{r['t_total']:.1f}</td>"
            f"<td style='text-align:right; color:{c}; font-weight:700;'>{r['ttk']:.1f}</td>"
            f"</tr>"
        )

    results_html = f"""
    <div style="font-size: 13px;">
      <table style="width:100%; border-collapse: collapse;">
        <thead>
          <tr>
            <th style="text-align:left; border-bottom:1px solid #666;">groupe</th>
            <th style="text-align:left; border-bottom:1px solid #666;">dest</th>
            <th style="text-align:right; border-bottom:1px solid #666;">t1 PI→M (s)</th>
            <th style="text-align:right; border-bottom:1px solid #666;">t2 M→target (s)</th>
            <th style="text-align:right; border-bottom:1px solid #666;">t_total (s)</th>
            <th style="text-align:right; border-bottom:1px solid #666;">TTK (s)</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows) if rows else ''}
        </tbody>
      </table>
      <div style="margin-top:6px; color:#aaa;">
        t_total = t1(BSP_approche_BAB={float(BSP_approche_BAB):.1f} km/h) + t2(BSP_retour_TRIB={float(BSP_retour_TRIB):.1f} km/h) + M_lost({float(M_lost):.1f}s) <br/>
        TTK = TTS_intersection({float(TTS_intersection):.1f}s) − t_total
      </div>
    </div>
    """

    out = {
        "TWD": float(TWD),
        "ANG_port": ANG_port,
        "ANG_UW": ANG_UW,
        "dir_UW": dir_UW,
        "M_buffer_BDY_xy": M_buffer_BDY_xy,
        "M_LL_SL1_xy": M_LL_SL1_xy,
        "lay_vis_SL1": lay_vis_SL1,
        "lay_vis_SL2": lay_vis_SL2,
        "first_leg_paths": first_leg_paths,
        "traj_second_segments": traj_second_segments,
        "results": results,
        "results_html": results_html,
        # new endpoints for viz
        "SL1_7m_xy": SL1_7m_xy,
        "SL2_7m_xy": SL2_7m_xy,
        "SP_xy": SP_xy,
    }
    return out
