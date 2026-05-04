"""
e-NABLE Hand STL Sizer — Streamlit App

Purpose
-------
A lean Streamlit app that helps makers enter recipient hand measurements,
estimate a conservative e-NABLE scale factor, upload an official e-NABLE STL
set, scale every STL, and download a ready-to-slice ZIP.

Important
---------
This app DOES NOT generate a clinically validated prosthetic hand from scratch.
It scales official STL templates that the user supplies. e-NABLE recommends
virtual fitting with Fusion 360/Blender for best results; two-measurement sizing
is only a basic guide.
"""

from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import trimesh


# -----------------------------
# Configuration
# -----------------------------

APP_TITLE = "e-NABLE Hand STL Sizer"
REFERENCE_WIDTH_MM = 80.0
REFERENCE_LENGTH_MM = 80.0
DEFAULT_CLEARANCE_MM = 3.0
DEFAULT_SCALE_OPTIONS = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]

DEVICE_PRESETS = {
    "Raptor Reloaded / Phoenix-style hand": {
        "width_weight": 0.65,
        "length_weight": 0.35,
        "notes": "For hand devices, width fit is often the limiting factor, especially when hand length is short.",
    },
    "Generic wrist-driven hand": {
        "width_weight": 0.55,
        "length_weight": 0.45,
        "notes": "Balanced width/length estimate. Confirm wrist clearance and tensioning in a virtual fit.",
    },
    "Custom / unknown hand design": {
        "width_weight": 0.60,
        "length_weight": 0.40,
        "notes": "Generic conservative estimate. Use the design-specific sizing guide when available.",
    },
}


@dataclass
class SizingResult:
    raw_width_scale: float
    raw_length_scale: float
    weighted_scale: float
    recommended_scale: int
    width_error_mm: float
    length_error_mm: float
    warning: str


# -----------------------------
# Sizing calculations
# -----------------------------

def nearest_scale(percent: float, options: List[int]) -> int:
    return int(min(options, key=lambda x: abs(x - percent)))


def calculate_sizing(
    hand_width_mm: float,
    hand_length_mm: float,
    reference_width_mm: float,
    reference_length_mm: float,
    clearance_mm: float,
    width_weight: float,
    length_weight: float,
    scale_options: List[int],
) -> SizingResult:
    """Estimate scale using width and length, with a conservative bias toward width fit."""
    if hand_width_mm <= 0 or hand_length_mm <= 0:
        raise ValueError("Hand width and hand length must be positive numbers.")
    if reference_width_mm <= 0 or reference_length_mm <= 0:
        raise ValueError("Reference dimensions must be positive numbers.")

    target_width = hand_width_mm + clearance_mm
    target_length = hand_length_mm + clearance_mm

    raw_width_scale = 100.0 * target_width / reference_width_mm
    raw_length_scale = 100.0 * target_length / reference_length_mm

    weighted_scale = width_weight * raw_width_scale + length_weight * raw_length_scale

    # Conservative rule: do not recommend a scale smaller than width fit.
    # This reflects e-NABLE guidance that a device too narrow can be a larger issue
    # than extra device length in some cases.
    conservative_scale = max(weighted_scale, raw_width_scale)
    recommended_scale = nearest_scale(conservative_scale, scale_options)

    scaled_width = reference_width_mm * recommended_scale / 100.0
    scaled_length = reference_length_mm * recommended_scale / 100.0
    width_error_mm = scaled_width - target_width
    length_error_mm = scaled_length - target_length

    warning = ""
    if width_error_mm < 0:
        warning = "Recommended scale may be too narrow. Increase scale or perform virtual fitting."
    elif abs(length_error_mm) > 20 and hand_length_mm <= hand_width_mm * 1.15:
        warning = "Length mismatch is large, but width may still be the safer sizing priority. Verify with virtual fitting."
    else:
        warning = "Use this as a starting point only. Confirm comfort, clearance, and function with virtual fitting."

    return SizingResult(
        raw_width_scale=raw_width_scale,
        raw_length_scale=raw_length_scale,
        weighted_scale=weighted_scale,
        recommended_scale=recommended_scale,
        width_error_mm=width_error_mm,
        length_error_mm=length_error_mm,
        warning=warning,
    )


def build_scale_matrix(
    hand_width_mm: float,
    hand_length_mm: float,
    reference_width_mm: float,
    reference_length_mm: float,
    clearance_mm: float,
    scale_options: List[int],
) -> pd.DataFrame:
    rows = []
    target_width = hand_width_mm + clearance_mm
    target_length = hand_length_mm + clearance_mm
    for scale in scale_options:
        scaled_width = reference_width_mm * scale / 100.0
        scaled_length = reference_length_mm * scale / 100.0
        width_error = scaled_width - target_width
        length_error = scaled_length - target_length
        score = abs(width_error) * 0.65 + abs(length_error) * 0.35
        rows.append(
            {
                "Scale %": scale,
                "Scaled width mm": round(scaled_width, 1),
                "Width error mm": round(width_error, 1),
                "Scaled length mm": round(scaled_length, 1),
                "Length error mm": round(length_error, 1),
                "Fit score lower is better": round(score, 1),
            }
        )
    return pd.DataFrame(rows).sort_values("Fit score lower is better")


# -----------------------------
# STL processing
# -----------------------------

def read_stl_bytes(uploaded_file) -> trimesh.Trimesh:
    raw = uploaded_file.read()
    mesh = trimesh.load_mesh(io.BytesIO(raw), file_type="stl")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not read {uploaded_file.name} as a mesh.")
    return mesh


def scale_mesh(mesh: trimesh.Trimesh, scale_percent: float) -> trimesh.Trimesh:
    scaled = mesh.copy()
    factor = scale_percent / 100.0
    scaled.apply_scale(factor)
    return scaled


def export_scaled_stls(uploaded_files, scale_percent: float, metadata: Dict[str, str]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest_lines = ["e-NABLE Hand STL Sizer Manifest", "", *[f"{k}: {v}" for k, v in metadata.items()], ""]

        for uploaded_file in uploaded_files:
            uploaded_file.seek(0)
            mesh = read_stl_bytes(uploaded_file)
            scaled = scale_mesh(mesh, scale_percent)
            stl_bytes = scaled.export(file_type="stl")
            clean_name = Path(uploaded_file.name).stem
            out_name = f"{clean_name}_scaled_{int(round(scale_percent))}pct.stl"
            zf.writestr(out_name, stl_bytes)
            bounds = scaled.bounds
            extents = scaled.extents
            manifest_lines.append(
                f"{out_name}: extents_mm approx X={extents[0]:.2f}, Y={extents[1]:.2f}, Z={extents[2]:.2f}; "
                f"bounds_min={bounds[0].round(2).tolist()}, bounds_max={bounds[1].round(2).tolist()}"
            )

        zf.writestr("SIZING_MANIFEST.txt", "\n".join(manifest_lines))
    return zip_buffer.getvalue()


def create_demo_calibration_stl(scale_percent: float) -> bytes:
    """Create a simple non-prosthetic calibration block so users can test the pipeline."""
    box = trimesh.creation.box(extents=(40, 20, 5))
    box.apply_scale(scale_percent / 100.0)
    return box.export(file_type="stl")


# -----------------------------
# Streamlit UI
# -----------------------------

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🖐️", layout="wide")
    st.title("🖐️ e-NABLE Hand STL Sizer")
    st.caption("Enter measurements, estimate a sizing scale, upload official e-NABLE STL files, and download a scaled STL set.")

    with st.expander("Important safety and fit notice", expanded=True):
        st.warning(
            "This app is a maker aid, not a medical-device fit validation tool. "
            "e-NABLE’s own sizing guidance recommends virtual fitting as the most reliable method. "
            "Use this app to produce a starting STL set, then verify fit, comfort, clearances, and function before use."
        )

    left, right = st.columns([1, 1])

    with left:
        st.header("1) Recipient measurements")
        device_type = st.selectbox("Device family", list(DEVICE_PRESETS.keys()))
        preset = DEVICE_PRESETS[device_type]
        st.info(preset["notes"])

        hand_width_mm = st.number_input(
            "Width of hand / residual palm area (mm)", min_value=20.0, max_value=160.0, value=80.0, step=1.0,
            help="Basic e-NABLE measurement sizing commonly uses hand width and hand length."
        )
        hand_length_mm = st.number_input(
            "Length of hand / residual hand area (mm)", min_value=20.0, max_value=220.0, value=80.0, step=1.0,
        )
        clearance_mm = st.number_input(
            "Added clearance allowance (mm)", min_value=0.0, max_value=15.0, value=DEFAULT_CLEARANCE_MM, step=0.5,
            help="Small allowance to avoid an overly tight shell. Adjust based on padding and design."
        )

        st.subheader("Template reference dimensions")
        reference_width_mm = st.number_input(
            "Reference STL hand width at 100% scale (mm)", min_value=10.0, max_value=200.0,
            value=REFERENCE_WIDTH_MM, step=1.0,
            help="Measure the unscaled design in your slicer/CAD if known. Defaults are placeholders."
        )
        reference_length_mm = st.number_input(
            "Reference STL hand length at 100% scale (mm)", min_value=10.0, max_value=260.0,
            value=REFERENCE_LENGTH_MM, step=1.0,
        )

        custom_options = st.text_input(
            "Allowed scale percentages", ", ".join(map(str, DEFAULT_SCALE_OPTIONS)),
            help="Comma-separated scale options to match your slicer workflow."
        )
        try:
            scale_options = sorted({int(x.strip()) for x in custom_options.split(",") if x.strip()})
            if not scale_options:
                scale_options = DEFAULT_SCALE_OPTIONS
        except Exception:
            st.error("Scale options must be comma-separated integers. Using defaults.")
            scale_options = DEFAULT_SCALE_OPTIONS

    with right:
        st.header("2) Estimated scale")
        result = calculate_sizing(
            hand_width_mm=hand_width_mm,
            hand_length_mm=hand_length_mm,
            reference_width_mm=reference_width_mm,
            reference_length_mm=reference_length_mm,
            clearance_mm=clearance_mm,
            width_weight=float(preset["width_weight"]),
            length_weight=float(preset["length_weight"]),
            scale_options=scale_options,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Recommended scale", f"{result.recommended_scale}%")
        m2.metric("Width-driven scale", f"{result.raw_width_scale:.1f}%")
        m3.metric("Length-driven scale", f"{result.raw_length_scale:.1f}%")

        if result.width_error_mm < 0:
            st.error(result.warning)
        else:
            st.info(result.warning)

        override = st.checkbox("Manually override scale")
        final_scale = result.recommended_scale
        if override:
            final_scale = st.number_input("Manual scale %", min_value=20.0, max_value=250.0, value=float(result.recommended_scale), step=1.0)

        matrix = build_scale_matrix(
            hand_width_mm, hand_length_mm, reference_width_mm, reference_length_mm, clearance_mm, scale_options
        )
        st.dataframe(matrix, use_container_width=True, hide_index=True)

    st.divider()
    st.header("3) Upload official STL set and download scaled set")
    st.write(
        "Upload the STL files for the e-NABLE design you are using. The app scales every STL by the selected percentage and packages them into a ZIP."
    )

    uploaded_files = st.file_uploader(
        "Upload one or more STL files", type=["stl"], accept_multiple_files=True
    )

    metadata = {
        "Device family": device_type,
        "Hand width mm": f"{hand_width_mm:.1f}",
        "Hand length mm": f"{hand_length_mm:.1f}",
        "Clearance mm": f"{clearance_mm:.1f}",
        "Reference width mm": f"{reference_width_mm:.1f}",
        "Reference length mm": f"{reference_length_mm:.1f}",
        "Final scale percent": f"{float(final_scale):.1f}",
    }

    if uploaded_files:
        try:
            zip_bytes = export_scaled_stls(uploaded_files, float(final_scale), metadata)
            st.success(f"Scaled {len(uploaded_files)} STL file(s) at {float(final_scale):.1f}%.")
            st.download_button(
                label="Download scaled STL ZIP",
                data=zip_bytes,
                file_name=f"enable_scaled_stl_set_{int(round(float(final_scale)))}pct.zip",
                mime="application/zip",
            )
        except Exception as exc:
            st.error(f"Could not process STL files: {exc}")
    else:
        st.info("No STL files uploaded yet. You can still download a small calibration STL to test your deployment pipeline.")
        demo_stl = create_demo_calibration_stl(float(final_scale))
        st.download_button(
            label="Download demo calibration STL only",
            data=demo_stl,
            file_name=f"demo_calibration_block_{int(round(float(final_scale)))}pct.stl",
            mime="model/stl",
        )

    st.divider()
    st.header("4) Maker checklist")
    st.checkbox("I verified the recipient measurements.")
    st.checkbox("I confirmed the selected e-NABLE design is appropriate for the limb presentation.")
    st.checkbox("I checked the scaled files in a slicer/CAD viewer before printing.")
    st.checkbox("I performed virtual fitting or physical test fitting before final use.")
    st.checkbox("I inspected all printed parts for sharp edges, weak layers, and hardware interference.")


if __name__ == "__main__":
    main()
