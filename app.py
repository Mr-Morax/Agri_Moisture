"""
Agri-Moisture Streamlit dashboard.

Maps the downstream flowchart stages:
  • Meteorological / ancillary integration (mock crop water balance)
  • Irrigation advisory map generation
  • Interactive geemap visualization with KPI summaries
"""
from __future__ import annotations

from streamlit_folium import st_folium
import sys
from importlib import metadata
class FakePkgResources:
    def get_distribution(self, name):
        return metadata.distribution(name)
sys.modules['pkg_resources'] = FakePkgResources()


from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import ee
import geemap.foliumap as geemap
import pandas as pd
import streamlit as st

import core_gee

# ---------------------------------------------------------------------------
# Ancillary meteorological mock data (historical pilot-area averages)
# ---------------------------------------------------------------------------

HISTORICAL_METEO: Dict[str, float] = {
    "rainfall_mm_week": 28.0,
    "et0_mm_week": 45.0,
}

# FAO-style crop coefficients Kc by [crop_class][growth_stage]
KC_COEFFICIENTS: Dict[int, Dict[int, float]] = {
    1: {1: 0.65, 2: 1.05, 3: 1.25, 4: 0.95},  # Rice
    2: {1: 0.45, 2: 0.85, 3: 1.15, 4: 0.75},  # Cotton
    3: {1: 0.30, 2: 0.40, 3: 0.50, 4: 0.35},  # Fallow
}

IRRIGATION_ADVISORY_LABELS: Dict[int, str] = {
    1: "No Action / Soil Moist",
    2: "Monitor / Light Deficit",
    3: "Immediate Irrigation / High Deficit",
}

MAP_PRODUCTS: Dict[str, str] = {
    "Crop Type Map": "crop_type_map",
    "Moisture Stress Level Map": "moisture_stress_map",
    "Irrigation Advisory Map": "irrigation_advisory_map",
}


@dataclass
class WaterBalanceSummary:
    """Pixel-aggregated water balance statistics for KPI display."""

    mean_cwr_mm: float
    mean_cwb_mm: float
    mean_deficit_mm: float
    rainfall_mm: float
    et0_mm: float


# ---------------------------------------------------------------------------
# Rule-based crop water balance (mock meteorological integration)
# ---------------------------------------------------------------------------


def crop_water_requirement(
    crop_class: int,
    growth_stage: int,
    et0_mm_week: float = HISTORICAL_METEO["et0_mm_week"],
) -> float:
    """Crop water requirement (mm/week): CWR = Kc × ET₀."""
    kc = KC_COEFFICIENTS.get(crop_class, {}).get(growth_stage, 0.5)
    return kc * et0_mm_week


def crop_water_balance(
    rainfall_mm_week: float,
    crop_water_requirement_mm: float,
) -> float:
    """Crop water balance (mm/week): CWB = rainfall − CWR."""
    return rainfall_mm_week - crop_water_requirement_mm


def _build_kc_image(
    crop_type_map: ee.Image,
    growth_stage_map: ee.Image,
) -> ee.Image:
    """Spatial Kc layer from crop type and phenology stage."""
    kc = ee.Image(0.5)
    for crop_id, stages in KC_COEFFICIENTS.items():
        for stage_id, kc_value in stages.items():
            mask = crop_type_map.eq(crop_id).And(growth_stage_map.eq(stage_id))
            kc = kc.where(mask, ee.Image(kc_value))
    return kc.rename("kc")


def build_water_balance_layers(
    crop_type_map: ee.Image,
    growth_stage_map: ee.Image,
    roi: ee.Geometry,
    rainfall_mm_week: float = HISTORICAL_METEO["rainfall_mm_week"],
    et0_mm_week: float = HISTORICAL_METEO["et0_mm_week"],
) -> Dict[str, ee.Image]:
    """
    Generate spatial crop water requirement, balance, and deficit layers.

    CWR = Kc × ET₀
    CWB = rainfall − CWR
    Deficit = max(0, CWR − rainfall)
    """
    kc = _build_kc_image(crop_type_map, growth_stage_map)
    rainfall = ee.Image(rainfall_mm_week).rename("rainfall")
    cwr = kc.multiply(et0_mm_week).rename("crop_water_requirement")
    cwb = rainfall.subtract(cwr).rename("crop_water_balance")
    deficit = cwr.subtract(rainfall).max(0).rename("water_deficit")
    return {
        "kc": kc.clip(roi),
        "crop_water_requirement": cwr.clip(roi),
        "crop_water_balance": cwb.clip(roi),
        "water_deficit": deficit.clip(roi),
    }


def build_irrigation_advisory_map(
    moisture_stress_map: ee.Image,
    water_deficit: ee.Image,
    roi: ee.Geometry,
    light_deficit_mm: float = 8.0,
    high_deficit_mm: float = 18.0,
) -> ee.Image:
    """
    Combine spectral moisture stress with crop water deficit logic.

    Classes
    -------
    1 — Green:  No action (low stress, balanced or surplus water)
    2 — Yellow: Monitor (moderate stress or light deficit)
    3 — Red:    Immediate irrigation (critical stress or high deficit)
    """
    no_action = moisture_stress_map.eq(1).And(water_deficit.lte(0))
    monitor = (
        moisture_stress_map.eq(2)
        .Or(water_deficit.gt(0).And(water_deficit.lt(light_deficit_mm)))
    )
    immediate = moisture_stress_map.eq(3).Or(water_deficit.gte(high_deficit_mm))

    advisory = (
        ee.Image(2)
        .where(no_action, 1)
        .where(monitor, 2)
        .where(immediate, 3)
        .rename("irrigation_advisory")
        .clip(roi)
    )
    return advisory.toInt()


# ---------------------------------------------------------------------------
# Area statistics helpers
# ---------------------------------------------------------------------------


def compute_class_area_hectares(
    class_image: ee.Image,
    roi: ee.Geometry,
    scale: int = 30,
) -> Dict[int, float]:
    """Grouped area (ha) per integer class value inside the ROI."""
    area_ha = ee.Image.pixelArea().divide(10000)
    grouped = (
        area_ha.addBands(class_image.rename("class"))
        .reduceRegion(
            reducer=ee.Reducer.sum().group(groupField=1, groupName="class"),
            geometry=roi,
            scale=scale,
            maxPixels=1e13,
        )
        .get("groups")
    )
    result: Dict[int, float] = {}
    for group in grouped.getInfo():
        result[int(group["class"])] = round(group["sum"], 1)
    return result


def compute_image_mean(
    image: ee.Image,
    roi: ee.Geometry,
    scale: int = 30,
) -> float:
    mean = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=scale,
        maxPixels=1e13,
    )
    value = mean.get(image.bandNames().get(0))
    return round(float(value.getInfo()), 2)


def summarize_water_balance(
    water_layers: Dict[str, ee.Image],
    roi: ee.Geometry,
) -> WaterBalanceSummary:
    return WaterBalanceSummary(
        mean_cwr_mm=compute_image_mean(water_layers["crop_water_requirement"], roi),
        mean_cwb_mm=compute_image_mean(water_layers["crop_water_balance"], roi),
        mean_deficit_mm=compute_image_mean(water_layers["water_deficit"], roi),
        rainfall_mm=HISTORICAL_METEO["rainfall_mm_week"],
        et0_mm=HISTORICAL_METEO["et0_mm_week"],
    )


def total_roi_hectares(roi: ee.Geometry, scale: int = 30) -> float:
    area = (
        ee.Image.pixelArea()
        .divide(10000)
        .rename("area_ha")
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=scale,
            maxPixels=1e13,
        )
        .get("area_ha")
    )
    return round(float(area.getInfo()), 1)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def get_layer_vis_params(layer_key: str) -> Dict[str, Any]:
    base = core_gee.get_visualization_params()
    if layer_key == "irrigation_advisory_map":
        return {
            "min": 1,
            "max": 3,
            "palette": ["#1a9850", "#fee08b", "#d73027"],
        }
    return {
        k: v
        for k, v in base.get(layer_key, {}).items()
        if k in ("min", "max", "palette")
    }

import folium

def build_map(
    outputs: core_gee.PipelineOutputs,
    irrigation_map: ee.Image,
    active_product: str,
) -> folium.Map:
    """Build a robust Folium map directly, bypassing geemap internal cache issues."""
    
    # 1. Initialize a native Folium Map centered on your data
    m = folium.Map(location=[31.0, 75.0], zoom_start=9, control_scale=True)
    
    # 2. Add standard OpenStreetMap tiles explicitly
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    # 3. Handle layer selections and fetch visualization parameters
    layer_key = MAP_PRODUCTS[active_product]
    layers = {
        "crop_type_map": outputs.crop_type_map,
        "moisture_stress_map": outputs.moisture_stress_map,
        "irrigation_advisory_map": irrigation_map,  # <-- Use the actual parameter name
    }
    ee_image = layers[layer_key]
    vis = get_layer_vis_params(layer_key)

    # 4. Convert the GEE Image into a standard Map Tile URL natively
    map_id_dict = ee.Image(ee_image).getMapId(vis)
    tile_url = map_id_dict["tile_fetcher"].url_format

    # 5. Inject the Earth Engine layer directly into the map
    folium.TileLayer(
        tiles=tile_url,
        attr="Google Earth Engine",
        name=active_product,
        overlay=True,
        control=True,
        opacity=0.85
    ).add_to(m)

    # 6. Add the ROI Bounding Box boundary line
    roi_vis = outputs.roi.bounds().getInfo()
    folium.GeoJson(
        roi_vis,
        name="Pilot ROI",
        style_function=lambda x: {
            "color": "#ffff00",
            "fillColor": "#00000000",
            "weight": 2
        }
    ).add_to(m)

    return m

def render_legend(active_product: str) -> None:
    if active_product == "Crop Type Map":
        labels = core_gee.CROP_CLASS_LABELS
        colors = ["#1a9850", "#fdae61", "#cccccc"]
    elif active_product == "Moisture Stress Level Map":
        labels = core_gee.STRESS_LEVEL_LABELS
        colors = ["#1a9850", "#fee08b", "#d73027"]
    else:
        labels = IRRIGATION_ADVISORY_LABELS
        colors = ["#1a9850", "#fee08b", "#d73027"]

    legend_html = "<div style='display:flex;gap:1rem;flex-wrap:wrap;margin-top:0.5rem;'>"
    for idx, (class_id, label) in enumerate(labels.items()):
        color = colors[idx] if idx < len(colors) else "#888888"
        legend_html += (
            f"<span style='display:inline-flex;align-items:center;gap:0.35rem;'>"
            f"<span style='width:14px;height:14px;background:{color};"
            f"display:inline-block;border:1px solid #333;'></span>{label}</span>"
        )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)


def render_growth_stage_chart(stage_areas: Dict[int, float]) -> None:
    rows = []
    for stage_id, label in core_gee.GROWTH_STAGE_LABELS.items():
        rows.append(
            {
                "Growth Stage": label,
                "Area (ha)": stage_areas.get(stage_id, 0.0),
            }
        )
    df = pd.DataFrame(rows)
    st.bar_chart(df.set_index("Growth Stage"))


# ---------------------------------------------------------------------------
# Pipeline orchestration for the dashboard
# ---------------------------------------------------------------------------


def run_dashboard_pipeline(
    roi_bounds: Tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    gee_project: Optional[str],
) -> Tuple[core_gee.PipelineOutputs, ee.Image, Dict[str, ee.Image], WaterBalanceSummary]:
    outputs = core_gee.run_pipeline(
        roi=list(roi_bounds),
        start_date=start_date,
        end_date=end_date,
        initialize=True,
        project=gee_project or None,
    )
    water_layers = build_water_balance_layers(
        outputs.crop_type_map,
        outputs.growth_stage_map,
        outputs.roi,
    )
    irrigation_map = build_irrigation_advisory_map(
        outputs.moisture_stress_map,
        water_layers["water_deficit"],
        outputs.roi,
    )
    summary = summarize_water_balance(water_layers, outputs.roi)
    return outputs, irrigation_map, water_layers, summary


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Agri-Moisture Advisory Dashboard",
        page_icon="🌾",
        layout="wide",
    )

    st.title("Agri-Moisture Advisory Dashboard")
    st.caption(
        "Satellite-driven crop monitoring, moisture stress detection, "
        "and rule-based irrigation advisory for the pilot ROI."
    )

    with st.sidebar:
        st.header("Configuration")
        gee_project = st.text_input(
            "Google Earth Engine project ID",
            value=st.session_state.get("gee_project", ""),
            help="Required if Earth Engine is not already initialized.",
        )
        st.subheader("Pilot ROI")
        roi_west = st.number_input("West", value=74.5, format="%.4f")
        roi_south = st.number_input("South", value=30.5, format="%.4f")
        roi_east = st.number_input("East", value=75.5, format="%.4f")
        roi_north = st.number_input("North", value=31.5, format="%.4f")
        start_date = st.date_input("Season start", value=pd.Timestamp("2024-06-01"))
        end_date = st.date_input("Season end", value=pd.Timestamp("2024-10-31"))

        st.subheader("Ancillary Meteorology (mock)")
        rainfall = st.number_input(
            "Historical avg. rainfall (mm/week)",
            value=HISTORICAL_METEO["rainfall_mm_week"],
        )
        et0 = st.number_input(
            "Historical avg. ET₀ (mm/week)",
            value=HISTORICAL_METEO["et0_mm_week"],
        )
        HISTORICAL_METEO["rainfall_mm_week"] = rainfall
        HISTORICAL_METEO["et0_mm_week"] = et0

        st.subheader("Map Product")
        active_product = st.radio(
            "Display layer",
            list(MAP_PRODUCTS.keys()),
            index=0,
        )

        run_clicked = st.button("Run Analysis Pipeline", type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("Running GEE pipeline and generating advisory layers…"):
            try:
                outputs, irrigation_map, water_layers, summary = run_dashboard_pipeline(
                    roi_bounds=(roi_west, roi_south, roi_east, roi_north),
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    gee_project=gee_project.strip() or None,
                )
                st.session_state["outputs"] = outputs
                st.session_state["irrigation_map"] = irrigation_map
                st.session_state["water_layers"] = water_layers
                st.session_state["water_summary"] = summary
                st.session_state["gee_project"] = gee_project
                st.session_state["analysis_ready"] = True
            except Exception as exc:
                st.session_state["analysis_ready"] = False
                st.error(f"Pipeline failed: {exc}")

    if not st.session_state.get("analysis_ready"):
        st.info(
            "Configure the pilot ROI and ancillary meteorology in the sidebar, "
            "then click **Run Analysis Pipeline** to load map products."
        )
        st.markdown(
            "**Flowchart stages covered:** Crop Type Map → Moisture Stress Level Map "
            "→ Crop Water Balance → Irrigation Advisory Map"
        )
        return

    outputs: core_gee.PipelineOutputs = st.session_state["outputs"]
    irrigation_map: ee.Image = st.session_state["irrigation_map"]
    summary: WaterBalanceSummary = st.session_state["water_summary"]

    stress_areas = compute_class_area_hectares(outputs.moisture_stress_map, outputs.roi)
    stage_areas = compute_class_area_hectares(outputs.growth_stage_map, outputs.roi)
    advisory_areas = compute_class_area_hectares(irrigation_map, outputs.roi)
    pilot_ha = total_roi_hectares(outputs.roi)
    critical_ha = stress_areas.get(3, 0.0)
    critical_pct = (critical_ha / pilot_ha * 100) if pilot_ha else 0.0
    irrigation_red_ha = advisory_areas.get(3, 0.0)

    st.subheader("Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Pilot area", f"{pilot_ha:,.1f} ha")
    kpi2.metric(
        "Critical moisture stress",
        f"{critical_ha:,.1f} ha",
        delta=f"{critical_pct:.1f}% of ROI",
        delta_color="inverse",
    )
    kpi3.metric(
        "Immediate irrigation zone",
        f"{irrigation_red_ha:,.1f} ha",
        delta="High deficit + critical stress",
        delta_color="inverse",
    )
    kpi4.metric(
        "Mean crop water deficit",
        f"{summary.mean_deficit_mm:.1f} mm/wk",
        delta=f"CWR {summary.mean_cwr_mm:.1f} | CWB {summary.mean_cwb_mm:.1f}",
    )

    st.subheader("Growth Stage Distribution")
    stage_cols = st.columns(len(core_gee.GROWTH_STAGE_LABELS))
    for idx, (stage_id, label) in enumerate(core_gee.GROWTH_STAGE_LABELS.items()):
        area = stage_areas.get(stage_id, 0.0)
        pct = (area / pilot_ha * 100) if pilot_ha else 0.0
        stage_cols[idx].metric(label, f"{area:,.1f} ha", delta=f"{pct:.1f}%")

    chart_col, meta_col = st.columns([2, 1])
    with chart_col:
        render_growth_stage_chart(stage_areas)
    with meta_col:
        st.markdown("**Crop water balance (rule-based mock)**")
        st.write(
            f"- Rainfall (avg): **{summary.rainfall_mm:.1f} mm/week**\n"
            f"- ET₀ (avg): **{summary.et0_mm:.1f} mm/week**\n"
            f"- Mean CWR: **{summary.mean_cwr_mm:.1f} mm/week**\n"
            f"- Mean CWB: **{summary.mean_cwb_mm:.1f} mm/week**"
        )
        st.markdown("**Advisory breakdown (ha)**")
        for class_id, label in IRRIGATION_ADVISORY_LABELS.items():
            st.write(f"- {label}: **{advisory_areas.get(class_id, 0.0):,.1f}**")

    st.subheader(active_product)
    render_legend(active_product)
    m = build_map(outputs, irrigation_map, active_product)
    # Add Folium's native layer toggle
    folium.LayerControl().add_to(m)
    
    # Render the native Folium map in Streamlit
    st_folium(m, use_container_width=True, height=620, returned_objects=[])


if __name__ == "__main__":
    main()
