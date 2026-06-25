"""
Google Earth Engine backend for Agri-Moisture.

Three-stage pipeline:
  1. Data Processing — Sentinel-2 / Sentinel-1 loading, QA60 masking, ARD stacks
  2. Crop Classification — multi-temporal spectral profiles + Random Forest
  3. Phenology & Moisture Stress — growth stages and stress classification
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import ee

# ---------------------------------------------------------------------------
# Constants & label maps (for frontend legend rendering)
# ---------------------------------------------------------------------------

CROP_CLASS_LABELS: Dict[int, str] = {
    1: "Rice",
    2: "Cotton",
    3: "Fallow",
}

GROWTH_STAGE_LABELS: Dict[int, str] = {
    1: "Sowing",
    2: "Vegetative",
    3: "Flowering",
    4: "Maturity",
}

STRESS_LEVEL_LABELS: Dict[int, str] = {
    1: "Low",
    2: "Moderate",
    3: "Critical",
}

# Crop-specific seasonal NDVI / NDWI baselines indexed by [crop_class][growth_stage]
_CROP_BASELINES: Dict[int, Dict[int, Dict[str, float]]] = {
    1: {  # Rice
        1: {"ndvi": 0.12, "ndwi": 0.05},
        2: {"ndvi": 0.42, "ndwi": 0.12},
        3: {"ndvi": 0.68, "ndwi": 0.18},
        4: {"ndvi": 0.50, "ndwi": 0.10},
    },
    2: {  # Cotton
        1: {"ndvi": 0.10, "ndwi": 0.04},
        2: {"ndvi": 0.38, "ndwi": 0.08},
        3: {"ndvi": 0.62, "ndwi": 0.14},
        4: {"ndvi": 0.45, "ndwi": 0.09},
    },
    3: {  # Fallow
        1: {"ndvi": 0.08, "ndwi": 0.03},
        2: {"ndvi": 0.15, "ndwi": 0.05},
        3: {"ndvi": 0.20, "ndwi": 0.06},
        4: {"ndvi": 0.12, "ndwi": 0.04},
    },
}

DEFAULT_ROI_COORDS = [74.5, 30.5, 75.5, 31.5]
DEFAULT_START_DATE = "2024-06-01"
DEFAULT_END_DATE = "2024-10-31"
DEFAULT_COMPOSITE_MONTHS = 5
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S1_COLLECTION = "COPERNICUS/S1_GRD"
CLASSIFICATION_SCALE = 10
SAR_SCALE = 10
TRAINING_ROI_INSET_FRACTION = 0.15


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def initialize_gee(project: Optional[str] = None) -> None:
    """Authenticate (if needed) and initialize the Earth Engine API."""
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except ee.EEException:
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()


def _as_geometry(roi: Union[ee.Geometry, Sequence[float]]) -> ee.Geometry:
    if isinstance(roi, ee.Geometry):
        return roi
    return ee.Geometry.Rectangle(list(roi))


def _inset_roi_geometry(
    geometry: ee.Geometry,
    inset_fraction: float = TRAINING_ROI_INSET_FRACTION,
) -> ee.Geometry:
    """
    Shrink the ROI toward its center so training points avoid edge artifacts.

    Uses a proportional inset on the bounding envelope (works for rectangles
    and general geometries).
    """
    envelope = geometry.bounds()
    ring = ee.List(envelope.coordinates().get(0))
    west = ee.Number(ee.List(ring.get(0)).get(0))
    south = ee.Number(ee.List(ring.get(0)).get(1))
    east = ee.Number(ee.List(ring.get(2)).get(0))
    north = ee.Number(ee.List(ring.get(2)).get(1))
    dx = east.subtract(west).multiply(inset_fraction)
    dy = north.subtract(south).multiply(inset_fraction)
    inner = ee.Geometry.Rectangle(
        [west.add(dx), south.add(dy), east.subtract(dx), north.subtract(dy)]
    )
    return geometry.intersection(inner, maxError=1)


def _clamped_min(image: ee.Image, minimum: float) -> ee.Image:
    """Server-safe floor clamp (clampedMin equivalent for ee.Image)."""
    return image.max(ee.Image.constant(minimum))


def _debug_log(location: str, message: str, data: dict, hypothesis_id: str) -> None:
    # region agent log
    import json
    import time

    try:
        with open(
            "/home/mrmorax/projects/Agri-Moisture/.cursor/debug-bb0b3c.log",
            "a",
            encoding="utf-8",
        ) as log_file:
            log_file.write(
                json.dumps(
                    {
                        "sessionId": "bb0b3c",
                        "runId": "pre-fix",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data,
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except OSError:
        pass
    # endregion


# ---------------------------------------------------------------------------
# Stage 1 — Data Processing Block (Analysis Ready Data)
# ---------------------------------------------------------------------------


def apply_qa60_cloud_mask(image: ee.Image) -> ee.Image:
    """
    Mask clouds and cirrus using the Sentinel-2 QA60 bitmask.

    Bits 10 (clouds) and 11 (cirrus) are cleared; surface reflectance is
    scaled to [0, 1].
    """
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    clear = (
        qa.bitwiseAnd(cloud_bit)
        .eq(0)
        .And(qa.bitwiseAnd(cirrus_bit).eq(0))
    )
    scaled = image.updateMask(clear).divide(10000)
    return scaled.copyProperties(image, ["system:time_start", "system:time_end"])


def _add_optical_indices(image: ee.Image) -> ee.Image:
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    return image.addBands([ndvi, ndwi])


def load_sentinel2_collection(
    roi: Union[ee.Geometry, Sequence[float]],
    start_date: str,
    end_date: str,
) -> ee.ImageCollection:
    """Load Sentinel-2 SR with proper scaling and index attachments."""
    geometry = _as_geometry(roi)
    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 95)) 
        .map(lambda img: img.select(["B.*", "QA60"])) # Keep explicit bands
        .map(lambda img: img.multiply(0.0001).copyProperties(img, ["system:time_start", "system:time_end"]))
        .map(_add_optical_indices)
    )
    return collection


def _preprocess_sentinel1(image: ee.Image) -> ee.Image:
    vv_db = image.select("VV").log10().multiply(10).rename("VV")
    vh_db = image.select("VH").log10().multiply(10).rename("VH")
    vh_vv_ratio = (
        image.select("VH")
        .divide(image.select("VV"))
        .rename("VH_VV_ratio")
    )
    # Drop all original bands before adding processed ones to avoid duplicates
    return (
        image.select([])
        .addBands(vv_db)
        .addBands(vh_db)
        .addBands(vh_vv_ratio)
        .copyProperties(image, ["system:time_start"])
    )


def load_sentinel1_collection(
    roi: Union[ee.Geometry, Sequence[float]],
    start_date: str,
    end_date: str,
) -> ee.ImageCollection:
    """Load Sentinel-1 GRD IW dual-pol scenes with VV, VH, and VH/VV ratio."""
    geometry = _as_geometry(roi)
    collection = (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .map(_preprocess_sentinel1)
    )
    return collection


def _monthly_composites(
    collection: ee.ImageCollection,
    start_date: str,
    num_months: int,
    band_names: Sequence[str],
    prefix: str,
) -> ee.Image:
    """Build a multi-temporal stack of monthly median composites."""
    result = None
    for i in range(num_months):
        month_start = ee.Date(start_date).advance(i, "month")
        month_end = month_start.advance(1, "month")
        suffix = f"_M{i}"
        new_names = [f"{prefix}{b}{suffix}" for b in band_names]
        median = collection.filterDate(month_start, month_end).median()
        selected = median.select(list(band_names)).rename(new_names)
        if result is None:
            result = selected
        else:
            result = result.addBands(selected)
    return result


def build_analysis_ready_data(
    roi: Union[ee.Geometry, Sequence[float]],
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    num_months: int = DEFAULT_COMPOSITE_MONTHS,
) -> Dict[str, ee.Image]:
    """
    Stage 1 output: unified multi-temporal optical and SAR stacks.

    Returns
    -------
    dict with keys:
        optical_stack, sar_stack, analysis_ready_stack, s2_collection, s1_collection
    """
    geometry = _as_geometry(roi)
    s2 = load_sentinel2_collection(geometry, start_date, end_date)
    s1 = load_sentinel1_collection(geometry, start_date, end_date)

    optical_stack = _monthly_composites(
        s2, start_date, num_months, ("NDVI", "NDWI"), prefix="OPT_"
    )
    sar_stack = _monthly_composites(
        s1,
        start_date,
        num_months,
        ("VV", "VH", "VH_VV_ratio"),
        prefix="SAR_",
    )

    analysis_ready_stack = optical_stack.addBands(sar_stack).clip(geometry)

    return {
        "optical_stack": optical_stack.clip(geometry),
        "sar_stack": sar_stack.clip(geometry),
        "analysis_ready_stack": analysis_ready_stack,
        "s2_collection": s2,
        "s1_collection": s1,
    }


def get_latest_clear_composite(
    s2_collection: ee.ImageCollection,
    roi: Union[ee.Geometry, Sequence[float]],
) -> ee.Image:
    """Most recent 30-day median composite for live index calculation."""
    geometry = _as_geometry(roi)
    latest_date = ee.Date(
        s2_collection.sort("system:time_start", False).first()
        .get("system:time_start")
    )
    return (
        s2_collection
        .filterDate(latest_date.advance(-30, "day"), latest_date.advance(1, "day"))
        .median()
        .clip(geometry)
    )


# ---------------------------------------------------------------------------
# Stage 2 — Crop Type Classification Model Block
# ---------------------------------------------------------------------------


def create_mock_training_samples(
    roi: Union[ee.Geometry, Sequence[float]],
    samples_per_class: int = 25,
    seed: int = 42,
    inset_fraction: float = TRAINING_ROI_INSET_FRACTION,
) -> ee.FeatureCollection:
    """
    Generate mock ground-truth points for Rice, Cotton, and Fallow classes.

    Points are randomly distributed within a center-buffered inset of the ROI
    to avoid sampling near boundaries where stacks are often masked empty.
    """
    geometry = _as_geometry(roi)
    sample_region = _inset_roi_geometry(geometry, inset_fraction)
    collections = []
    for class_id in CROP_CLASS_LABELS:
        points = ee.FeatureCollection.randomPoints(
            region=sample_region,
            points=samples_per_class,
            seed=seed + class_id,
        ).map(lambda feature: feature.set("crop_class", class_id))
        collections.append(points)
    return (
        ee.FeatureCollection(collections[0])
        .merge(collections[1])
        .merge(collections[2])
    )


def build_temporal_feature_collection(
    analysis_ready_stack: ee.Image,
    training_points: ee.FeatureCollection,
    scale: int = CLASSIFICATION_SCALE,
) -> ee.FeatureCollection:
    """Sample multi-temporal features safely by dynamically unmasking nulls."""
    band_names = analysis_ready_stack.bandNames()
    
    # 🟢 FIXED: unmask(0, False) broadcasts 0 to ALL 25 bands 
    # and forces a global footprint so nulls are mathematically impossible.
    safe_stack = analysis_ready_stack.unmask(0, False)

    sampled = safe_stack.sampleRegions(
        collection=training_points,
        properties=["crop_class"],
        scale=scale,
        tileScale=4,
        geometries=True,
    )
    
    # Filter out entries where the properties are entirely missing
    return sampled.filter(ee.Filter.notNull(band_names))

def train_crop_classifier(
    training_fc: ee.FeatureCollection,
    feature_band_names: Sequence[str],
    number_of_trees: int = 100,
) -> ee.Classifier:
    """Train a Smile Random Forest on temporal spectral profile features."""
    return ee.Classifier.smileRandomForest(numberOfTrees=number_of_trees).train(
        features=training_fc,
        classProperty="crop_class",
        inputProperties=list(feature_band_names),
    )


def classify_crops(
    classifier: ee.Classifier,
    analysis_ready_stack: ee.Image,
    roi: Union[ee.Geometry, Sequence[float]],
) -> ee.Image:
    """Apply the trained classifier and return an integer crop-type map."""
    geometry = _as_geometry(roi)
    crop_map = (
        analysis_ready_stack.classify(classifier)
        .toInt()
        .rename("crop_class")
        .clip(geometry)
    )
    return crop_map


def run_crop_classification_block(
    analysis_ready_stack: ee.Image,
    roi: Union[ee.Geometry, Sequence[float]],
    samples_per_class: int = 100,
) -> Dict[str, Any]:
    """
    Stage 2 orchestration: features → train → crop type map.

    Returns classifier, training_fc, and crop_type_map (ee.Image).
    """
    training_points = create_mock_training_samples(roi, samples_per_class)
    training_fc = build_temporal_feature_collection(
        analysis_ready_stack, training_points
    )
    # getInfo() pulls the server-side ee.List to a plain Python list
    band_names = analysis_ready_stack.bandNames().getInfo()
    classifier = train_crop_classifier(training_fc, band_names)
    crop_type_map = classify_crops(classifier, analysis_ready_stack, roi)
    return {
        "classifier": classifier,
        "training_fc": training_fc,
        "training_points": training_points,
        "crop_type_map": crop_type_map,
    }


# ---------------------------------------------------------------------------
# Stage 3 — Phenology & Moisture Stress Detection
# ---------------------------------------------------------------------------


def _baseline_image(
    crop_map: ee.Image,
    stage_map: ee.Image,
    index_name: str,
) -> ee.Image:
    """Build expected NDVI or NDWI baseline from crop class and growth stage."""
    baseline = ee.Image(0).rename(index_name)
    for crop_id, stages in _CROP_BASELINES.items():
        for stage_id, values in stages.items():
            mask = crop_map.eq(crop_id).And(stage_map.eq(stage_id))
            baseline = baseline.where(
                mask, ee.Image(values[index_name]).rename(index_name)
            )
    return baseline


def map_growth_stages(
    optical_stack: ee.Image,
    roi: Union[ee.Geometry, Sequence[float]],
) -> ee.Image:
    """
    Derive crop growth stages from temporal NDVI trajectory thresholds.

    Uses seasonal max NDVI and current NDVI to distinguish sowing through maturity.
    """
    geometry = _as_geometry(roi)

    ndvi_band_names = optical_stack.bandNames().filter(
        ee.Filter.stringContains("item", "NDVI")
    )
    ndvi_bands = optical_stack.select(ndvi_band_names)
    sorted_names = ndvi_band_names.sort()
    last_index = ndvi_band_names.size().subtract(1)
    current_ndvi = (
        ndvi_bands.select([sorted_names.get(last_index)]).rename("current_ndvi")
    )

    max_ndvi = ndvi_bands.reduce(ee.Reducer.max()).rename("max_ndvi")
    min_ndvi = ndvi_bands.reduce(ee.Reducer.min()).rename("min_ndvi")
    ndvi_range = max_ndvi.subtract(min_ndvi)

    sowing = current_ndvi.lt(0.22)
    vegetative = (
        current_ndvi.gte(0.22)
        .And(current_ndvi.lt(0.50))
        .And(current_ndvi.lt(max_ndvi.multiply(0.75)))
    )
    flowering = current_ndvi.gte(0.50).And(
        ndvi_range.gt(0.15).And(current_ndvi.gte(max_ndvi.multiply(0.70)))
    )
    maturity = current_ndvi.gte(0.30).And(
        ndvi_range.gt(0.10).And(current_ndvi.lt(max_ndvi.multiply(0.80)))
    )

    stage = (
        ee.Image(1)
        .where(vegetative, 2)
        .where(flowering, 3)
        .where(maturity, 4)
        .where(sowing, 1)
        .rename("growth_stage")
        .clip(geometry)
    )
    _debug_log(
        "core_gee.py:map_growth_stages",
        "growth stage graph built",
        {"method": "reduce_max_min"},
        "C",
    )
    return stage.toInt()


def compute_live_spectral_indices(
    s2_collection: ee.ImageCollection,
    roi: Union[ee.Geometry, Sequence[float]],
) -> Dict[str, ee.Image]:
    """Calculate live NDVI and NDWI from the most recent clear observation."""
    latest = get_latest_clear_composite(s2_collection, roi)
    ndvi = latest.select("NDVI").rename("NDVI_live")
    ndwi = latest.select("NDWI").rename("NDWI_live")
    return {"ndvi_live": ndvi, "ndwi_live": ndwi}


def build_moisture_stress_map(
    crop_type_map: ee.Image,
    growth_stage_map: ee.Image,
    ndvi_live: ee.Image,
    ndwi_live: ee.Image,
    roi: Union[ee.Geometry, Sequence[float]],
) -> ee.Image:
    """
    Flag moisture stress when live indices fall below crop-specific baselines.

    Stress levels:
        1 = Low, 2 = Moderate, 3 = Critical
    """
    geometry = _as_geometry(roi)
    expected_ndvi = _baseline_image(crop_type_map, growth_stage_map, "ndvi")
    expected_ndwi = _baseline_image(crop_type_map, growth_stage_map, "ndwi")

    safe_expected_ndvi = _clamped_min(expected_ndvi, 0.05)
    safe_expected_ndwi = _clamped_min(expected_ndwi, 0.05)
    ndvi_deficit = expected_ndvi.subtract(ndvi_live).divide(safe_expected_ndvi)
    ndwi_deficit = expected_ndwi.subtract(ndwi_live).divide(safe_expected_ndwi)
    combined_deficit = _clamped_min(
        ndvi_deficit.add(ndwi_deficit).divide(2),
        0.0,
    )

    low = combined_deficit.lt(0.15)
    moderate = combined_deficit.gte(0.15).And(combined_deficit.lt(0.35))
    critical = combined_deficit.gte(0.35)

    stress = (
        ee.Image(0)
        .where(low, 1)
        .where(moderate, 2)
        .where(critical, 3)
        .rename("moisture_stress")
        .clip(geometry)
    )
    _debug_log(
        "core_gee.py:build_moisture_stress_map",
        "moisture stress graph built",
        {"used_clamped_min": True},
        "B",
    )
    return stress.toInt()


def run_phenology_moisture_block(
    crop_type_map: ee.Image,
    optical_stack: ee.Image,
    s2_collection: ee.ImageCollection,
    roi: Union[ee.Geometry, Sequence[float]],
) -> Dict[str, ee.Image]:
    """Stage 3 orchestration: growth stages, live indices, moisture stress."""
    growth_stage_map = map_growth_stages(optical_stack, roi)
    live_indices = compute_live_spectral_indices(s2_collection, roi)
    moisture_stress_map = build_moisture_stress_map(
        crop_type_map,
        growth_stage_map,
        live_indices["ndvi_live"],
        live_indices["ndwi_live"],
        roi,
    )
    return {
        "growth_stage_map": growth_stage_map,
        "ndvi_live": live_indices["ndvi_live"],
        "ndwi_live": live_indices["ndwi_live"],
        "moisture_stress_map": moisture_stress_map,
    }


# ---------------------------------------------------------------------------
# Pipeline orchestrator & visualization helpers
# ---------------------------------------------------------------------------


@dataclass
class PipelineOutputs:
    """All Earth Engine image layers produced by the full pipeline."""

    # Stage 1
    optical_stack: ee.Image
    sar_stack: ee.Image
    analysis_ready_stack: ee.Image

    # Stage 2
    crop_type_map: ee.Image

    # Stage 3
    growth_stage_map: ee.Image
    ndvi_live: ee.Image
    ndwi_live: ee.Image
    moisture_stress_map: ee.Image

    # Reference collections (useful for time-series widgets)
    s2_collection: ee.ImageCollection
    s1_collection: ee.ImageCollection

    roi: ee.Geometry


def run_pipeline(
    roi: Union[ee.Geometry, Sequence[float]] = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    num_months: int = DEFAULT_COMPOSITE_MONTHS,
    samples_per_class: int = 100,
    initialize: bool = True,
    project: Optional[str] = None,
) -> PipelineOutputs:
    """
    Execute the full 3-stage Agri-Moisture GEE pipeline.

    All map layers are returned as ``ee.Image`` objects for Streamlit / geemap.
    """
    if initialize:
        initialize_gee(project=project)

    if roi is None:
        roi = ee.Geometry.Rectangle(DEFAULT_ROI_COORDS)
    elif not isinstance(roi, ee.Geometry):
        roi = ee.Geometry.Rectangle(list(roi))

    geometry = _as_geometry(roi)

    try:
        _debug_log(
            "core_gee.py:run_pipeline",
            "pipeline start",
            {"start_date": start_date, "end_date": end_date},
            "A",
        )
        stage1 = build_analysis_ready_data(
            geometry, start_date, end_date, num_months
        )
        _debug_log(
            "core_gee.py:run_pipeline",
            "stage1 complete",
            {"bands": stage1["analysis_ready_stack"].bandNames().getInfo()},
            "A",
        )
        stage2 = run_crop_classification_block(
            stage1["analysis_ready_stack"], geometry, samples_per_class
        )
        _debug_log(
            "core_gee.py:run_pipeline",
            "stage2 complete",
            {},
            "A",
        )
        stage3 = run_phenology_moisture_block(
            stage2["crop_type_map"],
            stage1["optical_stack"],
            stage1["s2_collection"],
            geometry,
        )
        _debug_log(
            "core_gee.py:run_pipeline",
            "stage3 complete",
            {},
            "A",
        )
    except Exception as exc:
        _debug_log(
            "core_gee.py:run_pipeline",
            "pipeline failed",
            {"error_type": type(exc).__name__, "error": str(exc)},
            "A",
        )
        raise

    return PipelineOutputs(
        optical_stack=stage1["optical_stack"],
        sar_stack=stage1["sar_stack"],
        analysis_ready_stack=stage1["analysis_ready_stack"],
        crop_type_map=stage2["crop_type_map"],
        growth_stage_map=stage3["growth_stage_map"],
        ndvi_live=stage3["ndvi_live"],
        ndwi_live=stage3["ndwi_live"],
        moisture_stress_map=stage3["moisture_stress_map"],
        s2_collection=stage1["s2_collection"],
        s1_collection=stage1["s1_collection"],
        roi=geometry,
    )


def get_visualization_params() -> Dict[str, Dict[str, Any]]:
    """Default symbology for each output layer (Streamlit / geemap legends)."""
    return {
        "crop_type_map": {
            "min": 1,
            "max": 3,
            "palette": ["#1a9850", "#fdae61", "#cccccc"],
            "classes": CROP_CLASS_LABELS,
        },
        "growth_stage_map": {
            "min": 1,
            "max": 4,
            "palette": ["#8dd3c7", "#80b1d3", "#fb8072", "#fdb462"],
            "classes": GROWTH_STAGE_LABELS,
        },
        "moisture_stress_map": {
            "min": 1,
            "max": 3,
            "palette": ["#1a9850", "#fee08b", "#d73027"],
            "classes": STRESS_LEVEL_LABELS,
        },
        "ndvi_live": {
            "min": 0,
            "max": 0.9,
            "palette": ["#d73027", "#fee08b", "#1a9850"],
        },
        "ndwi_live": {
            "min": -0.2,
            "max": 0.5,
            "palette": ["#d73027", "#ffffff", "#4575b4"],
        },
    }


def layers_for_frontend(outputs: PipelineOutputs) -> Dict[str, ee.Image]:
    """Flat dict of named layers ready for ``st.session_state`` or geemap."""
    return {
        "optical_stack": outputs.optical_stack,
        "sar_stack": outputs.sar_stack,
        "analysis_ready_stack": outputs.analysis_ready_stack,
        "crop_type_map": outputs.crop_type_map,
        "growth_stage_map": outputs.growth_stage_map,
        "ndvi_live": outputs.ndvi_live,
        "ndwi_live": outputs.ndwi_live,
        "moisture_stress_map": outputs.moisture_stress_map,
    }