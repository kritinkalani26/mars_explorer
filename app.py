"""
Mars Surface Explorer — Flask backend
Fetches NASA Perseverance rover images and classifies terrain using CLIP.
"""

import io
import json
import logging
import math
import os
import re
import threading
import time
from pathlib import Path

import requests
import torch
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────
NASA_API_KEY = os.getenv("NASA_API_KEY", "DEMO_KEY")
NASA_IMAGES_URL = "https://images-api.nasa.gov/search"  # No key needed
CACHE_FILE = Path("cache/classifications.json")
PAGE_SIZE = 12          # Images returned per API page
MAX_PHOTOS = 48         # Cap photos fetched per sol (keeps first-run manageable)
SOL_CACHE_TTL = 300     # Seconds before re-checking latest sol from NASA
CONFIDENCE_THRESHOLD = 0.30  # Below this the prediction is reported as "Unknown"

WAYPOINTS_URL  = "https://mars.nasa.gov/mmgis-maps/M20/Layers/json/M20_waypoints.json"
WAYPOINTS_FILE = Path("cache/waypoints.json")
WAYPOINTS_TTL  = 86400  # Refresh waypoints daily

# ── Satellite terrain overlay (Mars Trek orbital tiles + CLIP) ──────────────────
# Prompts tuned for orbital/satellite imagery — different viewpoint from surface photos.
SAT_TERRAIN_PROMPTS = [
    "sandy aeolian dunes and wind-blown deposits on Mars surface seen from orbit",
    "rocky bedrock outcrops and geological formations on Mars seen from satellite",
    "gravel and small debris field on Mars seen from above",
    "large boulders and rocky terrain scattered on Mars seen from orbit",
    "smooth fine-grained reddish soil plains and dust deposits on Mars from orbit",
]
TILE_BASE = (
    "https://trek.nasa.gov/tiles/Mars/EQ/"
    "Mars_Viking_MDIM21_ClrMosaic_global_232m/"
    "1.0.0/default/default028mm/{z}/{y}/{x}.jpg"
)
TILE_CACHE_FILE = Path("cache/tile_classifications.json")
TILE_ZOOM = 7    # max zoom supported by this dataset (verified from WMTSCapabilities)

# Human-readable terrain categories and the CLIP prompts used to classify them.
# More descriptive Mars-contextual prompts improve zero-shot accuracy.
TERRAIN_LABELS = ["Sand", "Bedrock", "Gravel", "Rock", "Soil"]
TERRAIN_PROMPTS = [
    "sandy terrain and wind-blown dunes on the surface of Mars",
    "flat exposed bedrock and geological outcrop on Mars",
    "loose gravel and small pebbles covering the ground on Mars",
    "large rock or boulder formation on Mars surface",
    "fine reddish soil and dust regolith on Mars terrain",
]

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Flask ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── CLIP Model (loaded once at startup) ────────────────────────────────────────
_model_ready = False
_model: CLIPModel = None
_processor: CLIPProcessor = None
_text_features: torch.Tensor = None      # surface photo text embeddings
_sat_text_features: torch.Tensor = None  # orbital/satellite tile text embeddings


def _load_clip() -> None:
    """Load CLIP in a background thread so Flask starts immediately."""
    global _model, _processor, _text_features, _sat_text_features, _model_ready
    try:
        log.info("BG  1/5 — Loading CLIPModel weights (cached locally after first download)…")
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        log.info("BG  2/5 — Loading CLIPProcessor…")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model.eval()

        def _encode_prompts(prompts):
            inp = _processor(text=prompts, return_tensors="pt", padding=True)
            out = _model.text_model(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
            )
            emb = _model.text_projection(out.pooler_output)
            return emb / emb.norm(p=2, dim=-1, keepdim=True)

        log.info("BG  3/5 — Pre-computing surface terrain text embeddings…")
        with torch.no_grad():
            _text_features = _encode_prompts(TERRAIN_PROMPTS)

        log.info("BG  4/5 — Pre-computing satellite terrain text embeddings…")
        with torch.no_grad():
            _sat_text_features = _encode_prompts(SAT_TERRAIN_PROMPTS)

        _model_ready = True
        log.info("BG  5/5 — CLIP ready (surface + satellite classification enabled).")
    except Exception as exc:
        log.error("CLIP model failed to load: %s — terrain labels will stay 'Unknown'", exc)


# Load in background so Flask is immediately available — images load without
# waiting; classification starts as soon as the model finishes.
threading.Thread(target=_load_clip, daemon=True).start()

# ── In-memory NASA response cache ──────────────────────────────────────────────
_latest_memo: dict = {"sol": None, "photos": [], "ts": 0.0}


# ── JSON classification cache helpers ──────────────────────────────────────────

def load_cache() -> dict:
    """Load the on-disk classification cache, returning {} on any error."""
    CACHE_FILE.parent.mkdir(exist_ok=True)
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


# ── Terrain classification ──────────────────────────────────────────────────────

def classify_terrain(image_url: str) -> dict:
    """
    Classify a single rover image into a terrain category using CLIP zero-shot.

    Returns a dict with keys: terrain, confidence, all_scores.
    Results are cached to disk so each image URL is only classified once.
    """
    cache = load_cache()
    if image_url in cache:
        return cache[image_url]

    fallback = {"terrain": "Unknown", "confidence": 0.0, "all_scores": {}}

    if not _model_ready:
        return fallback

    try:
        resp = requests.get(image_url, timeout=20)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")

        with torch.no_grad():
            img_inputs = _processor(images=image, return_tensors="pt")
            # Same pattern as text: use vision_model + visual_projection directly
            # for cross-version compatibility.
            vision_out = _model.vision_model(pixel_values=img_inputs["pixel_values"])
            img_feats = _model.visual_projection(vision_out.pooler_output)
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)

            # Scaled cosine similarity then softmax → class probabilities
            logits = (img_feats @ _text_features.T) * _model.logit_scale.exp()
            probs: list[float] = logits.softmax(dim=-1).squeeze().tolist()

        best_idx = probs.index(max(probs))
        confidence = probs[best_idx]
        terrain = TERRAIN_LABELS[best_idx] if confidence >= CONFIDENCE_THRESHOLD else "Unknown"

        result = {
            "terrain": terrain,
            "confidence": round(confidence * 100, 1),
            "all_scores": {
                label: round(p * 100, 1)
                for label, p in zip(TERRAIN_LABELS, probs)
            },
        }
        log.info("Classified %s → %s (%.0f%%)", image_url.split("/")[-1], terrain, confidence * 100)

    except Exception as exc:
        log.warning("Classification failed for %s: %s", image_url, exc)
        result = fallback

    # Reload cache before writing to reduce risk of concurrent overwrites
    cache = load_cache()
    cache[image_url] = result
    save_cache(cache)
    return result


# ── NASA helpers ───────────────────────────────────────────────────────────────
_SOL_RE = re.compile(r'\bsol\s*(\d+)', re.IGNORECASE)


def get_latest_photos() -> tuple[int, list]:
    """
    Fetch Perseverance images from the NASA Image Library (images-api.nasa.gov).
    No API key required. Results cached for SOL_CACHE_TTL seconds.
    Returns (max_sol, photo_list) where each photo dict is pre-normalised to
    the same shape build_item() expects.
    """
    now = time.time()
    if _latest_memo["sol"] is not None and now - _latest_memo["ts"] < SOL_CACHE_TTL:
        return _latest_memo["sol"], _latest_memo["photos"]

    log.info("Fetching Perseverance images from NASA Image Library…")
    resp = requests.get(
        NASA_IMAGES_URL,
        params={
            "q": "perseverance mars rover sol surface",
            "media_type": "image",
            "page_size": MAX_PHOTOS,
        },
        timeout=20,
    )
    resp.raise_for_status()

    raw_items = resp.json()["collection"]["items"]
    photos = []
    for i, item in enumerate(raw_items):
        meta = item["data"][0]
        links = item.get("links", [])
        if not links:
            continue

        img_url = links[0]["href"]  # already ~medium.jpg from this API
        title = meta.get("title", "")
        desc = meta.get("description", "")
        sol_match = _SOL_RE.search(title) or _SOL_RE.search(desc)
        sol = int(sol_match.group(1)) if sol_match else 0

        photos.append({
            "id": i,
            "img_src": img_url,
            "sol": sol,
            "camera": {
                "full_name": meta.get("center", "NASA/JPL-Caltech"),
                "name": meta.get("center", "NASA")[:8],
            },
            "earth_date": meta.get("date_created", "")[:10],
        })

    photos.sort(key=lambda p: p["sol"], reverse=True)
    max_sol = photos[0]["sol"] if photos else 0
    _latest_memo.update({"sol": max_sol, "photos": photos, "ts": now})
    log.info("Loaded %d images, latest sol: %d", len(photos), max_sol)
    return max_sol, photos


def build_item(photo: dict, classification: dict) -> dict:
    """Merge NASA photo metadata with classification results into one record."""
    return {
        "id": photo["id"],
        "img_src": photo["img_src"],
        "sol": photo["sol"],
        "camera": photo["camera"]["full_name"],
        "camera_abbr": photo["camera"]["name"],
        "earth_date": photo["earth_date"],
        **classification,
    }


# ── Waypoints ──────────────────────────────────────────────────────────────────

def load_waypoints() -> list:
    """
    Load Perseverance traverse waypoints from NASA MMGIS, caching locally.
    Returns a list of {sol, lat, lon, name} dicts sorted by sol.
    Falls back to empty list if the remote fetch fails.
    """
    if WAYPOINTS_FILE.exists():
        age = time.time() - WAYPOINTS_FILE.stat().st_mtime
        if age < WAYPOINTS_TTL:
            try:
                return json.loads(WAYPOINTS_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

    log.info("Fetching Perseverance waypoints from NASA MMGIS…")
    try:
        resp = requests.get(WAYPOINTS_URL, timeout=20)
        resp.raise_for_status()
        geojson = resp.json()
    except Exception as exc:
        log.warning("Waypoints fetch failed (%s) — map traverse will be empty", exc)
        return []

    waypoints = []
    for feat in geojson.get("features", []):
        props  = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [])
        sol    = props.get("sol")
        if sol is None or len(coords) < 2:
            continue
        waypoints.append({
            "sol":  int(float(sol)),
            "lon":  float(coords[0]),   # GeoJSON: lon first
            "lat":  float(coords[1]),
            "name": props.get("name", ""),
        })

    waypoints.sort(key=lambda w: w["sol"])
    WAYPOINTS_FILE.parent.mkdir(exist_ok=True)
    WAYPOINTS_FILE.write_text(json.dumps(waypoints, indent=2), encoding="utf-8")
    sol_range = f"sol {waypoints[0]['sol']}–{waypoints[-1]['sol']}" if waypoints else "none"
    log.info("Cached %d waypoints (%s)", len(waypoints), sol_range)
    return waypoints


def nearest_waypoint(sol: int, waypoints: list) -> dict | None:
    """Return the waypoint whose sol is closest to the requested value."""
    if not waypoints:
        return None
    return min(waypoints, key=lambda w: abs(w["sol"] - sol))


# ── Tile coordinate helpers (equirectangular, matches Mars Trek WMTS) ──────────

def _tile_coords(lat: float, lon: float, z: int) -> tuple[int, int]:
    """
    Convert geographic coords to WMTS tile (TileCol, TileRow) at zoom z.

    The Mars Trek WMTS tile matrix at zoom z has:
      MatrixWidth  = 2^(z+1)   (verified from WMTSCapabilities.xml)
      MatrixHeight = 2^z
    This gives a 2:1 equirectangular layout where each tile spans
    the same number of degrees in both axes (360/2^(z+1) per tile).
    """
    cols = 2 ** (z + 1)   # MatrixWidth
    rows = 2 ** z          # MatrixHeight
    x = min(int((lon + 180) / 360 * cols), cols - 1)
    y = min(int((90  - lat) / 180 * rows), rows - 1)
    return x, y


def _tile_bounds(tx: int, ty: int, z: int) -> dict:
    """Return geographic bounds {lat_min, lat_max, lon_min, lon_max} for tile (tx, ty, z)."""
    cols = 2 ** (z + 1)
    rows = 2 ** z
    return {
        "lon_min": round(tx       / cols * 360 - 180, 5),
        "lon_max": round((tx + 1) / cols * 360 - 180, 5),
        "lat_max": round(90 -  ty      / rows * 180,  5),
        "lat_min": round(90 - (ty + 1) / rows * 180,  5),
    }


def _classify_pil_sat(image) -> dict:
    """Run satellite-prompt CLIP on an already-loaded PIL image. No caching."""
    fallback = {"terrain": "Unknown", "confidence": 0.0}
    if not _model_ready or _sat_text_features is None:
        return fallback
    try:
        with torch.no_grad():
            inp       = _processor(images=image, return_tensors="pt")
            vis_out   = _model.vision_model(pixel_values=inp["pixel_values"])
            img_feats = _model.visual_projection(vis_out.pooler_output)
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
            logits    = (img_feats @ _sat_text_features.T) * _model.logit_scale.exp()
            probs: list[float] = logits.softmax(dim=-1).squeeze().tolist()
        best = probs.index(max(probs))
        conf = probs[best]
        return {
            "terrain":    TERRAIN_LABELS[best] if conf >= CONFIDENCE_THRESHOLD else "Unknown",
            "confidence": round(conf * 100, 1),
        }
    except Exception as exc:
        log.warning("_classify_pil_sat failed: %s", exc)
        return fallback


def classify_satellite_tile(tile_url: str) -> dict:
    """
    Download a Mars Trek orbital tile and classify its terrain via CLIP.
    Uses satellite-specific text embeddings for better orbital-view accuracy.
    Results are cached in TILE_CACHE_FILE so each URL is classified only once.
    """
    tile_cache: dict = {}
    if TILE_CACHE_FILE.exists():
        try:
            tile_cache = json.loads(TILE_CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    if tile_url in tile_cache:
        return tile_cache[tile_url]

    fallback = {"terrain": "Unknown", "confidence": 0.0}

    if not _model_ready or _sat_text_features is None:
        return fallback

    try:
        resp = requests.get(
            tile_url, timeout=20,
            headers={"User-Agent": "MarsExplorer/1.0 (terrain-research)"},
        )
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")

        with torch.no_grad():
            inp = _processor(images=image, return_tensors="pt")
            vision_out = _model.vision_model(pixel_values=inp["pixel_values"])
            img_feats  = _model.visual_projection(vision_out.pooler_output)
            img_feats  = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
            logits     = (img_feats @ _sat_text_features.T) * _model.logit_scale.exp()
            probs: list[float] = logits.softmax(dim=-1).squeeze().tolist()

        best_idx   = probs.index(max(probs))
        confidence = probs[best_idx]
        terrain    = TERRAIN_LABELS[best_idx] if confidence >= CONFIDENCE_THRESHOLD else "Unknown"
        result = {
            "terrain":    terrain,
            "confidence": round(confidence * 100, 1),
            "all_scores": {TERRAIN_LABELS[i]: round(p * 100, 1) for i, p in enumerate(probs)},
        }
        log.info("Sat tile %s → %s (%.0f%%)", tile_url.split("/")[-1], terrain, confidence * 100)

    except Exception as exc:
        log.warning("Satellite tile classification failed (%s): %s", tile_url.split("/")[-1], exc)
        result = fallback

    # Reload before writing to reduce race risk
    try:
        tile_cache = json.loads(TILE_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        tile_cache = {}
    tile_cache[tile_url] = result
    TILE_CACHE_FILE.parent.mkdir(exist_ok=True)
    TILE_CACHE_FILE.write_text(json.dumps(tile_cache, indent=2), encoding="utf-8")
    return result


# ── API routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/latest-sol")
def api_latest_sol():
    try:
        sol, _ = get_latest_photos()
        return jsonify({"sol": sol})
    except Exception as exc:
        log.exception("Error fetching latest sol")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/images")
def api_images():
    """
    GET /api/images?page=1&terrain=all

    Returns a paginated list of rover images with terrain classifications.
    On each call, up to PAGE_SIZE previously-unclassified images are classified
    and cached, so subsequent calls return immediately from cache.
    """
    try:
        page = max(1, int(request.args.get("page", 1)))
        terrain_filter = request.args.get("terrain", "all").strip().lower()

        sol, all_photos = get_latest_photos()
        all_photos = all_photos[:MAX_PHOTOS]

        cache = load_cache()
        classified: list[dict] = []
        unclassified: list[dict] = []

        for photo in all_photos:
            if photo["img_src"] in cache:
                classified.append(build_item(photo, cache[photo["img_src"]]))
            else:
                unclassified.append(photo)

        # Classify a fresh batch (one page at a time keeps response latency reasonable)
        for photo in unclassified[:PAGE_SIZE]:
            classified.append(build_item(photo, classify_terrain(photo["img_src"])))

        # How many photos still await classification after this request
        pending = max(0, len(unclassified) - PAGE_SIZE)

        # Apply terrain filter
        if terrain_filter == "all":
            results = classified
        else:
            results = [r for r in classified if r["terrain"].lower() == terrain_filter]

        # Stable ordering by photo id before slicing
        results.sort(key=lambda x: x["id"])

        total = len(results)
        pages = max(1, math.ceil(total / PAGE_SIZE))
        start = (page - 1) * PAGE_SIZE

        return jsonify({
            "images": results[start: start + PAGE_SIZE],
            "total": total,
            "page": page,
            "pages": pages,
            "sol": sol,
            "pending": pending,
            "model_ready": _model_ready,
        })

    except Exception as exc:
        log.exception("Error in /api/images")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/stats")
def api_stats():
    """
    GET /api/stats

    Returns total classified image count and a per-terrain breakdown,
    computed from the on-disk classification cache.
    """
    try:
        cache = load_cache()
        counts = {label: 0 for label in TERRAIN_LABELS + ["Unknown"]}
        for entry in cache.values():
            terrain = entry.get("terrain", "Unknown")
            counts[terrain] = counts.get(terrain, 0) + 1
        return jsonify({"total": sum(counts.values()), "by_terrain": counts})
    except Exception as exc:
        log.exception("Error in /api/stats")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/map-data")
def api_map_data():
    """
    GET /api/map-data

    Returns ALL rover waypoints as terrain-classified points.
    Each waypoint gets terrain from the nearest classified photo (by sol)
    within ±150 sols; waypoints with no nearby photo are labelled Unknown.

    Returns:
      photos           – every waypoint with terrain label + optional photo info
      traverse         – full rover path [{sol, lat, lon}] for the route line
      waypoint_count   – total waypoints
      classified_count – waypoints that got a non-Unknown terrain label
    """
    try:
        waypoints = load_waypoints()
        cache     = load_cache()
        _, photos = get_latest_photos()

        # Build sol → best classified photo lookup (only photos with valid sol)
        sol_photo: dict = {}
        for photo in photos:
            url = photo["img_src"]
            sol = photo["sol"]
            if sol <= 0 or url not in cache:
                continue
            existing = sol_photo.get(sol)
            if existing is None or cache[url].get("confidence", 0) > existing["cl"].get("confidence", 0):
                sol_photo[sol] = {"url": url, "cl": cache[url], "photo": photo}

        classified_sols = sorted(sol_photo.keys())

        # Assign terrain to every waypoint via nearest classified photo
        SOL_TOLERANCE = 150   # max sol distance to borrow a classification
        points = []
        for wp in waypoints:
            wp_sol     = wp["sol"]
            terrain    = "Unknown"
            confidence = 0.0
            img_url    = None
            earth_date = ""
            camera     = ""

            if classified_sols:
                nearest_sol = min(classified_sols, key=lambda s: abs(s - wp_sol))
                if abs(nearest_sol - wp_sol) <= SOL_TOLERANCE:
                    entry      = sol_photo[nearest_sol]
                    terrain    = entry["cl"].get("terrain", "Unknown")
                    confidence = entry["cl"].get("confidence", 0.0)
                    img_url    = entry["url"]
                    earth_date = entry["photo"]["earth_date"]
                    camera     = entry["photo"]["camera"]["full_name"]

            points.append({
                "image_url":    img_url,
                "terrain_type": terrain,
                "confidence":   confidence,
                "lat":          wp["lat"],
                "lon":          wp["lon"],
                "sol":          wp_sol,
                "earth_date":   earth_date,
                "camera":       camera,
            })

        traverse = [
            {"sol": w["sol"], "lat": w["lat"], "lon": w["lon"]}
            for w in waypoints
        ]
        classified_count = sum(1 for p in points if p["terrain_type"] != "Unknown")

        return jsonify({
            "photos":           points,
            "traverse":         traverse,
            "waypoint_count":   len(waypoints),
            "classified_count": classified_count,
        })

    except Exception as exc:
        log.exception("Error in /api/map-data")
        return jsonify({"error": str(exc)}), 500


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Forward azimuth in radians from point-1 to point-2 (0 = north)."""
    dlon   = math.radians(lon2 - lon1)
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon)
    return math.atan2(x, y)


def _offset(lat: float, lon: float, bearing_rad: float, metres: float) -> tuple[float, float]:
    """Return the point reached by travelling `metres` from (lat, lon) on bearing."""
    R = 3_389_500.0          # Mars mean radius, metres
    d = metres / R
    lat_r  = math.radians(lat)
    lat2   = math.asin(math.sin(lat_r) * math.cos(d) +
                        math.cos(lat_r) * math.sin(d) * math.cos(bearing_rad))
    lon2   = math.radians(lon) + math.atan2(
        math.sin(bearing_rad) * math.sin(d) * math.cos(lat_r),
        math.cos(d) - math.sin(lat_r) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


# Small in-memory cache for downloaded tile PIL images (url → PIL.Image)
_tile_img_cache: dict = {}


def _fetch_tile_image(url: str):
    """Download a tile and return a PIL Image, or None on failure. Cached in RAM."""
    if url in _tile_img_cache:
        return _tile_img_cache[url]
    try:
        resp = requests.get(url, timeout=20,
                            headers={"User-Agent": "MarsExplorer/1.0"})
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        _tile_img_cache[url] = img
        return img
    except Exception as exc:
        log.warning("Tile download failed %s: %s", url.split("/")[-1], exc)
        _tile_img_cache[url] = None
        return None


@app.route("/api/classify-area")
def api_classify_area():
    """
    GET /api/classify-area

    Returns 25 terrain-classified lat/lon points arranged in a 5 × 5 grid
    covering 5 km ahead of the rover's current position and heading.

    Layout:
      • Rows 1-5: 1 km, 2 km, 3 km, 4 km, 5 km ahead
      • Cols -2…+2: 2 km left to 2 km right (perpendicular to heading)

    Terrain is classified by cropping the sub-region of the orbital satellite
    tile that corresponds to each 1 km grid cell and running CLIP on it.
    Results are cached so subsequent calls are fast.
    """
    try:
        waypoints = load_waypoints()
        if not waypoints:
            return jsonify({"error": "No waypoints — cannot determine rover position"}), 503

        if not _model_ready:
            return jsonify({"error": "AI model still loading — try again in a moment"}), 503

        rover = waypoints[-1]
        lat, lon = rover["lat"], rover["lon"]
        mode  = request.args.get("mode", "ahead")

        # Load tile cache
        tile_cache: dict = {}
        if TILE_CACHE_FILE.exists():
            try:
                tile_cache = json.loads(TILE_CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Degrees-per-pixel at TILE_ZOOM for sub-tile cropping
        deg_per_px_lat = 180 / (2 ** TILE_ZOOM) / 256
        km_in_deg_lat  = 1000 / (math.pi / 180 * 3_389_500)
        crop_half      = max(4, int(0.5 * km_in_deg_lat / deg_per_px_lat))

        cache_before = len(tile_cache)

        def _classify_point(pt_lat, pt_lon):
            """Classify a lat/lon by cropping its sub-tile region with CLIP."""
            tx, ty   = _tile_coords(pt_lat, pt_lon, TILE_ZOOM)
            tile_url = TILE_BASE.format(z=TILE_ZOOM, y=ty, x=tx)
            bounds   = _tile_bounds(tx, ty, TILE_ZOOM)
            px = int((pt_lon - bounds["lon_min"]) / (bounds["lon_max"] - bounds["lon_min"]) * 256)
            py = int((bounds["lat_max"] - pt_lat)  / (bounds["lat_max"] - bounds["lat_min"]) * 256)
            px = max(crop_half, min(255 - crop_half, px))
            py = max(crop_half, min(255 - crop_half, py))
            key = f"{TILE_ZOOM}/{ty}/{tx}/{px}/{py}"
            if key not in tile_cache:
                img = _fetch_tile_image(tile_url)
                if img is not None:
                    crop = img.crop((px - crop_half, py - crop_half,
                                     px + crop_half, py + crop_half))
                    crop = crop.resize((224, 224), Image.LANCZOS)
                    tile_cache[key] = _classify_pil_sat(crop)
                else:
                    tile_cache[key] = {"terrain": "Unknown", "confidence": 0.0}
            return tile_cache[key]

        points = []

        if mode == "region":
            # Grid covering the full rover traverse + 10 km margin, 2 km spacing
            all_lats = [w["lat"] for w in waypoints]
            all_lons = [w["lon"] for w in waypoints]
            margin   = 10 * km_in_deg_lat
            step_lat = 2 * km_in_deg_lat
            step_lon = step_lat / math.cos(math.radians(lat))
            lat_min  = min(all_lats) - margin
            lat_max  = max(all_lats) + margin
            lon_min  = min(all_lons) - margin / math.cos(math.radians(lat))
            lon_max  = max(all_lons) + margin / math.cos(math.radians(lat))
            pt_lat   = lat_min
            while pt_lat <= lat_max + 1e-9:
                pt_lon = lon_min
                while pt_lon <= lon_max + 1e-9:
                    cl = _classify_point(pt_lat, pt_lon)
                    points.append({
                        "lat":        round(pt_lat, 6),
                        "lon":        round(pt_lon, 6),
                        "terrain":    cl.get("terrain", "Unknown"),
                        "confidence": cl.get("confidence", 0.0),
                    })
                    pt_lon += step_lon
                pt_lat += step_lat

        else:
            # 5×5 grid, 1 km spacing, 5 km ahead of rover heading
            ref_idx = max(0, len(waypoints) - 10)
            ref     = waypoints[ref_idx]
            fwd     = _bearing(ref["lat"], ref["lon"], lat, lon)
            perp    = fwd + math.pi / 2
            for row in range(1, 6):
                for col in range(-2, 3):
                    fwd_lat, fwd_lon = _offset(lat, lon, fwd,  row * 1000)
                    pt_lat,  pt_lon  = _offset(fwd_lat, fwd_lon, perp, col * 1000)
                    cl = _classify_point(pt_lat, pt_lon)
                    points.append({
                        "lat":        round(pt_lat, 6),
                        "lon":        round(pt_lon, 6),
                        "terrain":    cl.get("terrain", "Unknown"),
                        "confidence": cl.get("confidence", 0.0),
                        "km_ahead":   row,
                        "km_side":    col,
                    })

        if len(tile_cache) > cache_before:
            TILE_CACHE_FILE.parent.mkdir(exist_ok=True)
            TILE_CACHE_FILE.write_text(json.dumps(tile_cache, indent=2), encoding="utf-8")

        return jsonify({
            "points":    points,
            "rover_lat": lat,
            "rover_lon": lon,
            "rover_sol": rover["sol"],
        })

    except Exception as exc:
        log.exception("Error in /api/classify-area")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    if NASA_API_KEY == "DEMO_KEY":
        log.warning(
            "Using DEMO_KEY — NASA API is rate-limited to 30 req/hour. "
            "Set NASA_API_KEY in .env for full access."
        )
    app.run(debug=True, host="0.0.0.0", port=5000)
