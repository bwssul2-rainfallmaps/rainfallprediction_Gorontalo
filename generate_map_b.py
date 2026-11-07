"""
Improved Weather Map Generation script for location b: Paguyaman Area
- Heavy rainfall warnings (peak 3h + expected 24h)
- Safe rivers layer (no crash if GeoJSON missing)
- Writes warnings_b.json for the dashboard sidebar
- Better local / GitHub compatibility
"""

import folium
import requests
import json
import os
import logging
import tempfile
import shutil
import base64
from datetime import datetime, timezone, timedelta
from tenacity import retry, stop_after_attempt, wait_fixed
import numpy as np
import matplotlib.pyplot as plt
from folium.plugins import HeatMap

logging.basicConfig(
    filename="rainfall_maps_b.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ================== CONFIG ==================
API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not API_KEY:
    raise ValueError("Set OPENWEATHER_API_KEY before running this script")

CACHE_DURATION = 3600
FORECAST_INTERVALS = 8                     # 8 × 3h = 24h
TIMEZONE_OFFSET = timedelta(hours=8)       # WITA / Asia/Makassar
TIMEZONE_LABEL = "UTC+8"

# Do not use tile.openstreetmap.org as the default. Those servers return

# ---------- Heavy rainfall thresholds (mm) ----------
THRESHOLD_PEAK_MODERATE = 10.0   # mm in any 3-hour period → Moderate
THRESHOLD_PEAK_HEAVY   = 20.0   # mm in any 3-hour period → Heavy
THRESHOLD_PEAK_EXTREME = 30.0   # mm in any 3-hour period → Very Heavy / Extreme
THRESHOLD_24H_MODERATE = 15.0   # expected 24h rainfall → Moderate
THRESHOLD_24H_HEAVY    = 30.0   # expected 24h rainfall → Heavy
THRESHOLD_24H_EXTREME  = 50.0  # expected 24h rainfall → Very Heavy

LOCATIONS = [
    (0.649374,122.158923),(0.668016,122.623020),(0.845119,122.383486),
    (0.523573,122.398457),(0.677338,122.338574),(0.630731,122.009214),
    (0.518877,122.233778),(0.785022,122.488283),(0.779871,122.233778),
    (0.584125,122.518224),(0.630731,122.428399),(0.584125,122.308632),
    (0.751907,122.398457),(0.658695,122.248749),(0.521151,122.053629),
    (0.503020,122.323603),(0.733265,122.158923),(0.789192,122.323603),
    (0.677338,122.084069),(0.668016,122.533195)
]

MAP_CENTER = [0.682760, 122.381879]
ZOOM_START = 11
# ===========================================


# ────────────────────────────────────────────────
# Helper: Embed image as base64
# ────────────────────────────────────────────────
def embed_image_to_base64(image_path):
    if not os.path.exists(image_path):
        print(f"Warning: {image_path} not found → placeholder")
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"


# ────────────────────────────────────────────────
# API Helpers
# ────────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_api_data(url):
    resp = requests.get(url, timeout=12)
    resp.raise_for_status()
    return resp.json()


def load_cache(cache_file):
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, encoding="utf-8") as f:
            cache = json.load(f)
        if datetime.now().timestamp() - cache.get("timestamp", 0) < CACHE_DURATION:
            return cache.get("data", {})
        os.remove(cache_file)
    except Exception:
        pass
    return {}


def save_cache(cache_file, data):
    cache = {"timestamp": int(datetime.now().timestamp()), "data": data}
    tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, dir=os.path.dirname(cache_file) or ".")
    try:
        json.dump(cache, tmp)
        tmp.close()
        shutil.move(tmp.name, cache_file)
    except Exception:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


def save_weather_history(history_file, map_id, area, snapshot, last_update):
    if os.path.exists(history_file):
        with open(history_file, encoding="utf-8") as f:
            history = json.load(f)
        if (
            not isinstance(history, dict)
            or history.get("map") != map_id
            or not isinstance(history.get("runs"), list)
        ):
            raise ValueError(f"Invalid weather history format in {history_file}")
    else:
        history = {
            "map": map_id,
            "area": area,
            "timezone": TIMEZONE_LABEL,
            "runs": [],
        }

    latest_run = history["runs"][-1] if history["runs"] else None
    snapshot_content = {
        key: value for key, value in snapshot.items()
        if key not in ("generated", "last_update")
    }
    latest_content = (
        {
            key: value for key, value in latest_run.items()
            if key not in ("generated", "last_update")
        }
        if isinstance(latest_run, dict)
        else None
    )
    changed = (
        latest_content is None
        or json.dumps(snapshot_content, sort_keys=True) != json.dumps(latest_content, sort_keys=True)
    )

    history["last_update"] = last_update
    if changed:
        history["runs"].append(snapshot)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=os.path.dirname(history_file) or ".",
        ) as tmp:
            tmp_path = tmp.name
            json.dump(history, tmp, indent=2, ensure_ascii=False)
        os.replace(tmp_path, history_file)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    return changed


def get_3h_forecast(lat, lon, cache_file="forecast_3h_cache.json"):
    key = f"{lat:.6f},{lon:.6f}"
    cache = load_cache(cache_file)
    if key in cache:
        return cache[key]

    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        data = fetch_api_data(url)
        forecasts = []
        for item in data["list"][:FORECAST_INTERVALS]:
            dt = datetime.fromtimestamp(item["dt"], tz=timezone.utc) + TIMEZONE_OFFSET
            forecasts.append({
                "time": dt.strftime("%Y-%m-%d %H:%M"),
                "pop": round(item.get("pop", 0) * 100),
                "rainfall": item.get("rain", {}).get("3h", 0.0),
            })
        cache[key] = forecasts
        save_cache(cache_file, cache)
        return forecasts
    except Exception as e:
        print(f"Forecast error ({lat},{lon}): {e}")
        logging.error(f"Forecast error ({lat},{lon}): {e}")
        return None


def get_current_weather(lat, lon, cache_file="current_cache.json"):
    key = f"{lat:.6f},{lon:.6f}"
    cache = load_cache(cache_file)
    if key in cache:
        return cache[key]

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        data = fetch_api_data(url)
        result = {
            "rainfall": data.get("rain", {}).get("1h", 0.0),
            "temp": data.get("main", {}).get("temp", 0.0),
            "humidity": data.get("main", {}).get("humidity", 0),
            "timestamp": data.get("dt"),
        }
        cache[key] = result
        save_cache(cache_file, cache)
        return result
    except Exception as e:
        print(f"Current weather error ({lat},{lon}): {e}")
        logging.error(f"Current weather error ({lat},{lon}): {e}")
        return None


# ────────────────────────────────────────────────
# Safe Rivers layer
# ────────────────────────────────────────────────
def add_rivers_layer(m, geojson_path="RBI_Sulawesi2_sungai.geojson", name="Rivers"):
    """Add rivers GeoJSON. Never crashes if the file is missing."""
    if not os.path.exists(geojson_path):
        print(f"⚠️  Rivers GeoJSON not found: {geojson_path} → skipped")
        return

    try:
        with open(geojson_path, encoding="utf-8") as f:
            rivers_data = json.load(f)

        folium.GeoJson(
            rivers_data,
            name=name,
            style_function=lambda x: {
                "color": "#1e88e5",
                "weight": 2.8,
                "opacity": 0.85,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["NAMOBJ"],
                aliases=["River: "],
                localize=True,
            ),
            popup=folium.GeoJsonPopup(
                fields=["NAMOBJ", "NAMWS"],
                aliases=["River Name:", "Watershed:"],
                localize=True,
            ),
        ).add_to(m)
        print("✅ Rivers layer added successfully!")
    except Exception as e:
        print(f"❌ Failed to add rivers layer: {e}")
        # Minimal fallback – no tooltip
        try:
            with open(geojson_path, encoding="utf-8") as f:
                rivers_data = json.load(f)
            folium.GeoJson(
                rivers_data,
                name=name,
                style_function=lambda x: {"color": "#1e88e5", "weight": 2.5, "opacity": 0.8},
            ).add_to(m)
            print("✅ Rivers layer added (without tooltip)")
        except Exception as e2:
            print(f"⚠️  Could not add rivers layer at all: {e2}")


# ────────────────────────────────────────────────
# Draggable legend + colorbar helpers
# ────────────────────────────────────────────────
def add_draggable_legend(m, html_content: str, div_id: str):
    js = f"""
    <script>
    function makeResizableAndDraggable(id) {{
        const el = document.getElementById(id);
        if (!el) return;

        let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
        el.onmousedown = dragMouseDown;

        function dragMouseDown(e) {{
            if (e.target.classList.contains('resize-handle')) return;
            e = e || window.event;
            e.preventDefault();
            pos3 = e.clientX;
            pos4 = e.clientY;
            document.onmouseup = closeDragElement;
            document.onmousemove = elementDrag;
        }}

        function elementDrag(e) {{
            e = e || window.event;
            e.preventDefault();
            pos1 = pos3 - e.clientX;
            pos2 = pos4 - e.clientY;
            pos3 = e.clientX;
            pos4 = e.clientY;
            el.style.top = (el.offsetTop - pos2) + "px";
            el.style.left = (el.offsetLeft - pos1) + "px";
        }}

        function closeDragElement() {{
            document.onmouseup = null;
            document.onmousemove = null;
        }}

        let resizer = el.querySelector('.resize-handle');
        if (resizer) {{
            resizer.onmousedown = initResize;
            function initResize(e) {{
                e.preventDefault();
                window.addEventListener('mousemove', Resize);
                window.addEventListener('mouseup', stopResize);
            }}
            function Resize(e) {{
                el.style.width = (e.clientX - el.getBoundingClientRect().left) + 'px';
                el.style.height = (e.clientY - el.getBoundingClientRect().top) + 'px';
            }}
            function stopResize() {{
                window.removeEventListener('mousemove', Resize);
                window.removeEventListener('mouseup', stopResize);
            }}
        }}

        function autoSizeLegend() {{
            el.style.width = 'auto';
            el.style.height = 'auto';
            const rect = el.getBoundingClientRect();
            const vw = window.innerWidth;
            let targetWidth = Math.min(rect.width + 20, vw * 0.35, 140);
            el.style.width = targetWidth + 'px';
            const newRect = el.getBoundingClientRect();
            el.style.height = newRect.height + 'px';
        }}
        autoSizeLegend();
        window.addEventListener('resize', autoSizeLegend);
    }}

    window.addEventListener('load', () => {{
        makeResizableAndDraggable('legend-real');
        makeResizableAndDraggable('legend-pred');
    }});
    </script>
    """
    m.get_root().header.add_child(folium.Element(js))
    m.get_root().html.add_child(folium.Element(html_content))


def create_colorbar_png(filename: str, cmap_name: str, vmin: float, vmax: float, label: str):
    if vmax <= vmin:
        vmax = vmin + 1
    fig = plt.figure(figsize=(1.25, 3.3))
    ax = fig.add_axes([0.45, 0.12, 0.25, 0.76])
    norm = plt.Normalize(vmin, vmax)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_name), cax=ax, orientation="vertical")
    cb.set_label(label, fontsize=11, fontweight="bold")
    ticks = np.round(np.linspace(vmin, vmax, 7), 1 if vmax < 10 else 0)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:g}" for t in ticks])
    plt.savefig(filename, dpi=160, transparent=True, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def add_idw_overlay(fg: folium.FeatureGroup, points: list, values: list, filename_base: str,
                    cmap_name: str, layer_title: str, vmin=None, vmax=None):
    print(f"🛠️  Creating {layer_title} ...")
    try:
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        min_lat, max_lat = min(lats) - 0.015, max(lats) + 0.015
        min_lon, max_lon = min(lons) - 0.015, max(lons) + 0.015

        grid_lon = np.linspace(min_lon, max_lon, 250)
        grid_lat = np.linspace(min_lat, max_lat, 250)
        grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

        def idw_interpolate(x, y):
            dists = np.sqrt((np.array(lons) - x)**2 + (np.array(lats) - y)**2)
            dists[dists < 1e-10] = 1e-10
            weights = 1.0 / (dists ** 2)
            return np.sum(weights * np.array(values)) / np.sum(weights)

        grid_z = np.vectorize(idw_interpolate)(grid_x, grid_y)
        grid_z = np.ma.masked_invalid(grid_z)

        base_png = f"{filename_base}.png"
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis("off")
        cmap = plt.get_cmap(cmap_name)
        cmap.set_bad(alpha=0)
        ax.imshow(grid_z, extent=[min_lon, max_lon, min_lat, max_lat],
                  cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.75, origin="lower")
        plt.savefig(base_png, dpi=80, transparent=True, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        contour_png = f"{filename_base}_contours.png"
        data_max = np.nanmax(grid_z) if np.any(np.isfinite(grid_z)) else 0.0
        data_min = np.nanmin(grid_z) if np.any(np.isfinite(grid_z)) else 0.0
        data_range = data_max - data_min

        contours_created = False
        if data_max > 0.05 and data_range > 0.01:
            try:
                effective_max = max(data_max * 1.05, data_min + 0.5)
                levels_raw = np.linspace(data_min, effective_max, 8)
                levels = np.unique(np.round(levels_raw, 2))
                levels = levels[np.diff(np.concatenate(([data_min - 0.01], levels))) > 0]

                if len(levels) >= 2:
                    fig2, ax2 = plt.subplots(figsize=(10, 10))
                    ax2.axis("off")
                    ax2.set_xlim(min_lon, max_lon)
                    ax2.set_ylim(min_lat, max_lat)
                    ax2.set_position([0, 0, 1, 1])

                    ax2.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap_name, alpha=0.55)
                    cs = ax2.contour(grid_x, grid_y, grid_z, levels=levels, colors="black", linewidths=1.1)
                    ax2.clabel(cs, inline=True, fontsize=9, fmt="%.1f mm")

                    plt.savefig(contour_png, dpi=80, transparent=True, bbox_inches="tight", pad_inches=0)
                    plt.close(fig2)
                    contours_created = True
                    print(f"   → Contours created (max {data_max:.2f} mm, {len(levels)} levels)")
            except Exception as contour_err:
                print(f"   → Contour failed ({contour_err}) → skipping contours")
        else:
            print(f"   → Skipped contours: very low rain (max {data_max:.2f} mm)")

        bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        if os.path.exists(base_png):
            folium.raster_layers.ImageOverlay(
                image=base_png, bounds=bounds, opacity=0.65, interactive=True
            ).add_to(fg)
        if contours_created and os.path.exists(contour_png):
            folium.raster_layers.ImageOverlay(
                image=contour_png, bounds=bounds, opacity=0.75, interactive=True
            ).add_to(fg)

        print(f"✅ {layer_title} added successfully!")
    except Exception as e:
        print(f"❌ IDW failed ({e}). Using HeatMap fallback...")
        HeatMap(
            [[p[0], p[1], max(0.1, v)] for p, v in zip(points, values)],
            radius=15, blur=10,
            gradient={"0.4": "blue", "0.65": "lime", "0.9": "yellow", "1.0": "red"}
        ).add_to(fg)


# ────────────────────────────────────────────────
# Warning helpers
# ────────────────────────────────────────────────
def classify_peak(rain_mm: float) -> str:
    if rain_mm >= THRESHOLD_PEAK_EXTREME:
        return "extreme"
    if rain_mm >= THRESHOLD_PEAK_HEAVY:
        return "heavy"
    if rain_mm >= THRESHOLD_PEAK_MODERATE:
        return "moderate"
    return "normal"


def classify_24h(rain_mm: float) -> str:
    if rain_mm >= THRESHOLD_24H_EXTREME:
        return "extreme"
    if rain_mm >= THRESHOLD_24H_HEAVY:
        return "heavy"
    if rain_mm >= THRESHOLD_24H_MODERATE:
        return "moderate"
    return "normal"


def warning_level(peak_level: str, h24_level: str) -> str:
    """Return the highest severity among peak and 24h."""
    order = {"normal": 0, "moderate": 1, "heavy": 2, "extreme": 3}
    return max([peak_level, h24_level], key=lambda x: order.get(x, 0))


LEVEL_STYLE = {
    "extreme":  {"color": "#dc2626", "icon_color": "red",    "icon": "exclamation-triangle", "label": "EXTREME",  "radius": 12},
    "heavy":    {"color": "#ea580c", "icon_color": "orange", "icon": "cloud-showers-heavy",  "label": "HEAVY",    "radius": 10},
    "moderate": {"color": "#ca8a04", "icon_color": "beige",  "icon": "cloud",                "label": "MODERATE", "radius": 8},
}

def add_basemaps(m):
    """Showed layer is based on show boolean"""
    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors",
        name="OpenStreetMap",
        max_zoom=19,
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        attr='© OpenStreetMap contributors, Tiles style by Humanitarian OpenStreetMap Team',
        name="OSM HOT",
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Esri Streets",
        max_zoom=19,
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Esri Imagery",
        max_zoom=19,
        show=False,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="&copy; OpenStreetMap contributors",
        name="OpenStreetMap",
        max_zoom=19,
        show=True,
    ).add_to(m)

# ────────────────────────────────────────────────
# Prediction Map
# ────────────────────────────────────────────────
def create_prediction_map():
    m = folium.Map(location=MAP_CENTER, zoom_start=ZOOM_START, tiles=None, control_scale=True, zoom_snap=0.25, zoom_delta=0.5)
    add_basemaps(m)

    generated_utc = datetime.now(timezone.utc)
    generated_time = (generated_utc + TIMEZONE_OFFSET).strftime("%d %B %Y %H:%M")
    # One header, top-center. Timestamp lives inside it so the top-right
    # stays free for the (collapsed) layer control.
    title_html = f'''
    <div style="position:absolute; top:10px; left:50%; transform:translateX(-50%);
                background:rgba(255,255,255,0.94); padding:6px 16px 5px; border-radius:6px;
                box-shadow:0 2px 8px rgba(0,0,0,0.22); z-index:1000; text-align:center;
                border:1px solid #ccc; max-width:min(560px, 70vw); pointer-events:none;">
        <div style="font-size:17px; font-weight:700; line-height:1.2;">Rainfall Map – Paguyaman & Surroundings</div>
        <div style="font-size:11px; font-weight:500; color:#444; margin-top:2px;">
            {generated_time} {TIMEZONE_LABEL} · OpenWeatherMap · IDW
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    fg_markers   = folium.FeatureGroup(name="Stations", show=True)
    fg_warnings  = folium.FeatureGroup(name="⚠ Heavy Rain Warnings", show=True)
    fg_heatmap   = folium.FeatureGroup(name="Heatmap (Max PoP)", show=True)
    fg_idw_peak  = folium.FeatureGroup(name="IDW Peak Rain (mm)", show=False)
    fg_idw_24h   = folium.FeatureGroup(name="IDW Expected 24h (mm)", show=True)

    heat_data = []
    points_peak, values_peak = [], []
    points_24h, values_24h = [], []

    # Collect warnings for the banner + JSON
    warnings_list = []          # human readable
    warning_stations = []       # structured for JSON
    station_history = []
    unavailable_stations = []

    for lat, lon in LOCATIONS:
        fc = get_3h_forecast(lat, lon)
        if not fc:
            unavailable_stations.append({"lat": lat, "lon": lon})
            folium.CircleMarker(
                [lat, lon], radius=5, color="gray", fill=True, fill_opacity=0.6,
                popup="No data"
            ).add_to(fg_markers)
            continue

        max_pop = max(d["pop"] for d in fc)
        rain_at_max_pop = max(d["rainfall"] for d in fc)
        expected_24h = sum(d["rainfall"] for d in fc)

        peak_lvl = classify_peak(rain_at_max_pop)
        h24_lvl  = classify_24h(expected_24h)
        overall  = warning_level(peak_lvl, h24_lvl)

        station_history.append({
            "lat": lat,
            "lon": lon,
            "max_pop": max_pop,
            "peak_mm": round(rain_at_max_pop, 1),
            "expected_24h_mm": round(expected_24h, 1),
            "level": overall,
            "peak_level": peak_lvl,
            "h24_level": h24_lvl,
            "forecast": fc,
        })

        # ---------- Popup content ----------
        popup_lines = [
            f"<b>3-Hourly Precipitation Forecast</b><br><br>"
        ]
        for d in fc:
            popup_lines.append(
                f"<b>{d['time']}:</b> {d['pop']}% – {d['rainfall']:.1f} mm<br>"
            )

        if overall != "normal":
            style = LEVEL_STYLE[overall]
            popup_lines.insert(1,
                f"<div style='background:{style['color']};color:white;padding:4px 8px;"
                f"border-radius:4px;margin-bottom:6px;font-weight:bold;'>"
                f"⚠ {style['label']} RAINFALL</div>"
            )

        popup_content = f"<div style='font-size:13px;line-height:1.5;'>{''.join(popup_lines)}</div>"
        popup_obj = folium.Popup(popup_content, max_width=360)

        tooltip_text = (
            f"Max PoP: {max_pop}% | Peak: {rain_at_max_pop:.1f} mm | "
            f"Exp. 24h: {expected_24h:.1f} mm"
        )
        if overall != "normal":
            tooltip_text = f"⚠ {overall.upper()} | " + tooltip_text

        if overall in LEVEL_STYLE:
            style = LEVEL_STYLE[overall]
            icon_color = style["icon_color"]
            icon_name = style["icon"]
        else:
            icon_color = "blue" if max_pop > 50 else "orange"
            icon_name = "cloud-rain" if max_pop > 50 else "sun"

        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup_content, max_width=360),
            tooltip=folium.Tooltip(tooltip_text),
            icon=folium.Icon(icon=icon_name, prefix="fa", color=icon_color),
        ).add_to(fg_markers)

        # Extra warning circle for heavy/extreme stations
        if overall in LEVEL_STYLE:
            style = LEVEL_STYLE[overall]
            folium.CircleMarker(
                [lat, lon],
                radius=style["radius"],
                color=style["color"],
                weight=3,
                fill=True,
                fill_color=style["color"],
                fill_opacity=0.25,
                popup=popup_obj,
            ).add_to(fg_warnings)

            warnings_list.append(
                f"{overall.upper()}: Peak {rain_at_max_pop:.1f} mm / 24h {expected_24h:.1f} mm "
                f"@ ({lat:.3f}, {lon:.3f})"
            )
            warning_stations.append({
                "lat": lat,
                "lon": lon,
                "peak_mm": round(rain_at_max_pop, 1),
                "expected_24h_mm": round(expected_24h, 1),
                "max_pop": max_pop,
                "level": overall,
                "peak_level": peak_lvl,
                "h24_level": h24_lvl,
            })

        heat_data.append([lat, lon, max_pop / 100.0])
        points_peak.append((lat, lon))
        values_peak.append(rain_at_max_pop)
        points_24h.append((lat, lon))
        values_24h.append(expected_24h)

    if heat_data:
        HeatMap(
            heat_data, radius=10, blur=3,
            gradient={"0.4": "blue", "0.65": "lime", "0.9": "yellow", "1.0": "red"}
        ).add_to(fg_heatmap)

    fg_markers.add_to(m)
    fg_warnings.add_to(m)
    fg_heatmap.add_to(m)

    # IDW layers
    if values_peak:
        min_p = min(values_peak)
        max_p = max(values_peak)
        vmin_p = round(min_p * 0.9, 1) if min_p > 0 else 0.0
        vmax_p = round(max_p * 1.15, 1)

        add_idw_overlay(
            fg_idw_peak, points_peak, values_peak, "idw_peak_b",
            "YlOrRd", "IDW Peak Rain (mm)_b", vmin=vmin_p, vmax=vmax_p
        )
        create_colorbar_png("colorbar_peak_b.png", "YlOrRd", vmin_p, vmax_p, "Peak Rainfall (mm)")

    if values_24h:
        min_24 = min(values_24h)
        max_24 = max(values_24h)
        vmin_24 = round(min_24 * 0.9, 1) if min_24 > 0 else 0.0
        vmax_24 = round(max_24 * 1.10, 1)

        add_idw_overlay(
            fg_idw_24h, points_24h, values_24h, "idw_24h_b",
            "YlGnBu", "IDW Expected 24h (mm)_b", vmin=vmin_24, vmax=vmax_24
        )
        create_colorbar_png("colorbar_24h_b.png", "YlGnBu", vmin_24, vmax_24, "Expected 24h Rainfall (mm)")

    fg_idw_peak.add_to(m)
    fg_idw_24h.add_to(m)

    # Banner only for heavy / extreme. Moderate stays on the marker and in
    # warnings_b.json (sidebar), so the map is not covered on a normal wet day.
    n_extreme = sum(1 for s in warning_stations if s["level"] == "extreme")
    n_heavy = sum(1 for s in warning_stations if s["level"] == "heavy")
    if n_extreme or n_heavy:
        banner_color = "#b91c1c" if n_extreme else "#c2410c"
        banner_text = f"Heavy rainfall warning:<br>{n_extreme} extreme, {n_heavy} heavy station(s)"
        banner_html = f'''
        <div style="position:absolute; top:62px; left:12px; z-index:1001;
                    background:{banner_color}; color:white; padding:6px 12px;
                    border-radius:6px; box-shadow:0 2px 8px rgba(0,0,0,0.3);
                    font-size:13px; font-weight:600; max-width:340px;">
            {banner_text}
        </div>
        '''
        m.get_root().html.add_child(folium.Element(banner_html))
        print(f"Warning banner: {n_extreme} extreme, {n_heavy} heavy")
    else:
        print("No heavy/extreme rainfall — map banner hidden")

    # ---------- Legend ----------
    legend_html_pred = '''
    {% raw %}
    <style>
        #legend-pred {
            position: fixed; bottom: 60px; right: 10px; z-index: 1000;
            background: rgba(255,255,255,0.97); padding: 5px 7px;
            border: 1px solid #ccc; border-radius: 6px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
            width: 30vw; max-width: 115px; min-width: 90px;
            font-size: 9.2px; line-height: 1.3; user-select: none; cursor: move;
        }
        #legend-pred .resize-handle {
            position: absolute; bottom: 1px; right: 1px; width: 12px; height: 12px;
            background: linear-gradient(135deg, transparent 50%, #777 50%);
            cursor: nwse-resize; opacity: 0.5;
        }
        #legend-pred:hover .resize-handle { opacity: 0.8; }
        #legend-pred p { margin: 0 0 4px 0; font-weight: bold; text-align: center; font-size: 11px; }
        #legend-pred strong { font-size: 9.5px; }
        #legend-pred img { width: 85%; max-width: 78px; display: block; margin: 2px auto; }
        @media (max-width: 768px) {
            #legend-pred { width: 35vw; max-width: 130px; font-size: 9.5px; bottom: 50px; right: 8px; }
            #legend-pred p { font-size: 10px; }
            #legend-pred img { max-width: 80px; }
        }
    </style>
    <div id="legend-pred">
        <p>Forecast Rainfall</p>
        <strong>Peak Rainfall</strong><br>
        <img src="[PEAK_BASE64]"><br>
        <strong>Expected 24h</strong><br>
        <img src="[24H_BASE64]">
        <div class="resize-handle"></div>
    </div>
    {% endraw %}
    '''
    peak_b64 = embed_image_to_base64("colorbar_peak_b.png")
    _24h_b64 = embed_image_to_base64("colorbar_24h_b.png")
    legend_html_pred = legend_html_pred.replace("[PEAK_BASE64]", peak_b64).replace("[24H_BASE64]", _24h_b64)
    add_draggable_legend(m, legend_html_pred, "legend-pred")

    add_rivers_layer(m, "RBI_Sulawesi2_sungai.geojson", "Rivers")
    folium.LayerControl(collapsed=True).add_to(m)

    m.save("rainfall_prediction_map_b.html")
    print("✅ Saved: rainfall_prediction_map_b.html")

    last_update = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    history_snapshot = {
        "generated": generated_time,
        "last_update": last_update,
        "timezone": TIMEZONE_LABEL,
        "thresholds": {
            "peak_heavy_mm": THRESHOLD_PEAK_HEAVY,
            "peak_extreme_mm": THRESHOLD_PEAK_EXTREME,
            "peak_moderate_mm": THRESHOLD_PEAK_MODERATE,
            "h24_heavy_mm": THRESHOLD_24H_HEAVY,
            "h24_extreme_mm": THRESHOLD_24H_EXTREME,
            "h24_moderate_mm": THRESHOLD_24H_MODERATE,
        },
        "stations": station_history,
        "unavailable_stations": unavailable_stations,
    }
    history_changed = save_weather_history(
        "rainfall_history_b.json", "b", "Paguyaman",
        history_snapshot, last_update,
    )
    if history_changed:
        print("✅ New forecast snapshot saved: rainfall_history_b.json")
    else:
        print("ℹ️ Forecast unchanged; history not duplicated. Updated last_update.")

    # ---------- Write warnings JSON for the dashboard ----------
    warnings_payload = {
        "map": "b",
        "area": "Paguyaman",
        "generated": generated_time,
        "last_update": last_update,
        "timezone": TIMEZONE_LABEL,
        "thresholds": {
            "peak_heavy_mm": THRESHOLD_PEAK_HEAVY,
            "peak_extreme_mm": THRESHOLD_PEAK_EXTREME,
            "peak_moderate_mm": THRESHOLD_PEAK_MODERATE,
            "h24_heavy_mm": THRESHOLD_24H_HEAVY,
            "h24_extreme_mm": THRESHOLD_24H_EXTREME,
            "h24_moderate_mm": THRESHOLD_24H_MODERATE,
        },
        "has_warnings": len(warning_stations) > 0,
        "n_heavy": sum(1 for s in warning_stations if s["level"] == "heavy"),
        "n_extreme": sum(1 for s in warning_stations if s["level"] == "extreme"),
        "n_moderate": sum(1 for s in warning_stations if s["level"] == "moderate"),
        "stations": warning_stations,
        "summary": warnings_list,
    }
    with open("warnings_b.json", "w", encoding="utf-8") as f:
        json.dump(warnings_payload, f, indent=2, ensure_ascii=False)
    print("✅ Saved: warnings_b.json (for dashboard sidebar)")


# ────────────────────────────────────────────────
# Real-Time Map (optional – still disabled by default)
# ────────────────────────────────────────────────
def create_realtime_map():
    m = folium.Map(location=MAP_CENTER, zoom_start=ZOOM_START, tiles=None, control_scale=True)
    add_basemaps(m)

    generated_time = (datetime.now(timezone.utc) + TIMEZONE_OFFSET).strftime("%d %B %Y %H:%M")
    title_html = f'''
    <div style="position:absolute; top:10px; left:50%; transform:translateX(-50%);
                background:rgba(255,255,255,0.94); padding:6px 16px 5px; border-radius:6px;
                box-shadow:0 2px 8px rgba(0,0,0,0.22); z-index:1000; text-align:center;
                border:1px solid #ccc; max-width:min(560px, 70vw); pointer-events:none;">
        <div style="font-size:17px; font-weight:700; line-height:1.2;">Rainfall Map – Paguyaman (Real-time)</div>
        <div style="font-size:11px; font-weight:500; color:#444; margin-top:2px;">
            {generated_time} {TIMEZONE_LABEL} · OpenWeatherMap · IDW
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    fg_markers = folium.FeatureGroup(name="Stations", show=True)
    fg_idw = folium.FeatureGroup(name="IDW Current Rainfall (mm)", show=True)

    points, values = [], []

    for lat, lon in LOCATIONS:
        data = get_current_weather(lat, lon)
        rain = data["rainfall"] if data else 0.0
        temp = data["temp"] if data else 0.0
        hum = data["humidity"] if data else 0
        time_str = (
            (datetime.fromtimestamp(data["timestamp"], tz=timezone.utc) + TIMEZONE_OFFSET)
            .strftime("%Y-%m-%d %H:%M") if data else "N/A"
        )

        popup_content = f"""
        <div style="font-size:13px; line-height:1.5;">
            <b>Current Weather Observation (1-hour)</b><br><br>
            <b>Time:</b> {time_str} {TIMEZONE_LABEL}<br>
            <b>Rainfall:</b> {rain:.1f} mm<br>
            <b>Temperature:</b> {temp:.1f} °C<br>
            <b>Humidity:</b> {hum}%
        </div>
        """
        popup_obj = folium.Popup(popup_content, max_width=350)
        tooltip_text = f"Rainfall: {rain:.1f} mm | Temperature: {temp:.1f} °C"

        icon = folium.Icon(
            icon="cloud-rain" if rain > 0 else "sun",
            prefix="fa",
            color="blue" if rain > 0 else "orange",
        )
        folium.Marker(
            [lat, lon], popup=popup_obj, icon=icon, tooltip=folium.Tooltip(tooltip_text)
        ).add_to(fg_markers)

        points.append((lat, lon))
        values.append(rain)

    fg_markers.add_to(m)
    add_idw_overlay(fg_idw, points, values, "idw_current_b", "YlGnBu", "Current 1h Rainfall_a")

    rmax = max(values) if values else 0
    vmax = max(5.0, round(rmax + 0.8, 1)) if rmax > 0 else 3.0
    create_colorbar_png("colorbar_current_b.png", "YlGnBu", 0.0, vmax, "Current 1h Rainfall (mm)")

    fg_idw.add_to(m)

    legend_html_real = '''
    {% raw %}
    <style>
        #legend-real {
            position: fixed; bottom: 60px; right: 10px; z-index: 1000;
            background: rgba(255,255,255,0.97); padding: 6px 8px;
            border: 1px solid #ccc; border-radius: 6px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
            width: 30vw; max-width: 140px; min-width: 90px;
            font-size: 10px; line-height: 1.3; user-select: none; cursor: move;
        }
        #legend-real .resize-handle {
            position: absolute; bottom: 1px; right: 1px; width: 12px; height: 12px;
            background: linear-gradient(135deg, transparent 50%, #777 50%);
            cursor: nwse-resize; opacity: 0.5;
        }
        #legend-real p { margin: 0 0 4px 0; font-weight: bold; text-align: center; font-size: 11px; }
        #legend-real img { width: 85%; max-width: 90px; display: block; margin: 2px auto; }
    </style>
    <div id="legend-real">
        <p>Current Rainfall</p>
        <img src="[CURRENT_BASE64]" width="90">
        <div class="resize-handle"></div>
    </div>
    {% endraw %}
    '''
    current_b64 = embed_image_to_base64("colorbar_current_b.png")
    legend_html_real = legend_html_real.replace("[CURRENT_BASE64]", current_b64)
    add_draggable_legend(m, legend_html_real, "legend-real")

    folium.LayerControl().add_to(m)
    m.save("rainfall_realtime_map_b.html")
    print("✅ Saved: rainfall_realtime_map_b.html")


# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
if __name__ == "__main__":
    now_str = (datetime.now(timezone.utc) + TIMEZONE_OFFSET).strftime("%d %B %Y %H:%M")
    print(f"Generating maps at {now_str} {TIMEZONE_LABEL}...")
    create_prediction_map()
    # create_realtime_map()   # enable when you want the real-time map
    print("Done!")
    print("  • rainfall_prediction_map_b.html")
    print("  • warnings_b.json   ← used by the dashboard sidebar")
