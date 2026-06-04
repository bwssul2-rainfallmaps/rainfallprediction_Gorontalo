"""
Weather Map Generation script for location b: Pagyuaman Area
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

logging.basicConfig(filename="rainfall_maps.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ================== CONFIG ==================
API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not API_KEY:
    raise ValueError("OPENWEATHER_API_KEY environment variable is not set")
CACHE_DURATION = 3600
FORECAST_INTERVALS = 8
TIMEZONE_OFFSET = timedelta(hours=8)
TIMEZONE_LABEL = "WITA"

BASMAP = "OpenStreetMap" #"CartoDB positron" #"OpenStreetMap" "Stamen Terrain" "CartoDB positron" "CartoDB dark_matter" "Stamen Toner", etc

LOCATIONS = [
        (0.527443,123.060951), (0.551369,123.06236), (0.573717,123.01552),
        (0.62261,123.010997), (0.627603,123.08563), (0.53769,123.136951),
        (0.514899,123.221886), (0.506889,123.338041), (0.496038,123.433839),
        (0.71619,123.078239), (0.661621,122.98395), (0.606263,122.913432),
        (0.642131,122.851663), (0.705002,122.852499), (0.54153,122.888785),
        (0.63183,122.735449), (0.42106,123.22333), (0.577479,123.30391),
        (0.769264,122.941285), (0.587255,122.684044),   
]

MAP_CENTER = [0.573717,123.01552]
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
        encoded = base64.b64encode(img_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded}"

# ────────────────────────────────────────────────
# API Helpers (unchanged)
# ────────────────────────────────────────────────
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_api_data(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

def load_cache(cache_file):
    if not os.path.exists(cache_file): return {}
    try:
        with open(cache_file, encoding="utf-8") as f:
            cache = json.load(f)
        if datetime.now().timestamp() - cache.get("timestamp", 0) < CACHE_DURATION:
            return cache.get("data", {})
        os.remove(cache_file)
    except: pass
    return {}

def save_cache(cache_file, data):
    cache = {"timestamp": int(datetime.now().timestamp()), "data": data}
    tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, dir=os.path.dirname(cache_file) or ".")
    try:
        json.dump(cache, tmp)
        tmp.close()
        shutil.move(tmp.name, cache_file)
    except:
        if os.path.exists(tmp.name): os.remove(tmp.name)

def get_3h_forecast(lat, lon, cache_file="forecast_3h_cache.json"):
    key = f"{lat:.6f},{lon:.6f}"
    cache = load_cache(cache_file)
    if key in cache: return cache[key]

    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        data = fetch_api_data(url)
        forecasts = []
        for item in data["list"][:FORECAST_INTERVALS]:
            dt = datetime.fromtimestamp(item["dt"], tz=timezone.utc) + TIMEZONE_OFFSET
            forecasts.append({
                "time": dt.strftime("%Y-%m-%d %H:%M"),
                "pop": round(item.get("pop", 0) * 100),
                "rainfall": item.get("rain", {}).get("3h", 0),
            })
        cache[key] = forecasts
        save_cache(cache_file, cache)
        return forecasts
    except Exception as e:
        print(f"Forecast error: {e}")
        return None

def get_current_weather(lat, lon, cache_file="current_cache.json"):
    key = f"{lat:.6f},{lon:.6f}"
    cache = load_cache(cache_file)
    if key in cache: return cache[key]

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    try:
        data = fetch_api_data(url)
        result = {
            "rainfall": data.get("rain", {}).get("1h", 0),
            "temp": data.get("main", {}).get("temp", 0),
            "humidity": data.get("main", {}).get("humidity", 0),
            "timestamp": data.get("dt"),
        }
        cache[key] = result
        save_cache(cache_file, cache)
        return result
    except Exception as e:
        print(f"Current weather error: {e}")
        return None

# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────
def add_draggable_legend(m, html_content: str, div_id: str):
    js = f"""
    <script>
    function makeResizableAndDraggable(id) {{
        const el = document.getElementById(id);
        if (!el) return;

        // Draggable
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

        // Resizable
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

        // Auto-size to content + screen
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
    if vmax <= vmin: vmax = vmin + 1
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

        grid_lon = np.linspace(min_lon, max_lon, 700)
        grid_lat = np.linspace(min_lat, max_lat, 700)
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
        plt.savefig(base_png, dpi=140, transparent=True, bbox_inches="tight", pad_inches=0)
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
                levels = levels[np.diff(np.concatenate(([data_min-0.01], levels))) > 0]

                if len(levels) >= 2:
                    fig2, ax2 = plt.subplots(figsize=(10, 10))
                    ax2.axis("off")
                    ax2.set_xlim(min_lon, max_lon)
                    ax2.set_ylim(min_lat, max_lat)
                    ax2.set_position([0, 0, 1, 1])

                    ax2.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap_name, alpha=0.55)
                    cs = ax2.contour(grid_x, grid_y, grid_z, levels=levels, colors="black", linewidths=1.1)
                    ax2.clabel(cs, inline=True, fontsize=9, fmt="%.1f mm")

                    plt.savefig(contour_png, dpi=140, transparent=True, bbox_inches="tight", pad_inches=0)
                    plt.close(fig2)
                    contours_created = True
                    print(f"   → Contours created (max {data_max:.2f} mm, {len(levels)} levels)")
                else:
                    print("   → Skipped contours: not enough distinct levels")
            except Exception as contour_err:
                print(f"   → Contour failed ({contour_err}) → skipping contours")
        else:
            print(f"   → Skipped contours: very low rain (max {data_max:.2f} mm, range {data_range:.2f} mm)")

        bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        if os.path.exists(base_png):
            folium.raster_layers.ImageOverlay(image=base_png, bounds=bounds, opacity=0.65, interactive=True).add_to(fg)
        if contours_created and os.path.exists(contour_png):
            folium.raster_layers.ImageOverlay(image=contour_png, bounds=bounds, opacity=0.75, interactive=True).add_to(fg)

        print(f"✅ {layer_title} added successfully!")
    except Exception as e:
        print(f"❌ IDW failed ({e}). Using HeatMap fallback...")
        HeatMap([[p[0], p[1], max(0.1, v)] for p,v in zip(points, values)], 
                radius=15, blur=10,
                gradient={"0.4": "blue", "0.65": "lime", "0.9": "yellow", "1.0": "red"}).add_to(fg)

# ────────────────────────────────────────────────
# Prediction Map (with smarter, more accurate colorbar)
# ────────────────────────────────────────────────
def create_prediction_map():
    m = folium.Map(location=MAP_CENTER, zoom_start=ZOOM_START, tiles=BASMAP)

    # Title + Timestamp + Source
    title_html = '''
    <div style="position: absolute; top: 10px; left: 50%; transform: translateX(-50%);
                background: rgba(255,255,255,0.92); padding: 8px 18px; border-radius: 6px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.25); font-size: 19px; font-weight: bold;
                z-index: 1000; white-space: nowrap; border: 1px solid #ccc;">
        Rainfall Map – Paguyaman & Surroundings
    </div>
    '''

    timestamp_html = f'''
    <div style="position: absolute; top: 10px; right: 60px; background: rgba(255,255,255,0.92);
                padding: 6px 12px; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                font-size: 13px; z-index: 1000; text-align: right; border: 1px solid #ddd;">
        Generated: {(datetime.now(timezone.utc) + TIMEZONE_OFFSET).strftime("%d %B %Y %H:%M")} {TIMEZONE_LABEL}<br>
        <small style="color:#555; font-size:11px;">Data: OpenWeatherMap | Interpolation: IDW</small>
    </div>
    '''

    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element(timestamp_html))

    fg_markers  = folium.FeatureGroup(name="Stations", show=True)
    fg_heatmap  = folium.FeatureGroup(name="Heatmap (Max PoP)", show=True)
    fg_idw_peak = folium.FeatureGroup(name="IDW Peak Rain (mm)", show=False)
    fg_idw_24h  = folium.FeatureGroup(name="IDW Expected 24h (mm)", show=True)

    heat_data = []
    points_peak = []
    values_peak = []
    points_24h = []
    values_24h = []

    for lat, lon in LOCATIONS:
        fc = get_3h_forecast(lat, lon)
        if not fc:
            folium.CircleMarker([lat, lon], radius=5, color="gray", fill=True, fill_opacity=0.6,
                                popup="No data").add_to(fg_markers)
            continue

        max_pop = max(d["pop"] for d in fc)
        rain_at_max_pop = max(d["rainfall"] for d in fc)
        expected_24h = sum((d["pop"] / 100) * d["rainfall"] for d in fc)

        popup_content = f"""
        <div style="font-size:13px; line-height:1.5;">
            <b>3-Hourly Precipitation Forecast</b><br><br>
        """ + "".join(
            f"<b>{d['time']}:</b> {d['pop']}% probability – {d['rainfall']:.1f} mm<br>"
            for d in fc
        ) + "</div>"

        tooltip_text = f"Max probability: {max_pop}% | Peak rain: {rain_at_max_pop:.1f} mm"

        popup_obj = folium.Popup(popup_content, max_width=350)

        icon = folium.Icon(icon="cloud-rain" if max_pop > 50 else "sun", prefix="fa",
                           color="blue" if max_pop > 50 else "orange")

        folium.Marker([lat, lon], popup=popup_obj, tooltip=folium.Tooltip(tooltip_text), icon=icon).add_to(fg_markers)

        heat_data.append([lat, lon, max_pop / 100.0])
        points_peak.append((lat, lon))
        values_peak.append(rain_at_max_pop)
        points_24h.append((lat, lon))
        values_24h.append(expected_24h)

    if heat_data:
        HeatMap(heat_data, radius=10, blur=3, gradient={"0.4": "blue", "0.65": "lime", "0.9": "yellow", "1.0": "red"}).add_to(fg_heatmap)

    fg_markers.add_to(m)
    fg_heatmap.add_to(m)

    # ────────────────────────────────────────────────
    # Simple & accurate sync: Use real min/max for both map and legend
    # ────────────────────────────────────────────────
    if values_peak:
        min_p = min(values_peak)
        max_p = max(values_peak)
        vmin_p = round(min_p * 0.9, 1) if min_p > 0 else 0.0
        vmax_p = round(max_p * 1.15, 1)
        
        add_idw_overlay(fg_idw_peak, points_peak, values_peak, "idw_peak_b", "YlOrRd", "IDW Peak Rain (mm)_b", 
                        vmin=vmin_p, vmax=vmax_p)
        create_colorbar_png("colorbar_peak_b.png", "YlOrRd", vmin_p, vmax_p, "Peak Rainfall (mm)_b")

    if values_24h:
        min_24 = min(values_24h)
        max_24 = max(values_24h)
        vmin_24 = round(min_24 * 0.9, 1) if min_24 > 0 else 0.0
        vmax_24 = round(max_24 * 1.10, 1)
        
        add_idw_overlay(fg_idw_24h, points_24h, values_24h, "idw_24h_b", "YlGnBu", "IDW Expected 24h (mm)_b", 
                        vmin=vmin_24, vmax=vmax_24)
        create_colorbar_png("colorbar_24h_b.png", "YlGnBu", vmin_24, vmax_24, "Expected 24h Rainfall (mm)_b")

    fg_idw_peak.add_to(m)
    fg_idw_24h.add_to(m)

    legend_html_pred = '''
    {% raw %}
    <style>
        #legend-pred {
            position: fixed;
            bottom: 60px;
            right: 10px;
            z-index: 1000;
            background: rgba(255,255,255,0.97);
            padding: 6px 8px;
            border: 1px solid #ccc;
            border-radius: 6px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
            width: 30vw;
            max-width: 140px;
            min-width: 90px;
            font-size: 10px;
            line-height: 1.3;
            user-select: none;
            cursor: move;
            transition: all 0.2s ease;
        }
        #legend-pred .resize-handle {
            position: absolute;
            bottom: 1px;
            right: 1px;
            width: 12px;
            height: 12px;
            background: linear-gradient(135deg, transparent 50%, #777 50%);
            cursor: nwse-resize;
            opacity: 0.5;
        }
        #legend-pred:hover .resize-handle { opacity: 0.8; }
        #legend-pred p {
            margin: 0 0 4px 0;
            font-weight: bold;
            text-align: center;
            font-size: 11px;
        }
        #legend-pred strong {
            font-size: 9.5px;
        }
        #legend-pred img {
            width: 85%;
            max-width: 90px;
            display: block;
            margin: 2px auto;
        }
        @media (max-width: 768px) {
            #legend-pred {
                width: 35vw;
                max-width: 130px;
                font-size: 9.5px;
                padding: 5px 7px;
                bottom: 50px;
                right: 8px;
            }
            #legend-pred p { font-size: 10px; }
            #legend-pred img { max-width: 80px; }
        }
        @media (max-width: 480px) {
            #legend-pred {
                width: 40vw;
                max-width: 110px;
                font-size: 9px;
                padding: 4px 6px;
                bottom: 45px;
            }
            #legend-pred img { max-width: 70px; }
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

    folium.LayerControl().add_to(m)
    m.save("rainfall_prediction_map_b.html")
    print("✅ Saved: rainfall_prediction_map_b.html")
# ────────────────────────────────────────────────
# Real-Time Map
# ────────────────────────────────────────────────
def create_realtime_map():
    m = folium.Map(location=MAP_CENTER, zoom_start=ZOOM_START, tiles=BASMAP)

    # Title + Timestamp + Source
    title_html = '''
    <div style="position: absolute; top: 10px; left: 50%; transform: translateX(-50%);
                background: rgba(255,255,255,0.92); padding: 8px 18px; border-radius: 6px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.25); font-size: 19px; font-weight: bold;
                z-index: 1000; white-space: nowrap; border: 1px solid #ccc;">
        Rainfall Map – Paguyaman & Surroundings
    </div>
    '''

    timestamp_html = f'''
    <div style="position: absolute; top: 10px; right: 10px; background: rgba(255,255,255,0.92);
                padding: 6px 12px; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                font-size: 13px; z-index: 1000; text-align: right; border: 1px solid #ddd;">
        Generated: {datetime.now().strftime("%d %B %Y %H:%M")} {TIMEZONE_LABEL}<br>
        <small style="color:#555; font-size:11px;">Data: OpenWeatherMap | Interpolation: IDW</small>
    </div>
    '''

    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element(timestamp_html))

    fg_markers = folium.FeatureGroup(name="Stations", show=True)
    fg_idw     = folium.FeatureGroup(name="IDW Current Rainfall (mm)", show=True)

    points = []
    values = []

    for lat, lon in LOCATIONS:
        data = get_current_weather(lat, lon)
        rain = data["rainfall"] if data else 0.0
        temp = data["temp"] if data else 0.0
        hum  = data["humidity"] if data else 0
        time_str = (datetime.fromtimestamp(data["timestamp"], tz=timezone.utc) + TIMEZONE_OFFSET).strftime("%Y-%m-%d %H:%M") if data else "N/A"

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

        icon = folium.Icon(icon="cloud-rain" if rain > 0 else "sun", prefix="fa",
                           color="blue" if rain > 0 else "orange")

        folium.Marker([lat, lon], popup=popup_obj, icon=icon,
                      tooltip=folium.Tooltip(tooltip_text)).add_to(fg_markers)

        points.append((lat, lon))
        values.append(rain)

    fg_markers.add_to(m)

    add_idw_overlay(fg_idw, points, values, "idw_current_b", "YlGnBu", "Current 1h Rainfall_b")

    rmax = max(values) if values else 0
    vmax = max(5.0, round(rmax + 0.8, 1)) if rmax > 0 else 3.0
    create_colorbar_png("colorbar_current_b.png", "YlGnBu", 0.0, vmax, "Current 1h Rainfall (mm)_b")

    fg_idw.add_to(m)

    legend_html_real = '''
    {% raw %}
    <style>
        #legend-real {
            position: fixed;
            bottom: 60px;
            right: 10px;
            z-index: 1000;
            background: rgba(255,255,255,0.97);
            padding: 6px 8px;
            border: 1px solid #ccc;
            border-radius: 6px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
            width: 30vw;
            max-width: 140px;
            min-width: 90px;
            font-size: 10px;
            line-height: 1.3;
            user-select: none;
            cursor: move;
            transition: all 0.2s ease;
        }
        #legend-real .resize-handle {
            position: absolute;
            bottom: 1px;
            right: 1px;
            width: 12px;
            height: 12px;
            background: linear-gradient(135deg, transparent 50%, #777 50%);
            cursor: nwse-resize;
            opacity: 0.5;
        }
        #legend-real:hover .resize-handle { opacity: 0.8; }
        #legend-real p {
            margin: 0 0 4px 0;
            font-weight: bold;
            text-align: center;
            font-size: 11px;
        }
        #legend-real img {
            width: 85%;
            max-width: 90px;
            display: block;
            margin: 2px auto;
        }
        @media (max-width: 768px) {
            #legend-real {
                width: 35vw;
                max-width: 130px;
                font-size: 9.5px;
                padding: 5px 7px;
                bottom: 50px;
                right: 8px;
            }
            #legend-real p { font-size: 10px; }
            #legend-real img { max-width: 80px; }
        }
        @media (max-width: 480px) {
            #legend-real {
                width: 40vw;
                max-width: 110px;
                font-size: 9px;
                padding: 4px 6px;
                bottom: 45px;
            }
            #legend-real img { max-width: 70px; }
        }
    </style>

    <div id="legend-real">
        <p>Current Rainfall</p>
        <img src="[CURRENT_BASE64]" width="90">
        <div class="resize-handle"></div>
    </div>
    {% endraw %}
    '''

    current_b64 = embed_image_to_base64("colorbar_current.png")
    legend_html_real = legend_html_real.replace("[CURRENT_BASE64]", current_b64)

    add_draggable_legend(m, legend_html_real, "legend-real")

    folium.LayerControl().add_to(m)
    m.save("rainfall_realtime_map_b.html")
    print("✅ Saved: rainfall_realtime_map_b.html")

# ────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────
if __name__ == "__main__":
    now_str = datetime.now().strftime("%d %B %Y %H:%M")
    print(f"Generating maps at {now_str} {TIMEZONE_LABEL}...")
    create_prediction_map()
    # create_realtime_map() # disabled to save API calls during testing - enable when ready for real-time map
    print("Done! Files saved (single-file versions):")
    print("  • rainfall_prediction_map_b.html")
    print("  • rainfall_realtime_map.html")
