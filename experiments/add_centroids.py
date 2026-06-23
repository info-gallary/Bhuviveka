"""
BhuViveka — Add Patch Centroids to Risk CSV
============================================
The root cause of "Rows with corrected dist_km: 0" is that the patch
centroid coordinates (lat/lon) were never written into risk_labeled_all.csv.

This script extracts centroid coordinates from the patch .npy filenames
(which encode the UTM bounding box or lat/lon) and adds them to the CSV.
Then it recomputes dist_km via Haversine from centroid to GLOF epicentre.

Run BEFORE experiment_e2e_eval_v2.py and experiment_baselines.py.

Usage:
    uv run experiment_add_centroids.py
    uv run experiment_add_centroids.py --dry-run
"""
from __future__ import annotations
import argparse, csv, json, math, re
from pathlib import Path

ROOT     = Path.cwd()
RISK_CSV = ROOT / "data" / "processed" / "risk_from_seg" / "risk_labeled_all.csv"
SEG_IDX  = ROOT / "data" / "interim" / "training_index_seg.csv"
OUT_CSV  = ROOT / "data" / "processed" / "risk_from_seg" / "risk_labeled_with_centroids.csv"

GLOF_COORDS = {
    "sikkim_south_lhonak_2016": (27.916, 88.210),
    "sikkim_south_lhonak_2023": (27.916, 88.210),
    "uttarakhand_chamoli_2021": (30.370, 79.730),
    "uttarakhand_chorabari_2013": (30.740, 79.080),
    "uttarakhand_kedarnath_2013": (30.735, 79.067),
    "himachal_gepang_gath":     (32.530, 77.209),
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2-lat1)/2)**2
         + math.cos(phi1)*math.cos(phi2)*math.sin(math.radians(lon2-lon1)/2)**2)
    return 2 * R * math.asin(math.sqrt(max(0, a)))


def match_event(ev):
    ev = ev.lower().strip()
    for key in GLOF_COORDS:
        if key in ev or ev in key: return key
    return None


def try_parse_latlon_from_path(npy_path: str):
    """
    Multiple heuristics to extract lat/lon from patch path/filename.
    Patch paths typically look like:
      data/interim/patches_v4/event_name/27.916_88.210_tile001.npy
      data/interim/patches_v4/event_name/patch_27p916_88p210.npy
      data/interim/patches_v4/event_name/N27E088_001.npy  (degree tiles)
    Returns (lat, lon) or None.
    """
    name = Path(npy_path).stem
    full = npy_path.replace("\\", "/")

    # Pattern 1: float_float in filename  e.g. 27.916_88.210
    m = re.search(r'(2[0-9]\.\d+)_([6-9][0-9]\.\d+)', full)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if 20 <= lat <= 40 and 65 <= 100: return lat, lon

    # Pattern 2: dot replaced by p  e.g. 27p916_88p210
    m = re.search(r'(2[0-9]p\d+)_([6-9][0-9]p\d+)', name)
    if m:
        try:
            lat = float(m.group(1).replace("p", "."))
            lon = float(m.group(2).replace("p", "."))
            if 20 <= lat <= 40 and 65 <= lon <= 100: return lat, lon
        except: pass

    # Pattern 3: N##E### MGRS-style  e.g. N27E088
    m = re.search(r'N(\d{2})E(\d{3})', full, re.I)
    if m:
        lat = float(m.group(1)) + 0.5  # tile centre approximation
        lon = float(m.group(2)) + 0.5
        if 20 <= lat <= 40 and 65 <= lon <= 100: return lat, lon

    # Pattern 4: underscore-separated integers that look like scaled lat/lon
    # e.g. 27916_88210 (multiply by 0.001)
    m = re.search(r'\b(2[0-9]\d{3})_([6-9]\d{4})\b', name)
    if m:
        try:
            lat = int(m.group(1)) / 1000.0
            lon = int(m.group(2)) / 1000.0
            if 20 <= lat <= 40 and 65 <= lon <= 100: return lat, lon
        except: pass

    return None


def load_patch_index():
    """Load patch index and build a map: patch_id/npy_path -> (lat, lon)."""
    if not SEG_IDX.exists():
        print(f"WARNING: {SEG_IDX} not found")
        return {}

    centroid_map = {}
    rows = list(csv.DictReader(open(SEG_IDX, encoding="utf-8")))
    print(f"Patch index: {len(rows)} rows, columns: {list(rows[0].keys())[:8]}")

    for r in rows:
        pid  = r.get("patch_id", "")
        path = r.get("npy_path", "")

        # Try explicit lat/lon columns first
        for lat_col in ["patch_lat", "lat", "centroid_lat", "center_lat", "y"]:
            for lon_col in ["patch_lon", "lon", "centroid_lon", "center_lon", "x"]:
                if lat_col in r and lon_col in r:
                    try:
                        lat = float(r[lat_col]); lon = float(r[lon_col])
                        if 20 <= lat <= 40 and 65 <= lon <= 100:
                            if pid: centroid_map[pid] = (lat, lon)
                            if path: centroid_map[path] = (lat, lon)
                            break
                    except: pass

        # Fall back to parsing from path
        if pid not in centroid_map and path:
            coords = try_parse_latlon_from_path(path)
            if coords:
                if pid: centroid_map[pid] = coords
                centroid_map[path] = coords

    print(f"  Centroid map: {len(centroid_map)} entries")
    return centroid_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(RISK_CSV, encoding="utf-8")))
    print(f"Risk CSV: {len(rows)} rows")
    print(f"Columns: {list(rows[0].keys())[:10]}")

    centroid_map = load_patch_index()

    # Also try to parse centroid from the npy_path in the risk CSV itself
    n_from_index  = 0
    n_from_path   = 0
    n_from_event  = 0  # fallback: use event epicentre (distance = 0 approx)
    n_missing     = 0

    fieldnames = list(rows[0].keys())
    for col in ["patch_lat", "patch_lon", "dist_km", "risk_label"]:
        if col not in fieldnames: fieldnames.append(col)

    fixed = []
    for r in rows:
        r = dict(r)
        pid  = r.get("patch_id", "")
        path = r.get("npy_path", "")
        ev   = r.get("event", "")

        coords = None  # (patch_lat, patch_lon)

        # Try patch index
        if pid in centroid_map:
            coords = centroid_map[pid]
            n_from_index += 1
        elif path in centroid_map:
            coords = centroid_map[path]
            n_from_index += 1
        else:
            # Try parsing from path in risk CSV
            coords = try_parse_latlon_from_path(path)
            if coords: n_from_path += 1

        if coords:
            r["patch_lat"] = f"{coords[0]:.6f}"
            r["patch_lon"] = f"{coords[1]:.6f}"
        else:
            n_missing += 1

        # Compute dist_km
        epic = GLOF_COORDS.get(match_event(ev))
        if coords and epic:
            dist = haversine_km(coords[0], coords[1], epic[0], epic[1])
            r["dist_km"] = f"{dist:.3f}"
        elif epic:
            # No centroid — use 0 (on-lake patches are very close)
            r["dist_km"] = r.get("dist_km", "0")
            n_from_event += 1

        # Re-derive risk label
        area   = float(r.get("seg_area_km2", 0) or 0)
        change = float(r.get("change_pct", 0) or 0)
        dist   = float(r.get("dist_km", 0) or 0)
        if area > 0.15 and change > 15: r["risk_label"] = "HIGH"
        elif (0.05 <= area <= 0.15) or (5 <= change <= 15): r["risk_label"] = "MEDIUM"
        else: r["risk_label"] = "LOW"

        fixed.append(r)

    print(f"\nCentroid source breakdown:")
    print(f"  From patch index:  {n_from_index}")
    print(f"  From path parsing: {n_from_path}")
    print(f"  Using event centre: {n_from_event}")
    print(f"  No centroid:       {n_missing}")
    n_with_dist = sum(1 for r in fixed if float(r.get("dist_km",0) or 0) > 0)
    print(f"  Rows with dist_km > 0: {n_with_dist}/{len(fixed)}")

    if n_missing == len(rows):
        print("\n*** DIAGNOSIS: patch centroids cannot be extracted.")
        print("    The npy filenames do not encode lat/lon, and the patch index")
        print("    does not have lat/lon columns.")
        print("\n    REQUIRED ACTION:")
        print("    Add lat/lon to risk_from_seg.py during the labelling step.")
        print("    In risk_from_seg.py, when processing each patch, compute:")
        print("      import rasterio")
        print("      with rasterio.open(patch_path) as src:")
        print("          bounds = src.bounds")
        print("          crs    = src.crs")
        print("          from pyproj import Transformer")
        print("          t = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)")
        print("          cx = (bounds.left + bounds.right) / 2")
        print("          cy = (bounds.bottom + bounds.top) / 2")
        print("          lon, lat = t.transform(cx, cy)")
        print("    Then write patch_lat, patch_lon to the output CSV row.")
        print("\n    Alternative (if .npy is not georeferenced):")
        print("    Read the corresponding metadata .json sidecar file if one exists.")

    if args.dry_run:
        print("\nDRY RUN — not saving"); return

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(fixed)
    print(f"\nSaved to {OUT_CSV}")
    print("Use this CSV as RISK_CSV in all experiment scripts.")


if __name__ == "__main__": main()
