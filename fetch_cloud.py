"""Cloud fetcher for the Resonance Tower lamp (runs on GitHub Actions, Linux).

Same method as ~/resonance-tower/fetch.py, but the axis labels are read with Tesseract instead of macOS Vision:
  - Tomsk SOS70 charts (srf.jpg = frequencies, sra.jpg = amplitudes): read the auto-scaling left-axis labels,
    then follow the white F1 / A1 trace pixel by pixel.
  - NOAA SWPC planetary Kp.
Writes site/lamp.json  {f1, a1, kp, ts, fetched, src}  (+ site/data.json with the full 3-day series).
If a source fails, the last good value is kept from the currently published lamp.json, so the lamp never sees a gap.

Usage:  python fetch_cloud.py                 → site/
        python fetch_cloud.py --selftest      → compare Tesseract with the Vision reference in tests/
"""
import datetime as dt
import json
import pathlib
import sys

import numpy as np
import pytesseract
import requests
from PIL import Image, ImageOps

HERE = pathlib.Path(__file__).resolve().parent
SITE = HERE / "site"
PUBLISHED = "https://ademkadman.github.io/resonance-tower-data/lamp.json"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh)", "Referer": "https://sos70.ru/"}
TOMSK_URL = "https://sos70.ru/provider.php?file={}.jpg"
KP_URL = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"

# chart geometry (1006 x 340 images): hour 0 of the first day at x = 72, 288 px per day
X0, PX_PER_DAY, X_END = 72, 288, 936
TOMSK_UTC_OFFSET = 7
SCALE = 4
REGIONS = {  # name: (x0, y0, x1, y1) crop, chosen so 'F1'/'A1' and the other traces' labels stay out
    "axis": (22, 18, 72, 142),
    "dates": (0, 0, 1006, 22),
    "hours": (60, 318, 960, 340),
}


# ------------------------------------------------------------------ OCR (Tesseract)
def ocr(path):
    """Return [{text, x, y, w, h}] in image pixels, like the Vision helper on the Mac."""
    img = Image.open(path).convert("L")
    out = []
    for name, (x0, y0, x1, y1) in REGIONS.items():
        crop = img.crop((x0, y0, x1, y1)).resize(((x1 - x0) * SCALE, (y1 - y0) * SCALE), Image.LANCZOS)
        crop = ImageOps.invert(ImageOps.autocontrast(crop))           # dark text on white
        crop = crop.point(lambda v: 255 if v > 170 else 0)             # clean binary for Tesseract
        cfg = "--psm 11 -c tessedit_char_whitelist=0123456789."
        d = pytesseract.image_to_data(crop, config=cfg, output_type=pytesseract.Output.DICT)
        for t, l, tp, w, h, c in zip(d["text"], d["left"], d["top"], d["width"], d["height"], d["conf"]):
            if t.strip() and float(c) > 30:
                out.append({"text": t.strip(), "x": x0 + l / SCALE, "y": y0 + tp / SCALE, "w": w / SCALE, "h": h / SCALE})
    return out


# ------------------------------------------------------------------ digitising (same logic as fetch.py)
def parse_float(s):
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def grid_rows(path):
    """y of the dashed green horizontal grid lines in the top band (each axis label is centred on one)."""
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    green = (a[..., 1] > 90) & (a[..., 0] < 90) & (a[..., 2] < 90)
    cnt = green[:150, 80:880].sum(axis=1)
    return [y for y in range(15, 150) if cnt[y] > 60 and cnt[y] >= cnt[y - 1] and cnt[y] >= cnt[y + 1]]


def axis_fit(labels, path):
    """Linear y-pixel → value. Label centres are snapped to the grid line they sit on, so the result does not
    depend on how tightly the OCR engine draws its text boxes."""
    rows = grid_rows(path)
    pts = []
    for d in labels:
        v = parse_float(d["text"])
        if d["x"] < 70 and d["y"] < 140 and v is not None:
            yc = d["y"] + d["h"] / 2
            near = min(rows, key=lambda r: abs(r - yc)) if rows else None
            pts.append((near if near is not None and abs(near - yc) <= 6 else yc, v))
    if len(pts) < 3:
        raise RuntimeError(f"only {len(pts)} axis labels read")
    ys, vs = np.array(pts).T
    slope, icpt = np.polyfit(ys, vs, 1)
    resid = np.abs(np.polyval([slope, icpt], ys) - vs).max()
    step = np.median(np.abs(np.diff(np.sort(vs))))
    if slope >= 0 or resid > 0.25 * step:  # values must fall downward and lie on a line
        raise RuntimeError(f"axis labels not linear (resid {resid:.3f}): {pts}")
    return slope, icpt, (ys.min(), ys.max())


def first_day(labels):
    for _, t in sorted((d["x"], d["text"]) for d in labels if d["y"] < 20):
        try:
            return dt.datetime.strptime(t.strip(), "%d.%m.%Y")
        except ValueError:
            continue
    raise RuntimeError("no date label read")


def check_geometry(labels):
    zeros = sorted(d["x"] + d["w"] / 2 for d in labels if d["text"].strip() == "0" and d["y"] > 300)
    for z in zeros:
        k = round((z - X0) / PX_PER_DAY)
        if abs(z - (X0 + k * PX_PER_DAY)) > 6:
            raise RuntimeError(f"hour-0 tick at x={z:.0f} does not match chart geometry")


def trace(path, labels):
    slope, icpt, (ytop, ybot) = axis_fit(labels, path)
    check_geometry(labels)
    day0 = first_day(labels)
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    lo, hi = a.min(axis=2), a.max(axis=2)
    white = (lo > 150) & (hi - lo < 50)
    y0, y1 = int(ytop - 12), int(ybot + 12)
    series = []
    for x in range(X0 + 1, X_END):
        col = white[y0:y1, x].copy()
        if x > 890:  # the SOS70 logo in the top-right corner is white too
            col[: max(0, 90 - y0)] = False
        ys = np.nonzero(col)[0]
        if len(ys) == 0:
            continue
        y = y0 + np.median(ys)
        t = day0 + dt.timedelta(days=(x - X0) / PX_PER_DAY) - dt.timedelta(hours=TOMSK_UTC_OFFSET)
        series.append((t.strftime("%Y-%m-%dT%H:%M:%SZ"), round(float(slope * y + icpt), 3)))
    if len(series) < 50:
        raise RuntimeError(f"trace too short ({len(series)} columns)")
    return series


def sane(key, v):
    return {"f1": 7.0 <= v <= 8.7, "a1": 0 <= v <= 60}[key]


# ------------------------------------------------------------------ main
def run():
    SITE.mkdir(exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        prev = requests.get(PUBLISHED, timeout=20).json()
    except Exception:
        prev = {}
    lamp, full, errors = {}, {}, {}
    for key, name in (("f1", "srf"), ("a1", "sra")):
        try:
            r = requests.get(TOMSK_URL.format(name), headers=UA, timeout=60)
            r.raise_for_status()
            if r.content[:3] != b"\xff\xd8\xff":
                raise RuntimeError(f"not a JPEG ({len(r.content)} bytes)")
            path = HERE / f"{name}.jpg"
            path.write_bytes(r.content)
            series = trace(path, ocr(path))
            if not sane(key, series[-1][1]):
                raise RuntimeError(f"implausible value {series[-1][1]}")
            full[key] = series
            lamp[key], lamp[key + "_ts"] = series[-1][1], series[-1][0]
        except Exception as e:
            errors[key] = str(e)[:200]
            if key in prev:  # keep the last good value
                lamp[key], lamp[key + "_ts"] = prev[key], prev.get(key + "_ts", prev.get("ts"))
    try:
        r = requests.get(KP_URL, timeout=30)
        r.raise_for_status()
        last = r.json()[-1]
        lamp["kp"], lamp["kp_ts"] = last["estimated_kp"], last["time_tag"] + "Z"
    except Exception as e:
        errors["kp"] = str(e)[:200]
        if "kp" in prev:
            lamp["kp"], lamp["kp_ts"] = prev["kp"], prev.get("kp_ts")
    lamp["ts"] = max((lamp[k] for k in ("f1_ts", "a1_ts") if lamp.get(k)), default=None)
    lamp["fetched"] = now
    lamp["errors"] = errors
    (SITE / "lamp.json").write_text(json.dumps(lamp) + "\n")
    (SITE / "data.json").write_text(json.dumps({"fetched_utc": now, "errors": errors, **full}) + "\n")
    (SITE / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Resonance Tower data</title><pre>" + json.dumps(lamp, indent=2) + "</pre>")
    print(json.dumps(lamp, indent=2))
    if "f1" not in lamp or "a1" not in lamp:
        sys.exit(1)  # nothing usable at all: make the run show red


def selftest():
    ok = True
    for name in ("srf", "sra"):
        img = HERE / "tests" / f"{name}.jpg"
        ref = json.loads((HERE / "tests" / f"{name}.vision.json").read_text())
        mine = ocr(img)
        a_ref = sorted((round(d["y"]), d["text"]) for d in ref if d["x"] < 70 and d["y"] < 140 and parse_float(d["text"]))
        a_me = sorted((round(d["y"]), d["text"]) for d in mine if d["x"] < 70 and d["y"] < 140)
        s_ref, s_me = trace(img, ref), trace(img, mine)
        diff = max(abs(a[1] - b[1]) for a, b in zip(s_ref, s_me)) if len(s_ref) == len(s_me) else None
        print(f"{name}: axis Vision {a_ref}\n{name}: axis Tess.  {a_me}")
        print(f"{name}: date {first_day(mine).date()} | last {s_me[-1]} vs {s_ref[-1]} | max diff {diff}")
        ok &= diff is not None and diff < 1e-6 and first_day(mine) == first_day(ref)
    print("SELFTEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else run()
