# resonance-tower-data
Live inputs for the Resonance Tower lamp, published every 10 minutes at
**https://ademkadman.github.io/resonance-tower-data/lamp.json**

```json
{"f1": 7.84, "a1": 3.44, "kp": 0.0, "f1_ts": "…", "a1_ts": "…", "kp_ts": "…", "ts": "…", "fetched": "…", "errors": {}}
```
- **f1** Schumann fundamental frequency (Hz), **a1** its amplitude: digitised from the Tomsk SOS70 charts
  (`sos70.ru`, `srf.jpg` / `sra.jpg`). Tesseract reads the auto-scaling axis labels, which are snapped to the chart's
  grid lines; the white trace is followed pixel by pixel.
- **kp** NOAA SWPC planetary Kp (1-minute estimate).
- If a source fails, the previous value is kept (see `errors`). The full 3-day series is in `data.json`.

Workflows: `publish` (every 10 min → GitHub Pages), `selftest` (on push: Tesseract vs. the macOS Vision reference in
`tests/` must give identical traces), `keepalive` (monthly commit so the schedule is never paused).
