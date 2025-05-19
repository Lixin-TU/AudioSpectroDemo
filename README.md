# AudioSpectroDemo

Cross‑platform demo that converts a WAV file into a mel‑spectrogram visualization (PySide6) and ships as a one‑file Windows `.exe` built by GitHub Actions.

## Local quick start (macOS / Linux)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## CI build

Simply `git push` — the workflow in `.github/workflows/windows.yml` produces `AudioSpectroDemo.exe` and attaches it to the run artefacts.