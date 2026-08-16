# VisionGuide AI

A real-time AI visual companion for visually impaired users — point a phone or webcam at the world and it detects obstacles, reads text aloud, identifies currency, describes scenes, and answers spoken questions, all through a push-to-talk voice interface.

Built on a FastAPI backend (YOLO, EasyOCR, OpenCV face recognition, Gemini) and a single-page frontend that talks to it over a simple REST API.

---

## Screenshots

| Login | Live camera + voice | AI Log | Location |
|---|---|---|---|
| ![Login screen](docs/screenshots/login-screen.png) | ![Live tab](docs/screenshots/live-tab.png) | ![AI Log tab](docs/screenshots/ai-log-tab.png) | ![Location tab](docs/screenshots/location-tab.png) |

> Only the login screenshot above is a real capture. The other three are placeholders — run the app locally (see below), take a screenshot of each tab, and save it over the matching file in `docs/screenshots/` (`live-tab.png`, `ai-log-tab.png`, `location-tab.png`) to fill them in.

---

## Features

- **Object & obstacle detection** — YOLO-based detection with proximity (`close` / `medium` / `far`) and direction (`left` / `center` / `right`) estimated from bounding box size and position, so warnings can be spoken as "person close, left."
- **Scene description** — sends a snapshot to Gemini for a natural-language description of what's in front of the camera.
- **Text reading (OCR)** — reads signs, labels, and printed text aloud via EasyOCR.
- **Currency identification** — identifies banknotes/denominations from a snapshot.
- **Voice commands** — hold the mic, speak naturally ("what's in front of me?", "read this", "remember this person"), and Gemini classifies intent and routes to the right feature.
- **Face memory (on-device)** — save and recognize known people locally using OpenCV's LBPH recognizer. Nothing leaves the device; disabled by default (`enable_face_recognition` in config).
- **Fall detection & emergency alert** — watches device motion for a possible fall, asks the user to confirm by voice, and if there's no response, emails an emergency contact with the user's location via EmailJS.
- **Location / "where am I"** — resolves GPS coordinates into a readable address via the Google Maps Geocoding API, used both in the Location tab and in emergency alerts.
- **AI Log** — a running, timestamped record of everything the AI has detected this session (detections, scene descriptions, OCR text, currency reads, voice replies).

---

## Tech stack

| Layer | Tech |
|---|---|
| Backend | FastAPI, Uvicorn |
| Object detection | Ultralytics YOLO (`yolo11n.pt` / `yolo11s.pt`) |
| OCR | EasyOCR |
| Face recognition | OpenCV (Haar cascade + LBPH), local only |
| Scene description & voice intent | Google Gemini (`google-generativeai`) |
| Location | Google Maps Geocoding API |
| Frontend | Single-page HTML/CSS/JS, Web Speech API, MediaDevices, EmailJS |
| Auth | Signed HMAC session tokens, single demo account from `.env` |

---

## Project structure

```
visionguide-ai-main/
├── backend/
│   ├── app.py                  # FastAPI app + route registration
│   ├── core/                   # settings (.env-backed) and logging
│   ├── api/routes/             # one router per feature (detect, ocr, scene,
│   │                            currency, voice, location, auth)
│   └── services/                # request/response glue between routes and ai_engine
├── ai_engine/
│   ├── detector/                # YOLO wrapper + proximity/direction estimator
│   ├── ocr/                     # EasyOCR wrapper
│   ├── scene/                   # Gemini-based scene narrator
│   ├── currency/                # currency recognizer
│   └── face/                    # local face recognizer (LBPH)
├── frontend/
│   └── index.html                # login + Live / AI Log / Location tabs
├── models/                       # face cascade, saved face model/labels
├── tests/                        # pytest tests for detector/ocr/scene
├── docs/screenshots/              # README screenshots
├── start.ps1                      # one-click start (Windows)
└── .env                           # local secrets, gitignored
```

---

## Setup

### 1. Prerequisites

- Python 3.11+ (project was built/tested against 3.13)
- A webcam/mic-capable browser (Chrome recommended — Web Speech API support)
- A [Google AI Studio](https://aistudio.google.com/) API key for Gemini
- (Optional, for the Location tab) A [Google Cloud](https://console.cloud.google.com/) API key with the **Geocoding API** enabled — this is a *separate* key from the Gemini one

### 2. Configure environment variables

Create/edit `.env` in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL_NAME=gemini-2.0-flash
GOOGLE_MAPS_API_KEY=your_maps_geocoding_key   # optional — Location tab falls back to raw coordinates without it
APP_USER_EMAIL=test@example.com
APP_USER_PASSWORD=changeme
```

`.env` is gitignored — never commit real keys.

### 3. Install backend dependencies

```bash
python -m venv visionguide_env

# Windows
.\visionguide_env\Scripts\Activate.ps1
# macOS / Linux
source visionguide_env/bin/activate

pip install -r backend/requirements.txt
```

### 4. Run it

**Option A — one command (Windows):**
```powershell
.\start.ps1
```
This starts the backend, serves the frontend, and opens your browser automatically.

**Option B — manual (two terminals, any OS):**

Terminal 1 — backend:
```bash
uvicorn backend.app:app --reload
```
Runs at `http://127.0.0.1:8000`.

Terminal 2 — frontend:
```bash
cd frontend
python -m http.server 5500
```
Runs at `http://127.0.0.1:5500` — open this in your browser.

> Always access the frontend via `http://127.0.0.1:5500`, not as a `file://` path — camera/mic access and the API calls both require it to be served over HTTP from `127.0.0.1`, matching the backend's expected origin.

### 5. Log in

Use the credentials from `.env` (`APP_USER_EMAIL` / `APP_USER_PASSWORD`, `test@example.com` / `changeme` by default).

---

## API reference

All routes below (except `/api/login`) require `Authorization: Bearer <token>`, obtained from `/api/login`.

| Method | Route | Body | Returns |
|---|---|---|---|
| `POST` | `/api/login` | `{ email, password }` | `{ success, token }` |
| `POST` | `/api/detect` | `image` (multipart) | `{ count, detections: [{ label, confidence, box, center, proximity, direction }] }` |
| `POST` | `/api/read-text` | `image` (multipart) | `{ combined_text, ... }` |
| `POST` | `/api/describe-scene` | `image` (multipart) | `{ description }` |
| `POST` | `/api/identify-currency` | `image` (multipart) | `{ result }` |
| `POST` | `/api/voice-command` | `image`, `transcript` (multipart) | `{ intent, spoken_text, awaiting_name }` |
| `POST` | `/api/save-face` | `image`, `name` (multipart) | `{ spoken_text }` |
| `POST` | `/api/location` | `{ lat, lng }` | `{ lat, lng, address, locality?, resolved }` |
| `GET` | `/health` | — | `{ status, service }` |

---

## Testing

```bash
pytest tests/
```

Covers the detector, OCR, and scene narrator services.

---

## Privacy notes

- Face recognition runs entirely on-device (OpenCV LBPH) — no images are uploaded for this feature, and it's off by default (`enable_face_recognition = False` in `backend/core/config.py`).
- Scene description and voice-intent classification do send the current camera frame to the Gemini API, since that's how those features work.
- Location resolution sends only raw coordinates (no images) to the Google Maps Geocoding API.

---

## Known limitations

- Proximity/direction estimates are heuristic (based on bounding-box size and position), not physically calibrated depth.
- Speech recognition relies on the Web Speech API, which is best supported in Chrome.
- The demo login is a single hardcoded account from `.env`, not a real user system.
