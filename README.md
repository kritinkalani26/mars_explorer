# Mars Surface Explorer

A live browser for NASA's Perseverance rover images with automatic AI-powered terrain classification. Images are fetched in real-time from NASA's Mars Rover Photos API, and each image is classified into one of six terrain categories using OpenAI's CLIP vision model — entirely in your browser, with no proprietary cloud AI required.

---

## What It Does

- **Fetches the latest Perseverance rover images** from NASA's Mars Rover Photos API (most recent sol)
- **Classifies each image's terrain** using [CLIP](https://openai.com/research/clip) (a zero-shot Vision-Language model from OpenAI / HuggingFace) into: **Sand, Bedrock, Gravel, Rock, Soil, Unknown**
- **Caches all classifications** to a local JSON file — each image is only ever classified once
- **Visualises everything** in a dark, space-themed UI with terrain badges, confidence bars, filter controls, and live stats

---

## Why It Matters

### Rover Navigation
Understanding the terrain type at any given location is critical for safe rover navigation. Sandy or loose soil poses wheel-slip risk. Large rocks can damage actuators. Bedrock offers firm footing. Today's rovers rely on mission controllers manually reviewing images before each drive; AI terrain classification could automate this triage and enable faster, safer autonomous traversal.

### Planetary Science
Terrain distribution data — gathered at scale — helps geologists map surface composition across large areas far more quickly than manual annotation. Identifying transitions between bedrock and soil layers, or tracking the extent of sand dune fields, feeds directly into models of Martian geology, aeolian processes, and the planet's climatic history.

---

## Prerequisites

- Python 3.9 or newer
- A free NASA API key — get one at [api.nasa.gov](https://api.nasa.gov) (takes 30 seconds)
- Internet connection (for NASA API + HuggingFace model download on first run)
- ~1 GB free disk space (for the CLIP model weights, downloaded once)

---

## Setup & Running

**1. Clone / download this project**

```bash
git clone <your-repo-url>
cd mars_explorer
```

**2. Create your environment file**

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder with your NASA API key:

```
NASA_API_KEY=your_actual_key_here
```

> Without a key the app still works using `DEMO_KEY`, but NASA rate-limits that to 30 requests/hour.

**3. (Optional) Create a virtual environment**

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

> **Note on PyTorch:** The above installs the default PyTorch package, which may include CUDA support and be large (~2–3 GB). For a CPU-only install (smaller):
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

**5. Run the app**

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## First-Run Notes

- On the **very first run**, the CLIP model (~600 MB) is downloaded from HuggingFace and cached locally. This happens once.
- Each image is downloaded from NASA and run through the model. Expect **10–40 seconds** for the first page of 12 images (depends on your connection speed and CPU).
- **Subsequent loads are instant** — all results are cached in `cache/classifications.json`.
- The loading spinner stays visible while classification is in progress.

---

## API Reference

| Endpoint | Description |
|---|---|
| `GET /api/images?page=1&terrain=all` | Paginated images with terrain + metadata. `terrain` can be `all`, `sand`, `bedrock`, `gravel`, `rock`, `soil`, or `unknown`. |
| `GET /api/stats` | Count of classified images broken down by terrain type. |
| `GET /api/latest-sol` | The Martian sol number of the most recent available images. |

---

## Project Structure

```
mars_explorer/
├── app.py               # Flask backend: NASA API, CLIP classification, endpoints
├── requirements.txt     # Python dependencies
├── .env                 # Your secrets (never committed)
├── .env.example         # Template for .env
├── .gitignore
├── README.md
├── templates/
│   └── index.html       # Single-file frontend (vanilla JS, no build step)
└── cache/
    └── classifications.json   # Auto-generated classification cache
```

---

## How the Classification Works

CLIP (Contrastive Language–Image Pretraining) learns a joint embedding space for images and text. At classification time:

1. The rover image is encoded into a 512-dimensional vector by the CLIP image encoder.
2. Five terrain description strings (e.g. *"sandy terrain and wind-blown dunes on the surface of Mars"*) are encoded into the same space by the CLIP text encoder.
3. Cosine similarity is computed between the image vector and each terrain vector.
4. Softmax turns similarities into probabilities.
5. The highest-probability terrain wins. If the confidence is below 30%, the image is labelled **Unknown**.

This is zero-shot classification — CLIP was never trained on Mars images specifically, but it generalises well because its training on 400 million image-text pairs gives it a strong understanding of terrain vocabulary.

---

## Credits

- **Imagery:** NASA / JPL-Caltech — public domain
- **AI model:** [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) via HuggingFace Transformers
- **Data source:** [NASA Mars Rover Photos API](https://api.nasa.gov)
