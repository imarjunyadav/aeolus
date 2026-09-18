# Aeolus local end-to-end run

## Python services

Create/activate the virtual environment and install the development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Run tests:

```bash
python -m pytest
```

Start the cloud service:

```bash
AEOLUS_DATA_MODE=mock uvicorn cloud.api.main:app --reload --port 8000
```

Start the vessel service in a second terminal:

```bash
CLOUD_API_URL=http://localhost:8000 uvicorn vessel.api.main:app --reload --port 8001
```

Start the UI in a third terminal:

```bash
cd ui
npm install
npm run dev
```

Open `http://localhost:5173`.

The UI calls vessel `/sync`, which calls cloud `/forecast`, caches the resulting ForecastPackage, and then calls vessel `/navigate` for the safety-first route.

## Real provider mode

Do not enable this until the required provider credentials/network access are configured:

```bash
AEOLUS_DATA_MODE=real uvicorn cloud.api.main:app --reload --port 8000
```

Optional Indian NPDC export inputs:

```bash
export AEOLUS_NPDC_EXPORT_PATH=/absolute/path/to/npdc_export.csv
# or
export AEOLUS_NPDC_EXPORT_URL=https://your-authorized-export-url
```

The NPDC adapter intentionally accepts an export/download rather than inventing an undocumented public API.

## Docker

From the repository root:

```bash
docker compose up --build
```

Then open `http://localhost:5173`.

Docker currently uses mock data by default so the complete UI → vessel → cloud flow can be exercised without provider credentials. Switch the cloud service to real mode only after configuring the providers.
