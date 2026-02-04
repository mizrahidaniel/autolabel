# AutoLabel - ML-Assisted Data Labeling Tool

**60-80% of ML engineering time is spent labeling data. Let's fix that.**

## The Problem

Manual data labeling is slow and expensive:
- **Scale AI costs $$$** (starts at $0.08/image for classification, $$$$ for complex annotations)
- **Open source tools have no ML assistance** (labelImg, CVAT - pure manual work)
- **Active learning requires custom code** (which examples should I label next?)
- **Collaboration is clunky** (emailing CSV files, merge conflicts)

## The Solution

Web-based labeling tool with ML superpowers:

- 🤖 **ML Pre-Labeling** - Use CLIP/ResNet to suggest labels (saves 70%+ time)
- 🎯 **Active Learning** - Prioritize uncertain examples (label smarter, not harder)
- 🏷️ **Multi-Task Support** - Classification, object detection, segmentation
- 📤 **Export Formats** - COCO, YOLO, TF Record, CSV
- 👥 **Collaborative** - Multiple annotators, consensus tracking
- 🔒 **Self-Hosted** - Your data stays yours

## MVP (Week 1)

**Phase 1: Image Classification**
- Flask backend + React frontend
- Upload images (drag-drop, bulk upload)
- ML pre-labeling using CLIP (zero-shot classification)
- Manual correction interface
- Export to JSON/CSV
- SQLite for label storage

**Phase 2: Active Learning**
- Uncertainty scoring (which images need human review?)
- Smart batch suggestions
- Confidence visualization

**Phase 3: Object Detection**
- Bounding box annotation
- YOLO export format
- Pre-labeling with YOLOv8

## Monetization

- **Free (Self-Hosted):** Unlimited projects, basic ML models (ResNet, CLIP)
- **Pro ($19/mo):** Advanced models (SAM, Grounding DINO), active learning, team collaboration
- **Enterprise ($99/mo):** Custom model training, API access, SLA support

**Target:** 100 GitHub stars, 20 Pro users ($380 MRR) in 60 days

## Tech Stack

- **Backend:** Flask + SQLite (PostgreSQL for production)
- **Frontend:** React + Tailwind + shadcn/ui
- **ML:** CLIP (classification), SAM (segmentation), YOLOv8 (detection)
- **Deployment:** Docker + Fly.io

## Quickstart

### Backend (Flask + CLIP)

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
```

Server runs on `http://localhost:5000`

### Docker

```bash
docker build -t autolabel .
docker run -p 5000:5000 -v $(pwd)/uploads:/app/uploads autolabel
```

### API Usage

**1. Create a project**
```bash
curl -X POST http://localhost:5000/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "Animal Classifier", "labels": ["cat", "dog", "bird"]}'
```

**2. Upload images (auto-labeled with CLIP)**
```bash
curl -X POST http://localhost:5000/projects/1/upload \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

**3. Get images**
```bash
curl http://localhost:5000/projects/1/images
```

**4. Update label (manual correction)**
```bash
curl -X PUT http://localhost:5000/images/1/label \
  -H "Content-Type: application/json" \
  -d '{"label": "dog", "is_verified": true}'
```

**5. Export labels**
```bash
curl http://localhost:5000/projects/1/export
```

## Why This Matters

Every ML project needs labeled data. This tool saves weeks of manual work and thousands of dollars in labeling costs.

**Let's ship it.**
