"""
AutoLabel - ML-Assisted Data Labeling Tool
Flask Backend with CLIP Zero-Shot Classification
"""
import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = Path("uploads")
DB_PATH = "autolabel.db"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

UPLOAD_FOLDER.mkdir(exist_ok=True)

# Load CLIP model
print("Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print(f"CLIP model loaded on {device}")


# Database setup
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Projects table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            labels TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Images table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            label TEXT,
            confidence REAL,
            is_verified BOOLEAN DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects (id)
        )
    """)
    
    conn.commit()
    conn.close()


init_db()


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_label(image_path: str, labels: List[str]) -> Dict[str, float]:
    """
    Use CLIP to predict label probabilities
    Returns: {"label1": 0.8, "label2": 0.15, ...}
    """
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs
        inputs = processor(
            text=[f"a photo of a {label}" for label in labels],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0].cpu().numpy()
        
        # Return label:probability mapping
        predictions = {label: float(prob) for label, prob in zip(labels, probs)}
        return predictions
    
    except Exception as e:
        print(f"Error predicting label: {e}")
        return {}


# Routes

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "device": device})


@app.route("/projects", methods=["GET", "POST"])
def projects():
    """List or create projects"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if request.method == "GET":
        cursor.execute("SELECT id, name, labels, created_at FROM projects ORDER BY created_at DESC")
        rows = cursor.fetchall()
        
        projects_list = [
            {
                "id": row[0],
                "name": row[1],
                "labels": json.loads(row[2]),
                "created_at": row[3]
            }
            for row in rows
        ]
        
        conn.close()
        return jsonify({"projects": projects_list})
    
    # POST - create new project
    data = request.json
    name = data.get("name")
    labels = data.get("labels", [])
    
    if not name or not labels:
        return jsonify({"error": "Missing name or labels"}), 400
    
    cursor.execute(
        "INSERT INTO projects (name, labels) VALUES (?, ?)",
        (name, json.dumps(labels))
    )
    conn.commit()
    project_id = cursor.lastrowid
    conn.close()
    
    return jsonify({"id": project_id, "name": name, "labels": labels}), 201


@app.route("/projects/<int:project_id>/upload", methods=["POST"])
def upload_images(project_id: int):
    """Upload images and run ML pre-labeling"""
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist("files")
    
    # Get project labels
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT labels FROM projects WHERE id = ?", (project_id,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        return jsonify({"error": "Project not found"}), 404
    
    labels = json.loads(row[0])
    
    uploaded = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{filename}"
            
            filepath = UPLOAD_FOLDER / unique_filename
            file.save(filepath)
            
            # Run ML prediction
            predictions = predict_label(str(filepath), labels)
            
            # Get top prediction
            if predictions:
                top_label = max(predictions, key=predictions.get)
                confidence = predictions[top_label]
            else:
                top_label = None
                confidence = None
            
            # Save to database
            cursor.execute(
                """INSERT INTO images (project_id, filename, filepath, label, confidence)
                   VALUES (?, ?, ?, ?, ?)""",
                (project_id, filename, str(filepath), top_label, confidence)
            )
            
            uploaded.append({
                "id": cursor.lastrowid,
                "filename": filename,
                "label": top_label,
                "confidence": confidence,
                "predictions": predictions
            })
    
    conn.commit()
    conn.close()
    
    return jsonify({"uploaded": uploaded})


@app.route("/projects/<int:project_id>/images", methods=["GET"])
def get_images(project_id: int):
    """Get all images for a project"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, filename, filepath, label, confidence, is_verified
        FROM images
        WHERE project_id = ?
        ORDER BY created_at DESC
    """, (project_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    images = [
        {
            "id": row[0],
            "filename": row[1],
            "filepath": row[2],
            "label": row[3],
            "confidence": row[4],
            "is_verified": bool(row[5])
        }
        for row in rows
    ]
    
    return jsonify({"images": images})


@app.route("/images/<int:image_id>/label", methods=["PUT"])
def update_label(image_id: int):
    """Update image label"""
    data = request.json
    label = data.get("label")
    is_verified = data.get("is_verified", True)
    
    if not label:
        return jsonify({"error": "Missing label"}), 400
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "UPDATE images SET label = ?, is_verified = ? WHERE id = ?",
        (label, is_verified, image_id)
    )
    
    conn.commit()
    conn.close()
    
    return jsonify({"success": True})


@app.route("/projects/<int:project_id>/export", methods=["GET"])
def export_labels(project_id: int):
    """Export labels as JSON"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get project info
    cursor.execute("SELECT name, labels FROM projects WHERE id = ?", (project_id,))
    project_row = cursor.fetchone()
    
    if not project_row:
        conn.close()
        return jsonify({"error": "Project not found"}), 404
    
    project_name, labels_json = project_row
    
    # Get images
    cursor.execute("""
        SELECT filename, filepath, label, confidence, is_verified
        FROM images
        WHERE project_id = ?
    """, (project_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    export_data = {
        "project": {
            "name": project_name,
            "labels": json.loads(labels_json)
        },
        "images": [
            {
                "filename": row[0],
                "filepath": row[1],
                "label": row[2],
                "confidence": row[3],
                "is_verified": bool(row[4])
            }
            for row in rows
        ],
        "summary": {
            "total_images": len(rows),
            "verified": sum(1 for row in rows if row[4]),
            "unverified": sum(1 for row in rows if not row[4])
        }
    }
    
    return jsonify(export_data)


@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_image(filename):
    """Serve uploaded images"""
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
