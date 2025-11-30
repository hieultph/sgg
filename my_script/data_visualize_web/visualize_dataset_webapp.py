"""
Web App để visualize PSG Dataset với filtering

Usage:
    conda activate sgg_benchmark && python visualize_dataset_webapp.py

Features:
    - Browse all training images
    - Filter by object class
    - Filter by predicate
    - Show bounding boxes and relations
    - Interactive visualization

Access at: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify, send_file
import json
import cv2
import numpy as np
import os
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# Global data cache
DATASET = None
PREDICATES = None
OBJECTS = None
IMAGE_DIR = None

def load_dataset():
    """Load PSG dataset and labels"""
    global DATASET, PREDICATES, OBJECTS, IMAGE_DIR
    
    print("Loading PSG dataset...")
    
    # Load annotations
    with open('datasets/psg/psg/psg_train_val.json', 'r') as f:
        DATASET = json.load(f)
    
    # Load predicates
    with open('psg_predicates_list.json', 'r') as f:
        PREDICATES = json.load(f)
    
    # Load object classes
    with open('checkpoints/react_PSG/labels.json', 'r') as f:
        OBJECTS = json.load(f)
    
    IMAGE_DIR = 'datasets/psg/coco/coco/'  # Contains train2017/, val2017/ folders
    
    print(f"✓ Loaded {len(DATASET['data'])} images")
    print(f"✓ {len(PREDICATES)} predicates, {len(OBJECTS)} object classes")

def draw_scene_graph_on_image(image_entry, show_bbox=True, show_relations=True):
    """Draw bounding boxes and relations on image"""
    img_path = os.path.join(IMAGE_DIR, image_entry['file_name'])
    
    if not os.path.exists(img_path):
        # Return placeholder
        print(f"⚠ Image file not found: {img_path}")
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, 'Image file not found', (50, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f'{os.path.basename(img_path)}', (50, 270), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return img
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if not show_bbox and not show_relations:
        return img
    
    # Load segments (objects)
    segments = image_entry.get('segments_info', [])
    
    # Draw bounding boxes
    if show_bbox and 'annotations' in image_entry:
        for idx, ann in enumerate(image_entry['annotations']):
            if 'bbox' not in ann:
                continue
            
            x, y, w, h = ann['bbox']
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Get object class
            cat_id = segments[idx]['category_id'] if idx < len(segments) else 0
            obj_name = OBJECTS.get(cat_id, f"obj_{cat_id}")
            
            # Draw bbox
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"[{idx}] {obj_name}"
            cv2.putText(img, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw relations as lines
    if show_relations:
        relations = image_entry.get('relations', [])
        annotations = image_entry.get('annotations', [])
        
        for sub_idx, obj_idx, pred_id in relations:
            if sub_idx >= len(annotations) or obj_idx >= len(annotations):
                continue
            
            # Get centers of bounding boxes
            sub_bbox = annotations[sub_idx].get('bbox')
            obj_bbox = annotations[obj_idx].get('bbox')
            
            if not sub_bbox or not obj_bbox:
                continue
            
            sub_center = (int(sub_bbox[0] + sub_bbox[2]/2), 
                         int(sub_bbox[1] + sub_bbox[3]/2))
            obj_center = (int(obj_bbox[0] + obj_bbox[2]/2), 
                         int(obj_bbox[1] + obj_bbox[3]/2))
            
            # Draw arrow
            cv2.arrowedLine(img, sub_center, obj_center, (0, 255, 0), 2, tipLength=0.3)
            
            # Draw predicate label
            mid_x = (sub_center[0] + obj_center[0]) // 2
            mid_y = (sub_center[1] + obj_center[1]) // 2
            
            pred_name = PREDICATES[pred_id] if pred_id < len(PREDICATES) else f"pred_{pred_id}"
            cv2.putText(img, pred_name, (mid_x, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    return img

def image_to_base64(img):
    """Convert numpy image to base64 string"""
    img_pil = Image.fromarray(img)
    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """Get dataset statistics"""
    # Count predicate frequency
    pred_counts = {}
    obj_counts = {}
    
    for entry in DATASET['data'][:1000]:  # Sample for speed
        for sub, obj, pred in entry.get('relations', []):
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        
        for seg in entry.get('segments_info', []):
            cat_id = seg['category_id']
            obj_counts[cat_id] = obj_counts.get(cat_id, 0) + 1
    
    return jsonify({
        'total_images': len(DATASET['data']),
        'total_predicates': len(PREDICATES),
        'total_objects': len(OBJECTS),
        'predicate_counts': pred_counts,
        'object_counts': obj_counts
    })

@app.route('/api/predicates')
def get_predicates():
    """Get all predicates"""
    return jsonify(PREDICATES)

@app.route('/api/objects')
def get_objects():
    """Get all object classes"""
    return jsonify(OBJECTS)

@app.route('/api/images')
def get_images():
    """Get filtered images"""
    page = int(request.args.get('page', 0))
    per_page = int(request.args.get('per_page', 20))
    obj_filter = request.args.get('object', None)
    pred_filter = request.args.get('predicate', None)
    
    # Filter images
    filtered_data = []
    
    for entry in DATASET['data']:
        # Check object filter
        if obj_filter and obj_filter != 'all':
            obj_id = int(obj_filter)
            has_obj = any(seg['category_id'] == obj_id 
                         for seg in entry.get('segments_info', []))
            if not has_obj:
                continue
        
        # Check predicate filter
        if pred_filter and pred_filter != 'all':
            pred_id = int(pred_filter)
            has_pred = any(pred == pred_id 
                          for _, _, pred in entry.get('relations', []))
            if not has_pred:
                continue
        
        filtered_data.append(entry)
    
    # Pagination
    start = page * per_page
    end = start + per_page
    page_data = filtered_data[start:end]
    
    # Prepare response
    results = []
    for entry in page_data:
        results.append({
            'image_id': entry['image_id'],
            'file_name': entry['file_name'],
            'width': entry['width'],
            'height': entry['height'],
            'num_objects': len(entry.get('segments_info', [])),
            'num_relations': len(entry.get('relations', []))
        })
    
    return jsonify({
        'total': len(filtered_data),
        'page': page,
        'per_page': per_page,
        'images': results
    })

@app.route('/api/image/<image_id>')
def get_image_detail(image_id):
    """Get detailed info for specific image"""
    # Find image entry
    entry = None
    for e in DATASET['data']:
        if str(e['image_id']) == str(image_id):
            entry = e
            break
    
    if not entry:
        return jsonify({'error': 'Image not found'}), 404
    
    # Prepare relations with labels
    relations_data = []
    segments = entry.get('segments_info', [])
    
    for sub_idx, obj_idx, pred_id in entry.get('relations', []):
        sub_cat = segments[sub_idx]['category_id'] if sub_idx < len(segments) else 0
        obj_cat = segments[obj_idx]['category_id'] if obj_idx < len(segments) else 0
        
        relations_data.append({
            'subject_idx': sub_idx,
            'object_idx': obj_idx,
            'predicate_id': pred_id,
            'subject_name': OBJECTS.get(sub_cat, f"obj_{sub_cat}"),
            'object_name': OBJECTS.get(obj_cat, f"obj_{obj_cat}"),
            'predicate_name': PREDICATES[pred_id] if pred_id < len(PREDICATES) else f"pred_{pred_id}"
        })
    
    # Prepare objects
    objects_data = []
    for idx, seg in enumerate(segments):
        cat_id = seg['category_id']
        objects_data.append({
            'index': idx,
            'category_id': cat_id,
            'category_name': OBJECTS.get(cat_id, f"obj_{cat_id}"),
            'area': seg.get('area', 0)
        })
    
    return jsonify({
        'image_id': entry['image_id'],
        'file_name': entry['file_name'],
        'width': entry['width'],
        'height': entry['height'],
        'objects': objects_data,
        'relations': relations_data
    })

@app.route('/api/visualize/<image_id>')
def visualize_image(image_id):
    """Generate visualized image"""
    show_bbox = request.args.get('bbox', 'true').lower() == 'true'
    show_relations = request.args.get('relations', 'true').lower() == 'true'
    
    # Find image
    entry = None
    for e in DATASET['data']:
        if str(e['image_id']) == str(image_id):
            entry = e
            break
    
    if not entry:
        print(f"⚠ Image ID {image_id} not found in dataset")
        # Return placeholder image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, f'Image ID {image_id} not found', (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        img_base64 = image_to_base64(img)
        return jsonify({'image': img_base64, 'error': 'Image not found'})
    
    # Draw image
    try:
        img = draw_scene_graph_on_image(entry, show_bbox, show_relations)
        img_base64 = image_to_base64(img)
        return jsonify({'image': img_base64})
    except Exception as e:
        print(f"⚠ Error visualizing image {image_id}: {e}")
        # Return error image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, f'Error: {str(e)}', (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        img_base64 = image_to_base64(img)
        return jsonify({'image': img_base64, 'error': str(e)})

if __name__ == '__main__':
    # Load data
    load_dataset()
    
    print("\n" + "="*80)
    print("🚀 PSG Dataset Visualization Web App")
    print("="*80)
    print("\nStarting server...")
    print("Access at: http://localhost:5000")
    print("\nPress Ctrl+C to stop")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
