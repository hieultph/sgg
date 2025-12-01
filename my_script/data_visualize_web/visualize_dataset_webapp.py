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
    with open('../../datasets/psg/psg/psg_train_val.json', 'r') as f:
    # with open('../../tools/SGG-Annotate/my_custom_images/custom_psg_detections.json', 'r') as f:
        DATASET = json.load(f)
    
    # Load predicates directly from dataset
    PREDICATES = DATASET['predicate_classes']
    
    # Load object classes: thing_classes + stuff_classes
    # thing_classes are COCO objects (person, car, etc.)
    # stuff_classes are background/region classes (wall, sky, etc.)
    thing_classes = DATASET['thing_classes']
    stuff_classes = DATASET.get('stuff_classes', [])
    
    # Combine: thing classes first, then stuff classes
    # This matches how category_id is indexed in annotations
    OBJECTS = thing_classes + stuff_classes
    
    # Convert to dict for easy lookup: {category_id: class_name}
    OBJECTS = {i: name for i, name in enumerate(OBJECTS)}
    
    IMAGE_DIR = '../../datasets/psg/coco/coco/'  # Contains train2017/, val2017/ folders
    # IMAGE_DIR = '../../tools/SGG-Annotate/my_custom_images/images/'  # Contains train2017/, val2017/ folders
    
    print(f"✓ Loaded {len(DATASET['data'])} images")
    print(f"✓ {len(PREDICATES)} predicates, {len(OBJECTS)} object classes")
    print(f"  - thing_classes: {len(thing_classes)}")
    print(f"  - stuff_classes: {len(stuff_classes)}")

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
    
    # Load segments (objects) - contains category_id for each object
    segments = image_entry.get('segments_info', [])
    annotations = image_entry.get('annotations', [])
    
    # Draw bounding boxes
    if show_bbox and annotations:
        for idx, ann in enumerate(annotations):
            if 'bbox' not in ann:
                continue
            
            # Bbox format: [x1, y1, x2, y2] (XYXY format, bbox_mode=0)
            # Not [x, y, width, height]!
            x1, y1, x2, y2 = ann['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Convert to x, y, w, h for drawing
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            
            # Get object class from annotations (not segments_info)
            # category_id in annotations corresponds to thing_classes + stuff_classes index
            cat_id = ann.get('category_id', 0)
            obj_name = OBJECTS.get(cat_id, f"obj_{cat_id}")
            
            # Generate consistent color per object index
            np.random.seed(idx * 42)  # Consistent color
            color = tuple(np.random.randint(100, 255, 3).tolist())
            
            # Draw bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with background
            label = f"[{idx}] {obj_name}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x, y - label_h - 5), (x + label_w, y), color, -1)
            cv2.putText(img, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw relations as lines
    if show_relations:
        relations = image_entry.get('relations', [])
        
        for rel_idx, (sub_idx, obj_idx, pred_id) in enumerate(relations):
            if sub_idx >= len(annotations) or obj_idx >= len(annotations):
                continue
            
            # Get centers of bounding boxes
            sub_bbox = annotations[sub_idx].get('bbox')
            obj_bbox = annotations[obj_idx].get('bbox')
            
            if not sub_bbox or not obj_bbox:
                continue
            
            # Centers: bbox is [x1, y1, x2, y2] format
            sub_center = (int((sub_bbox[0] + sub_bbox[2])/2), 
                         int((sub_bbox[1] + sub_bbox[3])/2))
            obj_center = (int((obj_bbox[0] + obj_bbox[2])/2), 
                         int((obj_bbox[1] + obj_bbox[3])/2))
            
            # Generate consistent color per relation
            np.random.seed(rel_idx * 123)
            arrow_color = tuple(np.random.randint(50, 255, 3).tolist())
            
            # Draw arrow
            cv2.arrowedLine(img, sub_center, obj_center, arrow_color, 2, tipLength=0.2)
            
            # Draw predicate label at midpoint
            mid_x = (sub_center[0] + obj_center[0]) // 2
            mid_y = (sub_center[1] + obj_center[1]) // 2
            
            # pred_id is 0-indexed in PSG dataset
            pred_name = PREDICATES[pred_id] if pred_id < len(PREDICATES) else f"pred_{pred_id}"
            
            # Draw label with background for readability
            (text_w, text_h), _ = cv2.getTextSize(pred_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (mid_x - 2, mid_y - text_h - 2), 
                         (mid_x + text_w + 2, mid_y + 2), (0, 0, 0), -1)
            cv2.putText(img, pred_name, (mid_x, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
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
        
        for ann in entry.get('annotations', []):
            cat_id = ann.get('category_id', 0)
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
            has_obj = any(ann.get('category_id') == obj_id 
                         for ann in entry.get('annotations', []))
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
            'num_objects': len(entry.get('annotations', [])),
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
    annotations = entry.get('annotations', [])
    
    for sub_idx, obj_idx, pred_id in entry.get('relations', []):
        # Get category_id from annotations (not segments_info)
        sub_cat = annotations[sub_idx]['category_id'] if sub_idx < len(annotations) else 0
        obj_cat = annotations[obj_idx]['category_id'] if obj_idx < len(annotations) else 0
        
        relations_data.append({
            'subject_idx': sub_idx,
            'object_idx': obj_idx,
            'predicate_id': pred_id,
            'subject_name': OBJECTS.get(sub_cat, f"obj_{sub_cat}"),
            'object_name': OBJECTS.get(obj_cat, f"obj_{obj_cat}"),
            'predicate_name': PREDICATES[pred_id] if pred_id < len(PREDICATES) else f"pred_{pred_id}"
        })
    
    # Prepare objects from annotations (not segments_info)
    objects_data = []
    for idx, ann in enumerate(annotations):
        cat_id = ann.get('category_id', 0)
        bbox = ann.get('bbox', [0, 0, 0, 0])
        # bbox is [x1, y1, x2, y2], so area = (x2-x1) * (y2-y1)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if len(bbox) == 4 else 0
        
        objects_data.append({
            'index': idx,
            'category_id': cat_id,
            'category_name': OBJECTS.get(cat_id, f"obj_{cat_id}"),
            'bbox': bbox,
            'area': area
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
