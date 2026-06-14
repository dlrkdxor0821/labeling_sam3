"""Convert between YOLO labels and labelme JSON (rectangle shapes)."""
from utils.boxes import xyxy_to_yolo, yolo_to_xyxy


def yolo_to_labelme(yolo_lines, img_w, img_h, class_names, image_path):
    """YOLO label lines -> labelme JSON dict with rectangle shapes (pre-drawn boxes)."""
    shapes = []
    for line in yolo_lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        x1, y1, x2, y2 = yolo_to_xyxy((cx, cy, w, h), img_w, img_h)
        label = class_names[cls] if cls < len(class_names) else str(cls)
        shapes.append({
            "label": label,
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
        })
    return {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }


def _bbox_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def labelme_to_yolo(labelme_dict, class_to_id):
    """labelme JSON dict -> YOLO label lines (polygon/rectangle -> bounding box)."""
    img_w = labelme_dict["imageWidth"]
    img_h = labelme_dict["imageHeight"]
    lines = []
    for shape in labelme_dict.get("shapes", []):
        x1, y1, x2, y2 = _bbox_from_points(shape["points"])
        cx, cy, w, h = xyxy_to_yolo((x1, y1, x2, y2), img_w, img_h)
        cls = class_to_id.get(shape["label"], 0)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines
