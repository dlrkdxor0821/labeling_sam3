"""Pure bounding-box conversions between pixel-xyxy and normalized YOLO format."""


def xyxy_to_yolo(xyxy, img_w, img_h):
    """[x1,y1,x2,y2] pixels -> (cx, cy, w, h) normalized 0..1."""
    x1, y1, x2, y2 = xyxy
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return (cx, cy, w, h)


def yolo_to_xyxy(yolo, img_w, img_h):
    """(cx, cy, w, h) normalized -> [x1, y1, x2, y2] pixels."""
    cx, cy, w, h = yolo
    return (
        (cx - w / 2) * img_w,
        (cy - h / 2) * img_h,
        (cx + w / 2) * img_w,
        (cy + h / 2) * img_h,
    )


def yolo_label_lines(boxes_xyxy, img_w, img_h, class_id=0):
    """List of pixel xyxy boxes -> list of 'class cx cy w h' YOLO label strings."""
    lines = []
    for b in boxes_xyxy:
        cx, cy, w, h = xyxy_to_yolo(b, img_w, img_h)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines
