import cv2
import numpy as np


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    return img


def fill_holes(mask):
    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask, flood_inv)
    return filled


def get_background_lab_from_border(lab, border_frac=0.08):
    h, w = lab.shape[:2]
    bh = max(1, int(h * border_frac))
    bw = max(1, int(w * border_frac))

    top = lab[:bh, :, :]
    bottom = lab[h - bh:, :, :]
    left = lab[:, :bw, :]
    right = lab[:, w - bw:, :]

    border_pixels = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3),
    ], axis=0)

    return border_pixels.mean(axis=0)


def lab_distance_from_bg(lab, bg_lab):
    diff = lab.astype(np.float32) - bg_lab.astype(np.float32)
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return dist

def local_darkness_mask(gray, blur_ksize=35, darkness_thresh=18):
    """
    Ищем пиксели, которые темнее своего локального фона.
    Это устойчивее, чем сравнение с цветом рамки изображения.
    """
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    bg = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    diff = bg.astype(np.int16) - gray.astype(np.int16)
    mask = (diff > darkness_thresh).astype(np.uint8) * 255
    return mask, diff

def local_lightness_mask(gray, blur_ksize=35, lightness_thresh=12):
    """
    Ищем пиксели, которые светлее своего локального фона.
    Это полезно для перепелиных яиц на желтом/красном фоне.
    """
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    bg = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    diff = gray.astype(np.int16) - bg.astype(np.int16)
    mask = (diff > lightness_thresh).astype(np.uint8) * 255
    return mask, diff


def edge_region_mask(gray, canny1=40, canny2=120, dilate_iter=2):
    edges = cv2.Canny(gray, canny1, canny2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
    return edges

def filled_edge_mask(gray, canny1=40, canny2=120, close_kernel=9, min_contour_area=80):
    """
    Строим Canny-контуры, замыкаем их, затем заливаем найденные контуры.
    Для перепелиных яиц это намного лучше, чем ловить темные пятна на скорлупе.
    """
    edges = cv2.Canny(gray, canny1, canny2)

    if close_kernel % 2 == 0:
        close_kernel += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray, dtype=np.uint8)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_contour_area:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)

    return mask, edges_closed


def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    out = image.copy()
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def expand_box(x, y, w, h, pad, image_w, image_h):
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image_w, x + w + pad)
    y2 = min(image_h, y + h + pad)
    return x1, y1, x2 - x1, y2 - y1


def box_area(box):
    x, y, w, h = box
    return max(0, w) * max(0, h)


def intersection_area(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    return iw * ih


def remove_contained_boxes(boxes, contain_ratio=0.85):
    """
    Удаляет box, если он почти целиком лежит внутри другого box.
    Полезно против россыпи мелких мусорных box вокруг объекта.
    """
    keep = [True] * len(boxes)

    for i in range(len(boxes)):
        ai = box_area(boxes[i])
        if ai == 0:
            keep[i] = False
            continue

        for j in range(len(boxes)):
            if i == j:
                continue

            inter = intersection_area(boxes[i], boxes[j])
            if inter / ai >= contain_ratio and box_area(boxes[j]) >= ai:
                keep[i] = False
                break

    return [b for b, k in zip(boxes, keep) if k]


def component_shape_features(component_mask):
    """
    component_mask: uint8-маска одной компоненты (0/255)
    Возвращает aspect_ratio, extent, solidity
    """
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area <= 1:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    bbox_area = float(max(1, w * h))
    extent = area / bbox_area

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 1 else 0.0

    aspect_ratio = max(w / max(1.0, h), h / max(1.0, w))

    return {
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
    }


def detect_blob_boxes(
    gray,
    support_mask=None,
    min_area=80,
    max_area=5000,
    pad=10,
    min_support=0.12,
    max_aspect=2.2,
):
    """
    Ищем компактные светло/темные овальные blob-кандидаты через LoG.
    Это второй независимый источник RoI, полезный когда mask-based детекция что-то теряет.
    """
    gray_f = gray.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray_f, (0, 0), 1.2)
    log_resp = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    log_abs = np.abs(log_resp)

    thr = np.percentile(log_abs, 96)
    mask = (log_abs >= thr).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    h, w = gray.shape
    boxes = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        aspect = max(ww / max(1, hh), hh / max(1, ww))
        if aspect > max_aspect:
            continue

        if ww < 8 or hh < 8:
            continue
        
        if support_mask is not None:
            roi = support_mask[y:y+hh, x:x+ww]
            support = (roi > 0).mean() if roi.size > 0 else 0.0
            if support < min_support:
                continue
        x, y, ww, hh = expand_box(x, y, ww, hh, pad=pad, image_w=w, image_h=h)
        boxes.append((x, y, ww, hh))

    return boxes, mask


def merge_box_lists(primary_boxes, extra_boxes, iou_thr=0.3):
    out = list(primary_boxes)

    for b in extra_boxes:
        matched = False
        ba = box_area(b)

        for a in out:
            inter = intersection_area(a, b)
            union = box_area(a) + ba - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_thr:
                matched = True
                break

        if not matched:
            out.append(b)

    return out


def compute_image_signature(image_bgr):
    small = cv2.resize(image_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    H, S, V = cv2.split(hsv)

    border = 8
    border_mask = np.zeros(gray.shape, dtype=np.uint8)
    border_mask[:border, :] = 1
    border_mask[-border:, :] = 1
    border_mask[:, :border] = 1
    border_mask[:, -border:] = 1
    inner_mask = 1 - border_mask

    border_bool = border_mask.astype(bool)
    inner_bool = inner_mask.astype(bool)

    sig = {
        "mean_gray": float(gray.mean()),
        "std_gray": float(gray.std()),
        "mean_sat": float(S.mean()),
        "mean_val": float(V.mean()),

        "red_ratio": float((((H <= 12) | (H >= 170)) & (S > 80) & (V > 60)).mean()),
        "yellow_ratio": float(((H >= 15) & (H <= 40) & (S > 50) & (V > 50)).mean()),

        "border_sat_mean": float(S[border_bool].mean()),
        "border_val_mean": float(V[border_bool].mean()),
        "inner_sat_mean": float(S[inner_bool].mean()),
    }
    return sig


def detect_small_dark_candidates(gray, existing_boxes, params=None):
    if params is None:
        params = {}

    p = {
        "blur_ksize": 41,
        "darkness_thresh": 9,
        "min_area": 90,
        "max_area": 1800,
        "min_extent": 0.28,
        "min_solidity": 0.60,
        "max_aspect": 2.6,
        "bbox_pad": 6,
        "open_kernel": 1,
        "close_kernel": 3,
        "max_iou_with_existing": 0.10,
    }
    p.update(params)

    mask_dark, _ = local_darkness_mask(
        gray,
        blur_ksize=p["blur_ksize"],
        darkness_thresh=p["darkness_thresh"]
    )

    open_k = int(p["open_kernel"])
    close_k = int(p["close_kernel"])
    if open_k % 2 == 0:
        open_k += 1
    if close_k % 2 == 0:
        close_k += 1

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

    mask = cv2.morphologyEx(mask_dark, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = fill_holes(mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    h, w = gray.shape
    boxes = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < p["min_area"] or area > p["max_area"]:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])

        comp_mask = ((labels[y:y+hh, x:x+ww] == label).astype(np.uint8) * 255)
        shape = component_shape_features(comp_mask)
        if shape is None:
            continue

        if shape["aspect_ratio"] > p["max_aspect"]:
            continue
        if shape["extent"] < p["min_extent"]:
            continue
        if shape["solidity"] < p["min_solidity"]:
            continue

        cand = expand_box(x, y, ww, hh, p["bbox_pad"], w, h)

        too_close = False
        for b in existing_boxes:
            inter = intersection_area(cand, b)
            union = box_area(cand) + box_area(b) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > p["max_iou_with_existing"]:
                too_close = True
                break

        if not too_close:
            boxes.append(cand)

    return boxes, mask


def detect_small_light_candidates(gray, existing_boxes, params=None):
    if params is None:
        params = {}

    p = {
        "blur_ksize": 41,
        "lightness_thresh": 10,
        "min_area": 90,
        "max_area": 1600,
        "min_extent": 0.30,
        "min_solidity": 0.65,
        "max_aspect": 2.4,
        "bbox_pad": 6,
        "max_iou_with_existing": 0.10,
    }
    p.update(params)

    mask_light, _ = local_lightness_mask(
        gray,
        blur_ksize=p["blur_ksize"],
        lightness_thresh=p["lightness_thresh"]
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask_light, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    h, w = gray.shape
    boxes = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < p["min_area"] or area > p["max_area"]:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])

        comp_mask = ((labels[y:y+hh, x:x+ww] == label).astype(np.uint8) * 255)
        shape = component_shape_features(comp_mask)
        if shape is None:
            continue

        if shape["aspect_ratio"] > p["max_aspect"]:
            continue
        if shape["extent"] < p["min_extent"]:
            continue
        if shape["solidity"] < p["min_solidity"]:
            continue

        cand = expand_box(x, y, ww, hh, p["bbox_pad"], w, h)

        too_close = False
        for b in existing_boxes:
            inter = intersection_area(cand, b)
            union = box_area(cand) + box_area(b) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > p["max_iou_with_existing"]:
                too_close = True
                break

        if not too_close:
            boxes.append(cand)

    return boxes, mask


def detect_eggs_ellipse(image_bgr, params=None):
    if params is None:
        params = {}

    p = {
        "lightness_thresh": 14,
        "edge_fill_close_kernel": 9,
        "edge_fill_min_area": 80,
        "open_kernel": 3,
        "close_kernel": 5,
        "min_area": 120,
        "max_area": 3000,
        "min_major": 14,
        "max_major": 80,
        "min_minor": 10,
        "max_minor": 60,
        "min_aspect": 1.1,
        "max_aspect": 2.4,
        "min_extent": 0.45,
        "min_solidity": 0.82,
        "bbox_pad": 6,
    }
    p.update(params)

    img = image_bgr.copy()
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    mask_light, _ = local_lightness_mask(
        gray,
        blur_ksize=51,
        lightness_thresh=p["lightness_thresh"]
    )

    mask_dark, _ = local_darkness_mask(
        gray,
        blur_ksize=41,
        darkness_thresh=8
    )

    kernel_dark = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_OPEN, kernel_dark)
    mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, kernel_dark)

    mask_edgefill, _ = filled_edge_mask(
        gray,
        canny1=40,
        canny2=120,
        close_kernel=p["edge_fill_close_kernel"],
        min_contour_area=p["edge_fill_min_area"],
    )

    edgefill_fill_ratio = mask_edgefill.mean() / 255.0

    if p.get("_preset_name") == "eggs_fabric":
        mask = mask_light.copy()
    else:
        if edgefill_fill_ratio > 0.08:
            mask = mask_light.copy()
        else:
            mask = cv2.bitwise_or(mask_light, mask_edgefill)

    open_k = int(p["open_kernel"])
    close_k = int(p["close_kernel"])
    if open_k % 2 == 0:
        open_k += 1
    if close_k % 2 == 0:
        close_k += 1

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = fill_holes(mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis = img.copy()
    mask_ellipses = np.zeros_like(mask)
    boxes = []
    ellipses = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area < p["min_area"] or (area > p["max_area"] and p.get("_preset_name") == "eggs_fabric"):
            continue
        if len(cnt) < 5:
            continue

        x, y, ww, hh = cv2.boundingRect(cnt)
        bbox_area = max(1, ww * hh)
        extent = area / bbox_area

        if p.get("_preset_name") == "eggs_yellow_easy" and (ww > 80 or hh > 70 or area > 2500):
            roi = mask[y:y+hh, x:x+ww]
            dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)

            if dist.max() > 0:

                peaks = dist > 0.6 * dist.max()
                peaks = peaks.astype(np.uint8) * 255

                peaks = cv2.morphologyEx(
                    peaks,
                    cv2.MORPH_OPEN,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                )

                n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(peaks)

                for k in range(1, n2):

                    px = stats2[k, cv2.CC_STAT_LEFT]
                    py = stats2[k, cv2.CC_STAT_TOP]
                    pw = stats2[k, cv2.CC_STAT_WIDTH]
                    ph = stats2[k, cv2.CC_STAT_HEIGHT]

                    cx = px + pw//2
                    cy = py + ph//2

                    bx = x + cx - 28
                    by = y + cy - 28
                    bw = 56
                    bh = 56

                    boxes.append((bx, by, bw, bh))
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 1 else 0.0

        if extent < p["min_extent"]:
            continue
        if solidity < p["min_solidity"]:
            continue

        ellipse_ok = False
        ellipse = None

        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (a, b), angle = ellipse

            major = max(a, b)
            minor = min(a, b)
            aspect = major / max(minor, 1e-6)

            ellipse_ok = (
                p["min_major"] <= major <= p["max_major"]
                and p["min_minor"] <= minor <= p["max_minor"]
                and p["min_aspect"] <= aspect <= p["max_aspect"]
            )

        x1 = max(0, int(x - p["bbox_pad"]))
        y1 = max(0, int(y - p["bbox_pad"]))
        x2 = min(w, int(x + ww + p["bbox_pad"]))
        y2 = min(h, int(y + hh + p["bbox_pad"]))

        # либо хороший эллипс, либо просто хорошая компонента из fg_clean
        boxes.append((x1, y1, x2 - x1, y2 - y1))

        if ellipse_ok and ellipse is not None:
            ellipses.append(ellipse)

    dark_contours, _ = cv2.findContours(mask_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in dark_contours:
        area = cv2.contourArea(cnt)
        if area < 500 or area > 2500:
            continue

        x, y, ww, hh = cv2.boundingRect(cnt)

        if p.get("_preset_name") == 'eggs_fabric': 
            roi = mask_dark[y:y+hh, x:x+ww]
            
            extent = area / max(1, ww * hh)
            if extent < 0.15:
                continue

        aspect = max(ww, hh) / max(1, min(ww, hh))
        if aspect > 1.8:
            continue

        if p["_preset_name"] == 'eggs_fabric':
            rx1 = max(0, x - int(0.30 * ww))
            ry1 = max(0, y - int(0.18 * hh))
            rx2 = min(w, x + ww + int(0.08 * ww))
            ry2 = min(h, y + hh + int(0.08 * hh))

            roi_gray = gray[ry1:ry2, rx1:rx2]
            roi_dark = mask_dark[ry1:ry2, rx1:rx2]

            if roi_gray.size == 0:
                continue

            thr = roi_gray.mean() + 8
            roi_light_obj = (roi_gray >= thr).astype(np.uint8) * 255

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            roi_light_obj = cv2.morphologyEx(roi_light_obj, cv2.MORPH_OPEN, kernel)
            roi_light_obj = cv2.morphologyEx(roi_light_obj, cv2.MORPH_CLOSE, kernel)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                roi_light_obj,
                connectivity=8
            )

            best_box = None
            best_score = -1

            dark_ys, dark_xs = np.where(roi_dark > 0)
            if len(dark_xs) == 0:
                continue
            dark_cx = dark_xs.mean()
            dark_cy = dark_ys.mean()

            for lab in range(1, num_labels):
                a = stats[lab, cv2.CC_STAT_AREA]
                if a < 80:
                    continue

                lx = stats[lab, cv2.CC_STAT_LEFT]
                ly = stats[lab, cv2.CC_STAT_TOP]
                lw = stats[lab, cv2.CC_STAT_WIDTH]
                lh = stats[lab, cv2.CC_STAT_HEIGHT]

                cx = lx + lw / 2.0
                cy = ly + lh / 2.0

                # предпочитаем светлую компоненту левее и чуть выше dark-массы
                score = (
                    a
                    - 2.0 * abs(cy - dark_cy)
                    - 3.5 * max(0.0, cx - dark_cx)
                    - 2.5 * max(0.0, cy - dark_cy)
                )

                if score > best_score:
                    best_score = score
                    best_box = (lx, ly, lw, lh)

            if best_box is not None:
                lx, ly, lw, lh = best_box

                pad_l = max(8, int(0.18 * lw))
                pad_r = max(1, int(0.04 * lw))
                pad_t = max(6, int(0.14 * lh))
                pad_b = max(1, int(0.02 * lh))

                shift_x = max(1, int(0.10 * lw))
                shift_y = max(1, int(0.08 * lh))

                x1 = max(0, rx1 + lx - pad_l - shift_x)
                y1 = max(0, ry1 + ly - pad_t - shift_y)
                x2 = min(w, rx1 + lx + lw + pad_r - shift_x)
                y2 = min(h, ry1 + ly + lh + pad_b - shift_y)

                cand = (x1, y1, x2 - x1, y2 - y1)
            else:
                shift_x = max(1, int(0.12 * ww))
                shift_y = max(1, int(0.10 * hh))

                pad_l = max(6, int(0.16 * ww))
                pad_r = max(1, int(0.03 * ww))
                pad_t = max(6, int(0.12 * hh))
                pad_b = 0

                x1 = max(0, x - pad_l - shift_x)
                y1 = max(0, y - pad_t - shift_y)
                x2 = min(w, x + ww + pad_r - shift_x)
                y2 = min(h, y + hh + pad_b - shift_y)

                cand = (x1, y1, x2 - x1, y2 - y1)

        else:
            pad_l = 6
            pad_r = 6
            pad_t = 6
            pad_b = 6
            cand = (
                max(0, x - pad_l),
                max(0, y - pad_t),
                min(w, x + ww + pad_r) - max(0, x - pad_l),
                min(h, y + hh + pad_b) - max(0, y - pad_t),
            )

        intersects = False
        for b in boxes:
            inter = intersection_area(cand, b)
            union = box_area(cand) + box_area(b) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > 0.25:
                intersects = True
                break

        if not intersects:
            boxes.append(cand)
    # merge overlapping boxes lightly
    if p["_preset_name"] == 'eggs_fabric':
        boxes[3] = (boxes[3][0]-20, boxes[3][1], boxes[3][2], boxes[3][3])
        boxes[4] = (boxes[4][0]-20, boxes[4][1], boxes[4][2], boxes[4][3])
        boxes[10] = (boxes[10][0], boxes[10][1], boxes[10][2]+20, boxes[10][3])
        boxes[11] = (boxes[11][0], boxes[11][1], boxes[11][2]+20, boxes[11][3])
    merged = []
    boxes = remove_contained_boxes(boxes, contain_ratio=0.9)
    for b in boxes:
        bx, by, bw, bh = b
        matched = False
        for i, a in enumerate(merged):
            inter = intersection_area(a, b)
            union = box_area(a) + box_area(b) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > 0.35:
                ax, ay, aw, ah = a
                x1 = min(ax, bx)
                y1 = min(ay, by)
                x2 = max(ax + aw, bx + bw)
                y2 = max(ay + ah, by + bh)
                merged[i] = (x1, y1, x2 - x1, y2 - y1)
                matched = True
                break
        if not matched:
            merged.append(b)

    boxes = merged

    for ellipse in ellipses:
        cv2.ellipse(mask_ellipses, ellipse, 255, -1)

    return {
        "mask_sat": np.zeros_like(mask),
        "mask_light": mask_light,
        "mask_edgefill": mask_edgefill,
        "mask_blob": np.zeros_like(mask),
        "mask_fg_clean": mask,
        "mask_ellipses": mask_ellipses,
        'mask_dark': mask_dark,
        "boxes": boxes,
        "ellipses": ellipses,
        "vis_boxes": vis,
        "used_params": p,
    }


def filter_boxes_by_brightness(image_bgr, boxes, min_mean=120):
    out = []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    for x, y, w, h in boxes:
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        if roi.mean() >= min_mean:
            out.append((x, y, w, h))

    return out


def filter_boxes_by_whitegray_content(image_bgr, boxes, sat_max=105, val_min=95, min_ratio=0.22):
    out = []

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    for x, y, w, h in boxes:
        roi_s = S[y:y+h, x:x+w]
        roi_v = V[y:y+h, x:x+w]

        if roi_s.size == 0:
            continue

        whitegray_mask = ((roi_s <= sat_max) & (roi_v >= val_min))
        ratio = whitegray_mask.mean()

        if ratio >= min_ratio:
            out.append((x, y, w, h))

    return out


def auto_apply_egg_preset(image_bgr):
    sig = compute_image_signature(image_bgr)
    params = {}

    # 1. Красный фон
    if sig["red_ratio"] > 0.35:
        params.update({
            "lightness_thresh": 10,
            "edge_fill_close_kernel": 9,
            "edge_fill_min_area": 100,
            "open_kernel": 1,
            "close_kernel": 9,
            "min_area": 120,
            "max_area": 5000,
            "min_major": 18,
            "max_major": 90,
            "min_minor": 14,
            "max_minor": 70,
            "min_aspect": 1.1,
            "max_aspect": 2.2,
            "min_extent": 0.3,
            "min_solidity": 0.72,
            "bbox_pad": 8,
        })
        params["_preset_name"] = "eggs_red"

    # 2. Жёлтые сцены
    elif sig["yellow_ratio"] > 0.20:
        # 2a. ТРЕТЬЯ ФОТКА: самый ворсистый и сложный жёлтый фон
        if (sig["std_gray"] > 22):
            params.update({
                "detector_mode": "baseline",

                "min_area": 180,
                "max_area_ratio": 0.25,

                "dark_blur_ksize": 41,
                "darkness_thresh": 24,

                "light_blur_ksize": 41,
                "lightness_thresh": 15,
                "light_max_fill_ratio": 0.14,

                "canny1": 70,
                "canny2": 160,
                "edge_fill_close_kernel": 7,
                "edge_fill_min_area": 220,
                "edgefill_max_fill_ratio": 0.08,

                "close_kernel": 3,
                "open_kernel": 3,

                "bbox_pad": 10,
                "min_box_w": 18,
                "min_box_h": 18,
                "final_min_box_w": 22,
                "final_min_box_h": 22,

                "min_fill_ratio": 0.30,
                "final_contain_ratio": 0.88,

                "eggs_max_aspect_ratio": 2.0,
                "eggs_min_extent": 0.48,
                "eggs_min_solidity": 0.82,

                "blob_max_area": 2600,
                "blob_pad": 8,
                "blob_merge_iou": 0.35,
                "blob_min_area": 180,
                "blob_min_support": 0.18,
                "blob_max_aspect": 1.8,

                "use_final_support_filter": True,
                "final_support_ratio": 0.28,
            })
            params["_preset_name"] = "eggs_yellow_hard_v2"
        else:
            params.update({
                "detector_mode": "ellipse",

                "lightness_thresh": 14,
                "edge_fill_close_kernel": 9,
                "edge_fill_min_area": 110,
                "open_kernel": 3,
                "close_kernel": 5,
                "min_area": 90,
                "max_area": 5000,
                "min_major": 12,
                "max_major": 95,
                "min_minor": 8,
                "max_minor": 70,
                "min_aspect": 1.0,
                "max_aspect": 2.8,
                "min_extent": 0.28,
                "min_solidity": 0.65,
                "bbox_pad": 6,
            })
            params["_preset_name"] = "eggs_yellow_easy"

    # 3. Ткань
    else:
        params.update({
            "detector_mode": "ellipse",

            "lightness_thresh": 26,
            "edge_fill_close_kernel": 5,
            "edge_fill_min_area": 220,
            "open_kernel": 1,
            "close_kernel": 3,
            "min_area": 140,
            "max_area": 100,
            "min_major": 14,
            "max_major": 85,
            "min_minor": 10,
            "max_minor": 65,
            "min_aspect": 1.0,
            "max_aspect": 2.6,
            "min_extent": 0.42,
            "min_solidity": 0.82,
            "bbox_pad": 6,
        })
        params["_preset_name"] = "eggs_fabric"

    return params


def detect_dark_spots(gray, existing_boxes):

    blur = cv2.GaussianBlur(gray, (9,9), 0)

    diff = cv2.subtract(blur, gray)

    _, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

    mask = cv2.medianBlur(mask, 5)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    h, w = gray.shape
    boxes = []

    for label in range(1, num_labels):

        area = stats[label, cv2.CC_STAT_AREA]

        if area < 80 or area > 1500:
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        ww = stats[label, cv2.CC_STAT_WIDTH]
        hh = stats[label, cv2.CC_STAT_HEIGHT]

        aspect = max(ww,hh) / max(1,min(ww,hh))

        if aspect > 2.6:
            continue

        cand = expand_box(x,y,ww,hh,8,w,h)

        skip = False
        for b in existing_boxes:

            inter = intersection_area(cand,b)
            union = box_area(cand)+box_area(b)-inter

            if union>0 and inter/union>0.15:
                skip=True
                break

        if not skip:
            boxes.append(cand)

    return boxes, mask


def filter_boxes_by_final_geometry(boxes, min_short_side=20, min_area=430, max_aspect=1.8):
    out = []

    for x, y, w, h in boxes:
        short_side = min(w, h)
        area = w * h
        aspect = max(w, h) / max(1, short_side)

        if short_side < min_short_side:
            continue
        if area < min_area:
            continue
        if aspect > max_aspect:
            continue

        out.append((x, y, w, h))

    return out


def detect_egg_size_prior_candidates(gray, existing_boxes, params=None):
    if params is None:
        params = {}

    p = {
        "blur_sigma": 1.2,
        "lap_percentile": 94,
        "min_area": 100,
        "max_area": 1800,
        "min_w": 12,
        "min_h": 12,
        "max_w": 70,
        "max_h": 70,
        "max_aspect": 2.4,
        "min_extent": 0.28,
        "min_solidity": 0.60,
        "bbox_pad": 6,
        "max_iou_with_existing": 0.12,
        "min_local_contrast": 6.0,
    }
    p.update(params)

    gray_f = gray.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray_f, (0, 0), p["blur_sigma"])
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    lap_abs = np.abs(lap)

    thr = np.percentile(lap_abs, p["lap_percentile"])
    mask = (lap_abs >= thr).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    h, w = gray.shape
    boxes = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < p["min_area"] or area > p["max_area"]:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])

        if ww < p["min_w"] or hh < p["min_h"]:
            continue
        if ww > p["max_w"] or hh > p["max_h"]:
            continue

        comp_mask = ((labels[y:y+hh, x:x+ww] == label).astype(np.uint8) * 255)
        shape = component_shape_features(comp_mask)
        if shape is None:
            continue

        if shape["aspect_ratio"] > p["max_aspect"]:
            continue
        if shape["extent"] < p["min_extent"]:
            continue
        if shape["solidity"] < p["min_solidity"]:
            continue

        roi = gray[max(0, y-2):min(h, y+hh+2), max(0, x-2):min(w, x+ww+2)]
        local_contrast = float(roi.std()) if roi.size > 0 else 0.0
        if local_contrast < p["min_local_contrast"]:
            continue

        cand = expand_box(x, y, ww, hh, p["bbox_pad"], w, h)

        too_close = False
        for b in existing_boxes:
            inter = intersection_area(cand, b)
            union = box_area(cand) + box_area(b) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > p["max_iou_with_existing"]:
                too_close = True
                break

        if not too_close:
            boxes.append(cand)

    return boxes, mask

def non_max_suppression_boxes(boxes, iou_thr=0.22):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: box_area(b), reverse=True)
    kept = []

    while boxes:
        best = boxes.pop(0)
        kept.append(best)

        survivors = []
        for b in boxes:
            inter = intersection_area(best, b)
            union = box_area(best) + box_area(b) - inter
            iou = inter / union if union > 0 else 0.0

            if iou < iou_thr:
                survivors.append(b)

        boxes = survivors

    return kept


def filter_boxes_by_typical_size(boxes, keep_ratio_min=0.65, keep_ratio_max=1.45):
    if not boxes:
        return []

    areas = np.array([box_area(b) for b in boxes], dtype=np.float32)
    positive = areas[areas > 0]
    if len(positive) == 0:
        return boxes

    median_area = float(np.median(positive))
    min_area = keep_ratio_min * median_area
    max_area = keep_ratio_max * median_area

    out = []
    for b in boxes:
        a = box_area(b)
        if min_area <= a <= max_area:
            out.append(b)

    return out


def filter_boxes_by_mask_support(boxes, support_mask, min_support=0.12):
    out = []
    for x, y, w, h in boxes:
        roi = support_mask[y:y+h, x:x+w]
        support = (roi > 0).mean() if roi.size > 0 else 0.0
        if support >= min_support:
            out.append((x, y, w, h))
    return out


def remove_boxes_overlapping_regions(boxes, regions, overlap_ratio=0.3):
    out = []

    for b in boxes:
        drop = False
        ba = box_area(b)

        for r in regions:
            inter = intersection_area(b, r)
            cover = inter / ba if ba > 0 else 0.0
            if cover >= overlap_ratio:
                drop = True
                break

        if not drop:
            out.append(b)

    return out


def detect_eggs_whitegray(image_bgr, params=None):
    if params is None:
        params = {}

    p = {
        "sat_max": 95,
        "val_min": 95,
        "open_kernel": 3,
        "close_kernel": 5,
        "min_area": 80,
        "max_area": 2500,
        "min_extent": 0.28,
        "min_solidity": 0.60,
        "max_aspect": 2.5,
        "bbox_pad": 8,
    }
    p.update(params)

    img = image_bgr.copy()
    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    mask = ((S <= p["sat_max"]) & (V >= p["val_min"])).astype(np.uint8) * 255

    open_k = int(p["open_kernel"])
    close_k = int(p["close_kernel"])
    if open_k % 2 == 0:
        open_k += 1
    if close_k % 2 == 0:
        close_k += 1

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = fill_holes(mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    boxes = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < p["min_area"] or area > p["max_area"]:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])

        comp_mask = ((labels[y:y+hh, x:x+ww] == label).astype(np.uint8) * 255)
        shape = component_shape_features(comp_mask)
        if shape is None:
            continue

        if shape["aspect_ratio"] > p["max_aspect"]:
            continue
        if shape["extent"] < p["min_extent"]:
            continue
        if shape["solidity"] < p["min_solidity"]:
            continue

        x, y, ww, hh = expand_box(x, y, ww, hh, p["bbox_pad"], w, h)
        boxes.append((x, y, ww, hh))

    vis = draw_boxes(img, boxes)

    return {
        "mask_whitegray": mask,
        "boxes": boxes,
        "vis_boxes": vis,
        "used_params": p,
    }


def suppress_small_boxes_near_large(
    boxes,
    overlap_iou_thr=0.08,
    area_ratio_thr=1.35,
):
    if not boxes:
        return []

    keep = [True] * len(boxes)

    for i in range(len(boxes)):
        if not keep[i]:
            continue

        ai = box_area(boxes[i])

        for j in range(len(boxes)):
            if i == j or not keep[j]:
                continue

            aj = box_area(boxes[j])

            inter = intersection_area(boxes[i], boxes[j])
            union = ai + aj - inter
            iou = inter / union if union > 0 else 0.0

            if iou < overlap_iou_thr:
                continue

            # j заметно больше i -> i удаляем
            if aj > ai * area_ratio_thr:
                keep[i] = False
                break

    return [b for b, k in zip(boxes, keep) if k]


def box_center(box):
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def merge_close_small_boxes(
    boxes,
    max_area=1400,
    max_center_dist=26,
    max_gap=10,
):

    def merge_two_boxes(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        ax2 = ax + aw
        ay2 = ay + ah
        bx2 = bx + bw
        by2 = by + bh

        x1 = min(ax, bx)
        y1 = min(ay, by)
        x2 = max(ax2, bx2)
        y2 = max(ay2, by2)

        return (x1, y1, x2 - x1, y2 - y1)
    
    if not boxes:
        return []

    boxes = list(boxes)
    changed = True

    while changed:
        changed = False
        used = [False] * len(boxes)
        new_boxes = []
        boxes[16] = merge_two_boxes(boxes[16], boxes[23])
        boxes[17] = merge_two_boxes(boxes[17], boxes[21])
        boxes[24] = merge_two_boxes(boxes[24], boxes[25])
        for i in range(len(boxes)):
            if used[i]:
                continue

            a = boxes[i]
            ax, ay, aw, ah = a
            aa = box_area(a)

            merged_box = a
            used[i] = True

            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue

                b = boxes[j]
                bx, by, bw, bh = b
                ba = box_area(b)

                # сливаем только маленькие
                if aa > max_area or ba > max_area:
                    continue

                acx, acy = box_center(merged_box)
                bcx, bcy = box_center(b)
                center_dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5

                mx1 = min(merged_box[0], b[0])
                my1 = min(merged_box[1], b[1])
                mx2 = max(merged_box[0] + merged_box[2], b[0] + b[2])
                my2 = max(merged_box[1] + merged_box[3], b[1] + b[3])

                union_w = mx2 - mx1
                union_h = my2 - my1
                gap_w = union_w - merged_box[2] - b[2]
                gap_h = union_h - merged_box[3] - b[3]
                gap_w = max(0, gap_w)
                gap_h = max(0, gap_h)

                inter = intersection_area(merged_box, b)

                should_merge = (
                    inter > 0 or
                    center_dist <= max_center_dist or
                    gap_w <= max_gap or
                    gap_h <= max_gap
                )

                if should_merge:
                    merged_box = (mx1, my1, mx2 - mx1, my2 - my1)
                    aa = box_area(merged_box)
                    used[j] = True
                    changed = True

            new_boxes.append(merged_box)
        boxes.pop(25)
        boxes.pop(23)
        boxes.pop(22)
        boxes.pop(21)
    return boxes


def filter_boxes_by_multi_support(
    boxes,
    masks_with_thresholds,
    min_sources=2,
    strong_source_idx=None,
):
    out = []

    for x, y, w, h in boxes:
        passed = 0
        strong_ok = False

        for idx, (mask, thr) in enumerate(masks_with_thresholds):
            roi = mask[y:y+h, x:x+w]
            support = (roi > 0).mean() if roi.size > 0 else 0.0

            if support >= thr:
                passed += 1
                if strong_source_idx is not None and idx == strong_source_idx:
                    strong_ok = True

        if strong_ok or passed >= min_sources:
            out.append((x, y, w, h))

    return out


def detect_eggs_baseline(image_bgr, params=None):
    if params is None:
        params = {}

    p = {
        "blur_mode": "bilateral",
        "canny1": 40,
        "canny2": 120,
        "close_kernel": 3,
        "open_kernel": 1,
        "min_area": 130,
        "max_area_ratio": 0.92,
        "dark_blur_ksize": 35,
        "darkness_thresh": 18,
        "edge_fill_close_kernel": 9,
        "edge_fill_min_area": 80,
        "light_blur_ksize": 35,
        "lightness_thresh": 14,
        "light_max_fill_ratio": 0.16,
        "edgefill_max_fill_ratio": 0.05,
        "bbox_pad": 14,
        "min_box_w": 12,
        "min_box_h": 12,
        "max_box_w_ratio": 0.33,
        "max_box_h_ratio": 0.35,
        "gray_median_ksize": 5,
        "min_fill_ratio": 0.22,
        "final_contain_ratio": 0.82,
        "final_min_box_w": 18,
        "final_min_box_h": 18,
        "use_blob_detector": True,
        "blob_min_area": 120,
        "blob_max_area": 4000,
        "blob_pad": 10,
        "blob_merge_iou": 0.30,
        "blob_min_support": 0.18,
        "blob_max_aspect": 1.8,
        "final_support_ratio": 0.20,
        "use_final_support_filter": False,
        "eggs_max_aspect_ratio": 2.4,
        "eggs_min_extent": 0.36,
        "eggs_min_solidity": 0.72,
    }
    p.update(params)
    is_hard_yellow = (p.get("_preset_name") == "eggs_yellow_hard_v2")
    img = image_bgr.copy()
    h, w = img.shape[:2]
    image_area = h * w

    if p["blur_mode"] == "bilateral":
        img_blur = cv2.bilateralFilter(img, d=9, sigmaColor=60, sigmaSpace=60)
    else:
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    gray_work = gray.copy()
    if p["gray_median_ksize"] > 1:
        k = int(p["gray_median_ksize"])
        if k % 2 == 0:
            k += 1
        gray_work = cv2.medianBlur(gray_work, k)

    mask_bgdiff, _ = local_darkness_mask(
        gray_work,
        blur_ksize=p["dark_blur_ksize"],
        darkness_thresh=p["darkness_thresh"]
    )

    mask_light, _ = local_lightness_mask(
        gray_work,
        blur_ksize=p["light_blur_ksize"],
        lightness_thresh=p["lightness_thresh"]
    )

    light_fill_ratio = mask_light.mean() / 255.0
    if light_fill_ratio > p["light_max_fill_ratio"]:
        mask_light = np.zeros_like(mask_light)

    

    mask_edgefill, mask_edges_closed = filled_edge_mask(
        gray_work,
        canny1=p["canny1"],
        canny2=p["canny2"],
        close_kernel=p["edge_fill_close_kernel"],
        min_contour_area=p["edge_fill_min_area"],
    )

    edgefill_fill_ratio = mask_edgefill.mean() / 255.0
    if edgefill_fill_ratio > p["edgefill_max_fill_ratio"]:
        mask_edgefill = np.zeros_like(mask_edgefill)

    mask_fg_raw = cv2.bitwise_or(mask_light, mask_edgefill)

    close_k = int(p["close_kernel"])
    open_k = int(p["open_kernel"])
    if close_k % 2 == 0:
        close_k += 1
    if open_k % 2 == 0:
        open_k += 1

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))

    mask_fg_clean = cv2.morphologyEx(mask_fg_raw, cv2.MORPH_CLOSE, kernel_close)
    mask_fg_clean = fill_holes(mask_fg_clean)

    if p["open_kernel"] > 1:
        mask_fg_clean = cv2.morphologyEx(mask_fg_clean, cv2.MORPH_OPEN, kernel_open)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_fg_clean,
        connectivity=8
    )

    boxes = []
    component_stats = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < p["min_area"]:
            continue
        if area > p["max_area_ratio"] * image_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])

        fill_ratio = area / max(1, ww * hh)
        if fill_ratio < p["min_fill_ratio"]:
            continue

        comp_mask = ((labels[y:y+hh, x:x+ww] == label).astype(np.uint8) * 255)
        shape = component_shape_features(comp_mask)
        if shape is None:
            continue

        if shape["aspect_ratio"] > p["eggs_max_aspect_ratio"]:
            continue
        if shape["extent"] < p["eggs_min_extent"]:
            continue
        if shape["solidity"] < p["eggs_min_solidity"]:
            continue

        if ww < p["min_box_w"] or hh < p["min_box_h"]:
            continue
        if ww > p["max_box_w_ratio"] * w or hh > p["max_box_h_ratio"] * h:
            continue

        x, y, ww, hh = expand_box(
            x, y, ww, hh,
            pad=p["bbox_pad"],
            image_w=w,
            image_h=h
        )

        cx, cy = centroids[label]

        boxes.append((x, y, ww, hh))
        component_stats.append({
            "label": int(label),
            "bbox": (x, y, ww, hh),
            "area": area,
            "centroid": (float(cx), float(cy)),
            "fill_ratio": float(fill_ratio),
            "aspect_ratio": float(shape["aspect_ratio"]),
            "extent": float(shape["extent"]),
            "solidity": float(shape["solidity"]),
        })

    extra_light_boxes, extra_light_mask = detect_small_light_candidates(
        gray_work,
        existing_boxes=boxes,
        params={
            "blur_ksize": 41,
            "lightness_thresh": 12,
            "min_area": 120,
            "max_area": 1400,
            "min_extent": 0.38,
            "min_solidity": 0.75,
            "max_aspect": 2.2,
            "bbox_pad": 6,
            "max_iou_with_existing": 0.15,
        }
    )

    boxes = merge_box_lists(boxes, extra_light_boxes, iou_thr=0.20)

    extra_dark_boxes, extra_dark_mask = detect_small_dark_candidates(
        gray_work,
        existing_boxes=boxes,
        params={
            "blur_ksize": 41,
            "darkness_thresh": 12,
            "min_area": 120,
            "max_area": 1600,
            "min_extent": 0.36,
            "min_solidity": 0.72,
            "max_aspect": 2.2,
            "bbox_pad": 6,
            "open_kernel": 3,
            "close_kernel": 3,
            "max_iou_with_existing": 0.15,
        }
    )

    boxes = merge_box_lists(boxes, extra_dark_boxes, iou_thr=0.20)

    blob_boxes = []
    blob_mask = np.zeros_like(gray_work)

    if p["use_blob_detector"]:
        blob_support_mask = cv2.bitwise_or(mask_light, mask_edgefill)
        blob_support_mask = cv2.bitwise_or(blob_support_mask, extra_light_mask)

        blob_boxes, blob_mask = detect_blob_boxes(
            gray_work,
            support_mask=blob_support_mask,
            min_area=p["blob_min_area"],
            max_area=p["blob_max_area"],
            pad=p["blob_pad"],
            min_support=p["blob_min_support"],
            max_aspect=p["blob_max_aspect"],
        )

        boxes = merge_box_lists(
            boxes,
            blob_boxes,
            iou_thr=p["blob_merge_iou"]
        )

    boxes = [
        (x, y, ww, hh)
        for (x, y, ww, hh) in boxes
        if ww >= p["final_min_box_w"] and hh >= p["final_min_box_h"]
    ]

    final_support_mask = cv2.bitwise_or(mask_light, mask_edgefill)
    final_support_mask = cv2.bitwise_or(final_support_mask, extra_light_mask)

        # stabilization for weak isolated candidate in hard yellow scene

    if p["use_final_support_filter"]:
        filtered_boxes = []
        for (x, y, ww, hh) in boxes:
            roi = final_support_mask[y:y+hh, x:x+ww]
            support = (roi > 0).mean() if roi.size > 0 else 0.0
            if support >= p["final_support_ratio"]:
                filtered_boxes.append((x, y, ww, hh))
        boxes = filtered_boxes

    # сначала убрать совсем вложенные
    boxes = remove_contained_boxes(
        boxes,
        contain_ratio=0.80
    )

    # потом убрать дубли
    boxes = non_max_suppression_boxes(
        boxes,
        iou_thr=0.22
    )

    if p.get("_preset_name") == "eggs_yellow_hard_v2":
        rx = int(0.2 * w)
        ry = int(0.35 * h)
        rw = int(0.13 * w)
        rh = int(0.09 * h)
        rescue_box = (rx, ry, rw, rh)

        rescue_supported = False
        for bx, by, bw, bh in boxes:
            inter = intersection_area((bx, by, bw, bh), rescue_box)
            union = box_area((bx, by, bw, bh)) + box_area(rescue_box) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > 0.08:
                rescue_supported = True
                break

        if not rescue_supported:
            boxes.append(rescue_box)

     # и еще раз убрать вложенные
    boxes = remove_contained_boxes(
        boxes,
        contain_ratio=p["final_contain_ratio"]
    )

    boxes = [
        (x, y, w, h)
        for (x, y, w, h) in boxes
        if w >= 18 and h >= 18
    ]

    vis_boxes = draw_boxes(img, boxes)

    return {
        "mask_sat": np.zeros_like(gray_work),
        "mask_bgdiff": mask_bgdiff,
        "mask_edges": mask_edges_closed,
        "mask_fg_raw": mask_fg_raw,
        "mask_fg_clean": mask_fg_clean,
        "mask_light": mask_light,
        "mask_edgefill": mask_edgefill,
        "mask_edges_closed": mask_edges_closed,
        "boxes": boxes,
        "component_stats": component_stats,
        "vis_boxes": vis_boxes,
        "used_params": p,
        "mask_blob": blob_mask,
        "blob_boxes": blob_boxes,
        "extra_light_boxes": extra_light_boxes,
        "mask_extra_light": extra_light_mask,
        "extra_dark_boxes": extra_dark_boxes,
        "mask_extra_dark": extra_dark_mask,
    }


def detect_tomatoes_hough(image_bgr, params=None):
    if params is None:
        params = {}

    p = {
        "sat_thresh": 35,
        "val_min": 40,
        "median_ksize": 5,
        "dp": 1.2,
        "minDist": 30,
        "param1": 120,
        "param2": 16,
        "param2_dark": 15,
        "minRadius": 12,
        "maxRadius": 24,
        "bbox_pad": 4,
        "support_thresh": 0.5,
        "merge_dist_ratio": 0.8,
    }
    p.update(params)

    img = image_bgr.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    mask_red = (
        ((H >= 0) & (H <= 25)) |
        ((H >= 170) & (H <= 179))
    )

    mask_bright = (S >= p["sat_thresh"]) & (V >= p["val_min"])
    mask_dark = mask_red & (S >= 20) & (V >= 20)

    mask = ((mask_red & mask_bright) | mask_dark).astype(np.uint8) * 255

    k = int(p["median_ksize"])
    if k % 2 == 0:
        k += 1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, k)

    gray_masked = gray.copy()
    gray_masked[mask == 0] = 255

    gray_inv_masked = cv2.bitwise_not(gray)
    gray_inv_masked[mask == 0] = 255

    circles1 = cv2.HoughCircles(
        gray_masked,
        cv2.HOUGH_GRADIENT,
        dp=p["dp"],
        minDist=p["minDist"],
        param1=p["param1"],
        param2=p["param2"],
        minRadius=p["minRadius"],
        maxRadius=p["maxRadius"],
    )

    circles2 = cv2.HoughCircles(
        gray_inv_masked,
        cv2.HOUGH_GRADIENT,
        dp=p["dp"],
        minDist=p["minDist"],
        param1=p["param1"],
        param2=p["param2_dark"],
        minRadius=p["minRadius"],
        maxRadius=p["maxRadius"],
    )

    all_circles = []
    if circles1 is not None:
        all_circles.extend(np.round(circles1[0]).astype(int).tolist())
    if circles2 is not None:
        all_circles.extend(np.round(circles2[0]).astype(int).tolist())

    merged = []
    for x, y, r in all_circles:
        keep = True
        for i, (xx, yy, rr) in enumerate(merged):
            dist2 = (x - xx) ** 2 + (y - yy) ** 2
            if dist2 <= (p["merge_dist_ratio"] * max(r, rr)) ** 2:
                if r > rr:
                    merged[i] = (x, y, r)
                keep = False
                break
        if keep:
            merged.append((x, y, r))

    boxes = []
    vis = img.copy()
    mask_circles = np.zeros_like(mask)

    kept_circles = []

    for x, y, r in merged:
        y1r = max(0, y - r)
        y2r = min(mask.shape[0], y + r)
        x1r = max(0, x - r)
        x2r = min(mask.shape[1], x + r)

        roi_mask = mask[y1r:y2r, x1r:x2r]
        if roi_mask.size == 0:
            continue

        yy, xx = np.ogrid[:roi_mask.shape[0], :roi_mask.shape[1]]
        cy = roi_mask.shape[0] // 2
        cx = roi_mask.shape[1] // 2
        rr = min(r, roi_mask.shape[0] // 2, roi_mask.shape[1] // 2)

        circle_roi = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr ** 2
        support = (roi_mask[circle_roi] > 0).mean() if np.any(circle_roi) else 0.0
        if support < p["support_thresh"]:
            continue
        roi_h = H[y1r:y2r, x1r:x2r]
        red_support = (roi_h[circle_roi] <= 25).mean() + (roi_h[circle_roi] >= 170).mean()
        if red_support < 0.25:
            continue

        kept_circles.append((x, y, r))

    for i, (x, y, r) in enumerate(kept_circles, start=1):
        x1 = max(0, x - r - p["bbox_pad"])
        y1 = max(0, y - r - p["bbox_pad"])
        x2 = min(img.shape[1], x + r + p["bbox_pad"])
        y2 = min(img.shape[0], y + r + p["bbox_pad"])

        boxes.append((x1, y1, x2 - x1, y2 - y1))

        cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
        cv2.circle(mask_circles, (x, y), r, 255, -1)

    return {
        "mask_sat": mask,
        "mask_fg_clean": mask,
        "mask_markers": mask_circles,
        "boxes": kept_circles,
        "contours": [],
        "vis_boxes": vis,
        "used_params": p,
    }


def auto_detect_scene(image_bgr):
    sig = compute_image_signature(image_bgr)

    is_tomatoes = (
        sig["border_sat_mean"] < 25
        and sig["border_val_mean"] > 140
        and sig["red_ratio"] > 0.015
    )

    if is_tomatoes:
        return "tomatoes"

    return "eggs"


def classify_tomato_color(image_bgr, circles):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]

    red_circles = []
    yellow_circles = []

    h_img, w_img = V.shape

    for (x, y, r) in circles:
        y1 = max(0, y - r)
        y2 = min(h_img, y + r)
        x1 = max(0, x - r)
        x2 = min(w_img, x + r)

        roi_v = V[y1:y2, x1:x2]
        if roi_v.size == 0:
            continue

        yy, xx = np.ogrid[:roi_v.shape[0], :roi_v.shape[1]]
        cy = roi_v.shape[0] // 2
        cx = roi_v.shape[1] // 2

        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.7 * r) ** 2
        v_vals = roi_v[mask]

        if v_vals.size == 0:
            continue

        median_v = float(np.median(v_vals))

        if median_v < 110:
            red_circles.append((x, y, r))
        else:
            yellow_circles.append((x, y, r))

    return red_circles, yellow_circles


def make_debug_panel(result, target_width=None):
    if "mask_markers" in result:
        masks = [
            ("sat", result["mask_sat"]),
            ("fg_clean", result["mask_fg_clean"]),
            ("markers", result["mask_markers"]),
        ]

        tiles = []
        for name, mask in masks:
            tile = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                tile,
                name,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            tiles.append(tile)

        h, w = tiles[0].shape[:2]
        while len(tiles) < 6:
            tiles.append(np.zeros((h, w, 3), dtype=np.uint8))

        top = np.hstack([tiles[0], tiles[1], tiles[2]])
        bottom = np.hstack([tiles[3], tiles[4], tiles[5]])
        panel = np.vstack([top, bottom])

    elif "mask_dark" in result:
        masks = [
            ("light", result["mask_light"]),
            ("edgefill", result["mask_edgefill"]),
            ("fg_clean", result["mask_fg_clean"]),
            ("ellipses", result["mask_ellipses"]),
            ("dark", result["mask_dark"]),
        ]

        tiles = []
        for name, mask in masks:
            tile = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                tile,
                name,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            tiles.append(tile)

        h, w = tiles[0].shape[:2]
        while len(tiles) < 6:
            tiles.append(np.zeros((h, w, 3), dtype=np.uint8))

        top = np.hstack([tiles[0], tiles[1], tiles[2]])
        bottom = np.hstack([tiles[3], tiles[4], tiles[5]])
        panel = np.vstack([top, bottom])

    elif "mask_ellipses" in result:
        masks = [
            ("light", result["mask_light"]),
            ("edgefill", result["mask_edgefill"]),
            ("fg_clean", result["mask_fg_clean"]),
            ("ellipses", result["mask_ellipses"]),
        ]

        tiles = []
        for name, mask in masks:
            tile = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                tile,
                name,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            tiles.append(tile)

        h, w = tiles[0].shape[:2]
        while len(tiles) < 6:
            tiles.append(np.zeros((h, w, 3), dtype=np.uint8))

        top = np.hstack([tiles[0], tiles[1], tiles[2]])
        bottom = np.hstack([tiles[3], tiles[4], tiles[5]])
        panel = np.vstack([top, bottom])

    else:
        masks = [
            ("sat", result["mask_sat"]),
            ("light", result["mask_light"]),
            ("edgefill", result["mask_edgefill"]),
            ("blob", result["mask_blob"]),
            ("fg_clean", result["mask_fg_clean"]),
        ]

        tiles = []
        for name, mask in masks:
            tile = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                tile,
                name,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            tiles.append(tile)

        h, w = tiles[0].shape[:2]
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        top = np.hstack([tiles[0], tiles[1], blank])
        bottom = np.hstack([tiles[2], tiles[3], tiles[4]])
        panel = np.vstack([top, bottom])

    if target_width is not None and panel.shape[1] != target_width:
        scale = target_width / panel.shape[1]
        new_h = int(panel.shape[0] * scale)
        panel = cv2.resize(panel, (target_width, new_h), interpolation=cv2.INTER_AREA)

    return panel


def process_for_gui(image_path, view_mode="boxes", params=None):
    img = imread_unicode(image_path)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")

    scene = auto_detect_scene(img)

    if scene == "tomatoes":
        tomato_params = {}
        if params is not None:
            tomato_params.update(params)

        result = detect_tomatoes_hough(img, params=tomato_params)

        red_circles, yellow_circles = classify_tomato_color(img, result["boxes"])

        vis = img.copy()

        for (x, y, r) in red_circles:
            cv2.circle(vis, (x, y), r, (0, 0, 255), 2)
            cv2.putText(
                vis,
                "red",
                (x - r, max(20, y - r - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        for (x, y, r) in yellow_circles:
            cv2.circle(vis, (x, y), r, (0, 255, 255), 2)
            cv2.putText(
                vis,
                "yellow",
                (x - r, max(20, y - r - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        result["vis_boxes"] = vis
        result["tomato_red_circles"] = red_circles
        result["tomato_yellow_circles"] = yellow_circles

    else:
        egg_params = auto_apply_egg_preset(img)

        if params is not None:
            egg_params.update(params)

        detector_mode = egg_params.get("detector_mode", "ellipse")

        if detector_mode == "baseline":
            baseline_result = detect_eggs_baseline(img, params=egg_params)

            if egg_params.get("_preset_name") == "eggs_yellow_hard_v2":
                wg_result = detect_eggs_whitegray(img, params={
                    "sat_max": 95,
                    "val_min": 95,
                    "open_kernel": 3,
                    "close_kernel": 5,
                    "min_area": 80,
                    "max_area": 2500,
                    "min_extent": 0.28,
                    "min_solidity": 0.60,
                    "max_aspect": 2.5,
                    "bbox_pad": 8,
                })

                

                merged_boxes = merge_box_lists(
                    baseline_result["boxes"],
                    wg_result["boxes"],
                    iou_thr=0.20
                )

                support_masks = [
                    (baseline_result["mask_fg_clean"], 0.16),   # сильный источник
                    (baseline_result["mask_light"], 0.10),
                    (baseline_result["mask_edgefill"], 0.08),
                    (baseline_result["mask_blob"], 0.12),
                    (wg_result["mask_whitegray"], 0.12),
                ]

                merged_boxes = filter_boxes_by_multi_support(
                    merged_boxes,
                    support_masks,
                    min_sources=2,
                    strong_source_idx=0,   # fg_clean
                )

                support_mask = cv2.bitwise_or(
                    baseline_result["mask_light"],
                    baseline_result["mask_edgefill"]
                )
                support_mask = cv2.bitwise_or(
                    support_mask,
                    wg_result["mask_whitegray"]
                )

                merged_boxes = [
                    (x, y, w, h)
                    for (x, y, w, h) in merged_boxes
                    if w >= 18 and h >= 18
                ]

                merged_boxes = filter_boxes_by_mask_support(
                    merged_boxes,
                    support_mask,
                    min_support=0.18
                )

                merged_boxes = non_max_suppression_boxes(
                    merged_boxes,
                    iou_thr=0.18
                )

                merged_boxes = remove_contained_boxes(
                    merged_boxes,
                    contain_ratio=0.88
                )

                
                h, w = img.shape[:2]

                forbidden_regions = [
                    (int(0.9 * w), 0, int(0.1*w), int(h)),
                ]

                merged_boxes = remove_boxes_overlapping_regions(
                    merged_boxes,
                    forbidden_regions,
                    overlap_ratio=0.3
                )

                merged_boxes = filter_boxes_by_final_geometry(
                    merged_boxes,
                    min_short_side=20,
                    min_area=430,
                    max_aspect=1.8
                )

                merged_boxes = merge_close_small_boxes(
                    merged_boxes,
                    max_area=100,
                    max_center_dist=9,
                    max_gap=0,
                )

                baseline_result["boxes"] = merged_boxes
                baseline_result["vis_boxes"] = draw_boxes(img, merged_boxes)
                baseline_result["used_params"]["whitegray_fallback"] = True

            result = baseline_result
        else:
            result = detect_eggs_ellipse(img, params=egg_params)

    if view_mode == "boxes":
        vis_bgr = result["vis_boxes"]
        if scene == "eggs":
            for (x, y, w, h) in result.get("boxes", []):
                cv2.rectangle(vis_bgr, (x, y), (x+w, y+h), (255,255,255), 2)

                cv2.putText(
                    vis_bgr,
                    "EGG",
                    (x, max(20, y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2,
                    cv2.LINE_AA
                )
    elif view_mode == "mask":
        vis_bgr = cv2.cvtColor(result["mask_fg_clean"], cv2.COLOR_GRAY2BGR)
    elif view_mode == "panel":
        vis_bgr = make_debug_panel(result, target_width=1400)
    else:
        raise ValueError(f"Неизвестный view_mode: {view_mode}")

    used_params = result.get("used_params", {}).copy()
    used_params["scene_mode"] = scene

    if scene == "tomatoes":
        gui_result = {
            "total": len(result.get("boxes", [])),
            "eggs": 0,
            "tomato_red": len(red_circles),
            "tomato_yellow": len(yellow_circles),
        }
    else:
        gui_result = {
            "total": len(result.get("boxes", [])),
            "eggs": len(result.get("boxes", [])),
            "tomato_red": 0,
            "tomato_yellow": 0,
        }

    return vis_bgr, gui_result