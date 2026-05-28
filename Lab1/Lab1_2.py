import cv2
import numpy as np
from ultralytics import FastSAM


import cv2
import numpy as np


def preprocess_image(image, target_size=1024):
    TARGET_SIZE = 1024
    BLUR_KERNEL = 5
    GAMMA = 0.6
    BRIGHTNESS_BETA = 10

    h, w = image.shape[:2]
    max_dim = max(h, w)

    if max_dim > TARGET_SIZE:
        scale = TARGET_SIZE / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        img = image.copy()

    img = cv2.GaussianBlur(img, (BLUR_KERNEL, BLUR_KERNEL), 0)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    v_float = v_ch.astype(np.float32) / 255.0
    v_corr = np.power(v_float, GAMMA)
    v_corr = np.clip(v_corr * 255.0, 0, 255).astype(np.uint8)

    hsv_corr = cv2.merge([h_ch, s_ch, v_corr])
    img_corr = cv2.cvtColor(hsv_corr, cv2.COLOR_HSV2BGR)

    img_out = cv2.convertScaleAbs(img_corr, beta=BRIGHTNESS_BETA)

    return img_out


def apply_fastsam(image):
    MODEL_PATH = "FastSAM-x.pt"
    DEVICE = "cpu"
    IMGSZ = 1024
    CONF = 0.45
    IOU = 0.3

    model = FastSAM(MODEL_PATH)

    results = model(
        image,
        device=DEVICE,
        retina_masks=True,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        verbose=False,
    )

    return results[0]


def filter_masks(masks):
    MIN_AREA = 350
    MIN_REL_AREA = 0.5
    MAX_REL_AREA = 1.8

    MIN_INTERSECTION = 50
    CONTAINMENT_THR = 0.9
    STRONG_OVERLAP_THR = 0.5
    BIG_RATIO_THR = 1.3
    SMALL_INSIDE_BIG_THR = 0.4

    binary_masks = [(mask > 0.5).astype(np.uint8) for mask in masks]
    areas = [int(m.sum()) for m in binary_masks]

    keep = [i for i, area in enumerate(areas) if area >= MIN_AREA]
    binary_masks = [binary_masks[i] for i in keep]
    areas = [areas[i] for i in keep]

    if not binary_masks:
        return []

    if len(binary_masks) > 3:
        median_area = np.median(areas)
        keep = [
            i for i, area in enumerate(areas)
            if median_area * MIN_REL_AREA <= area <= median_area * MAX_REL_AREA
        ]
        binary_masks = [binary_masks[i] for i in keep]
        areas = [areas[i] for i in keep]

    if not binary_masks:
        return []

    to_remove = set()
    n = len(binary_masks)

    for i in range(n):
        if i in to_remove:
            continue

        for j in range(i + 1, n):
            if j in to_remove:
                continue

            inter = int(np.logical_and(binary_masks[i], binary_masks[j]).sum())
            if inter < MIN_INTERSECTION:
                continue

            min_area = min(areas[i], areas[j])
            max_area = max(areas[i], areas[j])

            if min_area == 0:
                continue

            overlap_ratio = inter / min_area
            area_ratio = min_area / max_area

            if overlap_ratio > CONTAINMENT_THR:
                if area_ratio < SMALL_INSIDE_BIG_THR:
                    to_remove.add(i if areas[i] < areas[j] else j)
                else:
                    to_remove.add(i if areas[i] > areas[j] else j)

            elif overlap_ratio > STRONG_OVERLAP_THR:
                if areas[i] > areas[j] * BIG_RATIO_THR:
                    to_remove.add(i)
                elif areas[j] > areas[i] * BIG_RATIO_THR:
                    to_remove.add(j)

    binary_masks = [m for k, m in enumerate(binary_masks) if k not in to_remove]
    return binary_masks


def erode_region(region, ksize=7):
    mask = region.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    eroded = cv2.erode(mask, kernel, iterations=1) > 0

    if eroded.sum() < 30:
        return region

    return eroded


def make_outer_ring(region, outer_ksize=21):
    mask = region.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_ksize, outer_ksize))
    outer = cv2.dilate(mask, kernel, iterations=1) > 0
    ring = outer & (~region)
    return ring


def classify_object_type(h, s, v, bg_s, bg_v):
    LOW_S_THRESHOLD = 40
    BRIGHT_Q = 0.95
    MIN_GOOD_PIXELS = 20

    WHITE_S_MAX = 75
    WHITE_V_MIN = 120
    WHITE_FRAC_MIN = 0.04

    NEUTRAL_S_MAX = 80
    NEUTRAL_FRAC_MIN = 0.4

    EGG_DELTA_S = 25
    EGG_DELTA_V = 18

    RED_H_MAX = 10
    RED_H_MIN2 = 170
    RED_FRAC_MAX_FOR_BG_EGG = 0.12

    if len(h) == 0:
        return "unknown", (255, 0, 255)

    v_thr = np.quantile(v, BRIGHT_Q)
    good = v <= v_thr
    if good.sum() > MIN_GOOD_PIXELS:
        h = h[good]
        s = s[good]
        v = v[good]

    white_mask = (s <= WHITE_S_MAX) & (v >= WHITE_V_MIN)
    white_frac = float(white_mask.mean())

    neutral_mask = s <= NEUTRAL_S_MAX
    neutral_frac = float(neutral_mask.mean())

    colored = s >= LOW_S_THRESHOLD
    if colored.sum() > MIN_GOOD_PIXELS:
        hc = h[colored]
    else:
        hc = h

    red_frac = float(((hc <= RED_H_MAX) | (hc >= RED_H_MIN2)).mean()) if len(hc) else 0.0

    s_med = float(np.median(s))
    v_med = float(np.median(v))

    bg_s_med = float(np.median(bg_s)) if len(bg_s) else s_med
    bg_v_med = float(np.median(bg_v)) if len(bg_v) else v_med

    bg_egg = (
        s_med <= bg_s_med - EGG_DELTA_S and
        v_med <= bg_v_med - EGG_DELTA_V and
        red_frac <= RED_FRAC_MAX_FOR_BG_EGG
    )

    if white_frac >= WHITE_FRAC_MIN or neutral_frac >= NEUTRAL_FRAC_MIN:
        return "egg", (255, 255, 255)

    if bg_egg:
        return "egg", (255, 255, 255)

    return "tomato", (0, 0, 255)


def get_tomato_features(inner_region, hsv, lab):
    s = hsv[..., 1][inner_region]
    v = hsv[..., 2][inner_region]
    b = lab[..., 2][inner_region]

    if len(v) == 0:
        return None

    # убираем совсем блеклые пиксели, чтобы блики меньше мешали
    colored = s >= 40
    if colored.sum() >= 20:
        v = v[colored]
        b = b[colored]

    return {
        "v_med": float(np.median(v)),
        "b_med": float(np.median(b)),
    }


def assign_tomato_labels(tomato_infos):
    if not tomato_infos:
        return tomato_infos

    # если томат всего один, относительное разделение невозможно
    if len(tomato_infos) == 1:
        t = tomato_infos[0]
        score = t["v_med"] + 0.5 * t["b_med"]
        if score >= 190:
            t["label"] = "tomato_yellow"
            t["color"] = (0, 255, 255)
        else:
            t["label"] = "tomato_red"
            t["color"] = (0, 0, 255)
        return tomato_infos

    v_all = np.array([t["v_med"] for t in tomato_infos], dtype=np.float32)
    b_all = np.array([t["b_med"] for t in tomato_infos], dtype=np.float32)

    v_q25, v_q75 = np.quantile(v_all, [0.25, 0.75])
    b_q25, b_q75 = np.quantile(b_all, [0.25, 0.75])

    thr_v = 0.5 * (v_q25 + v_q75)
    thr_b = 0.5 * (b_q25 + b_q75)

    v_spread = float(v_q75 - v_q25)

    # "серая зона" около порога по яркости:
    # если объект рядом с порогом, решаем по b_med
    margin = max(4.0, 0.15 * v_spread)

    for t in tomato_infos:
        if t["v_med"] <= thr_v - margin:
            label = "tomato_red"
        elif t["v_med"] >= thr_v + margin:
            label = "tomato_yellow"
        else:
            label = "tomato_yellow" if t["b_med"] >= thr_b+4 else "tomato_red"

        t["label"] = label
        t["color"] = (0, 255, 255) if label == "tomato_yellow" else (0, 0, 255)

    return tomato_infos


def process_for_gui(image_path, view_mode="mask", params=None):
    ALPHA = 0.5

    data = np.fromfile(image_path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    prep = preprocess_image(bgr)

    result = apply_fastsam(prep)

    counts = {
        "eggs": 0,
        "tomato_yellow": 0,
        "tomato_red": 0,
        "total": 0,
    }

    if result.masks is None or result.masks.data is None:
        return bgr, counts

    masks = result.masks.data.cpu().numpy()
    masks = filter_masks(masks)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    overlay = bgr.copy()
    egg_regions = []
    tomato_infos = []

    for mask in masks:
        region = mask > 0.5

        inner_region = erode_region(region, ksize=7)
        ring_region = make_outer_ring(region, outer_ksize=21)

        h = hsv[..., 0][inner_region]
        s = hsv[..., 1][inner_region]
        v = hsv[..., 2][inner_region]

        bg_s = hsv[..., 1][ring_region]
        bg_v = hsv[..., 2][ring_region]

        base_label, _ = classify_object_type(h, s, v, bg_s, bg_v)

        if base_label == "egg":
            egg_regions.append(region)
        else:
            feats = get_tomato_features(inner_region, hsv, lab)
            if feats is None:
                continue

            tomato_infos.append({
                "region": region,
                "v_med": feats["v_med"],
                "b_med": feats["b_med"],
            })

    tomato_infos = assign_tomato_labels(tomato_infos)

    for region in egg_regions:
        overlay[region] = (255, 255, 255)
        counts["eggs"] += 1
        counts["total"] += 1

    for t in tomato_infos:
        overlay[t["region"]] = t["color"]
        if t["label"] == "tomato_red":
            counts["tomato_red"] += 1
        else:
            counts["tomato_yellow"] += 1
        counts["total"] += 1

    mask_view = cv2.addWeighted(overlay, ALPHA, bgr, 1.0 - ALPHA, 0.0)

    if view_mode == "panel":
        vis = np.hstack([bgr, mask_view])
    else:
        vis = mask_view

    return vis, counts