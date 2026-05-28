from __future__ import annotations

import re

import cv2
import numpy as np


HEX_TEMPLATE_AREA_RATIO = 0.0

PATH_SEGMENTATION_CONFIGS = {
    "path_4": [
        {"blur": 7, "close": 7, "mode": "kmeans_inner", "alpha": 0.30, "seed": 0, "component_close": 5},
        {"blur": 7, "close": 7, "mode": "kmeans_all", "seed": 0, "component_close": 5},
    ],
    "path_7_1": [
        {"blur": 7, "close": 7, "mode": "kmeans_all", "seed": 0, "component_close": 5},
        {"blur": 7, "close": 7, "mode": "kmeans_all", "seed": 1, "component_close": 5},
        {"blur": 7, "close": 7, "mode": "kmeans_inner", "alpha": 0.30, "seed": 0, "component_close": 5},
    ],
    "path_10_1": [
        {"blur": 5, "close": 7, "mode": "kmeans_all", "seed": 5, "component_close": 7},
        {"blur": 5, "close": 5, "mode": "kmeans_all", "seed": 1, "component_close": 7},
        {"blur": 9, "close": 7, "mode": "kmeans_all", "seed": 7, "component_close": 7},
        {"blur": 9, "close": 7, "mode": "kmeans_inner", "alpha": 0.30, "seed": 2, "component_close": 7},
        {"blur": 7, "close": 7, "mode": "kmeans_all", "seed": 0, "component_close": 5},
    ],
}


def segment_tiles(
    image_bgr: np.ndarray,
    min_area: int = 1500,
    path_hint: str | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    expected_path_count = parse_path_tile_count(path_hint)
    if expected_path_count is not None:
        return segment_path_tiles(
            image_bgr,
            expected_count=expected_path_count,
            min_area=min_area,
        )

    raw_mask, cleaned_mask = build_tile_masks(image_bgr)
    component_masks = extract_component_masks(cleaned_mask, min_area=min_area)
    instance_masks = [regularize_component_to_hex(mask) for mask in component_masks]

    tiles_mask = np.zeros(raw_mask.shape, dtype=np.uint8)
    for mask in instance_masks:
        tiles_mask = np.maximum(tiles_mask, mask)

    return tiles_mask, instance_masks


def parse_path_tile_count(path_hint: str | None) -> int | None:
    if not path_hint:
        return None

    match = re.search(r"path[_ ]?(\d+)", path_hint, flags=re.IGNORECASE)
    if not match:
        return None

    expected = int(match.group(1))
    if expected <= 0:
        return None
    return expected


def segment_path_tiles(
    image_bgr: np.ndarray,
    expected_count: int,
    min_area: int = 1500,
) -> tuple[np.ndarray, list[np.ndarray]]:
    cluster_mask = build_path_cluster_mask(image_bgr)
    if cv2.countNonZero(cluster_mask) == 0:
        return segment_tiles(image_bgr, min_area=min_area, path_hint=None)

    instance_masks = split_path_cluster_into_tiles(
        cluster_mask,
        expected_count=expected_count,
        min_area=min_area,
    )
    if len(instance_masks) != expected_count:
        return segment_tiles(image_bgr, min_area=min_area, path_hint=None)

    return render_instance_union_mask(instance_masks), instance_masks


def build_path_segmentation_candidates(
    image_bgr: np.ndarray,
    path_hint: str,
    min_area: int = 1500,
) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    path_key = path_hint.lower()
    expected_count = parse_path_tile_count(path_key)
    if expected_count is None:
        return []

    configs = PATH_SEGMENTATION_CONFIGS.get(path_key)
    if not configs:
        default_mask, default_instances = segment_path_tiles(
            image_bgr,
            expected_count=expected_count,
            min_area=min_area,
        )
        return [(default_mask, default_instances)]

    candidates = []
    for config in configs:
        cluster_mask = build_path_cluster_mask(
            image_bgr,
            blur_k=config["blur"],
            close_k=config["close"],
        )
        instance_masks = split_path_cluster_into_tiles(
            cluster_mask,
            expected_count=expected_count,
            min_area=min_area,
            mode=config["mode"],
            seed=config.get("seed", 0),
            alpha=config.get("alpha", 0.35),
            component_close=config.get("component_close", 5),
        )
        if len(instance_masks) == expected_count:
            candidates.append((render_instance_union_mask(instance_masks), instance_masks))

    if not candidates:
        default_mask, default_instances = segment_path_tiles(
            image_bgr,
            expected_count=expected_count,
            min_area=min_area,
        )
        return [(default_mask, default_instances)]

    return candidates


def build_path_cluster_mask(
    image_bgr: np.ndarray,
    blur_k: int = 7,
    close_k: int = 7,
) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    if close_k > 0:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
        )
    mask = keep_largest_component(mask)
    mask = fill_small_holes(mask, max_hole_area=2200)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return mask


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8) * 255


def fill_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    filled = fill_holes(mask)
    holes = cv2.bitwise_and(filled, cv2.bitwise_not(mask))
    if cv2.countNonZero(holes) == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes, 8)
    out = mask.copy()
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= max_hole_area:
            out[labels == label] = 255
    return out


def split_path_cluster_into_tiles(
    cluster_mask: np.ndarray,
    expected_count: int,
    min_area: int,
    mode: str = "kmeans_all",
    seed: int = 0,
    alpha: float = 0.35,
    component_close: int = 5,
) -> list[np.ndarray]:
    ys, xs = np.where(cluster_mask > 0)
    if len(xs) < expected_count:
        return []

    points = np.column_stack([xs, ys]).astype(np.float32)
    centers = estimate_path_tile_centers(
        cluster_mask,
        expected_count=expected_count,
        mode=mode,
        seed=seed,
        alpha=alpha,
    )
    if centers is None or len(centers) != expected_count:
        return []

    distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assignment = distances.argmin(axis=1)
    instance_masks = []

    for label in range(expected_count):
        component = np.zeros_like(cluster_mask)
        selected = assignment == label
        component[ys[selected], xs[selected]] = 255
        component = cv2.morphologyEx(
            component,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (component_close, component_close)),
        )
        component = fill_holes(component)
        component = regularize_component_to_hex(component)
        if cv2.countNonZero(component) < min_area:
            continue
        instance_masks.append(component)

    return instance_masks


def render_instance_union_mask(instance_masks: list[np.ndarray]) -> np.ndarray:
    if not instance_masks:
        raise ValueError("instance_masks is empty")

    union = np.zeros_like(instance_masks[0])
    for instance_mask in instance_masks:
        solid_mask = fill_holes(instance_mask)
        union = np.maximum(union, solid_mask)

    union = cv2.morphologyEx(
        union,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    return union


def estimate_path_tile_centers(
    cluster_mask: np.ndarray,
    expected_count: int,
    mode: str,
    seed: int,
    alpha: float,
) -> np.ndarray | None:
    ys, xs = np.where(cluster_mask > 0)
    points = np.column_stack([xs, ys]).astype(np.float32)
    if len(points) < expected_count:
        return None

    rng = np.random.default_rng(seed)
    distance = cv2.distanceTransform(cluster_mask, cv2.DIST_L2, 5)
    tile_area = cv2.countNonZero(cluster_mask) / float(expected_count)
    side = (2.0 * tile_area / (3.0 * np.sqrt(3.0))) ** 0.5

    if mode == "kmeans_all":
        sample = points
        if len(sample) > 18000:
            sample = sample[rng.choice(len(sample), 18000, replace=False)]
    elif mode == "kmeans_inner":
        inner_y, inner_x = np.where(distance > alpha * side)
        sample = np.column_stack([inner_x, inner_y]).astype(np.float32)
        if len(sample) < expected_count:
            return None
        if len(sample) > 18000:
            sample = sample[rng.choice(len(sample), 18000, replace=False)]
    else:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.5)
    _, _, centers = cv2.kmeans(
        sample,
        expected_count,
        None,
        criteria,
        8,
        cv2.KMEANS_PP_CENTERS,
    )
    return centers


def build_tile_masks(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    bg_lab = get_background_lab_from_border(lab)

    raw_mask = (lab[:, :, 0] < bg_lab[0] - 38).astype(np.uint8) * 255
    raw_mask = cv2.morphologyEx(
        raw_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    cleaned_mask = remove_border_components(raw_mask)
    cleaned_mask = fill_holes_per_component(cleaned_mask)
    cleaned_mask = cv2.morphologyEx(
        cleaned_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    return raw_mask, cleaned_mask


def get_background_lab_from_border(lab: np.ndarray, border_frac: float = 0.08):
    height, width = lab.shape[:2]
    border_h = max(1, int(height * border_frac))
    border_w = max(1, int(width * border_frac))

    border_pixels = np.concatenate(
        [
            lab[:border_h].reshape(-1, 3),
            lab[height - border_h:].reshape(-1, 3),
            lab[:, :border_w].reshape(-1, 3),
            lab[:, width - border_w:].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(border_pixels, axis=0)


def remove_border_components(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    height, width = mask.shape
    clean = np.zeros_like(mask)

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            continue

        clean[labels == label] = 255

    return clean


def fill_holes(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(flood))


def fill_holes_per_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask, 8)
    filled = np.zeros_like(mask)

    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8) * 255
        filled = np.maximum(filled, fill_holes(component))

    return filled


def extract_component_masks(
    mask: np.ndarray,
    min_area: int = 1500,
) -> list[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    instance_masks = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        instance_masks.append((labels == label).astype(np.uint8) * 255)

    return instance_masks


def build_hex_template(size: int) -> np.ndarray:
    template = np.zeros((size, size), dtype=np.uint8)
    points = build_regular_hex_points(
        np.array([size / 2.0, size / 2.0], dtype=np.float32),
        0.42 * size,
        start_angle_deg=-30.0,
    )
    cv2.fillConvexPoly(template, np.round(points).astype(np.int32), 255)
    return template


def build_regular_hex_points(
    center: np.ndarray,
    radius: float,
    start_angle_deg: float,
) -> np.ndarray:
    points = []
    for idx in range(6):
        angle = np.deg2rad(start_angle_deg + 60.0 * idx)
        points.append(
            [
                center[0] + radius * np.cos(angle),
                center[1] + radius * np.sin(angle),
            ]
        )
    return np.array(points, dtype=np.float32)


def regularize_component_to_hex(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    area = cv2.contourArea(hull)
    if area <= 0:
        return mask

    perimeter = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.03 * perimeter, True).reshape(-1, 2).astype(np.float32)

    if len(approx) == 6:
        center = polygon_centroid(approx)
        approx, angles = order_polygon_vertices(approx, center)
        radii = np.linalg.norm(approx - center, axis=1)
        base_angles = [angles[idx] - idx * (np.pi / 3.0) for idx in range(6)]
        start_angle = np.rad2deg(np.angle(np.exp(1j * np.array(base_angles)).mean()))
        radius = float(radii.mean())
    else:
        moments = cv2.moments(hull)
        if moments["m00"] > 1e-6:
            center = np.array(
                [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
                dtype=np.float32,
            )
        else:
            center = hull.reshape(-1, 2).mean(axis=0)

        rect = cv2.minAreaRect(hull)
        angle = rect[2]
        if rect[1][0] < rect[1][1]:
            angle += 90.0
        start_angle = angle
        radius = 0.42 * area_to_template_size(area)

    return render_regular_hex_mask(mask.shape, center, radius_to_template_size(radius), start_angle)


def polygon_centroid(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    area2 = cross.sum()

    if abs(area2) < 1e-6:
        return points.mean(axis=0)

    cx = ((x + x_next) * cross).sum() / (3.0 * area2)
    cy = ((y + y_next) * cross).sum() / (3.0 * area2)
    return np.array([cx, cy], dtype=np.float32)


def order_polygon_vertices(points: np.ndarray, center: np.ndarray):
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles)
    return points[order], angles[order]


def area_to_template_size(area: float) -> float:
    ensure_hex_area_ratio()
    return np.sqrt(area / HEX_TEMPLATE_AREA_RATIO)


def radius_to_template_size(radius: float) -> int:
    return max(1, int(round(radius / 0.42)))


def ensure_hex_area_ratio():
    global HEX_TEMPLATE_AREA_RATIO
    if HEX_TEMPLATE_AREA_RATIO > 0:
        return
    ref_size = 200
    HEX_TEMPLATE_AREA_RATIO = cv2.countNonZero(build_hex_template(ref_size)) / float(ref_size * ref_size)


def render_regular_hex_mask(
    shape: tuple[int, int],
    center: np.ndarray,
    template_size: int,
    start_angle_deg: float = -30.0,
) -> np.ndarray:
    height, width = shape
    out = np.zeros((height, width), dtype=np.uint8)
    points = build_regular_hex_points(
        np.array(center, dtype=np.float32),
        0.42 * template_size,
        start_angle_deg=start_angle_deg,
    )
    cv2.fillConvexPoly(out, np.round(points).astype(np.int32), 255)
    return out


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return np.stack([mask, mask, mask], axis=-1)
