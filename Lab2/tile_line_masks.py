from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TileLineMasks:
    blue_mask: np.ndarray
    red_mask: np.ndarray
    yellow_mask: np.ndarray


def extract_tile_line_masks(tile_rgb: np.ndarray, tile_mask: np.ndarray) -> TileLineMasks:
    hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2LAB)
    rgb = tile_rgb.astype(np.int16)
    inside_tile = tile_mask > 0
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    blue_dom = blue - np.maximum(red, green)
    red_dom = red - np.maximum(green, blue)
    yellow_dom = np.minimum(red, green) - blue

    blue_seed = (
        (hsv[:, :, 0] >= 90)
        & (hsv[:, :, 0] <= 150)
        & (hsv[:, :, 2] >= 18)
        & (blue_dom >= 1)
        & (blue >= red)
        & (blue >= green)
        & inside_tile
    )

    yellow_mask = (
        (hsv[:, :, 0] >= 10)
        & (hsv[:, :, 0] <= 26)
        & (hsv[:, :, 1] >= 110)
        & (hsv[:, :, 2] >= 40)
        & (yellow_dom >= 28)
        & (lab[:, :, 2] >= 165)
        & inside_tile
    )

    red_mask = (
        (
            ((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 168))
            & (hsv[:, :, 1] >= 110)
            & (hsv[:, :, 2] >= 35)
            & (red_dom >= 40)
        )
        & inside_tile
    )

    overlap_red_yellow = red_mask & yellow_mask
    if np.any(overlap_red_yellow):
        prefer_yellow = overlap_red_yellow & (
            (green * 100 >= red * 58)
            | (yellow_dom > red_dom + 6)
        )
        prefer_red = overlap_red_yellow & ~prefer_yellow

        red_mask = (red_mask & ~overlap_red_yellow) | prefer_red
        yellow_mask = (yellow_mask & ~overlap_red_yellow) | prefer_yellow

    blue_grow = (
        (hsv[:, :, 2] <= 125)
        & (lab[:, :, 2] <= 134)
        & (blue >= red - 8)
        & (blue >= green - 8)
        & inside_tile
        & ~red_mask
        & ~yellow_mask
    )
    blue_mask = reconstruct_color_mask(
        blue_seed.astype(np.uint8) * 255,
        blue_grow.astype(np.uint8) * 255,
    )
    blue_support = np.maximum(
        blue_grow.astype(np.uint8) * 255,
        blue_seed.astype(np.uint8) * 255,
    )
    blue_support = np.maximum(
        blue_support,
        build_blue_cool_fallback_mask(
            hsv=hsv,
            lab=lab,
            inside_tile=inside_tile,
            red_mask=red_mask,
            yellow_mask=yellow_mask,
        ),
    )
    blue_mask = np.maximum(
        blue_mask,
        blue_support,
    )

    return TileLineMasks(
        blue_mask=cleanup_blue_mask(blue_mask, blue_support, tile_mask),
        red_mask=cleanup_red_mask(red_mask.astype(np.uint8) * 255, tile_mask),
        yellow_mask=cleanup_yellow_mask(yellow_mask.astype(np.uint8) * 255, tile_mask),
    )


def cleanup_color_mask(mask: np.ndarray, tile_mask: np.ndarray) -> np.ndarray:
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_mid)
    clean = keep_large_components(clean, min_area=35)
    clean = cv2.bitwise_and(clean, tile_mask)
    return clean


def cleanup_red_mask(mask: np.ndarray, tile_mask: np.ndarray) -> np.ndarray:
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_mid)
    clean = cv2.bitwise_and(clean, tile_mask)
    return keep_red_line_components(clean, tile_mask)


def cleanup_blue_mask(
    mask: np.ndarray,
    support_mask: np.ndarray,
    tile_mask: np.ndarray,
) -> np.ndarray:
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_mid)
    clean = cv2.bitwise_and(clean, tile_mask)
    clean = keep_blue_line_components(clean, tile_mask)

    if 0 < cv2.countNonZero(clean) < 1200:
        clean = cv2.dilate(
            clean,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        clean = cv2.bitwise_and(clean, tile_mask)
        clean = keep_blue_line_components(clean, tile_mask)

    clean = extend_mask_to_edges(clean, support_mask, tile_mask)
    clean = keep_blue_line_components(clean, tile_mask)

    return clean


def cleanup_yellow_mask(mask: np.ndarray, tile_mask: np.ndarray) -> np.ndarray:
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_mid)
    clean = cv2.bitwise_and(clean, tile_mask)
    return keep_yellow_line_components(clean, tile_mask)


def reconstruct_color_mask(seed_mask: np.ndarray, grow_mask: np.ndarray) -> np.ndarray:
    prev = np.zeros_like(seed_mask)
    current = seed_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while cv2.countNonZero(cv2.absdiff(current, prev)) > 0:
        prev = current.copy()
        current = cv2.dilate(current, kernel)
        current = cv2.bitwise_and(current, grow_mask)

    return current


def build_blue_cool_fallback_mask(
    hsv: np.ndarray,
    lab: np.ndarray,
    inside_tile: np.ndarray,
    red_mask: np.ndarray,
    yellow_mask: np.ndarray,
) -> np.ndarray:
    neutral_mask = (
        inside_tile
        & ~red_mask
        & ~yellow_mask
        & (hsv[:, :, 2] <= 140)
    )

    if not np.any(neutral_mask):
        return np.zeros(inside_tile.shape, dtype=np.uint8)

    cool_threshold = float(np.quantile(lab[:, :, 2][neutral_mask], 0.12))
    fallback_mask = neutral_mask & (lab[:, :, 2] <= cool_threshold + 1.0)
    return fallback_mask.astype(np.uint8) * 255


def extend_mask_to_edges(
    mask: np.ndarray,
    support_mask: np.ndarray,
    tile_mask: np.ndarray,
    near_edge_px: float = 18.0,
) -> np.ndarray:
    if cv2.countNonZero(mask) == 0:
        return mask

    edge_distance = cv2.distanceTransform(tile_mask, cv2.DIST_L2, 5)
    edge_band = ((tile_mask > 0) & (edge_distance <= 4.0)).astype(np.uint8) * 255
    allowed = cv2.dilate(
        support_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    allowed = cv2.bitwise_and(allowed, tile_mask)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue

        component = (labels == label).astype(np.uint8) * 255
        component_dist = edge_distance[component > 0]
        if component_dist.size == 0:
            continue

        current = component.copy()
        if float(component_dist.min()) <= near_edge_px:
            for _ in range(20):
                if cv2.countNonZero(cv2.bitwise_and(current, edge_band)) > 0:
                    break
                nxt = cv2.dilate(current, kernel, iterations=1)
                nxt = cv2.bitwise_and(nxt, allowed)
                if cv2.countNonZero(cv2.absdiff(nxt, current)) == 0:
                    break
                current = nxt

        out = np.maximum(out, current)

    return out


def keep_red_line_components(mask: np.ndarray, tile_mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(120, int(0.005 * cv2.countNonZero(tile_mask)))
    min_minor_axis = max(16.0, 0.08 * min(mask.shape))
    candidates = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        minor_axis = min(width, height)
        if minor_axis < min_minor_axis:
            continue

        candidates.append((area, component_mask))

    if not candidates:
        return keep_large_components(mask, min_area=min_area)

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:2]

    clean = np.zeros_like(mask)
    for _, component_mask in selected:
        clean = np.maximum(clean, component_mask)

    return clean


def keep_blue_line_components(mask: np.ndarray, tile_mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(45, int(0.002 * cv2.countNonZero(tile_mask)))
    min_minor_axis = max(8.0, 0.04 * min(mask.shape))
    candidates = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        minor_axis = min(width, height)
        if minor_axis < min_minor_axis:
            continue

        candidates.append((area, component_mask))

    if not candidates:
        return keep_large_components(mask, min_area=min_area)

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:2]

    clean = np.zeros_like(mask)
    for _, component_mask in selected:
        clean = np.maximum(clean, component_mask)

    return clean


def keep_yellow_line_components(mask: np.ndarray, tile_mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(90, int(0.004 * cv2.countNonZero(tile_mask)))
    min_minor_axis = max(12.0, 0.06 * min(mask.shape))
    candidates = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component_mask = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        minor_axis = min(width, height)
        if minor_axis < min_minor_axis:
            continue

        candidates.append((area, component_mask))

    if not candidates:
        return keep_large_components(mask, min_area=min_area)

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:2]

    clean = np.zeros_like(mask)
    for _, component_mask in selected:
        clean = np.maximum(clean, component_mask)

    return clean


def keep_large_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            clean[labels == label] = 255
    return clean


def color_mask_to_rgb(mask: np.ndarray, color_rgb: tuple[int, int, int]) -> np.ndarray:
    rgb = np.full((mask.shape[0], mask.shape[1], 3), 240, dtype=np.uint8)
    for channel, value in enumerate(color_rgb):
        rgb[:, :, channel] = np.where(mask > 0, value, rgb[:, :, channel])
    return rgb
