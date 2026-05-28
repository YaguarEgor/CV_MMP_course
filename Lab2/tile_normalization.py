from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class NormalizedTile:
    tile_rgb: np.ndarray
    tile_mask: np.ndarray
    source_center: tuple[int, int]


def extract_normalized_tiles_from_instances(
    image_bgr: np.ndarray,
    instance_masks: list[np.ndarray],
    min_area: int = 1500,
    out_size: int = 220,
) -> list[NormalizedTile]:
    normalized_tiles = []

    for mask in instance_masks:
        normalized_tile = normalize_tile_from_mask(
            image_bgr,
            mask,
            min_area=min_area,
            out_size=out_size,
        )
        if normalized_tile is not None:
            normalized_tiles.append(normalized_tile)

    return normalized_tiles


def normalize_tile_from_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    min_area: int,
    out_size: int,
) -> NormalizedTile | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < min_area:
        return None

    moments = cv2.moments(contour)
    if moments["m00"] > 1e-6:
        source_center = (
            int(round(moments["m10"] / moments["m00"])),
            int(round(moments["m01"] / moments["m00"])),
        )
    else:
        pts = contour.reshape(-1, 2)
        source_center = (int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1])))

    src_vertices = extract_hex_vertices(mask)
    if src_vertices is None:
        return None

    dst_vertices = build_canonical_hex_vertices(out_size)
    homography, _ = cv2.findHomography(src_vertices, dst_vertices, 0)
    if homography is None:
        return None

    warped_bgr = cv2.warpPerspective(
        image_bgr,
        homography,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(240, 240, 240),
    )
    canonical_mask = build_canonical_hex_mask(out_size)

    tile_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
    tile_rgb[canonical_mask == 0] = 240

    return NormalizedTile(
        tile_rgb=tile_rgb,
        tile_mask=canonical_mask,
        source_center=source_center,
    )


def extract_hex_vertices(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True).reshape(-1, 2).astype(np.float32)

    if len(approx) != 6:
        return None

    ordered = ensure_matching_orientation(approx, build_canonical_hex_vertices(220))
    return rotate_vertices_to_bottom_edge(ordered)


def build_canonical_hex_vertices(out_size: int, margin: int = 18) -> np.ndarray:
    center = np.array([out_size / 2.0, out_size / 2.0], dtype=np.float32)
    radius = (out_size / 2.0) - margin
    vertices = []

    for angle_deg in (90, 30, 330, 270, 210, 150):
        angle_rad = np.deg2rad(angle_deg)
        vertices.append(
            [
                center[0] + radius * np.cos(angle_rad),
                center[1] + radius * np.sin(angle_rad),
            ]
        )

    return np.array(vertices, dtype=np.float32)


def build_canonical_hex_mask(out_size: int) -> np.ndarray:
    mask = np.zeros((out_size, out_size), dtype=np.uint8)
    vertices = np.round(build_canonical_hex_vertices(out_size)).astype(np.int32)
    cv2.fillConvexPoly(mask, vertices, 255)
    return mask


def ensure_matching_orientation(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if np.sign(polygon_signed_area(points)) != np.sign(polygon_signed_area(reference)):
        return points[::-1].copy()
    return points.copy()


def rotate_vertices_to_bottom_edge(points: np.ndarray) -> np.ndarray:
    bottom_vertex_idx = int(np.argmax(points[:, 1]))
    return np.roll(points, -bottom_vertex_idx, axis=0)


def polygon_signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def get_canonical_edge_segments(out_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
    vertices = build_canonical_hex_vertices(out_size)
    return [
        (vertices[idx], vertices[(idx + 1) % len(vertices)])
        for idx in range(len(vertices))
    ]
