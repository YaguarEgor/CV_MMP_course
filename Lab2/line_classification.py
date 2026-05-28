from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from tile_normalization import get_canonical_edge_segments


@dataclass
class ColorLineClassification:
    color_name: str
    edge_a: int
    edge_b: int
    shape_name: str

    @property
    def summary(self) -> str:
        return f"{self.color_name}: {self.shape_name} (edges {self.edge_a}-{self.edge_b})"

    @property
    def display_summary(self) -> str:
        return f"{self.color_name}: {self.shape_name}"


def classify_tile_lines(
    red_mask: np.ndarray,
    yellow_mask: np.ndarray,
    blue_mask: np.ndarray,
    tile_mask: np.ndarray,
) -> list[ColorLineClassification]:
    return [
        classify_color_line("Red line", red_mask, tile_mask),
        classify_color_line("Blue line", blue_mask, tile_mask),
        classify_color_line("Yellow line", yellow_mask, tile_mask),
    ]


def classify_color_line(
    color_name: str,
    mask: np.ndarray,
    tile_mask: np.ndarray,
) -> ColorLineClassification:
    edge_scores = score_edges_for_mask(mask, tile_mask)
    top_edges = np.argsort(edge_scores)[::-1][:2]
    edge_a = int(top_edges[0]) + 1
    edge_b = int(top_edges[1]) + 1

    if edge_a > edge_b:
        edge_a, edge_b = edge_b, edge_a

    shape_name = classify_edge_pair(edge_a, edge_b)
    return ColorLineClassification(
        color_name=color_name,
        edge_a=edge_a,
        edge_b=edge_b,
        shape_name=shape_name,
    )


def score_edges_for_mask(mask: np.ndarray, tile_mask: np.ndarray) -> np.ndarray:
    if cv2.countNonZero(mask) == 0:
        return np.zeros(6, dtype=np.float32)

    height, width = mask.shape
    edge_segments = get_canonical_edge_segments(width)
    dilated_mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    edge_bands = build_edge_band_masks(tile_mask)
    ys, xs = np.where(mask > 0)
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    scores = np.zeros(6, dtype=np.float32)

    for idx, ((p0, p1), edge_band) in enumerate(zip(edge_segments, edge_bands)):
        overlap = cv2.countNonZero(cv2.bitwise_and(dilated_mask, edge_band))
        min_dist = min_distance_to_segment(points, p0, p1)
        scores[idx] = float(overlap) + 12.0 * max(0.0, 22.0 - min_dist)

    return scores


def build_edge_band_masks(tile_mask: np.ndarray) -> list[np.ndarray]:
    height, width = tile_mask.shape
    edge_segments = get_canonical_edge_segments(width)
    edge_distance = cv2.distanceTransform(tile_mask, cv2.DIST_L2, 5)
    inner_band = ((tile_mask > 0) & (edge_distance <= 26.0)).astype(np.uint8) * 255
    edge_bands = []

    for p0, p1 in edge_segments:
        band = np.zeros((height, width), dtype=np.uint8)
        cv2.line(
            band,
            tuple(np.round(p0).astype(np.int32)),
            tuple(np.round(p1).astype(np.int32)),
            255,
            thickness=24,
        )
        band = cv2.bitwise_and(band, inner_band)
        edge_bands.append(band)

    return edge_bands


def min_distance_to_segment(points: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float:
    if len(points) == 0:
        return 1e6

    segment = p1 - p0
    denom = float(np.dot(segment, segment))
    if denom <= 1e-6:
        return float(np.linalg.norm(points - p0[None, :], axis=1).min())

    t = np.clip(((points - p0[None, :]) @ segment) / denom, 0.0, 1.0)
    projection = p0[None, :] + t[:, None] * segment[None, :]
    return float(np.linalg.norm(points - projection, axis=1).min())


def classify_edge_pair(edge_a: int, edge_b: int) -> str:
    delta = abs(edge_a - edge_b)
    step = min(delta, 6 - delta)
    if step == 1:
        return "short arc"
    if step == 2:
        return "long arc"
    if step == 3:
        return "straight"
    return "unknown"
