from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from line_classification import ColorLineClassification
from tile_normalization import extract_hex_vertices


@dataclass
class ClosedRoute:
    color_name: str | None
    tile_sequence: list[int]

    @property
    def summary(self) -> str:
        if not self.tile_sequence:
            return "Closed route: not found"

        sequence_text = " -> ".join(str(tile_number) for tile_number in self.tile_sequence)
        if self.color_name:
            return f"Closed route ({self.color_name}): {sequence_text}"
        return f"Closed route: {sequence_text}"


@dataclass
class RouteStartCandidate:
    color_key: str
    start_edge: int
    closing_edge: int
    next_idx: int
    next_entry_edge: int


def find_closed_route(
    instance_masks: list[np.ndarray],
    analyses: list[tuple[object, object, list[ColorLineClassification]]],
    tile_numbers: list[int],
) -> ClosedRoute | None:
    if len(instance_masks) != len(analyses) or len(analyses) != len(tile_numbers):
        return None
    if not tile_numbers or any(tile_number is None for tile_number in tile_numbers):
        return None

    adjacency = build_touching_adjacencies(instance_masks)
    if not adjacency:
        return None

    edge_neighbors = build_edge_neighbor_map(len(tile_numbers), adjacency)
    color_edges = [build_color_edge_map(classifications) for _, _, classifications in analyses]
    centers = [np.array(tile.source_center, dtype=np.float32) for tile, _, _ in analyses]

    start_idx = find_lowest_tile_index(centers)
    start_candidates = build_start_candidates(start_idx, edge_neighbors, color_edges)
    if not start_candidates:
        return None

    best_route: ClosedRoute | None = None
    best_rank: tuple[int, float, list[int]] | None = None

    for candidate in start_candidates:
        route_indices = walk_route(
            start_idx=start_idx,
            candidate=candidate,
            edge_neighbors=edge_neighbors,
            color_edges=color_edges,
            tile_count=len(tile_numbers),
        )
        if route_indices is None:
            continue

        tile_sequence = [tile_numbers[idx] for idx in route_indices]
        turn_score = compute_turn_score(route_indices, centers)
        rank = (len(tile_sequence), turn_score, tile_sequence)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_route = ClosedRoute(candidate.color_key, tile_sequence)

    return best_route


def build_touching_adjacencies(instance_masks: list[np.ndarray]) -> list[tuple[int, int, int, int]]:
    vertices = [extract_hex_vertices(mask) for mask in instance_masks]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    adjacency = []

    for idx_a in range(len(instance_masks)):
        if vertices[idx_a] is None:
            continue
        for idx_b in range(idx_a + 1, len(instance_masks)):
            if vertices[idx_b] is None:
                continue

            contact = cv2.bitwise_and(
                cv2.dilate(instance_masks[idx_a], kernel),
                cv2.dilate(instance_masks[idx_b], kernel),
            )
            if cv2.countNonZero(contact) < 50:
                continue

            ys, xs = np.where(contact > 0)
            contact_center = np.array([xs.mean(), ys.mean()], dtype=np.float32)
            edge_a = find_closest_edge(vertices[idx_a], contact_center)
            edge_b = find_closest_edge(vertices[idx_b], contact_center)
            adjacency.append((idx_a, edge_a, idx_b, edge_b))

    return adjacency


def build_edge_neighbor_map(
    tile_count: int,
    adjacency: list[tuple[int, int, int, int]],
) -> list[dict[int, list[tuple[int, int]]]]:
    edge_neighbors = [{edge: [] for edge in range(1, 7)} for _ in range(tile_count)]
    for tile_a, edge_a, tile_b, edge_b in adjacency:
        edge_neighbors[tile_a][edge_a].append((tile_b, edge_b))
        edge_neighbors[tile_b][edge_b].append((tile_a, edge_a))
    return edge_neighbors


def build_color_edge_map(classifications: list[ColorLineClassification]) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for item in classifications:
        color_key = item.color_name.split()[0].lower()
        out[color_key] = (item.edge_a, item.edge_b)
    return out


def find_lowest_tile_index(centers: list[np.ndarray]) -> int:
    return max(
        range(len(centers)),
        key=lambda idx: (float(centers[idx][1]), -float(centers[idx][0])),
    )


def build_start_candidates(
    start_idx: int,
    edge_neighbors: list[dict[int, list[tuple[int, int]]]],
    color_edges: list[dict[str, tuple[int, int]]],
) -> list[RouteStartCandidate]:
    candidates: list[RouteStartCandidate] = []
    start_contacts = edge_neighbors[start_idx]

    for color_key, edge_pair in color_edges[start_idx].items():
        edge_a, edge_b = edge_pair
        neighbors_a = start_contacts.get(edge_a, [])
        neighbors_b = start_contacts.get(edge_b, [])
        if len(neighbors_a) != 1 or len(neighbors_b) != 1:
            continue

        next_idx_a, next_entry_edge_a = neighbors_a[0]
        next_idx_b, next_entry_edge_b = neighbors_b[0]

        candidates.append(
            RouteStartCandidate(
                color_key=color_key,
                start_edge=edge_a,
                closing_edge=edge_b,
                next_idx=next_idx_a,
                next_entry_edge=next_entry_edge_a,
            )
        )
        candidates.append(
            RouteStartCandidate(
                color_key=color_key,
                start_edge=edge_b,
                closing_edge=edge_a,
                next_idx=next_idx_b,
                next_entry_edge=next_entry_edge_b,
            )
        )

    return candidates


def walk_route(
    start_idx: int,
    candidate: RouteStartCandidate,
    edge_neighbors: list[dict[int, list[tuple[int, int]]]],
    color_edges: list[dict[str, tuple[int, int]]],
    tile_count: int,
) -> list[int] | None:
    route_indices = [start_idx]
    visited = {start_idx}

    prev_idx = start_idx
    current_idx = candidate.next_idx
    entry_edge = candidate.next_entry_edge

    while True:
        if current_idx in visited:
            return None

        route_indices.append(current_idx)
        visited.add(current_idx)

        line_edges = color_edges[current_idx].get(candidate.color_key)
        if line_edges is None or entry_edge not in line_edges:
            return None

        exit_edge = line_edges[1] if line_edges[0] == entry_edge else line_edges[0]
        next_contacts = [
            item
            for item in edge_neighbors[current_idx].get(exit_edge, [])
            if item[0] != prev_idx
        ]
        if len(next_contacts) != 1:
            return None

        next_idx, next_entry_edge = next_contacts[0]
        if next_idx == start_idx:
            if next_entry_edge != candidate.closing_edge:
                return None
            return route_indices

        if len(route_indices) >= tile_count:
            return None

        prev_idx = current_idx
        current_idx = next_idx
        entry_edge = next_entry_edge


def compute_turn_score(route_indices: list[int], centers: list[np.ndarray]) -> float:
    if len(route_indices) < 3:
        return -1e6

    start_center = centers[route_indices[0]]
    second_center = centers[route_indices[1]]
    last_center = centers[route_indices[-1]]

    second_dx = float(second_center[0] - start_center[0])
    last_dx = float(last_center[0] - start_center[0])
    return second_dx - last_dx


def find_closest_edge(vertices: np.ndarray, point: np.ndarray) -> int:
    best_edge = 1
    best_distance = float("inf")

    for edge_idx in range(6):
        start = vertices[edge_idx]
        end = vertices[(edge_idx + 1) % 6]
        distance = point_to_segment_distance(point, start, end)
        if distance < best_distance:
            best_distance = distance
            best_edge = edge_idx + 1

    return best_edge


def point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-6:
        return float(np.linalg.norm(point - start))

    t = np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0)
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))
