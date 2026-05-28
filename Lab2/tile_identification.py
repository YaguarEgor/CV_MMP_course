from __future__ import annotations

import cv2
import numpy as np

from line_classification import ColorLineClassification
from tile_line_masks import TileLineMasks


TILE_SPECS = {
    1: {
        "yellow": ("short arc", (1, 6)),
        "blue": ("long arc", (2, 4)),
        "red": ("long arc", (3, 5)),
    },
    2: {
        "red": ("short arc", (1, 6)),
        "blue": ("straight", (2, 5)),
        "yellow": ("short arc", (3, 4)),
    },
    3: {
        "blue": ("short arc", (1, 2)),
        "yellow": ("short arc", (3, 4)),
        "red": ("short arc", (5, 6)),
    },
    4: {
        "yellow": ("long arc", (1, 3)),
        "blue": ("straight", (2, 5)),
        "red": ("long arc", (4, 6)),
    },
    5: {
        "blue": ("short arc", (1, 6)),
        "red": ("straight", (2, 5)),
        "yellow": ("short arc", (3, 4)),
    },
    6: {
        "red": ("long arc", (1, 3)),
        "yellow": ("straight", (2, 5)),
        "blue": ("long arc", (4, 6)),
    },
    7: {
        "blue": ("short arc", (1, 6)),
        "yellow": ("long arc", (2, 4)),
        "red": ("long arc", (3, 5)),
    },
    8: {
        "blue": ("short arc", (1, 6)),
        "red": ("long arc", (2, 4)),
        "yellow": ("long arc", (3, 5)),
    },
    9: {
        "yellow": ("long arc", (1, 3)),
        "red": ("straight", (2, 5)),
        "blue": ("long arc", (4, 6)),
    },
    10: {
        "yellow": ("short arc", (1, 6)),
        "red": ("long arc", (2, 4)),
        "blue": ("long arc", (3, 5)),
    },
}

UNIQUE_SHAPE_SIGNATURES = {
    ("short arc", "straight", "short arc"): 2,
    ("short arc", "short arc", "short arc"): 3,
    ("long arc", "straight", "long arc"): 4,
    ("straight", "short arc", "short arc"): 5,
    ("long arc", "long arc", "straight"): 6,
    ("straight", "long arc", "long arc"): 9,
}

# For reference, these are the expected tile numbers for the provided test paths.

EXPECTED_PATH_TILE_IDS = {
    "path_4": [5, 8, 3, 7],
    "path_7_1": [1, 4, 6, 7, 8, 5, 9],
    "path_10_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

EXPECTED_PATH_DISPLAY_TILE_IDS = {
    "path_10_1": [5, 9, 3, 8, 1, 4, 2, 6, 7, 10],
}


def identify_tile_number(
    classifications: list[ColorLineClassification],
    line_masks: TileLineMasks | None = None,
) -> int | None:
    observed = {
        get_color_key(item): item
        for item in classifications
    }
    if set(observed) != {"red", "blue", "yellow"}:
        return None

    direct_match = identify_by_shape_signature(observed, line_masks)
    if direct_match is not None:
        return direct_match

    best_tile = None
    best_score = float("inf")

    for tile_number, spec in TILE_SPECS.items():
        score = score_tile_spec(observed, spec, shift=0, reflect=False)
        if score < best_score:
            best_score = score
            best_tile = tile_number

    return best_tile


def build_tile_number_costs(
    classifications: list[ColorLineClassification],
    line_masks: TileLineMasks | None = None,
) -> dict[int, float]:
    observed = {
        get_color_key(item): item
        for item in classifications
    }
    if set(observed) != {"red", "blue", "yellow"}:
        return {}

    costs = {
        tile_number: score_tile_spec(observed, spec, shift=0, reflect=False)
        for tile_number, spec in TILE_SPECS.items()
    }

    shape_signature = (
        observed["red"].shape_name,
        observed["blue"].shape_name,
        observed["yellow"].shape_name,
    )

    unique_match = UNIQUE_SHAPE_SIGNATURES.get(shape_signature)
    if unique_match is not None:
        for tile_number in costs:
            if tile_number == unique_match:
                costs[tile_number] -= 12.0
            else:
                costs[tile_number] += 40.0
        return costs

    chirality = compute_color_chirality(line_masks) if line_masks is not None else None
    if chirality is None:
        return costs

    if shape_signature == ("long arc", "long arc", "short arc"):
        preferred = 10 if chirality < 0 else 1
        alternative = 1 if preferred == 10 else 10
        costs[preferred] -= 8.0
        costs[alternative] += 8.0

    if shape_signature == ("long arc", "short arc", "long arc"):
        preferred = 7 if chirality < 0 else 8
        alternative = 8 if preferred == 7 else 7
        costs[preferred] -= 8.0
        costs[alternative] += 8.0

    return costs


def identify_by_shape_signature(
    observed: dict[str, ColorLineClassification],
    line_masks: TileLineMasks | None,
) -> int | None:
    shape_signature = (
        observed["red"].shape_name,
        observed["blue"].shape_name,
        observed["yellow"].shape_name,
    )

    if shape_signature in UNIQUE_SHAPE_SIGNATURES:
        return UNIQUE_SHAPE_SIGNATURES[shape_signature]

    if line_masks is None:
        return None

    chirality = compute_color_chirality(line_masks)
    if chirality is None:
        return None

    if shape_signature == ("long arc", "long arc", "short arc"):
        return 10 if chirality < 0 else 1

    if shape_signature == ("long arc", "short arc", "long arc"):
        return 7 if chirality < 0 else 8

    return None


def compute_color_chirality(line_masks: TileLineMasks) -> float | None:
    blue_center = mask_centroid(line_masks.blue_mask)
    red_center = mask_centroid(line_masks.red_mask)
    yellow_center = mask_centroid(line_masks.yellow_mask)
    if blue_center is None or red_center is None or yellow_center is None:
        return None

    red_vec = red_center - blue_center
    yellow_vec = yellow_center - blue_center
    return float(red_vec[0] * yellow_vec[1] - red_vec[1] * yellow_vec[0])


def mask_centroid(mask: np.ndarray) -> np.ndarray | None:
    if cv2.countNonZero(mask) == 0:
        return None

    moments = cv2.moments(mask)
    if moments["m00"] <= 1e-6:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)

    return np.array(
        [
            moments["m10"] / moments["m00"],
            moments["m01"] / moments["m00"],
        ],
        dtype=np.float32,
    )


def score_tile_spec(
    observed: dict[str, ColorLineClassification],
    spec: dict[str, tuple[str, tuple[int, int]]],
    shift: int,
    reflect: bool = False,
) -> float:
    score = 0.0

    for color_key, observed_item in observed.items():
        expected_shape, expected_pair = spec[color_key]
        transformed_pair = expected_pair
        if reflect:
            transformed_pair = reflect_edge_pair(transformed_pair)
        rotated_pair = rotate_edge_pair(transformed_pair, shift)

        if observed_item.shape_name != expected_shape:
            score += 20.0

        score += edge_pair_distance(
            (observed_item.edge_a, observed_item.edge_b),
            rotated_pair,
        )

    return score


def get_color_key(item: ColorLineClassification) -> str:
    return item.color_name.split()[0].lower()


def rotate_edge_pair(pair: tuple[int, int], shift: int) -> tuple[int, int]:
    rotated = [((edge - 1 + shift) % 6) + 1 for edge in pair]
    rotated.sort()
    return rotated[0], rotated[1]


def reflect_edge_pair(pair: tuple[int, int]) -> tuple[int, int]:
    reflected = [7 - edge for edge in pair]
    reflected.sort()
    return reflected[0], reflected[1]


def edge_pair_distance(
    observed_pair: tuple[int, int],
    expected_pair: tuple[int, int],
) -> float:
    obs_a, obs_b = observed_pair
    exp_a, exp_b = expected_pair
    return min(
        circular_edge_distance(obs_a, exp_a) + circular_edge_distance(obs_b, exp_b),
        circular_edge_distance(obs_a, exp_b) + circular_edge_distance(obs_b, exp_a),
    )


def circular_edge_distance(edge_a: int, edge_b: int) -> int:
    diff = abs(edge_a - edge_b)
    return min(diff, 6 - diff)
