from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class LayerPaths:
    silhouette: Path
    essential: Path
    structure: Path
    segmentation: Path
    detail: Path
    shade: Path
    texture: Path


def _clean_binary(gray: np.ndarray) -> np.ndarray:
    # Ink is dark; keep anti-aliased line edges while ignoring near-white background.
    ink = (gray < 242).astype(np.uint8)
    # Remove isolated 1-2 px specks without changing the long line geometry.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    clean = np.zeros_like(ink)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        span = max(int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        if area >= 5 or span >= 6:
            clean[labels == i] = 1
    return clean


def _save_transparent(mask: np.ndarray, gray: np.ndarray, target: Path) -> None:
    """Save original line darkness on transparency so recomposition preserves tonal hatching."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = 0
    darkness = (255 - gray).astype(np.float32)
    # Preserve anti-aliasing and soft engraved lines while keeping faint near-white noise out.
    alpha = np.clip((darkness - 4) * 1.08, 0, 255).astype(np.uint8)
    rgba[..., 3] = alpha * mask.astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(target)


def split_stencil_into_stable_layers(full_png: Path, output_dir: Path) -> LayerPaths:
    """
    Split a finished artistic stencil into stable, non-overlapping line groups.

    This is intentionally NOT an edge detector on the source photo. It only groups
    linework that has already been produced by the generative stencil model.
    Controls can therefore hide/show groups without moving facial geometry.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    gray = cv2.imread(str(full_png), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not read generated stencil: {full_png}")

    ink = _clean_binary(gray)
    h, w = ink.shape

    # Stroke thickness estimate. Thick cores correlate with main contours.
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 5)

    # Local line density separates hatching from isolated fine detail.
    density = cv2.GaussianBlur(ink.astype(np.float32), (0, 0), sigmaX=7.0, sigmaY=7.0)

    # Long connected components receive a higher structural importance.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    component_importance = np.zeros_like(ink, dtype=np.float32)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        span = max(ww, hh)
        score = min(1.0, (span / max(h, w)) * 5.0 + min(area / 5000.0, 0.5))
        component_importance[labels == i] = score

    # Main line core -> expand only within original ink to recover full stroke width.
    main_core = ((dist >= 2.15) & (ink > 0)).astype(np.uint8)
    main_grown = cv2.dilate(main_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
    silhouette = ((main_grown > 0) & (ink > 0) & ((component_importance > 0.30) | (dist >= 2.8))).astype(np.uint8)

    remaining = (ink & (1 - silhouette)).astype(np.uint8)

    # Strong/long remaining lines = internal structure.
    structure = (
        (remaining > 0)
        & ((dist >= 1.15) | (component_importance >= 0.20))
        & (density < 0.46)
    ).astype(np.uint8)

    # Essential structure is an intentionally sparse subset for the "Muy limpio" mode.
    # It keeps only the strongest/longest internal contours, while the outer silhouette
    # remains in the silhouette layer. This is still derived ONLY from the finished
    # artistic stencil, never from photo edge detection.
    essential = (
        (structure > 0)
        & ((component_importance >= 0.34) | (dist >= 1.55))
        & (density < 0.34)
    ).astype(np.uint8)
    remaining = (remaining & (1 - structure)).astype(np.uint8)

    # Tonal-segmentation guides are intentionally sparse, thin, elongated broken
    # strokes. The generator is explicitly prompted to draw them as dashed/broken
    # boundaries between value zones. We isolate a conservative subset here so
    # these guides can remain visible regardless of detail/texture sliders.
    #
    # This classifier operates ONLY on the finished artistic stencil; it never
    # detects edges from the user's photograph.
    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(remaining, 8)
    segmentation = np.zeros_like(remaining)
    for i in range(1, n2):
        area = int(stats2[i, cv2.CC_STAT_AREA])
        ww = int(stats2[i, cv2.CC_STAT_WIDTH])
        hh = int(stats2[i, cv2.CC_STAT_HEIGHT])
        span = max(ww, hh)
        minor = max(1, min(ww, hh))
        elong = span / minor
        if not (5 <= area <= 420 and 8 <= span <= max(90, int(max(h, w) * 0.16)) and elong >= 1.7):
            continue
        component = labels2 == i
        mean_density = float(density[component].mean()) if np.any(component) else 1.0
        mean_dist = float(dist[component].mean()) if np.any(component) else 99.0
        # Sparse + thin is the key signature. Keep this intentionally strict to
        # avoid promoting dense engraving texture into the permanent guide layer.
        if mean_density < 0.25 and mean_dist < 1.45:
            segmentation[component] = 1

    remaining = (remaining & (1 - segmentation)).astype(np.uint8)

    # Dense thin lines are typically crosshatching/shading.
    shade = ((remaining > 0) & (density >= 0.10)).astype(np.uint8)
    remaining = (remaining & (1 - shade)).astype(np.uint8)

    # Of the remaining fine marks, more clustered marks behave like texture;
    # isolated short lines behave like detail.
    texture_density = cv2.GaussianBlur(remaining.astype(np.float32), (0, 0), sigmaX=3.0, sigmaY=3.0)
    texture = ((remaining > 0) & (texture_density >= 0.045)).astype(np.uint8)
    detail = (remaining & (1 - texture)).astype(np.uint8)

    # Anything accidentally unassigned stays in detail so the composite never loses ink.
    assigned = np.clip(silhouette + structure + segmentation + shade + texture + detail, 0, 1)
    missing = (ink & (1 - assigned)).astype(np.uint8)
    detail = np.clip(detail + missing, 0, 1).astype(np.uint8)

    paths = LayerPaths(
        silhouette=output_dir / "silhouette.png",
        essential=output_dir / "essential.png",
        structure=output_dir / "structure.png",
        segmentation=output_dir / "segmentation.png",
        detail=output_dir / "detail.png",
        shade=output_dir / "shade.png",
        texture=output_dir / "texture.png",
    )
    _save_transparent(silhouette, gray, paths.silhouette)
    _save_transparent(essential, gray, paths.essential)
    _save_transparent(structure, gray, paths.structure)
    _save_transparent(segmentation, gray, paths.segmentation)
    _save_transparent(detail, gray, paths.detail)
    _save_transparent(shade, gray, paths.shade)
    _save_transparent(texture, gray, paths.texture)
    return paths
