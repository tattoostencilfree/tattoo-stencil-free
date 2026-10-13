from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import uuid

import cv2
import numpy as np
from PIL import Image


@dataclass
class PosterizeResult:
    job_id: str
    posterized_png: Path
    guided_png: Path
    guides_png: Path
    guided_detailed_png: Path
    guided_simplified_png: Path
    guided_clean_png: Path
    guides_detailed_png: Path
    guides_simplified_png: Path
    guides_clean_png: Path
    width: int
    height: int
    levels: int


def _resize_to_max(rgb: np.ndarray, max_side: int = 1600) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale >= 0.999:
        return rgb.copy(), 1.0
    out = cv2.resize(
        rgb,
        (max(2, int(round(w * scale))), max(2, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return out, scale


def _cleanup_small_regions(labels: np.ndarray, min_area: int, passes: int = 3) -> np.ndarray:
    """Merge tiny islands into the dominant touching region without redrawing big shapes."""
    labels = labels.copy().astype(np.int32)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(max(1, passes)):
        changed = False
        for value in [int(v) for v in np.unique(labels)]:
            mask = (labels == value).astype(np.uint8)
            n, cc, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area >= min_area:
                    continue
                x = int(stats[i, cv2.CC_STAT_LEFT])
                y = int(stats[i, cv2.CC_STAT_TOP])
                ww = int(stats[i, cv2.CC_STAT_WIDTH])
                hh = int(stats[i, cv2.CC_STAT_HEIGHT])
                comp = (cc[y:y + hh, x:x + ww] == i).astype(np.uint8)
                dil = cv2.dilate(comp, kernel, 1).astype(bool)
                region = labels[y:y + hh, x:x + ww]
                neighbors = region[dil & (comp == 0)]
                neighbors = neighbors[neighbors != value]
                if neighbors.size == 0:
                    continue
                vals, counts = np.unique(neighbors, return_counts=True)
                replacement = int(vals[np.argmax(counts)])
                region[comp > 0] = replacement
                labels[y:y + hh, x:x + ww] = region
                changed = True
        if not changed:
            break
    return labels.astype(np.uint8)


def _luminance_posterize(work_rgb: np.ndarray, intensity: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a clean cel/posterized image while preserving the source color family.

    The old full-color clustering shifted skin/hair tones too much. This version clusters
    only luminance, keeps smoothly filtered chroma from the source, then applies a light
    saturation lift. Result: cleaner graphic regions, but much closer to the original hues.
    """
    intensity = int(np.clip(intensity, 0, 100))

    # Smooth photographic micro-noise but keep facial/object boundaries.
    bgr = cv2.cvtColor(work_rgb, cv2.COLOR_RGB2BGR)
    sigma_s = float(np.interp(intensity, [0, 100], [18, 72]))
    sigma_r = float(np.interp(intensity, [0, 100], [0.08, 0.32]))
    smooth_bgr = cv2.edgePreservingFilter(bgr, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
    smooth_bgr = cv2.bilateralFilter(smooth_bgr, 7, 26 + intensity * 0.16, 26 + intensity * 0.16)
    smooth_rgb = cv2.cvtColor(smooth_bgr, cv2.COLOR_BGR2RGB)

    lab = cv2.cvtColor(smooth_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]

    # More posterization to the right, more nuanced to the left.
    levels = int(round(np.interp(intensity, [0, 100], [14, 6])))
    samples = L.reshape(-1, 1).astype(np.float32)
    stride = max(1, len(samples) // 110000)
    sample = samples[::stride]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.08)
    _compact, _labels, centers = cv2.kmeans(sample, levels, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    centers = np.sort(centers.ravel().astype(np.float32))

    # Assign every pixel to the nearest luminance band.
    distances = np.abs(L[:, :, None] - centers[None, None, :])
    labels = np.argmin(distances, axis=2).astype(np.uint8)

    # Clean little speckles while keeping actual shapes.
    labels = cv2.medianBlur(labels, 3)
    if intensity >= 62:
        labels = cv2.medianBlur(labels, 5)
    min_area = int(round(np.interp(intensity, [0, 100], [14, 55])))
    labels = _cleanup_small_regions(labels, min_area=min_area, passes=2)

    # Recalculate representative luminance from the smoothed source.
    region_l = np.zeros(levels, dtype=np.float32)
    for k in range(levels):
        px = L[labels == k]
        region_l[k] = float(np.median(px)) if px.size else float(centers[k])
    region_l.sort()

    # Keep a little within-region light variation so it looks like a polished illustration,
    # not a crude 8-color cutout.
    qL = region_l[labels]
    poster_strength = float(np.interp(intensity, [0, 100], [0.20, 0.88]))
    outL = qL * poster_strength + L * (1.0 - poster_strength)

    out_lab = lab.copy()
    out_lab[:, :, 0] = outL
    # Preserve chroma; give a restrained saturation lift like the approved reference.
    chroma_boost = float(np.interp(intensity, [0, 100], [1.02, 1.10]))
    out_lab[:, :, 1] = 128.0 + (out_lab[:, :, 1] - 128.0) * chroma_boost
    out_lab[:, :, 2] = 128.0 + (out_lab[:, :, 2] - 128.0) * chroma_boost

    poster_rgb = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Very small brightness/saturation polish, still respecting source hues.
    hsv = cv2.cvtColor(poster_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.05, 0, 255)
    gamma = float(np.interp(intensity, [0, 100], [0.98, 0.93]))
    hsv[:, :, 2] = np.clip((hsv[:, :, 2] / 255.0) ** gamma * 255.0, 0, 255)
    poster_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return poster_rgb, labels, region_l


def _reduce_labels(labels: np.ndarray, centers: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Build the stencil region map from the SAME posterization, but remove micro-zones.

    The posterized preview keeps all of its useful tonal richness. For the stencil we
    deliberately merge only very small isolated regions so the tattoo guide does not
    become saturated with tiny contours.
    """
    n = len(centers)
    total_area = int(labels.shape[0] * labels.shape[1])

    # Minimum surviving region size as a fraction of the image.
    # Detailed keeps many regions but drops tiny speckles; simplified/clean merge more.
    min_ratio = {
        "detailed": 0.00032,    # 0.032%
        "simplified": 0.00072,  # 0.072%
        "clean": 0.00170,       # 0.170%
    }[mode]
    min_area = max(28, int(round(total_area * min_ratio)))

    if mode == "detailed":
        mapping = cv2.medianBlur(labels.astype(np.uint8), 3)
        mapping = _cleanup_small_regions(mapping, min_area=min_area, passes=4)
        return mapping, centers.copy()

    if mode == "simplified":
        target = max(6, int(round(n * 0.62)))
        blur = 3
    else:
        target = max(4, int(round(n * 0.38)))
        blur = 5

    mapping = np.floor(labels.astype(np.float32) * target / max(n, 1)).astype(np.uint8)
    mapping = np.clip(mapping, 0, target - 1)
    mapping = cv2.medianBlur(mapping, blur)
    mapping = _cleanup_small_regions(mapping, min_area=min_area, passes=4)

    raw_group = np.floor(np.arange(n) * target / max(n, 1)).astype(int)
    new_centers = np.zeros(target, np.float32)
    for k in range(target):
        members = centers[raw_group == k]
        new_centers[k] = float(members.mean()) if len(members) else float(centers[min(k, n - 1)])
    return mapping, new_centers


def _boundary(labels: np.ndarray) -> np.ndarray:
    out = np.zeros(labels.shape, np.uint8)
    out[:, 1:] |= (labels[:, 1:] != labels[:, :-1]).astype(np.uint8)
    out[1:, :] |= (labels[1:, :] != labels[:-1, :]).astype(np.uint8)
    return out


def _remove_small_lines(mask: np.ndarray, min_area: int, min_span: int) -> np.ndarray:
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask, np.uint8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        span = max(int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        if area >= min_area or span >= min_span:
            out[cc == i] = 1
    return out


def _mode_guides(
    poster_rgb: np.ndarray,
    source_rgb: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Map posterized zones into solid structure + dashed tonal boundaries.

    DETAILED: every meaningful posterized zone boundary is represented.
    SIMPLIFIED/CLEAN: use progressively merged versions of the same exact region map.
    """
    labels_m, centers_m = _reduce_labels(labels, centers, mode)
    h, w = labels_m.shape
    boundaries = _boundary(labels_m)

    # Detect truly strong structural borders from source contrast/color difference.
    gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), 0.75)

    src = source_rgb.astype(np.int16)
    color_gap = np.zeros((h, w), np.float32)
    color_gap[:, 1:] = np.maximum(color_gap[:, 1:], np.max(np.abs(src[:, 1:] - src[:, :-1]), axis=2))
    color_gap[1:, :] = np.maximum(color_gap[1:, :], np.max(np.abs(src[1:] - src[:-1]), axis=2))

    vals = grad[boundaries > 0]
    pct = {"detailed": 66, "simplified": 78, "clean": 84}[mode]
    gthr = float(np.percentile(vals, pct)) if vals.size else 9999.0
    cthr = {"detailed": 26.0, "simplified": 34.0, "clean": 42.0}[mode]
    solid = ((boundaries > 0) & ((grad >= gthr) | (color_gap >= cthr))).astype(np.uint8)

    # IMPORTANT: detailed preserves every meaningful remaining region boundary as a tonal guide.
    # Suppress only near-black-to-near-black changes (background/suit micro-banding), which
    # are not useful tattoo value guides and made the earlier stencil look dirty.
    bright_side = np.zeros((h, w), np.float32)
    bright_side[:, 1:] = np.maximum(bright_side[:, 1:], np.maximum(centers_m[labels_m[:, 1:]], centers_m[labels_m[:, :-1]]))
    bright_side[1:, :] = np.maximum(bright_side[1:, :], np.maximum(centers_m[labels_m[1:, :]], centers_m[labels_m[:-1, :]]))
    min_brightness = {"detailed": 34.0, "simplified": 56.0, "clean": 64.0}[mode]
    if mode == "detailed":
        # The Detailed stencil must inherit every retained posterization-zone boundary.
        # Micro-regions were already merged in _reduce_labels(), so at this point a
        # remaining boundary represents a meaningful medium/large tonal region.
        # This is what preserves cheek, forehead, eyelid, neck, fabric and background
        # value shapes instead of letting the artistic model silently omit them.
        tonal = ((boundaries > 0) & (solid == 0)).astype(np.uint8)
    else:
        tonal = ((boundaries > 0) & (solid == 0) & (bright_side >= min_brightness)).astype(np.uint8)

    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    tonal = cv2.morphologyEx(tonal, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    solid = _remove_small_lines(
        solid,
        {"detailed": 10, "simplified": 16, "clean": 24}[mode],
        {"detailed": 14, "simplified": 20, "clean": 28}[mode],
    )
    tonal = _remove_small_lines(
        tonal,
        {"detailed": 8, "simplified": 24, "clean": 34}[mode],
        {"detailed": 11, "simplified": 28, "clean": 38}[mode],
    )

    # Make continuous structure a touch heavier than tonal guides.
    if mode == "detailed":
        solid = cv2.dilate(solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)), 1)
    elif mode == "clean":
        solid = cv2.dilate(solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), 1)

    # Convert tonal boundaries to clean broken guides.
    yy, xx = np.indices((h, w))
    period = {"detailed": 12, "simplified": 14, "clean": 18}[mode]
    on = {"detailed": 9, "simplified": 8, "clean": 9}[mode]
    pattern = ((xx + (yy // 2)) % period) < on
    dashed = (tonal > 0) & pattern

    # User preview: clean posterization with guides over it.
    guided = poster_rgb.copy().astype(np.float32)
    guided[solid > 0] *= 0.12
    guided[dashed] *= 0.55
    guided = np.clip(guided, 0, 255).astype(np.uint8)

    # Semantic guide map for stencil generation. Structural lines are black.
    # Tonal boundaries are stored CONTINUOUS in gray so the provider can redraw them
    # as clean anti-aliased dashed tattoo guides at the final output resolution.
    # The user-facing posterization preview above still shows them dashed.
    overlay = np.zeros((h, w, 4), np.uint8)
    overlay[solid > 0, :3] = 0
    overlay[solid > 0, 3] = 255
    overlay[tonal > 0, :3] = 88
    overlay[tonal > 0, 3] = 255
    return guided, overlay


def posterize_for_tattoo(image_path: Path, output_root: Path, intensity: int = 25) -> PosterizeResult:
    intensity = int(np.clip(intensity, 0, 100))
    job_id = uuid.uuid4().hex
    out_dir = output_root / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    master = np.array(Image.open(image_path).convert("RGB"))
    H, W = master.shape[:2]
    work, scale = _resize_to_max(master, 1600)

    poster_small, labels, centers = _luminance_posterize(work, intensity)

    guided_by_mode: dict[str, np.ndarray] = {}
    guides_by_mode: dict[str, np.ndarray] = {}
    for mode in ("detailed", "simplified", "clean"):
        guided_by_mode[mode], guides_by_mode[mode] = _mode_guides(
            poster_small, work, labels, centers, mode
        )

    def up_rgb(arr: np.ndarray) -> np.ndarray:
        return cv2.resize(arr, (W, H), interpolation=cv2.INTER_CUBIC) if scale < 0.999 else arr

    def up_rgba(arr: np.ndarray) -> np.ndarray:
        return cv2.resize(arr, (W, H), interpolation=cv2.INTER_NEAREST) if scale < 0.999 else arr

    poster = up_rgb(poster_small)
    guided = {k: up_rgb(v) for k, v in guided_by_mode.items()}
    guides = {k: up_rgba(v) for k, v in guides_by_mode.items()}

    poster_path = out_dir / "posterized.png"
    Image.fromarray(poster, "RGB").save(poster_path, "PNG", optimize=True)

    guided_paths: dict[str, Path] = {}
    guide_paths: dict[str, Path] = {}
    for mode in ("detailed", "simplified", "clean"):
        gp = out_dir / f"guided-{mode}.png"
        op = out_dir / f"guides-{mode}.png"
        Image.fromarray(guided[mode], "RGB").save(gp, "PNG", optimize=True)
        Image.fromarray(guides[mode], "RGBA").save(op, "PNG", optimize=True)
        guided_paths[mode] = gp
        guide_paths[mode] = op

    guided_alias = out_dir / "guided.png"
    guides_alias = out_dir / "guides.png"
    Image.open(guided_paths["detailed"]).save(guided_alias, "PNG", optimize=True)
    Image.open(guide_paths["detailed"]).save(guides_alias, "PNG", optimize=True)

    return PosterizeResult(
        job_id=job_id,
        posterized_png=poster_path,
        guided_png=guided_alias,
        guides_png=guides_alias,
        guided_detailed_png=guided_paths["detailed"],
        guided_simplified_png=guided_paths["simplified"],
        guided_clean_png=guided_paths["clean"],
        guides_detailed_png=guide_paths["detailed"],
        guides_simplified_png=guide_paths["simplified"],
        guides_clean_png=guide_paths["clean"],
        width=W,
        height=H,
        levels=len(centers),
    )
