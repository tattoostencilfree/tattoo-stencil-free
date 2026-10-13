"""Image-generation provider for the artistic stencil engine.

The source photo is always the first image. The second image is a style-only
reference chosen for the requested stencil mode.

Important: the detailed mode intentionally keeps the same golden reference and
prompt language as v5. The new work in v6 is focused on Simplificado and Muy limpio.
"""
from __future__ import annotations

import base64
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import cv2
from PIL import Image

from .layerizer import split_stencil_into_stable_layers


@dataclass
class StencilGenerationResult:
    job_id: str
    full_png: Path
    silhouette_png: Path | None = None
    essential_png: Path | None = None
    structure_png: Path | None = None
    segmentation_png: Path | None = None
    detail_png: Path | None = None
    shade_png: Path | None = None
    texture_png: Path | None = None
    model: str | None = None
    mode: str = "detailed"


@dataclass
class PhotoEnhancementResult:
    job_id: str
    enhanced_png: Path
    model: str
    width: int
    height: int


@dataclass
class PosterizationMasterResult:
    job_id: str
    posterized_png: Path
    model: str
    width: int
    height: int


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "backend" / "generated"
ENHANCED = ROOT / "backend" / "enhanced"
REFERENCE_DIR = ROOT / "backend" / "reference"
POSTER_MASTERS = ROOT / "backend" / "poster_masters"
REFERENCES = {
    # All stencil modes use the SAME approved clean transfer style.
    # Mode prompts control the amount of retained information; the style itself stays consistent.
    "detailed": REFERENCE_DIR / "clean-stencil-approved.png",
    "simplified": REFERENCE_DIR / "clean-stencil-approved.png",
    "clean": REFERENCE_DIR / "clean-stencil-approved.png",
}
GENERATED.mkdir(parents=True, exist_ok=True)
ENHANCED.mkdir(parents=True, exist_ok=True)
POSTER_MASTERS.mkdir(parents=True, exist_ok=True)

# EXACT detailed base prompt from v5. Do not change without deliberately changing
# the already-approved Detailed mode.
DETAILED_BASE_PROMPT = """
EDIT the FIRST image into a CLEAN PROFESSIONAL TATTOO TRANSFER STENCIL.
The FIRST image is the user's clean posterized subject.
The SECOND image is a GUIDE MAP for geometry only: black/gray marks show approved structural and tonal boundaries. DO NOT copy the guide map's pixel texture, roughness, thickness or gray color into the final artwork.
The THIRD image is ONLY a STYLE REFERENCE for clean line hierarchy and white-space economy.

ABSOLUTE FIDELITY:
- Preserve the exact subject, identity, facial geometry, expression, pose, crop, clothing, objects and composition.
- Preserve the main background layout and major background shapes when they are visible in the source / posterization.
- Never duplicate, add, remove or relocate an object, badge, flag, ornament, facial feature or background element.
- Follow the visible posterized guide map. Those guide boundaries come from the user's own posterization.

DESIRED STENCIL STYLE:
- pure white background
- pure black linework only; no gray strokes, no gray shadows, no gray debris
- bold, smooth continuous outer silhouette and major structural contours
- thinner clean internal contours for eyes, nose, lips, ears, folds and important shape landmarks
- thin BROKEN / SEGMENTED black lines for ALL medium and large posterized light/midtone/shadow region changes, plus a SMALL amount of extra clean line detail for clearly readable secondary zones
- generous white space
- clean tattoo-transfer readability at normal viewing size
- retain every medium and large tonal zone from the guide map, including meaningful background zones; allow a small number of clear secondary zones when they improve readability; combine ONLY genuinely tiny regions into larger readable forms

STRICTLY FORBIDDEN:
- dense engraving
- crosshatching
- large fields of parallel hatching
- photographic texture tracing
- pores, tiny wrinkles, tiny hair fragments or micro-contours
- scratchy noise, dots, debris or broken accidental marks
- black shading fills or gray wash
- Canny / threshold / edge-detection appearance
- invented lines or duplicated objects
- copying blocky/pixelated guide-map artifacts
- gray duplicated outlines beside black outlines

DETAILED DOES NOT MEAN DENSE HATCHING.
It means: preserve the clean style while adding the medium and large posterized-zone boundaries that define facial planes, hair masses, clothing folds and major light/shadow shapes. Do not omit medium zones. You may include a few additional clear secondary zones, but do not trace micro-zones. Keep the drawing open, legible and suitable for tattoo transfer.
""".strip()

# Shared content rules for the two NEW clean modes.

DIRECT_CLEAN_BASE_PROMPT = """
EDIT the FIRST image into a cleaner version of the SAME tattoo transfer stencil.
The SECOND image is a fidelity check and the THIRD image is only a clean-line style reference.
Preserve exact subject geometry and composition. Remove information; do not redraw anatomy.
Use smooth continuous structural contours and sparse segmented tonal guides.
No hatching, no crosshatching, no texture fields, no micro-lines, no noise and no invented objects.
""".strip()

CLEAN_BASE_PROMPT = """
EDIT the FIRST image, which is the already-approved DETAILED STENCIL MASTER for this exact subject.
The SECOND image is the original source photo and is ONLY a fidelity check for identity/anatomy.
The THIRD image is ONLY a visual reference for the requested degree of cleanliness and line economy.

THIS IS A CLEANUP EDIT OF THE FIRST IMAGE, NOT A NEW REDRAW FROM SCRATCH.

ABSOLUTE GEOMETRY LOCK:
- Keep the exact same canvas framing, crop, subject scale, pose, orientation and composition as the FIRST image.
- Keep the same recognizable silhouette and the same positions/shapes of all major contours.
- Keep eyes, nose, mouth, ears, hands/paws, folds and other identity-defining landmarks in the same places.
- Do not redesign, restyle or reinterpret the anatomy.
- Do not move, widen, narrow, rotate or reshape the head/body/objects/background masses.
- Think of the operation as carefully erasing secondary ink from the detailed stencil while preserving its master drawing.

GLOBAL OUTPUT RULES:
- white background; clean black linework
- remove line density selectively while preserving structural continuity
- no edge-detection / Canny / threshold look
- no random scratches, chaotic short dashes, noisy micro-contours or disconnected debris
- keep only meaningful medium/large background shapes; no random background artifacts or tiny background texture
- no new decorative marks or invented objects
- the result must visibly be the SAME stencil drawing at a cleaner level of information
""".strip()

MODE_PROMPTS = {
    "simplified": """
MODE: SIMPLIFICADO.
- Preserve the same subject and all major structure.
- Keep major and medium posterized tonal boundaries only.
- Merge small neighboring tonal regions into larger readable shapes.
- Use far fewer internal lines than Detallado.
- Keep eyes, nose, mouth, hair mass, clothing edges and key folds clear.
- No crosshatching, no texture fields, no micro-lines, no dense hair-strand drawing.
- Large clean white spaces are required.
""".strip(),
    "clean": """
MODE: MUY LIMPIO.
- Preserve only the essential outer silhouette and indispensable internal landmarks.
- Keep only the largest posterized tonal divisions that materially help tattoo placement.
- Use very few broken tonal guides.
- No hatching, no crosshatching, no texture, no microdetail, no tonal fill.
- The stencil must be immediately readable from a distance and easy to transfer.
""".strip(),
}


def _detailed_control_prompt(
    preparation: str = "balanced",
    enhance: int = 50,
    silhouette: int = 70,
    detail: int = 75,
    shadows: int = 72,
    texture: int = 68,
    simplify: int = 0,
) -> str:
    return f"""
SOURCE PREPARATION:
- preparation preset: {preparation}
- source detail recovery: {enhance}/100

USER ADJUSTMENTS (0-100):
- main silhouette emphasis: {silhouette}
- internal structural detail: {detail}
- tonal-guide presence: {shadows}
- decorative texture: {texture}
- simplification: {simplify}

IMPORTANT INTERPRETATION:
- Detail means meaningful structural contours and useful posterized-zone guides, NOT hatching density.
- Shadow means the number/strength of broken tonal-boundary guides, NOT shaded fill.
- Texture must remain extremely restrained in every mode; never create dense hair/fabric/skin line fields.
- Preserve clean white space and transfer readability.
""".strip()


def _clean_mode_control_prompt(
    mode: str,
    preparation: str = "balanced",
    enhance: int = 50,
    silhouette: int = 70,
    detail: int = 75,
    shadows: int = 72,
    texture: int = 68,
    simplify: int = 0,
) -> str:
    if mode == "simplified":
        guardrail = (
            "The SIMPLIFIED mode definition has priority. Treat detail/shadow/texture sliders only as small fine-tuning; "
            "never return to dense engraved texture or dense hatching."
        )
    else:
        guardrail = (
            "The VERY CLEAN mode definition has absolute priority. Ignore requests that would reintroduce texture, "
            "crosshatching, tonal shading or many secondary lines."
        )
    return f"""
SOURCE PREPARATION:
- preparation preset: {preparation}
- source detail recovery: {enhance}/100
Recover enough source information to understand form, but do not translate photographic microtexture into stencil lines.

FINE ADJUSTMENTS (0-100):
- main silhouette emphasis: {silhouette}
- internal detail: {detail}
- shadow amount: {shadows}
- texture amount: {texture}
- simplification: {simplify}

{guardrail}
Never deform, relocate or change important geometry because a value changes.
""".strip()





POSTERIZED_GUIDE_SEMANTICS = """
GUIDE SEMANTICS:
- dark continuous guides = structural contours
- broken guides = tonal/value boundaries from the user's posterization
- preserve their position and meaning
- simplify tiny regions rather than tracing every small fragment
- NEVER convert guide lines into hatching, scratches, fur texture or engraving
""".strip()
TONAL_SEGMENTATION_RULES = """
TONAL GUIDE RULES:
- Broken guides are functional tattoo value-zone markers, not hatching.
- Keep them thin, sparse and clean.
- Omit tiny closed loops and very short isolated segments.
- DETAILED: EVERY retained medium or large tonal boundary in the guide map is mandatory. This includes facial plane zones on cheeks, forehead, eyelids, under-eyes, nose, chin, neck, fabric and meaningful background masses. Omit only small isolated micro-zones.
- SIMPLIFIED: retain major tonal zones only.
- VERY CLEAN: retain only the principal shadow/light separation.
- Never fill tonal regions with lines.
""".strip()



POSTERIZATION_REGION_LOCK = """
POSTERIZATION REGION LOCK:
The FIRST image contains the user's approved posterized design and its guide geometry.
Preserve ALL medium and large guide boundaries. In Detailed mode, none of those retained boundaries may be omitted, even when they lie inside smooth skin areas such as cheeks, forehead or eyelids. DO NOT preserve tiny isolated islands or micro-regions.
If several tiny adjacent regions would create clutter, merge them into the nearest larger tonal region. Medium regions must not be merged away.
Continuous guide geometry represents structural form. Broken guide geometry represents tonal-zone changes.
Never add a boundary that is not supported by the FIRST image.
""".strip()


POSTERIZATION_MASTER_PROMPT = """
EDIT the FIRST image into a polished posterized / cel-shaded illustration while preserving the exact subject.
The SECOND image is ONLY a STYLE REFERENCE for the desired posterized finish.

ABSOLUTE FIDELITY:
- Preserve the exact same subject, identity, facial geometry, expression, gaze, pose, crop, camera angle, clothing, objects and composition from the FIRST image.
- Preserve the source color family: skin stays skin-toned, hair stays its source color, clothing keeps its source colors, background stays consistent.
- Do not add, remove, duplicate or relocate any object, symbol, badge, jewelry, flag, text or background element.

POSTERIZATION STYLE:
- clean professional cel-shaded / illustrated posterization
- broad, coherent shapes for highlights, light, midtones, shadows and deep shadows
- smooth, intentional color regions with enough nuance to preserve likeness and volume
- crisp but natural shape transitions; no dirty speckles, pepper noise, muddy blotches or accidental tiny islands
- retain selective real detail in eyes, brows, lips, hair direction and important material folds
- rich, faithful color; do not turn skin/hair into muddy brown or gray
- no black contour drawing over the image; this is a color posterization, not a stencil
- no text, watermark, decorative graphics or invented objects

The result should look like a high-end editorial poster illustration derived directly from the source photograph, with clean tonal masses and faithful colors.
""".strip()


async def create_posterization_master(image_path: Path) -> PosterizationMasterResult:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    reference = REFERENCE_DIR / "posterization-reference.png"
    if not reference.exists():
        raise RuntimeError("Posterization style reference is missing")

    model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2.5-sunburst")
    quality = os.getenv("OPENAI_POSTERIZE_QUALITY", os.getenv("OPENAI_IMAGE_QUALITY", "high"))
    size = os.getenv("OPENAI_POSTERIZE_SIZE", "auto")
    timeout_s = float(os.getenv("OPENAI_IMAGE_TIMEOUT", "300"))

    source_bytes = image_path.read_bytes()
    ref_bytes = reference.read_bytes()
    source_mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".png": "image/png",
    }.get(image_path.suffix.lower(), "image/png")

    files = [
        ("image[]", (image_path.name, source_bytes, source_mime)),
        ("image[]", (reference.name, ref_bytes, "image/png")),
    ]
    data = {
        "model": model,
        "prompt": POSTERIZATION_MASTER_PROMPT,
        "quality": quality,
        "size": size,
        "output_format": "png",
        "background": "opaque",
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                "https://api.openai.com/v1/images/edits",
                headers=headers,
                data=data,
                files=files,
            )
    except httpx.TimeoutException as exc:
        raise RuntimeError("La posterización tardó demasiado. Inténtalo nuevamente.") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"No se pudo conectar con el generador de imágenes: {exc}") from exc

    if response.status_code >= 400:
        try:
            payload = response.json()
            err = payload.get("error", {}) or {}
            msg = err.get("message") or str(payload)
            code = err.get("code") or err.get("type") or "unknown"
        except Exception:
            msg = response.text[:1000]
            code = "unknown"
        raise RuntimeError(f"OpenAI image error {response.status_code} [{code}]: {msg}")

    payload = response.json()
    items = payload.get("data") or []
    if not items or not items[0].get("b64_json"):
        raise RuntimeError("Image provider returned no posterized image")

    job_id = uuid.uuid4().hex
    job_dir = POSTER_MASTERS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    target = job_dir / "master.png"
    target.write_bytes(base64.b64decode(items[0]["b64_json"]))

    # Force exact source dimensions so the stencil map and final download keep the same canvas.
    with Image.open(image_path) as src_im:
        sw, sh = src_im.size
    with Image.open(target).convert("RGB") as out_im:
        if out_im.size != (sw, sh):
            out_im = out_im.resize((sw, sh), Image.Resampling.LANCZOS)
            out_im.save(target, "PNG", optimize=True)
    return PosterizationMasterResult(job_id=job_id, posterized_png=target, model=model, width=sw, height=sh)


ENHANCE_BASE_PROMPT = """
RESTORE AND UPSCALE the FIRST image as a high-quality photographic restoration.
This is an enhancement task, NOT a creative redesign.

ABSOLUTE PRESERVATION RULES:
- Preserve the exact same person/subject, identity, facial geometry, expression, gaze, pose, anatomy and proportions.
- Preserve the exact same crop, framing, camera angle, clothing, veil/fabric arrangement, hands, objects and background layout.
- Preserve the source colors and overall lighting character unless correcting obvious color cast.
- Do not beautify, age-shift, change ethnicity, change facial features, change hairstyle, change clothing, or invent new objects.

QUALITY GOAL:
- Recover clean high-resolution detail from blur, low resolution and JPEG/compression artifacts.
- Improve the clarity of eyes, eyelashes, eyebrows, lips, hair, fabric weave, folds and other real details that are supported by the source.
- Improve local contrast and tonal separation while keeping natural skin and realistic photographic texture.
- Reduce blur, ringing, blockiness and noise without plastic skin or oversharpening halos.
- Produce a natural, photorealistic, high-resolution version that looks like the same original photograph captured with a much better camera.

DO NOT stylize, illustrate, repaint, add dramatic makeup, add jewelry, change the background, or alter composition.
The source image is ground truth. When uncertain, preserve rather than invent.
""".strip()


def _enhance_control_prompt(preparation: str, enhance: int) -> str:
    strength = {
        "soft": "gentle restoration; prioritize exact fidelity and subtle cleanup",
        "balanced": "balanced restoration; recover clear detail while preserving exact identity",
        "strong": "strong restoration; recover as much plausible source detail as possible without redesigning the subject",
    }.get(preparation, "balanced restoration")
    return f"""
RESTORATION PRESET: {preparation} — {strength}.
DETAIL RECOVERY: {enhance}/100.
Higher values may recover more fine detail, but must never change facial geometry, pose, clothing, or composition.
Keep edges natural and photographic; avoid crunchy sharpening and fake texture.
""".strip()


def _snap16(v: int) -> int:
    return max(16, int(round(v / 16.0)) * 16)


def _enhance_output_size(image_path: Path) -> str:
    """Return a robust output size for restoration.

    GPT Image 2.5 supports custom sizes, but for an end-user upload flow the
    safest default is ``auto`` so unusual source aspect ratios do not make the
    enhancement request fail. Advanced users can override this with
    OPENAI_ENHANCE_SIZE (for example 1024x1536 or 2048x2048).
    """
    configured = os.getenv("OPENAI_ENHANCE_SIZE", "auto").strip()
    return configured or "auto"


async def enhance_photo(
    image_path: Path,
    *,
    preparation: str = "balanced",
    enhance: int = 50,
) -> PhotoEnhancementResult:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2.5-sunburst")
    quality = os.getenv("OPENAI_ENHANCE_QUALITY", "high")
    size = _enhance_output_size(image_path)
    timeout_s = float(os.getenv("OPENAI_IMAGE_TIMEOUT", "300"))

    source_bytes = image_path.read_bytes()
    source_mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".png": "image/png",
    }.get(image_path.suffix.lower(), "image/png")

    prompt = ENHANCE_BASE_PROMPT + "\n\n" + _enhance_control_prompt(preparation, enhance)
    data = {
        "model": model,
        "prompt": prompt,
        "quality": quality,
        "size": size,
        "output_format": "png",
        "background": "opaque",
    }
    files = [("image[]", (image_path.name, source_bytes, source_mime))]
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                "https://api.openai.com/v1/images/edits",
                headers=headers,
                data=data,
                files=files,
            )
    except httpx.TimeoutException as exc:
        raise RuntimeError("La mejora tardó demasiado y agotó el tiempo de espera. Inténtalo nuevamente.") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"No se pudo conectar con el generador de imágenes: {exc}") from exc

    if response.status_code >= 400:
        try:
            payload = response.json()
            err = payload.get("error", {}) or {}
            msg = err.get("message") or str(payload)
            code = err.get("code") or err.get("type") or "unknown"
        except Exception:
            msg = response.text[:1000]
            code = "unknown"
        raise RuntimeError(f"OpenAI image error {response.status_code} [{code}]: {msg}")

    payload = response.json()
    items = payload.get("data") or []
    if not items or not items[0].get("b64_json"):
        raise RuntimeError("Image provider returned no enhanced image")

    job_id = uuid.uuid4().hex
    job_dir = ENHANCED / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    target = job_dir / "enhanced.png"
    target.write_bytes(base64.b64decode(items[0]["b64_json"]))
    with Image.open(target) as im:
        width, height = im.size
    return PhotoEnhancementResult(job_id=job_id, enhanced_png=target, model=model, width=width, height=height)


def _reference_for_mode(mode: str) -> Path:
    if mode not in REFERENCES:
        raise RuntimeError(f"Unsupported stencil mode: {mode}")
    ref = REFERENCES[mode]
    if not ref.exists():
        raise RuntimeError(f"Style reference for mode '{mode}' is missing")
    return ref


def provider_status() -> dict[str, Any]:
    return {
        "provider": "openai",
        "configured": bool(os.getenv("OPENAI_API_KEY")),
        "model": os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2.5-sunburst"),
        "references": {mode: path.exists() for mode, path in REFERENCES.items()},
        "reference_exists": all(path.exists() for path in REFERENCES.values()),
    }



def _save_mask_rgba(mask: np.ndarray, target: Path, value: int = 0, alpha_value: int = 255) -> None:
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = value
    rgba[..., 3] = (mask.astype(np.uint8) * alpha_value)
    Image.fromarray(rgba, "RGBA").save(target, "PNG", optimize=True)


def _save_empty_rgba(shape: tuple[int, int], target: Path) -> None:
    h, w = shape
    Image.fromarray(np.zeros((h, w, 4), dtype=np.uint8), "RGBA").save(target, "PNG", optimize=True)


def _generate_local_stencil_from_guides(
    image_path: Path,
    guide_path: Path,
    mode: str,
) -> StencilGenerationResult:
    """Render the exact approved posterized guide map as the stencil.

    Fallback renderer used only when OpenAI is unavailable or STENCIL_FORCE_LOCAL=1.
    Continuous black guides remain continuous; gray broken guides remain tonal
    segmentation guides. This prevents invented objects and guarantees that every
    guide shown in the posterization step survives.
    """
    with Image.open(guide_path).convert("RGBA") as im:
        guide = np.array(im)

    h, w = guide.shape[:2]
    rgb = guide[:, :, :3]
    alpha = guide[:, :, 3]
    maxc = rgb.max(axis=2)

    solid = ((alpha >= 180) & (maxc <= 20)).astype(np.uint8)
    dashed = ((alpha >= 180) & (maxc > 20) & (maxc <= 110)).astype(np.uint8)

    # Keep guide geometry exact. Only a tiny close pass reconnects 1px interruptions.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, k)

    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    canvas[dashed > 0] = (72, 72, 72)
    canvas[solid > 0] = (0, 0, 0)

    job_id = uuid.uuid4().hex
    job_dir = GENERATED / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    full_path = job_dir / "full.png"
    Image.fromarray(canvas, "RGB").save(full_path, "PNG", optimize=True)

    # Build deterministic layers directly from the semantic guide masks instead of
    # re-classifying the finished line art. This keeps the on-screen composite identical
    # to the full stencil and avoids losing tonal-zone boundaries.
    silhouette_path = job_dir / "silhouette.png"
    essential_path = job_dir / "essential.png"
    structure_path = job_dir / "structure.png"
    segmentation_path = job_dir / "segmentation.png"
    detail_path = job_dir / "detail.png"
    shade_path = job_dir / "shade.png"
    texture_path = job_dir / "texture.png"

    _save_mask_rgba(solid, silhouette_path, value=0, alpha_value=255)
    _save_empty_rgba((h, w), essential_path)
    _save_empty_rgba((h, w), structure_path)
    _save_mask_rgba(dashed, segmentation_path, value=72, alpha_value=255)
    _save_empty_rgba((h, w), detail_path)
    _save_empty_rgba((h, w), shade_path)
    _save_empty_rgba((h, w), texture_path)

    return StencilGenerationResult(
        job_id=job_id,
        full_png=full_path,
        silhouette_png=silhouette_path,
        essential_png=essential_path,
        structure_png=structure_path,
        segmentation_png=segmentation_path,
        detail_png=detail_path,
        shade_png=shade_path,
        texture_png=texture_path,
        model="local-guides",
        mode=mode,
    )

def _derive_clean_mode_from_master(base_stencil_path: Path, mode: str) -> StencilGenerationResult:
    """Derive Simplificado/Muy limpio locally from the paid Detailed master.

    This avoids a second OpenAI image charge when the user switches stencil modes.
    """
    job_id = uuid.uuid4().hex
    job_dir = GENERATED / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    layers = split_stencil_into_stable_layers(base_stencil_path, job_dir)

    with Image.open(base_stencil_path).convert("RGB") as im:
        w, h = im.size
    canvas = Image.new("RGB", (w, h), "white")

    if mode == "simplified":
        keep = [layers.silhouette, layers.structure, layers.segmentation]
    else:
        keep = [layers.silhouette, layers.essential, layers.segmentation]

    for lp in keep:
        with Image.open(lp).convert("RGBA") as layer:
            canvas = Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")

    full_path = job_dir / "full.png"
    canvas.save(full_path, "PNG", optimize=True)

    # Zero out secondary layers so frontend sliders cannot reintroduce dense ink.
    if mode in {"simplified", "clean"}:
        for lp in (layers.detail, layers.shade, layers.texture):
            _save_empty_rgba((h, w), lp)
    if mode == "clean":
        _save_empty_rgba((h, w), layers.structure)

    return StencilGenerationResult(
        job_id=job_id,
        full_png=full_path,
        silhouette_png=layers.silhouette,
        essential_png=layers.essential,
        structure_png=layers.structure,
        segmentation_png=layers.segmentation,
        detail_png=layers.detail,
        shade_png=layers.shade,
        texture_png=layers.texture,
        model="local-cleanup",
        mode=mode,
    )


def _overlay_mandatory_guides(full_path: Path, guide_path: Path | None, mode: str) -> None:
    """Restore missing posterization tonal zones as clean tattoo guide lines.

    The semantic guide map stores structural lines in black and tonal boundaries as
    CONTINUOUS gray curves. We redraw only the retained tonal curves here as clean,
    anti-aliased black dashes. This guarantees that medium/large posterized regions
    (cheeks, forehead, eyelids, neck, fabric, background masses, etc.) survive the
    artistic AI pass without pasting the old blocky/pixelated guide map over the result.
    """
    if guide_path is None or not guide_path.exists() or not full_path.exists():
        return

    with Image.open(full_path).convert("RGB") as base_im, Image.open(guide_path).convert("RGBA") as guide_im:
        base = np.array(base_im)
        bh, bw = base.shape[:2]
        g = np.array(guide_im)
        rgb = g[..., :3]
        alpha = g[..., 3]
        maxc = rgb.max(axis=2)

        # Tonal boundaries are encoded as medium gray in the semantic guide file.
        tonal_src = ((alpha >= 150) & (maxc >= 35) & (maxc <= 150)).astype(np.uint8)
        if not tonal_src.any():
            return

        # Resize the semantic mask smoothly to final AI output size. Do not resize the
        # finished dashed pixels; dashes are created AFTER resizing to avoid blockiness.
        tonal_f = cv2.resize(tonal_src.astype(np.float32), (bw, bh), interpolation=cv2.INTER_LINEAR)
        tonal = (tonal_f >= 0.28).astype(np.uint8)
        tonal = cv2.morphologyEx(tonal, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

        # Keep useful medium/large line runs. Tiny regions were already merged at the
        # posterization stage, so Detailed uses a deliberately permissive line filter.
        total = bh * bw
        min_area, min_span = {
            "detailed": (max(5, int(total * 0.000003)), max(10, int(max(bh, bw) * 0.007))),
            "simplified": (max(14, int(total * 0.000010)), max(18, int(max(bh, bw) * 0.012))),
            "clean": (max(24, int(total * 0.000022)), max(26, int(max(bh, bw) * 0.018))),
        }[mode]

        n, cc, stats, _ = cv2.connectedComponentsWithStats(tonal, 8)
        kept = np.zeros_like(tonal, np.uint8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            span = max(int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
            if area >= min_area or span >= min_span:
                kept[cc == i] = 1
        if not kept.any():
            return

        # Create clean dash cadence at final resolution. More detailed mode has shorter
        # gaps so medium facial zones remain easy to read, but these are still guides,
        # not hatching or shading.
        maxdim = max(bh, bw)
        period = {
            "detailed": max(11, int(round(maxdim * 0.0090))),
            "simplified": max(14, int(round(maxdim * 0.0115))),
            "clean": max(18, int(round(maxdim * 0.0140))),
        }[mode]
        on_len = {
            "detailed": max(7, int(round(period * 0.68))),
            "simplified": max(8, int(round(period * 0.58))),
            "clean": max(9, int(round(period * 0.50))),
        }[mode]
        yy, xx = np.indices((bh, bw))
        dash_pattern = ((xx + (yy // 2)) % period) < on_len
        dashed = (kept > 0) & dash_pattern

        # Add only guide portions that are genuinely MISSING from the artistic stencil.
        # This prevents duplicated outlines and keeps already-detailed hair/fabric areas
        # from becoming crowded, while open areas such as cheeks, forehead and eyelids
        # receive the missing tonal-zone guides.
        base_gray = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
        existing_ink = (base_gray < 205).astype(np.uint8)
        near_existing = cv2.dilate(existing_ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
        local_density = cv2.GaussianBlur(existing_ink.astype(np.float32), (0, 0), sigmaX=7.0, sigmaY=7.0)
        guide_density = cv2.GaussianBlur(kept.astype(np.float32), (0, 0), sigmaX=7.0, sigmaY=7.0)
        density_limit = {"detailed": 0.105, "simplified": 0.085, "clean": 0.065}[mode]
        guide_density_limit = {"detailed": 0.155, "simplified": 0.130, "clean": 0.110}[mode]
        missing = dashed & (near_existing == 0) & (local_density < density_limit) & (guide_density < guide_density_limit)

        # Remove tiny isolated leftover pieces created by the missing-line comparison.
        missing_u8 = missing.astype(np.uint8)
        n2, cc2, stats2, _ = cv2.connectedComponentsWithStats(missing_u8, 8)
        clean_missing = np.zeros_like(missing_u8)
        final_min_span = {"detailed": 7, "simplified": 10, "clean": 13}[mode]
        for i in range(1, n2):
            area = int(stats2[i, cv2.CC_STAT_AREA])
            span = max(int(stats2[i, cv2.CC_STAT_WIDTH]), int(stats2[i, cv2.CC_STAT_HEIGHT]))
            if area >= 3 or span >= final_min_span:
                clean_missing[cc2 == i] = 1

        if not clean_missing.any():
            return

        # Anti-aliased black line overlay. Core pixels stay black; only edge pixels are
        # softened. This avoids the gray/pixel-block artifacts from older versions.
        core = clean_missing.astype(np.float32)
        edge = cv2.GaussianBlur(core, (0, 0), sigmaX=0.55, sigmaY=0.55)
        coverage = np.maximum(core, edge * 0.72)
        coverage = np.clip(coverage, 0.0, 1.0)[..., None]
        out = np.clip(base.astype(np.float32) * (1.0 - coverage), 0, 255).astype(np.uint8)
        Image.fromarray(out, "RGB").save(full_path, "PNG", optimize=True)

def _normalize_final_stencil(full_path: Path) -> None:
    """Clean faint gray debris while preserving smooth black tattoo linework."""
    if not full_path.exists():
        return
    with Image.open(full_path).convert("L") as im:
        gray = np.array(im)

    # Remove near-white haze and push real linework toward black without hard jagged thresholding.
    g = gray.astype(np.float32)
    out = np.full_like(g, 255.0)
    strong = g <= 205
    transition = (g > 205) & (g < 242)
    out[strong] = np.clip((g[strong] - 35.0) * 0.82, 0, 190)
    out[transition] = 215 + (g[transition] - 205) / 37.0 * 40.0
    cleaned = np.clip(out, 0, 255).astype(np.uint8)

    ink = (cleaned < 235).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros_like(ink)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        span = max(int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        if area >= 5 or span >= 7:
            keep[cc == i] = 1
    cleaned[keep == 0] = 255
    Image.fromarray(cleaned, "L").convert("RGB").save(full_path, "PNG", optimize=True)


async def generate_stencil(
    image_path: Path,
    *,
    mode: str = "detailed",
    base_stencil_path: Path | None = None,
    mandatory_guide_path: Path | None = None,
    preparation: str = "balanced",
    enhance: int = 50,
    silhouette: int = 70,
    detail: int = 75,
    shadows: int = 72,
    texture: int = 68,
    simplify: int = 0,
 ) -> StencilGenerationResult:
    if mode in {"simplified", "clean"} and base_stencil_path is not None and base_stencil_path.exists():
        return _derive_clean_mode_from_master(base_stencil_path, mode)

    api_key = os.getenv("OPENAI_API_KEY")
    force_local_guides = os.getenv("STENCIL_FORCE_LOCAL", "0").strip().lower() in {"1", "true", "yes", "on"}
    if mandatory_guide_path is not None and mandatory_guide_path.exists() and (force_local_guides or not api_key):
        return _generate_local_stencil_from_guides(image_path, mandatory_guide_path, mode)

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    reference = _reference_for_mode(mode)

    model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2.5-sunburst")
    quality = os.getenv("OPENAI_IMAGE_QUALITY", "high")
    size = os.getenv("OPENAI_IMAGE_SIZE", "1024x1536")
    timeout_s = float(os.getenv("OPENAI_IMAGE_TIMEOUT", "180"))

    if mode == "detailed":
        # Preserve the v5 prompt behavior exactly for Detailed.
        prompt = "\n\n".join([
            DETAILED_BASE_PROMPT,
            POSTERIZATION_REGION_LOCK,
            POSTERIZED_GUIDE_SEMANTICS,
            TONAL_SEGMENTATION_RULES,
            _detailed_control_prompt(
                preparation=preparation,
                enhance=enhance,
                silhouette=silhouette,
                detail=detail,
                shadows=shadows,
                texture=texture,
                simplify=simplify,
            ),
        ])
    else:
        has_master = base_stencil_path is not None and base_stencil_path.exists()
        mode_prompt = MODE_PROMPTS[mode]
        if not has_master:
            mode_prompt = (
                mode_prompt
                .replace("CLEAN THE EXISTING MASTER", "REDRAW THE SOURCE")
                .replace("DETAILED MASTER'S", "SOURCE IMAGE'S")
                .replace("same stencil drawing", "same subject")
            )
        prompt = "\n\n".join([
            CLEAN_BASE_PROMPT if has_master else DIRECT_CLEAN_BASE_PROMPT,
            mode_prompt,
            POSTERIZATION_REGION_LOCK,
            POSTERIZED_GUIDE_SEMANTICS,
            TONAL_SEGMENTATION_RULES,
            _clean_mode_control_prompt(
                mode=mode,
                preparation=preparation,
                enhance=enhance,
                silhouette=silhouette,
                detail=detail,
                shadows=shadows,
                texture=texture,
                simplify=simplify,
            ),
        ])

    source_bytes = image_path.read_bytes()
    ref_bytes = reference.read_bytes()
    source_mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".png": "image/png",
    }.get(image_path.suffix.lower(), "image/png")

    if mode == "detailed":
        files = [("image[]", (image_path.name, source_bytes, source_mime))]
        if mandatory_guide_path is not None and mandatory_guide_path.exists():
            files.append(("image[]", ("approved-guide-map.png", mandatory_guide_path.read_bytes(), "image/png")))
        files.append(("image[]", (reference.name, ref_bytes, "image/png")))
    else:
        if base_stencil_path is not None and base_stencil_path.exists():
            master_bytes = base_stencil_path.read_bytes()
            files = [
                ("image[]", ("detailed-master.png", master_bytes, "image/png")),
                ("image[]", (image_path.name, source_bytes, source_mime)),
                ("image[]", (reference.name, ref_bytes, "image/png")),
            ]
        else:
            files = [("image[]", (image_path.name, source_bytes, source_mime))]
            if mandatory_guide_path is not None and mandatory_guide_path.exists():
                files.append(("image[]", ("approved-guide-map.png", mandatory_guide_path.read_bytes(), "image/png")))
            files.append(("image[]", (reference.name, ref_bytes, "image/png")))
        # Keep the requested canvas whenever possible.
        try:
            from PIL import Image
            size_source = base_stencil_path if (base_stencil_path is not None and base_stencil_path.exists()) else image_path
            with Image.open(size_source) as master_im:
                mw, mh = master_im.size
            if mw % 16 == 0 and mh % 16 == 0 and mw <= 3840 and mh <= 3840:
                size = f"{mw}x{mh}"
        except Exception:
            pass

    data = {
        "model": model,
        "prompt": prompt,
        "quality": quality,
        "size": size,
        "output_format": "png",
        "background": "opaque",
    }

    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.post(
            "https://api.openai.com/v1/images/edits",
            headers=headers,
            data=data,
            files=files,
        )

    if response.status_code >= 400:
        try:
            payload = response.json()
            err = payload.get("error", {}) or {}
            msg = err.get("message") or str(payload)
            code = err.get("code") or err.get("type") or "unknown"
        except Exception:
            msg = response.text[:1000]
            code = "unknown"
        raise RuntimeError(f"OpenAI image error {response.status_code} [{code}]: {msg}")

    payload = response.json()
    items = payload.get("data") or []
    if not items or not items[0].get("b64_json"):
        raise RuntimeError("Image provider returned no image")

    job_id = uuid.uuid4().hex
    job_dir = GENERATED / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    full_path = job_dir / "full.png"
    full_path.write_bytes(base64.b64decode(items[0]["b64_json"]))

    # Normalize the artistic result first, then restore ONLY clean semantic tonal guides.
    # The raw guide bitmap is never pasted over the result; _overlay_mandatory_guides
    # rebuilds anti-aliased dashed lines at final resolution so medium/large posterized
    # zones cannot disappear while the stencil stays clean.
    _normalize_final_stencil(full_path)
    _overlay_mandatory_guides(full_path, mandatory_guide_path, mode)

    layers = split_stencil_into_stable_layers(full_path, job_dir)

    return StencilGenerationResult(
        job_id=job_id,
        full_png=full_path,
        silhouette_png=layers.silhouette,
        essential_png=layers.essential,
        structure_png=layers.structure,
        segmentation_png=layers.segmentation,
        detail_png=layers.detail,
        shade_png=layers.shade,
        texture_png=layers.texture,
        model=model,
        mode=mode,
    )
