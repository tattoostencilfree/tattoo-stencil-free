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


ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "backend" / "generated"
ENHANCED = ROOT / "backend" / "enhanced"
REFERENCE_DIR = ROOT / "backend" / "reference"
REFERENCES = {
    "detailed": REFERENCE_DIR / "golden-stencil.png",
    "simplified": REFERENCE_DIR / "simplified-reference.png",
    "clean": REFERENCE_DIR / "clean-reference.png",
}
GENERATED.mkdir(parents=True, exist_ok=True)
ENHANCED.mkdir(parents=True, exist_ok=True)

# EXACT detailed base prompt from v5. Do not change without deliberately changing
# the already-approved Detailed mode.
DETAILED_BASE_PROMPT = """
EDIT the FIRST image (the user's source photo) into a refined black-and-white artistic tattoo stencil.
The SECOND image is ONLY a visual style reference for the desired line hierarchy, engraving language,
cleanliness, hatching quality and professional finish. Do not copy the second image's person, pose,
facial features, clothing or ornaments into the source image.

Preserve the source subject faithfully: identity/likeness when a person is present, proportions, pose,
expression, gaze, anatomy, important objects and composition. Keep distinctive features recognizable.

STYLE REQUIREMENTS:
- clean white background; black and dark-gray linework only
- strong, elegant outer silhouette and major boundaries, slightly heavier than inner lines
- thin, controlled structural contours for eyes, nose, lips, hands, folds and other important forms
- shadows and midtones translated into refined flowing hatching / segmented engraving lines
- hatching follows the volume and curvature of the subject instead of random directions
- textures are simplified artistically rather than traced photographically
- generous white space; the drawing must breathe and remain readable as a stencil
- preserve high-value details while avoiding clutter

STRICTLY AVOID:
- edge-detection / Canny / threshold appearance
- random scratches, chaotic short dashes, noisy micro-contours or photographic texture tracing
- large uncontrolled black blobs
- background artifacts
- facial/anatomical distortion
- changing the source pose or inventing unrelated decorations

The final result should look hand-designed by a highly skilled tattoo stencil illustrator: polished,
clean, hierarchical, readable and suitable for tattoo transfer/reference.
""".strip()

# Shared content rules for the two NEW clean modes.
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
- Do not move, widen, narrow, rotate or reshape the head/body/objects.
- Think of the operation as carefully erasing secondary ink from the detailed stencil while preserving its master drawing.

GLOBAL OUTPUT RULES:
- white background; clean black linework
- remove line density selectively while preserving structural continuity
- no edge-detection / Canny / threshold look
- no random scratches, chaotic short dashes, noisy micro-contours or disconnected debris
- no background artifacts
- no new decorative marks or invented objects
- the result must visibly be the SAME stencil drawing at a cleaner level of information
""".strip()

MODE_PROMPTS = {
    "simplified": """
MODE: SIMPLIFIED — CLEAN INTERMEDIATE REDRAW.
The target is like the simplified example: recognizably the same subject, but with much fewer lines and much cleaner open areas.

REQUIRED BEHAVIOR:
- Preserve the complete recognizable silhouette and the important internal forms.
- CLEAN THE EXISTING MASTER with fewer, smoother, more deliberate lines. Remove secondary line families coherently, never by random erasure.
- Keep enough selected contour and form lines to preserve volume and character.
- Keep eyes, nose, mouth, ears, hands/paws and other identity-defining features crisp and recognizable.
- Reduce repetitive texture strongly: do NOT draw every hair, fur strand, fabric fiber, pore, tiny wrinkle or tiny ornament.
- Use only selected directional texture strokes where they help explain form.
- Reduce hatching substantially. Use sparse, controlled shading accents only in important shadow/form zones.
- Keep selected clean broken tonal-boundary guides so the tattoo artist can still see the main shadow zones.
- Leave broad, clean white spaces between line groups.
- Avoid broken-looking remnants, accidental gaps, scattered dots or half-erased hatching.

The final result must look like a PURPOSE-BUILT clean intermediate stencil: elegant, clear, moderately detailed and easy to transfer.
""".strip(),
    "clean": """
MODE: VERY CLEAN — ESSENTIAL PROFESSIONAL OUTLINE REDRAW.
The target is like the very-clean example: the same subject reduced to its essential silhouette and indispensable internal landmarks.

REQUIRED BEHAVIOR:
- Preserve the DETAILED MASTER'S exact recognizable silhouette, proportions, pose and identity/species at the same pixel locations.
- Use bold, smooth major contours and only a very small number of essential internal structural lines.
- Keep only key lines for eyes, nose, mouth, ears, major folds/joints and other indispensable landmarks.
- NO fur/hair strand texture.
- NO fabric/material texture.
- NO decorative microdetail.
- NO crosshatching and NO tonal fill.
- Keep only the essential BROKEN TONAL SEGMENTATION GUIDES required by TONAL_SEGMENTATION_RULES; these are guide boundaries, not shading fill.
- Almost no shading; use sparse form accents plus the essential tonal-boundary guides.
- Require large clean white interior areas.
- Remove short dashes, duplicate contours, secondary wrinkles and small marks.
- Do not turn the source into a generic cartoon or icon: likeness/anatomy must stay faithful even with very few lines.

The final result must read instantly as a clean stencil and be very easy to transfer.
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
    # EXACT v5 control language for the approved detailed behavior.
    return f"""
SOURCE PREPARATION:
- preparation preset: {preparation}
- source detail recovery: {enhance}/100
Recover useful source detail before translating it into linework, but do not create oversharpening halos or photographic micro-noise.

USER ADJUSTMENTS (0-100):
- main silhouette emphasis: {silhouette}
- internal detail: {detail}
- hatching/shadow density: {shadows}
- decorative/material texture: {texture}
- simplification: {simplify}

Interpret these conservatively. Higher simplification removes secondary information while preserving
all essential contours and likeness. Do not deform or relocate main lines just because a value changes.

SIMPLIFICATION LANGUAGE:
- If simplification is below 30: keep the existing highly detailed engraved stencil language.
- If simplification is 30-79: create a CLEAN SIMPLIFIED stencil. Keep the outer silhouette and important
  structural contours, greatly reduce hatching and material texture, remove most short secondary marks,
  and leave broad clean white areas. Lines should be deliberate, smooth and easy to transfer.
- If simplification is 80 or higher: create a VERY CLEAN stencil. Keep ONLY the strong outer silhouette
  plus a very small number of essential internal form lines. No texture, no crosshatching, no tonal fill,
  no micro-lines, and almost no shading. The result should read like a clean professional outline drawing.
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



TONAL_SEGMENTATION_RULES = """
MANDATORY TONAL SEGMENTATION GUIDES — REQUIRED IN EVERY STENCIL MODE:
- Derive these guides from the tonal structure of the CURRENT USER IMAGE, never from the style reference.
- Add clean broken/dashed guide contours that mark meaningful transitions between highlight, midtone, shadow and deep shadow zones.
- These are NOT random hatching strokes and NOT texture. They are deliberate tonal-boundary guides for a tattoo artist.
- Guide lines must follow the subject's real form and curvature: face planes, muscles, folds, fur masses, fabric volumes and object surfaces.
- Keep them visually distinct from the main silhouette: thinner, lighter and usually broken/segmented rather than solid.
- Do not outline every tiny gradient. Mark only useful tonal regions that help the artist understand where one value zone changes into another.
- Never place segmentation guides across key facial features or important contours in a way that obscures them.
- Preserve clean white space between tonal zones.

MODE DENSITY:
- DETAILED: include a rich but orderly set of tonal segmentation guides for major and secondary value changes.
- SIMPLIFIED: include only major tonal-zone boundaries; fewer, longer, cleaner segmented guides.
- VERY CLEAN: include only the most essential tonal separation guides needed to mark the principal shadow masses.

The segmentation guides are part of the functional tattoo stencil and must remain present in the final output.
""".strip()


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
            msg = payload.get("error", {}).get("message") or str(payload)
        except Exception:
            msg = response.text[:1000]
        raise RuntimeError(f"Image provider error ({response.status_code}): {msg}")

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


async def generate_stencil(
    image_path: Path,
    *,
    mode: str = "detailed",
    base_stencil_path: Path | None = None,
    preparation: str = "balanced",
    enhance: int = 50,
    silhouette: int = 70,
    detail: int = 75,
    shadows: int = 72,
    texture: int = 68,
    simplify: int = 0,
) -> StencilGenerationResult:
    api_key = os.getenv("OPENAI_API_KEY")
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
        prompt = "\n\n".join([
            CLEAN_BASE_PROMPT,
            MODE_PROMPTS[mode],
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
        # Detailed remains exactly the approved source-photo -> stencil workflow.
        files = [
            ("image[]", (image_path.name, source_bytes, source_mime)),
            ("image[]", (reference.name, ref_bytes, "image/png")),
        ]
    else:
        if base_stencil_path is None or not base_stencil_path.exists():
            raise RuntimeError("Simplified/Clean requires the approved Detailed stencil master")
        master_bytes = base_stencil_path.read_bytes()
        # IMPORTANT ORDER:
        # 1) exact detailed master to EDIT, 2) original photo for fidelity checking,
        # 3) style reference for line economy only.
        files = [
            ("image[]", ("detailed-master.png", master_bytes, "image/png")),
            ("image[]", (image_path.name, source_bytes, source_mime)),
            ("image[]", (reference.name, ref_bytes, "image/png")),
        ]
        # Keep the same canvas as the detailed master whenever possible.
        try:
            from PIL import Image
            with Image.open(base_stencil_path) as master_im:
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
            msg = payload.get("error", {}).get("message") or str(payload)
        except Exception:
            msg = response.text[:1000]
        raise RuntimeError(f"Image provider error ({response.status_code}): {msg}")

    payload = response.json()
    items = payload.get("data") or []
    if not items or not items[0].get("b64_json"):
        raise RuntimeError("Image provider returned no image")

    job_id = uuid.uuid4().hex
    job_dir = GENERATED / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    full_path = job_dir / "full.png"
    full_path.write_bytes(base64.b64decode(items[0]["b64_json"]))

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
