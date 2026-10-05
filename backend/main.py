from __future__ import annotations

from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .provider import generate_stencil, enhance_photo, provider_status, GENERATED, ENHANCED

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
UPLOADS = ROOT / "backend" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}

app = FastAPI(title="Tattoo Stencil Lab MVP")
app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")


@app.get("/")
def home():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/config")
def config():
    status = provider_status()
    # Never expose credentials.
    return {
        "engine": status,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
    }


async def _save_and_validate_upload(file: UploadFile) -> Path:
    ext = Path(file.filename or "image.png").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "Formato no soportado. Usa JPG, PNG o WEBP.")

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "La imagen supera el límite de 20 MB.")

    file_id = uuid.uuid4().hex
    dst = UPLOADS / f"{file_id}{ext}"
    dst.write_bytes(raw)

    try:
        with Image.open(dst) as im:
            im.verify()
    except Exception:
        dst.unlink(missing_ok=True)
        raise HTTPException(400, "El archivo no parece ser una imagen válida.")
    return dst


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    dst = await _save_and_validate_upload(file)
    return {"id": dst.stem, "name": file.filename}




@app.post("/api/enhance")
async def enhance(
    file: UploadFile = File(...),
    preparation: str = Form("balanced"),
    enhance: int = Form(50),
):
    if preparation not in {"soft", "balanced", "strong"}:
        raise HTTPException(400, "preparation no válido")
    if not 0 <= enhance <= 100:
        raise HTTPException(400, "enhance debe estar entre 0 y 100")

    status = provider_status()
    if not status["configured"]:
        return JSONResponse(
            status_code=503,
            content={
                "error": "generator_not_configured",
                "message": "Configura OPENAI_API_KEY en el servidor para activar la mejora real de imagen.",
            },
        )

    image_path = await _save_and_validate_upload(file)
    try:
        result = await enhance_photo(
            image_path,
            preparation=preparation,
            enhance=enhance,
        )
    except RuntimeError as exc:
        return JSONResponse(
            status_code=502,
            content={"error": "enhancement_failed", "message": str(exc)},
        )
    finally:
        image_path.unlink(missing_ok=True)

    return {
        "ok": True,
        "job_id": result.job_id,
        "model": result.model,
        "width": result.width,
        "height": result.height,
        "enhanced": f"/api/enhanced/{result.job_id}/enhanced.png",
    }


@app.get("/api/enhanced/{job_id}/{filename}")
def enhanced_file(job_id: str, filename: str):
    if not job_id.isalnum() or filename != "enhanced.png":
        raise HTTPException(404)
    path = ENHANCED / job_id / filename
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/generate")
async def generate(
    file: UploadFile = File(...),
    mode: str = Form("detailed"),
    preparation: str = Form("balanced"),
    enhance: int = Form(50),
    silhouette: int = Form(70),
    detail: int = Form(75),
    shadows: int = Form(72),
    texture: int = Form(68),
    simplify: int = Form(0),
    base_job_id: str | None = Form(None),
):
    if mode not in {"detailed", "simplified", "clean"}:
        raise HTTPException(400, "mode no válido")
    if preparation not in {"soft", "balanced", "strong"}:
        raise HTTPException(400, "preparation no válido")
    if not 0 <= enhance <= 100:
        raise HTTPException(400, "enhance debe estar entre 0 y 100")

    for name, value in {
        "silhouette": silhouette,
        "detail": detail,
        "shadows": shadows,
        "texture": texture,
        "simplify": simplify,
    }.items():
        if not 0 <= value <= 100:
            raise HTTPException(400, f"{name} debe estar entre 0 y 100")

    status = provider_status()
    if not status["configured"]:
        return JSONResponse(
            status_code=503,
            content={
                "error": "generator_not_configured",
                "message": "Configura OPENAI_API_KEY en el servidor para activar la generación real.",
            },
        )

    base_stencil_path = None
    if mode in {"simplified", "clean"}:
        if not base_job_id or not base_job_id.isalnum():
            raise HTTPException(400, "Primero genera el modo Detallado; Simplificado y Muy limpio se derivan de ese stencil maestro.")
        candidate = GENERATED / base_job_id / "full.png"
        if not candidate.exists():
            raise HTTPException(400, "No se encontró el stencil Detallado maestro. Vuelve a generarlo.")
        base_stencil_path = candidate

    image_path = await _save_and_validate_upload(file)
    try:
        result = await generate_stencil(
            image_path,
            mode=mode,
            base_stencil_path=base_stencil_path,
            preparation=preparation,
            enhance=enhance,
            silhouette=silhouette,
            detail=detail,
            shadows=shadows,
            texture=texture,
            simplify=simplify,
        )
    except RuntimeError as exc:
        return JSONResponse(
            status_code=502,
            content={"error": "generation_failed", "message": str(exc)},
        )
    finally:
        image_path.unlink(missing_ok=True)

    base = f"/api/generated/{result.job_id}"
    return {
        "ok": True,
        "job_id": result.job_id,
        "model": result.model,
        "mode": result.mode,
        "full": f"{base}/full.png",
        "layers": {
            "silhouette": f"{base}/silhouette.png",
            "essential": f"{base}/essential.png",
            "structure": f"{base}/structure.png",
            "segmentation": f"{base}/segmentation.png",
            "detail": f"{base}/detail.png",
            "shade": f"{base}/shade.png",
            "texture": f"{base}/texture.png",
        },
    }


@app.get("/api/generated/{job_id}/{filename}")
def generated_file(job_id: str, filename: str):
    if not job_id.isalnum():
        raise HTTPException(404)
    allowed = {"full.png", "silhouette.png", "essential.png", "structure.png", "segmentation.png", "detail.png", "shade.png", "texture.png"}
    if filename not in allowed:
        raise HTTPException(404)
    path = GENERATED / job_id / filename
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/health")
def health():
    return {"ok": True, "engine": provider_status()}
