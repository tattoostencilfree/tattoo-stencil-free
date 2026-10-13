from __future__ import annotations

from pathlib import Path
import base64
import hashlib
import hmac
import json
import os
import time
import uuid

import httpx

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .provider import generate_stencil, enhance_photo, create_posterization_master, provider_status, GENERATED, ENHANCED, POSTER_MASTERS
from .posterizer import posterize_for_tattoo

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
UPLOADS = ROOT / "backend" / "uploads"
POSTERIZED = ROOT / "backend" / "posterized"
UPLOADS.mkdir(parents=True, exist_ok=True)
POSTERIZED.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}

PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID", "").strip()
PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET", "").strip()
PAYPAL_ENV = os.getenv("PAYPAL_ENV", "live").strip().lower()
ADMIN_PREVIEW = os.getenv("ADMIN_PREVIEW", "false").strip().lower() in {"1", "true", "yes", "on"}
PAYMENT_REQUIRED = (os.getenv("PAYMENT_REQUIRED", "true").strip().lower() not in {"0", "false", "no", "off"}) and not ADMIN_PREVIEW
PAYMENT_PRICE_USD = os.getenv("PAYMENT_PRICE_USD", "1.79").strip() or "1.79"
ACCESS_HOURS = max(1, int(os.getenv("PAYMENT_ACCESS_HOURS", "24")))
ACCESS_COOKIE = "tattoo_stencil_access"


def _paypal_base() -> str:
    return "https://api-m.sandbox.paypal.com" if PAYPAL_ENV == "sandbox" else "https://api-m.paypal.com"


def _payment_configured() -> bool:
    return bool(PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET)


def _token_key() -> bytes:
    # Derive a dedicated signing key without adding another secret to Railway.
    source = PAYPAL_CLIENT_SECRET or os.getenv("OPENAI_API_KEY", "local-dev-key")
    return hashlib.sha256((source + ":tattoo-stencil-access").encode()).digest()


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _issue_access_payload() -> dict:
    now = int(time.time())
    return {
        "v": 1,
        "paid": True,
        "iat": now,
        "exp": now + ACCESS_HOURS * 3600,
        "uses": {"enhance": 0, "posterize": 0, "detailed": 0, "simplified": 0, "clean": 0},
    }


def _encode_access(payload: dict) -> str:
    body = _b64e(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    sig = _b64e(hmac.new(_token_key(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def _decode_access(token: str | None) -> dict | None:
    if not token or "." not in token:
        return None
    try:
        body, sig = token.rsplit(".", 1)
        expected = _b64e(hmac.new(_token_key(), body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_b64d(body))
        if not payload.get("paid") or int(payload.get("exp", 0)) <= int(time.time()):
            return None
        return payload
    except Exception:
        return None


def _get_access(request: Request) -> dict | None:
    if not PAYMENT_REQUIRED:
        return {"paid": True, "exp": int(time.time()) + 365 * 86400, "uses": {"enhance": 0, "posterize": 0, "detailed": 0, "simplified": 0, "clean": 0}}
    return _decode_access(request.cookies.get(ACCESS_COOKIE))


def _set_access_cookie(response: Response, payload: dict) -> None:
    max_age = max(60, int(payload["exp"]) - int(time.time()))
    response.set_cookie(
        ACCESS_COOKIE,
        _encode_access(payload),
        max_age=max_age,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )


def _locked_response(message: str = "Paga US$1.79 para procesar este diseño y desbloquear las herramientas.") -> JSONResponse:
    return JSONResponse(status_code=402, content={"error": "payment_required", "message": message})


def _quota_response(kind: str) -> JSONResponse:
    labels = {
        "enhance": "La mejora incluida en este acceso ya fue utilizada.",
        "detailed": "El stencil Detallado incluido ya fue generado.",
        "simplified": "El stencil Simplificado incluido ya fue generado.",
        "clean": "El stencil Muy limpio incluido ya fue generado.",
    }
    return JSONResponse(status_code=429, content={"error": "paid_quota_used", "message": labels.get(kind, "Esta generación incluida ya fue utilizada.")})


async def _paypal_access_token() -> str:
    if not _payment_configured():
        raise RuntimeError("PayPal no está configurado en el servidor.")
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{_paypal_base()}/v1/oauth2/token",
            data={"grant_type": "client_credentials"},
            auth=(PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET),
            headers={"Accept": "application/json", "Accept-Language": "en_US"},
        )
    if r.status_code >= 400:
        raise RuntimeError(f"PayPal OAuth error ({r.status_code}).")
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError("PayPal no devolvió un access token.")
    return token

app = FastAPI(title="Tattoo Stencil Lab MVP")
app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")


@app.get("/")
def home():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/config")
def config(request: Request):
    status = provider_status()
    access = _get_access(request)
    return {
        "engine": status,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "preview_mode": ADMIN_PREVIEW,
        "payment": {
            "required": PAYMENT_REQUIRED,
            "configured": _payment_configured(),
            "unlocked": bool(access),
            "price": PAYMENT_PRICE_USD,
            "currency": "USD",
            "access_hours": ACCESS_HOURS,
            # PayPal Client ID is public by design; never expose the Client Secret.
            "paypal_client_id": PAYPAL_CLIENT_ID if _payment_configured() else "",
            "uses": (access or {}).get("uses", {}),
        },
    }


@app.post("/api/paypal/create-order")
async def paypal_create_order():
    if not PAYMENT_REQUIRED:
        return {"id": "payment-disabled"}
    try:
        access_token = await _paypal_access_token()
        payload = {
            "intent": "CAPTURE",
            "purchase_units": [
                {
                    "reference_id": "tattoo-stencil-design",
                    "description": "TATTOO STENCIL - procesamiento de diseño",
                    "amount": {"currency_code": "USD", "value": PAYMENT_PRICE_USD},
                }
            ],
            "application_context": {"shipping_preference": "NO_SHIPPING", "user_action": "PAY_NOW"},
        }
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{_paypal_base()}/v2/checkout/orders",
                json=payload,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                    "PayPal-Request-Id": uuid.uuid4().hex,
                },
            )
        data = r.json() if r.content else {}
        if r.status_code >= 400 or not data.get("id"):
            return JSONResponse(status_code=502, content={"error": "paypal_create_failed", "message": "No se pudo iniciar el pago con PayPal."})
        return {"id": data["id"]}
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": "paypal_create_failed", "message": str(exc)})


@app.post("/api/paypal/capture-order/{order_id}")
async def paypal_capture_order(order_id: str, response: Response):
    if not order_id or len(order_id) > 80:
        raise HTTPException(400, "Order ID no válido")
    try:
        access_token = await _paypal_access_token()
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{_paypal_base()}/v2/checkout/orders/{order_id}/capture",
                headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json", "PayPal-Request-Id": f"capture-{order_id}"},
            )
        data = r.json() if r.content else {}
        if r.status_code >= 400 or data.get("status") != "COMPLETED":
            return JSONResponse(status_code=402, content={"error": "paypal_capture_failed", "message": "El pago no pudo confirmarse."})

        purchase_units = data.get("purchase_units") or []
        captures = ((purchase_units[0].get("payments") or {}).get("captures") or []) if purchase_units else []
        capture = captures[0] if captures else {}
        amount = capture.get("amount") or {}
        if amount.get("currency_code") != "USD" or amount.get("value") != PAYMENT_PRICE_USD:
            return JSONResponse(status_code=400, content={"error": "amount_mismatch", "message": "El monto confirmado no coincide con el acceso solicitado."})

        access_payload = _issue_access_payload()
        _set_access_cookie(response, access_payload)
        return {"ok": True, "unlocked": True, "expires_at": access_payload["exp"], "uses": access_payload["uses"]}
    except Exception as exc:
        return JSONResponse(status_code=502, content={"error": "paypal_capture_failed", "message": str(exc)})


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
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    preparation: str = Form("balanced"),
    enhance: int = Form(50),
):
    access = _get_access(request)
    if not access:
        return _locked_response()
    if PAYMENT_REQUIRED and int((access.get("uses") or {}).get("enhance", 0)) >= 1:
        return _quota_response("enhance")

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

    if PAYMENT_REQUIRED:
        access.setdefault("uses", {})["enhance"] = 1
        _set_access_cookie(response, access)

    return {
        "ok": True,
        "job_id": result.job_id,
        "model": result.model,
        "width": result.width,
        "height": result.height,
        "uses": access.get("uses", {}),
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


@app.post("/api/posterize-master")
async def posterize_master(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
):
    access = _get_access(request)
    if not access:
        return _locked_response()

    # Reuse the already-created master within the paid session instead of spending again.
    cached_job = str(access.get("poster_master_job_id") or "")
    if cached_job.isalnum():
        cached = POSTER_MASTERS / cached_job / "master.png"
        if cached.exists():
            with Image.open(cached) as im:
                w, h = im.size
            return {
                "ok": True,
                "job_id": cached_job,
                "master": f"/api/poster-master/{cached_job}/master.png",
                "width": w,
                "height": h,
                "cached": True,
                "uses": access.get("uses", {}),
            }

    image_path = await _save_and_validate_upload(file)
    try:
        status = provider_status()
        if not status["configured"]:
            # Local fallback keeps the workflow usable, but the normal configured path uses
            # the artistic reference-guided posterization master for production quality.
            job_id = uuid.uuid4().hex
            out_dir = POSTER_MASTERS / job_id
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / "master.png"
            with Image.open(image_path).convert("RGB") as im:
                im.save(target, "PNG", optimize=True)
            model = "local-fallback"
            with Image.open(target) as im:
                w, h = im.size
        else:
            result = await create_posterization_master(image_path)
            job_id = result.job_id
            target = result.posterized_png
            model = result.model
            w, h = result.width, result.height
    except RuntimeError as exc:
        return JSONResponse(status_code=502, content={"error": "poster_master_failed", "message": str(exc)})
    finally:
        image_path.unlink(missing_ok=True)

    access.setdefault("uses", {})["posterize"] = 1
    access["poster_master_job_id"] = job_id
    if PAYMENT_REQUIRED:
        _set_access_cookie(response, access)

    return {
        "ok": True,
        "job_id": job_id,
        "master": f"/api/poster-master/{job_id}/master.png",
        "width": w,
        "height": h,
        "model": model,
        "cached": False,
        "uses": access.get("uses", {}),
    }


@app.get("/api/poster-master/{job_id}/{filename}")
def poster_master_file(job_id: str, filename: str):
    if not job_id.isalnum() or filename != "master.png":
        raise HTTPException(404)
    path = POSTER_MASTERS / job_id / filename
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/posterize")
async def posterize(
    request: Request,
    file: UploadFile = File(...),
    intensity: int = Form(35),
):
    access = _get_access(request)
    if not access:
        return _locked_response()
    if not 0 <= intensity <= 100:
        raise HTTPException(400, "intensity debe estar entre 0 y 100")

    image_path = await _save_and_validate_upload(file)
    try:
        result = posterize_for_tattoo(image_path, POSTERIZED, intensity=intensity)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": "posterize_failed", "message": f"No se pudo posterizar la imagen: {exc}"},
        )
    finally:
        image_path.unlink(missing_ok=True)

    base = f"/api/posterized/{result.job_id}"
    return {
        "ok": True,
        "job_id": result.job_id,
        "width": result.width,
        "height": result.height,
        "levels": result.levels,
        "intensity": intensity,
        "posterized": f"{base}/posterized.png",
        "guided": f"{base}/guided.png",
        "guides": f"{base}/guides.png",
        "guided_by_mode": {
            "detailed": f"{base}/guided-detailed.png",
            "simplified": f"{base}/guided-simplified.png",
            "clean": f"{base}/guided-clean.png",
        },
        "guides_by_mode": {
            "detailed": f"{base}/guides-detailed.png",
            "simplified": f"{base}/guides-simplified.png",
            "clean": f"{base}/guides-clean.png",
        },
        "cost": "local",
    }


@app.get("/api/posterized/{job_id}/{filename}")
def posterized_file(job_id: str, filename: str):
    if not job_id.isalnum():
        raise HTTPException(404)
    allowed = {
        "posterized.png", "guided.png", "guides.png",
        "guided-detailed.png", "guided-simplified.png", "guided-clean.png",
        "guides-detailed.png", "guides-simplified.png", "guides-clean.png",
    }
    if filename not in allowed:
        raise HTTPException(404)
    path = POSTERIZED / job_id / filename
    if not path.exists():
        raise HTTPException(404)
    media = "image/png"
    return FileResponse(path, media_type=media, headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/generate")
async def generate(
    request: Request,
    response: Response,
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
    poster_job_id: str | None = Form(None),
):
    access = _get_access(request)
    if not access:
        return _locked_response()
    # Once the posterized master exists, switching modes remains allowed for the same paid
    # design. Each mode uses the same posterized guide map and can be cached independently.
    if PAYMENT_REQUIRED and int((access.get("uses") or {}).get("stencil", 0)) >= 1 and not poster_job_id:
        return _quota_response("stencil")

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

    base_stencil_path = None
    # A paid design produces exactly one stencil mode. Simplified/Clean can therefore
    # be generated directly from the prepared/posterized source instead of secretly
    # spending a second AI generation on a Detailed master.
    if mode in {"simplified", "clean"} and base_job_id and base_job_id.isalnum():
        candidate = GENERATED / base_job_id / "full.png"
        if candidate.exists():
            base_stencil_path = candidate

    image_path = await _save_and_validate_upload(file)
    poster_source_path = None
    mandatory_guide_path = None
    if poster_job_id and poster_job_id.isalnum():
        candidate_source = POSTERIZED / poster_job_id / "posterized.png"
        candidate_guide = POSTERIZED / poster_job_id / f"guides-{mode}.png"
        candidate_guided = POSTERIZED / poster_job_id / f"guided-{mode}.png"
        if candidate_source.exists() and candidate_guide.exists():
            # Send the clean posterized image as the visual source. The guide map is sent
            # separately to the image model so rough guide pixels are not copied as artwork.
            poster_source_path = candidate_source
            mandatory_guide_path = candidate_guide

    if mandatory_guide_path is None:
        status = provider_status()
        if not status["configured"]:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "generator_not_configured",
                    "message": "Configura OPENAI_API_KEY en el servidor para activar la generación real.",
                },
            )

    try:
        result = await generate_stencil(
            poster_source_path or image_path,
            mode=mode,
            base_stencil_path=base_stencil_path,
            mandatory_guide_path=mandatory_guide_path,
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

    if PAYMENT_REQUIRED:
        access.setdefault("uses", {})["stencil"] = 1
        access.setdefault("uses", {})[mode] = 1
        _set_access_cookie(response, access)

    base = f"/api/generated/{result.job_id}"
    return {
        "ok": True,
        "job_id": result.job_id,
        "model": result.model,
        "mode": result.mode,
        "uses": access.get("uses", {}),
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
