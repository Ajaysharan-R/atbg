"""POST /extract-conversation — screenshot OCR and conversation reconstruction."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional

from api.services.ocr_service import process_image

router = APIRouter(prefix="/extract-conversation", tags=["screenshot"])


class MessageOut(BaseModel):
    turn_index: int
    speaker: str
    text: str
    timestamp: Optional[str] = None


class ExtractResponse(BaseModel):
    text: str
    messages: list[MessageOut]
    confidence: float
    warnings: list[str]


@router.post("", response_model=ExtractResponse)
async def extract_conversation(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    file_bytes = await file.read()

    try:
        result = process_image(file_bytes, content_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image processing failed: {str(e)}"
        )

    return ExtractResponse(
        text=result["text"],
        messages=[MessageOut(**m) for m in result["messages"]],
        confidence=result["confidence"],
        warnings=result["warnings"],
    )
