import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

import requests


# JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
# MODEL = "PaddleOCR-VL-1.6"
# DEFAULT_TOKEN = os.getenv("PADDLEOCR_TOKEN", "7e51b8813491aa7579876177a09342944b026368")

JOB_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
DEFAULT_TOKEN = "7e51b8813491aa7579876177a09342944b026368"
MODEL = "PP-OCRv6"

DEFAULT_OPTIONAL_PAYLOAD = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useChartRecognition": False,
}


class PaddleOCRClientError(RuntimeError):
    pass


def _auth_headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"bearer {token}"}


def submit_ocr_job(
    file_path: str,
    token: str,
    model: str = MODEL,
    optional_payload: Optional[dict] = None,
) -> str:
    optional_payload = optional_payload or DEFAULT_OPTIONAL_PAYLOAD
    headers = _auth_headers(token)

    if file_path.startswith("http"):
        headers["Content-Type"] = "application/json"
        payload = {
            "fileUrl": file_path,
            "model": model,
            "optionalPayload": optional_payload,
        }
        response = requests.post(JOB_URL, json=payload, headers=headers, timeout=60)
    else:
        if not os.path.exists(file_path):
            raise PaddleOCRClientError(f"File tidak ditemukan: {file_path}")

        data = {
            "model": model,
            "optionalPayload": json.dumps(optional_payload),
        }
        with open(file_path, "rb") as file:
            response = requests.post(
                JOB_URL,
                headers=headers,
                data=data,
                files={"file": file},
                timeout=120,
            )

    if response.status_code != 200:
        raise PaddleOCRClientError(
            f"Gagal submit OCR ({response.status_code}): {response.text}"
        )

    body = response.json()
    try:
        return body["data"]["jobId"]
    except KeyError as exc:
        raise PaddleOCRClientError(f"Response submit tidak valid: {body}") from exc


def poll_ocr_job(
    job_id: str,
    token: str,
    interval_seconds: int = 5,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    headers = _auth_headers(token)

    while True:
        response = requests.get(f"{JOB_URL}/{job_id}", headers=headers, timeout=60)
        if response.status_code != 200:
            raise PaddleOCRClientError(
                f"Gagal membaca status job ({response.status_code}): {response.text}"
            )

        body = response.json()
        data = body.get("data", {})
        state = data.get("state")

        if progress_callback:
            progress_callback(data)

        if state == "done":
            return data

        if state == "failed":
            raise PaddleOCRClientError(data.get("errorMsg", "Job OCR gagal."))

        if state not in {"pending", "running"}:
            raise PaddleOCRClientError(f"State job tidak dikenal: {state}")

        time.sleep(interval_seconds)


def download_ocr_results(jsonl_url: str, output_dir: str) -> list:
    response = requests.get(jsonl_url, timeout=120)
    response.raise_for_status()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pages = []
    page_num = 0

    for line_num, raw_line in enumerate(response.text.strip().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        try:
            result = json.loads(line)["result"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise PaddleOCRClientError(f"JSONL tidak valid pada baris {line_num}") from exc

        for layout_result in result.get("layoutParsingResults", []):
            markdown_data = layout_result.get("markdown", {})
            markdown_text = markdown_data.get("text", "")
            markdown_file = output_path / f"doc_{page_num}.md"
            markdown_file.write_text(markdown_text, encoding="utf-8")

            markdown_images = []
            for image_path, image_url in markdown_data.get("images", {}).items():
                full_image_path = output_path / image_path
                full_image_path.parent.mkdir(parents=True, exist_ok=True)
                image_response = requests.get(image_url, timeout=120)
                image_response.raise_for_status()
                full_image_path.write_bytes(image_response.content)
                markdown_images.append(
                    {
                        "name": image_path,
                        "file": str(full_image_path),
                        "url": image_url,
                    }
                )

            output_images = []
            for image_name, image_url in layout_result.get("outputImages", {}).items():
                image_response = requests.get(image_url, timeout=120)
                image_response.raise_for_status()
                image_file = output_path / f"{image_name}_{page_num}.jpg"
                image_file.write_bytes(image_response.content)
                output_images.append(
                    {
                        "name": image_name,
                        "file": str(image_file),
                        "url": image_url,
                    }
                )

            pages.append(
                {
                    "page": page_num + 1,
                    "markdown": markdown_text,
                    "markdown_file": str(markdown_file),
                    "markdown_images": markdown_images,
                    "output_images": output_images,
                }
            )
            page_num += 1

    return pages


def build_api_response(
    *,
    filename: str,
    model: str,
    job_id: str,
    job_data: dict,
    pages: list,
    output_dir: str,
    latency_seconds: float,
) -> dict:
    progress = job_data.get("extractProgress", {})
    result_url = job_data.get("resultUrl", {})

    return {
        "status": "success",
        "message": "OCR selesai diproses.",
        "data": {
            "jobId": job_id,
            "state": job_data.get("state"),
            "totalPages": progress.get("totalPages"),
            "extractedPages": progress.get("extractedPages"),
            "startTime": progress.get("startTime"),
            "endTime": progress.get("endTime"),
            "resultUrl": result_url,
            "outputDir": output_dir,
            "pages": pages,
        },
        "meta": {
            "filename": filename,
            "model": model,
            "latencySeconds": round(latency_seconds, 3),
            "latencyMs": round(latency_seconds * 1000),
            "generatedAt": datetime.now(timezone.utc).isoformat(),
        },
    }


def process_ocr_file(
    file_path: str,
    token: str = DEFAULT_TOKEN,
    model: str = MODEL,
    optional_payload: Optional[dict] = None,
    output_dir: str = "output",
    interval_seconds: int = 5,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    if not token:
        raise PaddleOCRClientError("Token PaddleOCR belum diisi.")

    started_at = time.perf_counter()
    job_id = submit_ocr_job(file_path, token, model, optional_payload)
    if progress_callback:
        progress_callback({"state": "submitted", "jobId": job_id})

    job_data = poll_ocr_job(
        job_id,
        token,
        interval_seconds=interval_seconds,
        progress_callback=progress_callback,
    )
    jsonl_url = job_data.get("resultUrl", {}).get("jsonUrl")
    if not jsonl_url:
        raise PaddleOCRClientError("URL hasil JSON tidak ditemukan dari response OCR.")

    pages = download_ocr_results(jsonl_url, output_dir)
    latency_seconds = time.perf_counter() - started_at
    return build_api_response(
        filename=os.path.basename(file_path),
        model=model,
        job_id=job_id,
        job_data=job_data,
        pages=pages,
        output_dir=output_dir,
        latency_seconds=latency_seconds,
    )
