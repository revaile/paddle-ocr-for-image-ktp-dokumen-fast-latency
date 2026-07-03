import json
import time
from pathlib import Path

import requests
import streamlit as st

from ocr_client import (
    DEFAULT_OPTIONAL_PAYLOAD,
    DEFAULT_TOKEN,
    MODEL,
    PaddleOCRClientError,
    process_ocr_file,
)


UPLOAD_DIR = Path("tmp_uploads")
OUTPUT_DIR = Path("output")


def format_size(size_bytes: int) -> str:
    size_mb = size_bytes / (1024 * 1024)
    return f"{size_mb:.2f} MB"


st.set_page_config(page_title="PaddleOCR Upload", layout="wide")

st.title("PaddleOCR Upload")
st.caption("Upload PDF atau gambar, lalu lihat hasil OCR dalam bentuk preview dan response JSON.")

with st.sidebar:
    st.header("Pengaturan API")
    token = st.text_input("Token PaddleOCR", value=DEFAULT_TOKEN, type="password")
    model = st.text_input("Model", value=MODEL)
    poll_interval = st.number_input("Interval polling (detik)", min_value=2, max_value=30, value=5)

    st.divider()
    st.subheader("Optional payload")
    use_orientation = st.checkbox(
        "Doc orientation classify",
        value=DEFAULT_OPTIONAL_PAYLOAD["useDocOrientationClassify"],
    )
    use_unwarping = st.checkbox(
        "Doc unwarping",
        value=DEFAULT_OPTIONAL_PAYLOAD["useDocUnwarping"],
    )
    use_chart = st.checkbox(
        "Chart recognition",
        value=DEFAULT_OPTIONAL_PAYLOAD["useChartRecognition"],
    )

uploaded_file = st.file_uploader(
    "Pilih file",
    type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "webp"],
)

if uploaded_file:
    st.info(f"File siap diproses: {uploaded_file.name} ({format_size(uploaded_file.size)})")

run_button = st.button("Proses OCR", type="primary", disabled=uploaded_file is None)

if run_button and uploaded_file:
    started_at = time.perf_counter()
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    safe_name = Path(uploaded_file.name).name
    upload_path = UPLOAD_DIR / safe_name
    upload_path.write_bytes(uploaded_file.getbuffer())

    optional_payload = {
        "useDocOrientationClassify": use_orientation,
        "useDocUnwarping": use_unwarping,
        "useChartRecognition": use_chart,
    }

    progress = st.progress(0)
    status_box = st.empty()

    def update_progress(data: dict) -> None:
        state = data.get("state", "running")
        job_id = data.get("jobId")

        if state == "submitted":
            status_box.write(f"Job terkirim. Job ID: `{job_id}`")
            progress.progress(10)
            return

        extract_progress = data.get("extractProgress", {})
        total_pages = extract_progress.get("totalPages") or 0
        extracted_pages = extract_progress.get("extractedPages") or 0

        if state == "pending":
            status_box.write("Menunggu antrian OCR...")
            progress.progress(20)
        elif state == "running" and total_pages:
            percent = min(90, 20 + int((extracted_pages / total_pages) * 70))
            status_box.write(f"Memproses halaman {extracted_pages}/{total_pages}...")
            progress.progress(percent)
        elif state == "running":
            status_box.write("OCR sedang berjalan...")
            progress.progress(45)
        elif state == "done":
            status_box.write("Mengunduh hasil OCR...")
            progress.progress(95)

    try:
        with st.spinner("Mengirim file ke PaddleOCR..."):
            response = process_ocr_file(
                file_path=str(upload_path),
                token=token,
                model=model,
                optional_payload=optional_payload,
                output_dir=str(OUTPUT_DIR),
                interval_seconds=int(poll_interval),
                progress_callback=update_progress,
            )

        progress.progress(100)
        status_box.success("OCR selesai.")

        pages = response["data"]["pages"]
        st.metric("Latency", f"{response['meta']['latencySeconds']} detik")
        tab_preview, tab_json, tab_files = st.tabs(["Preview", "Response API", "File Output"])

        with tab_preview:
            if not pages:
                st.warning("Tidak ada halaman yang berhasil diekstrak.")
            for page in pages:
                st.subheader(f"Halaman {page['page']}")
                if page["markdown"]:
                    st.markdown(page["markdown"])
                else:
                    st.caption("Markdown kosong.")

                for image in page["output_images"]:
                    st.image(image["file"], caption=image["name"], use_container_width=True)

        with tab_json:
            st.json(response)
            st.download_button(
                "Download response JSON",
                data=json.dumps(response, ensure_ascii=False, indent=2),
                file_name="ocr_response.json",
                mime="application/json",
            )

        with tab_files:
            st.write(f"Output folder: `{response['data']['outputDir']}`")
            for page in pages:
                st.write(f"Halaman {page['page']}: `{page['markdown_file']}`")

    except (PaddleOCRClientError, requests.RequestException) as exc:
        progress.progress(0)
        status_box.error("OCR gagal diproses.")
        st.json(
            {
                "status": "error",
                "message": str(exc),
                "data": None,
                "meta": {
                    "filename": uploaded_file.name,
                    "model": model,
                    "latencySeconds": round(time.perf_counter() - started_at, 3),
                },
            }
        )
