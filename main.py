import json
import os

from ocr_client import PaddleOCRClientError, process_ocr_file


def print_progress(data: dict) -> None:
    state = data.get("state")
    if state == "submitted":
        print(f"Job submitted successfully. job id: {data.get('jobId')}")
        print("Start polling for results")
    elif state == "pending":
        print("The current status of the job is pending")
    elif state == "running":
        progress = data.get("extractProgress", {})
        total_pages = progress.get("totalPages")
        extracted_pages = progress.get("extractedPages")
        if total_pages:
            print(
                "The current status of the job is running, "
                f"total pages: {total_pages}, extracted pages: {extracted_pages}"
            )
        else:
            print("The current status of the job is running...")
    elif state == "done":
        progress = data.get("extractProgress", {})
        print(
            "Job completed, "
            f"successfully extracted pages: {progress.get('extractedPages')}, "
            f"start time: {progress.get('startTime')}, "
            f"end time: {progress.get('endTime')}"
        )


if __name__ == "__main__":
    file_path = os.path.join("file", "abstrak.pdf")
    print(f"Processing file: {file_path}")

    try:
        response = process_ocr_file(file_path=file_path, progress_callback=print_progress)
        for page in response["data"]["pages"]:
            print(f"Markdown document saved at {page['markdown_file']}")
            for image in page["markdown_images"] + page["output_images"]:
                print(f"Image saved to: {image['file']}")

        print(json.dumps(response, ensure_ascii=False, indent=2))
    except PaddleOCRClientError as exc:
        print(f"Error: {exc}")
