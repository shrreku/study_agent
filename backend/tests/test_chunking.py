import os
import requests


def test_chunk_sample_pdf():
    url = os.getenv("BASE_URL", "http://localhost:8000")
    token = "test-token"
    # first upload file
    sample_path = os.path.join(os.getcwd(), "sample", "Fundamentals of Heat and Mass Transfer Chapter 6 (1).pdf")
    if not os.path.exists(sample_path):
        return
    with open(sample_path, "rb") as f:
        files = {"file": (os.path.basename(sample_path), f, "application/pdf")}
        data = {"title": "Sample"}
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.post(f"{url}/api/resources/upload", files=files, data=data, headers=headers)
        assert resp.status_code in (200,201)
        rid = resp.json()["resource_id"]
        # call chunk endpoint
        resp2 = requests.post(f"{url}/api/resources/{rid}/chunk", headers=headers)
        assert resp2.status_code == 200
        assert resp2.json().get("chunks_created", 0) > 0


