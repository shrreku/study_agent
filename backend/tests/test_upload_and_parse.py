import os
import io
import requests


def test_upload_sample_pdf():
    # quick integration-ish test if backend is running locally
    url = os.getenv("BASE_URL", "http://localhost:8000")
    token = "test-token"
    sample_path = os.path.join(os.getcwd(), "sample", "Fundamentals of Heat and Mass Transfer Chapter 6 (1).pdf")
    if not os.path.exists(sample_path):
        return
    with open(sample_path, "rb") as f:
        files = {"file": (os.path.basename(sample_path), f, "application/pdf")}
        data = {"title": "Sample PDF"}
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.post(f"{url}/api/resources/upload", files=files, data=data, headers=headers)
        assert resp.status_code == 200 or resp.status_code == 201
        j = resp.json()
        assert "resource_id" in j


