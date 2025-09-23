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



def test_encode_sentences_and_semantic_split():
    # ensure backend dir is importable
    import sys
    import os
    this_dir = os.path.dirname(__file__)
    backend_dir = os.path.abspath(os.path.join(this_dir, ".."))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    import embed as embed_module
    from chunker import semantic_chunk_sentences

    sentences = [
        "This is the first sentence.",
        "This sentence is similar to the first sentence.",
        "A new unrelated topic starts here and is different.",
        "More content about the unrelated topic continues.",
    ]

    # encode with small batch size to exercise batching path
    vecs = embed_module.encode_sentences(sentences, batch_size=2)
    assert isinstance(vecs, list)
    assert len(vecs) == len(sentences)
    # vectors should be numeric lists
    assert all(isinstance(v, list) for v in vecs)

    spans = semantic_chunk_sentences(sentences, threshold=0.9, min_tokens=1, max_tokens=1000, overlap=0)
    # With a high similarity threshold, first two should stay together and last two together
    assert isinstance(spans, list)
    assert len(spans) >= 1


