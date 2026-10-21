"""
Integration tests for Crop Disease Detection & Advisory API.
"""
import io
from PIL import Image
from fastapi.testclient import TestClient

from app import app


def create_test_image_bytes():
    """Create a dummy 224x224 RGB green leaf-like image for testing."""
    img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


def test_health():
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["status"] == "ok"
    assert data["total_classes"] == 110
    print("✓ /health passed:", data)


def test_info():
    client = TestClient(app)
    res = client.get("/api/v1/info")
    assert res.status_code == 200, res.text
    data = res.json()
    assert "supported_crops" in data
    assert len(data["supported_crops"]) > 10
    print("✓ /api/v1/info passed, supported crops:", len(data["supported_crops"]))


def test_classes():
    client = TestClient(app)
    res = client.get("/api/v1/classes")
    assert res.status_code == 200, res.text
    classes = res.json()
    assert len(classes) == 110
    assert classes[0]["label"] == "Apple_Black_Rot"
    print("✓ /api/v1/classes passed, total classes:", len(classes))


def test_crops():
    client = TestClient(app)
    res = client.get("/api/v1/crops")
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["total_crops"] > 10
    print("✓ /api/v1/crops passed, total crops:", data["total_crops"])


def test_predict():
    client = TestClient(app)
    img_bytes = create_test_image_bytes()
    files = {"file": ("test_leaf.jpg", img_bytes, "image/jpeg")}
    res = client.post("/api/v1/predict?top_k=5", files=files)
    assert res.status_code == 200, res.text
    data = res.json()
    assert "top_prediction" in data
    assert len(data["top_k"]) == 5
    print("✓ /api/v1/predict passed, top prediction:", data["top_prediction"], data["confidence_pct"])


def test_advisory():
    client = TestClient(app)
    # Disease advisory
    res = client.post("/api/v1/advisory", json={"disease": "Rice_Blast"})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["disease"] == "Rice_Blast"
    assert not data["is_healthy"]
    assert "markdown" in data

    # Healthy advisory
    res_healthy = client.get("/api/v1/advisory/Cotton_Healthy")
    assert res_healthy.status_code == 200, res_healthy.text
    data_healthy = res_healthy.json()
    assert data_healthy["is_healthy"] is True
    print("✓ /api/v1/advisory passed (both disease & healthy)")


def test_analyze():
    client = TestClient(app)
    img_bytes = create_test_image_bytes()
    files = {"file": ("test_leaf.jpg", img_bytes, "image/jpeg")}
    res = client.post("/api/v1/analyze", files=files)
    assert res.status_code == 200, res.text
    data = res.json()
    assert "prediction" in data
    assert "advisory" in data
    print("✓ /api/v1/analyze passed:", data["prediction"]["top_prediction"])


def test_kb_lookup():
    client = TestClient(app)
    res = client.get("/api/v1/kb/Rice_Blast")
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["found"] is True
    print("✓ /api/v1/kb passed, kb_key:", data["kb_key"])


def test_docs_and_ui():
    client = TestClient(app)
    res_docs = client.get("/docs")
    assert res_docs.status_code == 200
    res_openapi = client.get("/openapi.json")
    assert res_openapi.status_code == 200
    res_ui = client.get("/")
    assert res_ui.status_code == 200
    print("✓ /docs and UI passed")


if __name__ == "__main__":
    print("\nRunning API test suite...\n")
    test_health()
    test_info()
    test_classes()
    test_crops()
    test_predict()
    test_advisory()
    test_analyze()
    test_kb_lookup()
    test_docs_and_ui()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!\n")
