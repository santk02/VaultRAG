import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def test_health_check(client):
    """Test the health check endpoint.

    Requires live Postgres + Qdrant (as CI provides via docker services) since
    health_router.py now probes both rather than hardcoding "ok" — this test will
    fail in an environment without those services reachable.
    """
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "services" in data
    assert data["services"]["api"] == "running"


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "name" in data
    assert "version" in data
    assert data["status"] == "running"


def test_cors_headers(client):
    """Test that CORS headers are present."""
    response = client.get("/health", headers={"Origin": "http://localhost"})
    assert response.status_code == 200
    # CORS headers should be present (allow_origins=["*"] in main.py's CORSMiddleware)
    assert "access-control-allow-origin" in response.headers
