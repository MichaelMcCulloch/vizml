from fastapi.testclient import TestClient

from backend.main import app


def sample_graph(name="Example"):
    return {"version": 1, "name": name, "nodes": [
        {"id": "input", "position": {"x": 0, "y": 0}, "data": {"kind": "feature_input", "label": "Input", "params": {"features": 3}}},
    ], "edges": []}


def test_projects_and_templates_persist_independently(tmp_path, monkeypatch):
    monkeypatch.setenv("VIZML_DATA_DIR", str(tmp_path))
    client = TestClient(app)
    graph = sample_graph()
    created = client.post("/api/workspace/projects", json={"name": "My project", "graph": graph})
    assert created.status_code == 200
    project = created.json()
    assert project["graph"]["nodes"][0]["id"] == "input"
    saved = client.post("/api/workspace/templates", json={"name": "My template", "graph": graph}).json()
    graph["name"] = "Changed"
    updated = client.put(f"/api/workspace/projects/{project['id']}", json={"name": "Changed", "graph": graph})
    assert updated.status_code == 200
    assert client.get(f"/api/workspace/projects/{project['id']}").json()["graph"]["name"] == "Changed"
    assert client.get(f"/api/workspace/templates/{saved['id']}").json()["graph"]["name"] == "Example"
    assert [item["name"] for item in client.get("/api/workspace/projects").json()["items"]] == ["Changed"]
    assert (tmp_path / "projects" / f"{project['id']}.json").is_file()
    assert client.delete(f"/api/workspace/templates/{saved['id']}").status_code == 200
    assert client.get(f"/api/workspace/templates/{saved['id']}").status_code == 404
    assert client.get(f"/api/workspace/projects/{project['id']}").status_code == 200


def test_workspace_rejects_bad_names_ids_missing_updates_and_oversize(tmp_path, monkeypatch):
    monkeypatch.setenv("VIZML_DATA_DIR", str(tmp_path))
    client = TestClient(app)
    graph = sample_graph()
    assert client.post("/api/workspace/projects", json={"name": "  ", "graph": graph}).status_code == 400
    assert client.get("/api/workspace/projects/not-a-uuid").status_code == 404
    missing = "11111111-1111-4111-8111-111111111111"
    assert client.put(f"/api/workspace/projects/{missing}", json={"name": "x", "graph": graph}).status_code == 404
    graph["metadata"] = {"large": "x" * 2_000_000}
    assert client.post("/api/workspace/projects", json={"name": "x", "graph": graph}).status_code == 400
