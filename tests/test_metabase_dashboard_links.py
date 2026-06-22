from app.services.bi import metabase_dashboard


def test_dashboard_links_reuses_existing_public_uuid(monkeypatch):
    calls = []

    def fake_json_request(method, path, **kwargs):
        calls.append((method, path))
        return {"id": 7, "public_uuid": "existing-public-id"}

    monkeypatch.setattr(metabase_dashboard, "_json_request", fake_json_request)
    monkeypatch.setattr(metabase_dashboard, "_metabase_base_url", lambda: "http://localhost:3000")

    result = metabase_dashboard._dashboard_links(
        "session",
        7,
        ensure_public_link=True,
    )

    assert result == {
        "dashboard_id": 7,
        "dashboard_url": "http://localhost:3000/dashboard/7",
        "embed_url": "http://localhost:3000/public/dashboard/existing-public-id",
    }
    assert calls == [("GET", "/api/dashboard/7")]


def test_dashboard_links_creates_public_link_when_missing(monkeypatch):
    calls = []

    def fake_json_request(method, path, **kwargs):
        calls.append((method, path))
        if method == "GET":
            return {"id": 8, "public_uuid": None}
        return {"uuid": "new-public-id"}

    monkeypatch.setattr(metabase_dashboard, "_json_request", fake_json_request)
    monkeypatch.setattr(metabase_dashboard, "_metabase_base_url", lambda: "http://localhost:3000")

    result = metabase_dashboard._dashboard_links(
        "session",
        8,
        ensure_public_link=True,
    )

    assert result["embed_url"] == "http://localhost:3000/public/dashboard/new-public-id"
    assert calls == [
        ("GET", "/api/dashboard/8"),
        ("POST", "/api/dashboard/8/public_link"),
    ]
