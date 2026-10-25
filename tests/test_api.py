"""HTTP API: auth, observability, every endpoint, limits and malformed inputs."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from causa.api.app import create_app
from causa.data.campaign import CampaignConfig, simulate_campaign
from causa.data.experiments import ExperimentConfig, simulate_experiment
from causa.features import FEATURE_NAMES, feature_matrix


def experiment_payload(n: int = 6000, **spec: object) -> dict[str, object]:
    frame = simulate_experiment(ExperimentConfig(n_users=n, lift=0.08, seed=9))
    return {"spec": {"name": "api-test", **spec}, "data": frame.to_dict("list")}


def observational_payload(n: int = 1500) -> dict[str, object]:
    data = simulate_campaign(CampaignConfig(n=n, seed=4))
    X = feature_matrix(data.customers)
    columns = {name: X[:, j].tolist() for j, name in enumerate(FEATURE_NAMES)}
    columns |= {"coupon": data.treatment.tolist(), "revenue": data.revenue.tolist()}
    return {
        "covariates": list(FEATURE_NAMES),
        "treatment": "coupon",
        "outcome": "revenue",
        "data": columns,
        "n_boot": 30,
    }


def test_health_reports_the_loaded_model(client, small_model):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_learner"] == small_model.learner_name


def test_request_ids_are_echoed_generated_and_sanitised(client):
    assert client.get("/health", headers={"X-Request-ID": "abc-123"}).headers["X-Request-ID"] == "abc-123"
    generated = client.get("/health").headers["X-Request-ID"]
    assert len(generated) == 32
    hostile = client.get("/health", headers={"X-Request-ID": "x" * 500}).headers["X-Request-ID"]
    assert hostile != "x" * 500


def test_metrics_are_prometheus_text_with_route_templates(client):
    client.post("/v1/experiments/simulate", json={"n_users": 4000})
    text = client.get("/metrics").text
    assert "causa_requests_total" in text and 'path="/v1/experiments/simulate"' in text
    assert "causa_experiment_decisions_total" in text


def test_api_key_is_enforced_when_configured(settings, small_model, monkeypatch):
    secured = settings.model_copy(update={"api_key": "s3cret"})
    with TestClient(create_app(secured, model=small_model)) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/").status_code == 200
        assert c.get("/v1/targeting/model").status_code == 401
        assert c.get("/v1/targeting/model", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/v1/targeting/model", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_simulate_store_get_and_list(client):
    created = client.post("/v1/experiments/simulate", json={"n_users": 20_000, "lift": 0.1, "seed": 3})
    assert created.status_code == 201
    record = created.json()
    assert record["decision"] in {"ship", "keep_running", "stop_no_effect", "do_not_ship", "invalid"}
    fetched = client.get(f"/v1/experiments/{record['id']}").json()
    assert fetched["report"]["primary"] == record["report"]["primary"]
    assert [row["id"] for row in client.get("/v1/experiments").json()] == [record["id"]]


def test_unknown_experiment_is_404_with_request_id(client):
    response = client.get("/v1/experiments/doesnotexist")
    assert response.status_code == 404 and response.json()["request_id"]


def test_json_upload_round_trip(client):
    response = client.post("/v1/experiments", json=experiment_payload(mde=40.0))
    assert response.status_code == 201
    assert response.json()["report"]["power"]["mde"] == 40.0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["data"].update({"revenue": p["data"]["revenue"][:-1]}),
        lambda p: p.update({"data": {}}),
        lambda p: p["spec"].update({"unknown_field": 1}),
        lambda p: p["spec"].update({"alpha": 0.9}),
        lambda p: p["data"].update({"variant": ["x"] * len(p["data"]["variant"])}),
    ],
)
def test_malformed_experiment_uploads_are_422(client, mutate):
    payload = experiment_payload(2000)
    mutate(payload)
    assert client.post("/v1/experiments", json=payload).status_code == 422


def test_oversized_upload_is_413(client):
    response = client.post("/v1/experiments", json=experiment_payload(n=61_000))
    assert response.status_code == 413


def test_csv_upload_and_its_failure_modes(client):
    csv = simulate_experiment(ExperimentConfig(n_users=5000, seed=2)).to_csv(index=False).encode()
    ok = client.post(
        "/v1/experiments/csv", files={"file": ("e.csv", csv)}, data={"spec": json.dumps({"name": "csv"})}
    )
    assert ok.status_code == 201 and ok.json()["name"] == "csv"
    garbage = client.post("/v1/experiments/csv", files={"file": ("e.csv", b"\x00\x01not,a\ncsv")})
    assert garbage.status_code == 422
    bad_spec = client.post("/v1/experiments/csv", files={"file": ("e.csv", csv)}, data={"spec": "{not json"})
    assert bad_spec.status_code == 422


def test_targeting_respects_budget_and_ranks(client):
    customers = client.get("/v1/customers/sample?n=200&seed=1").json()
    body = client.post("/v1/targeting", json={"customers": customers, "budget": 0.1}).json()
    assert body["n_customers"] == 200 and body["n_treated"] <= 20
    assert sorted(r["rank"] for r in body["results"]) == list(range(1, 201))
    assert [r["customer_id"] for r in body["results"]] == [c["customer_id"] for c in customers]


@pytest.mark.parametrize(
    ("field", "value"),
    [("email_open_rate", 2.0), ("city_tier", 7), ("recency_days", 0), ("avg_order_value", "big")],
)
def test_invalid_customers_are_422(client, field, value):
    customer = client.get("/v1/customers/sample?n=1").json()[0]
    customer[field] = value
    assert client.post("/v1/targeting", json={"customers": [customer]}).status_code == 422


def test_targeting_without_a_model_is_503(settings):
    with TestClient(create_app(settings)) as c:
        customers = c.get("/v1/customers/sample?n=2").json()
        assert c.post("/v1/targeting", json={"customers": customers}).status_code == 503
        assert c.get("/health").json()["model_loaded"] is False


def test_model_info(client, small_model):
    info = client.get("/v1/targeting/model").json()
    assert info["learner"] == small_model.learner_name and set(info["validation"]) == set(
        small_model.validation
    )


def test_observational_ate_returns_every_estimator_and_diagnostics(client):
    body = client.post("/v1/observational/ate", json=observational_payload()).json()
    assert body["recommended"] == "aipw"
    assert {"ipw", "aipw", "aipw_wald"} <= set(body["result"]["estimates"])
    assert [row["feature"] for row in body["result"]["overlap"]["smd"]] == list(FEATURE_NAMES)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["data"].update({"coupon": [2] * len(p["data"]["coupon"])}),
        lambda p: p["data"].update({"revenue": [None] * len(p["data"]["revenue"])}),
        lambda p: p.update({"covariates": [*p["covariates"], "ghost"]}),
        lambda p: p.update({"treatment": "revenue"}),
        lambda p: p.update({"n_boot": 5}),
    ],
)
def test_malformed_observational_uploads_are_422(client, mutate):
    payload = observational_payload(300)
    mutate(payload)
    assert client.post("/v1/observational/ate", json=payload).status_code == 422


def test_too_many_covariates_is_rejected(client):
    payload = observational_payload(300)
    rng = np.random.default_rng(0)
    for j in range(5):
        payload["data"][f"noise{j}"] = rng.standard_normal(300).tolist()
        payload["covariates"].append(f"noise{j}")
    response = client.post("/v1/observational/ate", json=payload)
    assert response.status_code == 422 and "at most" in response.json()["detail"]


def test_benchmark_endpoint_reads_the_results_file(client, settings):
    assert client.get("/v1/benchmark").status_code == 404
    settings.benchmark_path.write_text(json.dumps({"seed": 42}))
    assert client.get("/v1/benchmark").json() == {"seed": 42}


def test_dashboard_and_static_assets_are_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "causa" in page.text
    assert client.get("/static/app.js").status_code == 200
