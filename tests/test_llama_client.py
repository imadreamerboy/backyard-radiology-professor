from radiology_trainer.adapters.llama_client import LlamaCppClient


def test_load_model_is_idempotent(monkeypatch) -> None:
    client = LlamaCppClient("http://127.0.0.1:8080/v1")
    monkeypatch.setattr(
        LlamaCppClient,
        "list_models",
        lambda self: [{"id": "professor", "status": {"value": "loaded"}}],
    )
    monkeypatch.setattr(
        "radiology_trainer.adapters.llama_client.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected POST")),
    )

    client.load_model("professor")


def test_unload_model_is_idempotent(monkeypatch) -> None:
    client = LlamaCppClient("http://127.0.0.1:8080/v1")
    monkeypatch.setattr(
        LlamaCppClient,
        "list_models",
        lambda self: [{"id": "professor", "status": {"value": "unloaded"}}],
    )
    monkeypatch.setattr(
        "radiology_trainer.adapters.llama_client.requests.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected POST")),
    )

    client.unload_model("professor")
