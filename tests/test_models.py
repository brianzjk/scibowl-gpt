from scibowl.generate.local_model import HeuristicWriterModel, build_writer_model


def test_build_writer_model_defaults_to_heuristic(monkeypatch) -> None:
    monkeypatch.delenv("SCIBOWL_WRITER_PROVIDER", raising=False)
    monkeypatch.delenv("SCIBOWL_WRITER_MODEL", raising=False)
    monkeypatch.delenv("SCIBOWL_WRITER_BASE_URL", raising=False)

    model = build_writer_model()

    assert isinstance(model, HeuristicWriterModel)
