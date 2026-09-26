"""Unit tests for the Graphical TUI Selector and Model Catalog Helper."""

from aglibol.cli.interactive.selector import ModelCatalogHelper, SelectorItem, TUISelector
from aglibol.ollama.models import ModelInfo


def test_model_catalog_helper_classification():
    models = [
        ModelInfo(
            name="nomic-embed-text:latest", size_bytes=300_000_000, digest="d1", modified_at="2026"
        ),
        ModelInfo(
            name="qwen2.5-coder:7b",
            size_bytes=4_700_000_000,
            digest="d2",
            modified_at="2026",
            parameter_size="7.0B",
            quantization_level="Q4_K_M",
        ),
        ModelInfo(
            name="glm4:9b",
            size_bytes=5_500_000_000,
            digest="d3",
            modified_at="2026",
            parameter_size="9.0B",
            quantization_level="Q4_0",
        ),
        ModelInfo(name="glm-ocr:latest", size_bytes=900_000_000, digest="d4", modified_at="2026"),
    ]

    # Test Coder ranking
    coder_items = ModelCatalogHelper.classify_and_build_items(
        models, role="coder", current_model="qwen2.5-coder:7b"
    )
    assert len(coder_items) == 4

    # Top item should be qwen2.5-coder:7b
    top_coder = coder_items[0]
    assert top_coder.key == "qwen2.5-coder:7b"
    assert "BEST FOR CODER" in top_coder.badge
    assert "7.0B" in top_coder.specs

    # Bottom item should be nomic-embed-text
    bottom_item = coder_items[-1]
    assert bottom_item.key == "nomic-embed-text:latest"
    assert "Embedding Only" in bottom_item.badge
    assert bottom_item.is_disabled is True


def test_tui_selector_fallback_numeric_selection(monkeypatch):
    items = [
        SelectorItem(key="planner", label="Planner", specs="Current model: qwen2.5"),
        SelectorItem(key="coder", label="Coder", specs="Current model: qwen2.5-coder"),
        SelectorItem(key="reviewer", label="Reviewer", specs="Current model: qwen2.5"),
    ]

    # Mock user input "2" (Coder)
    monkeypatch.setattr("rich.prompt.Prompt.ask", lambda prompt, default="": "2")

    chosen = TUISelector.choose(
        title="Test Role Selection",
        items=items,
        default_index=0,
    )
    assert chosen == "coder"


def test_tui_selector_fallback_cancel(monkeypatch):
    items = [
        SelectorItem(key="opt1", label="Option 1"),
        SelectorItem(key="opt2", label="Option 2"),
    ]

    # Mock user input "0" or "cancel"
    monkeypatch.setattr("rich.prompt.Prompt.ask", lambda prompt, default="": "0")

    chosen = TUISelector.choose(
        title="Test Cancel",
        items=items,
    )
    assert chosen is None


def test_tui_selector_fallback_custom_input(monkeypatch):
    items = [
        SelectorItem(key="m1", label="Model 1"),
    ]

    # Simulate choosing custom, then entering tag
    prompts = iter(["2", "my-custom-model:latest"])  # 2 is custom item index
    monkeypatch.setattr("rich.prompt.Prompt.ask", lambda prompt, default="": next(prompts))

    chosen = TUISelector.choose(
        title="Test Custom",
        items=items,
        allow_custom=True,
    )
    assert chosen == "my-custom-model:latest"
