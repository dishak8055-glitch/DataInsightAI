# tests/test_multimodal.py

import os
import io
import pandas as pd
from PIL import Image
from modules.multimodal import (
    analyze_chart_image,
    generate_audio_summary,
    generate_dataset_infographic_card,
    get_multimodal_genai_survey
)
import pytest


def test_generate_dataset_infographic_card():
    df = pd.DataFrame({
        "Age": [22, 35, 28, 42],
        "Gender": ["Female", "Male", "Female", "Male"],
        "SpendingScore": [80, 45, 90, 30]
    })
    img_bytes = generate_dataset_infographic_card(df, target_column="SpendingScore")
    assert img_bytes is not None
    assert len(img_bytes) > 1000  # PNG image binary


def test_generate_audio_summary():
    res = generate_audio_summary("Testing audio narration for DataInsight AI dataset summary.")
    assert "audio_bytes" in res
    assert len(res["audio_bytes"]) > 0
    assert os.path.exists(res["file_path"])
    # Clean up temporary test file
    try:
        os.remove(res["file_path"])
    except Exception:
        pass


def test_multimodal_genai_survey():
    survey = get_multimodal_genai_survey()
    assert len(survey) == 4
    modalities = [s["Modality"] for s in survey]
    assert any("Tabular" in m for m in modalities)
    assert any("Image" in m for m in modalities)
    assert any("Video" in m for m in modalities)
    assert any("Music" in m for m in modalities)


def test_chart_image_analysis_rejects_empty_image():
    with pytest.raises(ValueError, match="non-empty"):
        analyze_chart_image(b"", "image/png")


def test_chart_image_analysis_rejects_unsupported_mime_type():
    with pytest.raises(ValueError, match="Supported image types"):
        analyze_chart_image(b"image bytes", "application/pdf")


def test_chart_image_analysis_sends_valid_image_to_gemini(monkeypatch):
    image_buffer = io.BytesIO()
    Image.new("RGB", (10, 10), color="white").save(image_buffer, format="PNG")
    calls = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.update(kwargs)
            return type("Response", (), {"text": "The chart is a line chart."})()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr("modules.multimodal.HAS_GENAI", True)
    monkeypatch.setattr("modules.multimodal.get_gemini_client", lambda: FakeClient())
    result = analyze_chart_image(
        image_buffer.getvalue(), "image/png", "Describe the trend"
    )
    assert result == "The chart is a line chart."
    assert calls["model"] == "gemini-3.8-flash"
