# tests/test_multimodal.py

import os
import pandas as pd
from modules.multimodal import (
    generate_audio_summary,
    generate_dataset_infographic_card,
    get_multimodal_genai_survey
)


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
