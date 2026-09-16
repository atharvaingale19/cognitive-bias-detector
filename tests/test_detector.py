from src.bias_detector import analyze_text


def test_analyze_text_returns_result():

    result = analyze_text(
        "He ignored me because he must hate me."
    )

    assert result is not None


def test_result_contains_required_fields():

    result = analyze_text(
        "I failed one exam, so I will always fail."
    )

    assert "prediction" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "important_features" in result


def test_confidence_is_valid():

    result = analyze_text(
        "Everyone in that group is terrible."
    )

    assert 0.0 <= result["confidence"] <= 1.0


def test_empty_input():

    result = analyze_text("")

    assert result is None