from cardimech.validation import run_reference_validation


def test_reference_validation_passes() -> None:
    result = run_reference_validation()
    assert result["passed"] is True
    assert result["checks"]
    assert all(result["checks"].values())
