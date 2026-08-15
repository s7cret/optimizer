from pathlib import Path


def test_openpine_contracts_pin_is_exact_git_sha() -> None:
    text = Path(__file__).resolve().parents[2].joinpath("pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert "openpine-contracts==" not in text
    assert "git+https://github.com/s7cret/openpine-contracts.git@" in text
    marker = "openpine-contracts.git@"
    start = text.index(marker) + len(marker)
    sha = "".join(ch for ch in text[start:] if ch.isalnum())[:40]
    assert len(sha) == 40
    assert all(ch in "0123456789abcdef" for ch in sha)
