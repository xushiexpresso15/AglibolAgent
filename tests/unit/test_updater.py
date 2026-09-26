"""Unit tests for automated update manager and version comparison."""

from pathlib import Path

from aglibol.utils.updater import UpdateManager, parse_version_tuple


def test_parse_version_tuple():
    assert parse_version_tuple("0.1.0") == (0, 1, 0)
    assert parse_version_tuple("v0.2.1") == (0, 2, 1)
    assert parse_version_tuple("1.10.4-beta") == (1, 10, 4)
    assert parse_version_tuple("v0.2.0") > parse_version_tuple("0.1.9")
    assert parse_version_tuple("0.1.0") == parse_version_tuple("v0.1.0")


def test_updater_cache_ttl(tmp_path: Path):
    updater = UpdateManager(cache_dir=tmp_path)
    assert updater._read_cache() is None

    updater._write_cache(
        {
            "timestamp": 9999999999.0,  # far future
            "latest_version": "9.9.9",
            "release_notes": "Future release",
        }
    )

    # Should read from cache and detect as newer without network
    info = updater.check_for_updates(force=False)
    assert info.is_newer is True
    assert info.latest_version == "9.9.9"
    assert info.source == "cache"


def test_updater_detect_install_method():
    updater = UpdateManager()
    method = updater.detect_install_method()
    assert method in ("git", "uv", "pip")
