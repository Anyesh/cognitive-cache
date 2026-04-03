import json
import os
import subprocess
import sys
import tempfile

import pytest

from cognitive_cache.cli import main


def _create_fake_repo(tmpdir):
    subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t.com"], cwd=tmpdir, capture_output=True
    )
    subprocess.run(["git", "config", "user.name", "T"], cwd=tmpdir, capture_output=True)

    os.makedirs(os.path.join(tmpdir, "src"))
    with open(os.path.join(tmpdir, "src/auth.py"), "w") as f:
        f.write("def login(username, password):\n    return True\n")
    with open(os.path.join(tmpdir, "src/db.py"), "w") as f:
        f.write("def get_user(name):\n    return name\n")

    subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmpdir, capture_output=True)


def test_cli_default_output(capsys, monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cognitive-cache",
                "select",
                "--repo",
                tmpdir,
                "--task",
                "fix the login bug",
            ],
        )
        main()
        output = capsys.readouterr().out
        assert "src/auth.py" in output
        assert "files selected" in output


def test_cli_json_output(capsys, monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cognitive-cache",
                "select",
                "--repo",
                tmpdir,
                "--task",
                "fix login",
                "--json",
            ],
        )
        main()
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "files" in data
        assert "total_tokens" in data
        assert "budget" in data


def test_cli_output_file(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        output_path = os.path.join(tmpdir, "context.txt")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cognitive-cache",
                "select",
                "--repo",
                tmpdir,
                "--task",
                "fix login",
                "--output",
                output_path,
            ],
        )
        main()
        assert os.path.exists(output_path)
        content = open(output_path).read()
        assert "login" in content


def test_cli_custom_budget(capsys, monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        _create_fake_repo(tmpdir)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cognitive-cache",
                "select",
                "--repo",
                tmpdir,
                "--task",
                "fix login",
                "--budget",
                "50",
                "--json",
            ],
        )
        main()
        output = capsys.readouterr().out
        data = json.loads(output)
        assert data["total_tokens"] <= 50


def test_cli_nonexistent_repo(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cognitive-cache",
            "select",
            "--repo",
            "/nonexistent/repo",
            "--task",
            "fix bug",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1
