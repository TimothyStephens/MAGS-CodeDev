"""Tests for container Dockerfile generation.

P0-2 regression: ``_generate_dockerfile_content`` used to call
``backend.dockerfile_content()`` which was undefined on ``PythonBackend``
(and absent from the ``LanguageBackend`` Protocol), raising
``AttributeError`` on the podman/docker image-build path. These tests pin
that the method exists, returns the Python toolchain, and is actually
emitted into the generated Dockerfile without raising.
"""

from mags_codedev.backends.python import PythonBackend
from mags_codedev.utils.docker_ops import _generate_dockerfile_content


class TestDockerfileContent:
    def test_python_backend_implements_dockerfile_content(self):
        """PythonBackend must expose dockerfile_content() returning toolchain deps."""
        backend = PythonBackend()
        content = backend.dockerfile_content()
        assert isinstance(content, str)
        assert "pytest" in content
        assert "flake8" in content
        assert "mypy" in content

    def test_generate_dockerfile_includes_backend_content(self, tmp_path, monkeypatch):
        """Generated Dockerfile must include the backend's dockerfile_content()."""
        (tmp_path / "requirements.txt").write_text("pydantic\n")
        monkeypatch.chdir(tmp_path)

        backend = PythonBackend()
        dockerfile = _generate_dockerfile_content({"settings": {}}, backend)

        assert "FROM python:3.11-slim" in dockerfile
        assert "WORKDIR /project" in dockerfile
        assert "COPY requirements.txt ." in dockerfile
        assert backend.dockerfile_content() in dockerfile

    def test_generate_dockerfile_without_requirements(self, tmp_path, monkeypatch):
        """No requirements.txt -> no COPY/RUN requirements lines, backend extras still present."""
        monkeypatch.chdir(tmp_path)

        backend = PythonBackend()
        dockerfile = _generate_dockerfile_content({"settings": {}}, backend)

        assert "FROM python:3.11-slim" in dockerfile
        assert "COPY requirements.txt" not in dockerfile
        assert backend.dockerfile_content() in dockerfile
