import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen


class Ripgrep:
    """Cross-platform ripgrep helper that locates or installs rg."""

    RG_BIN_NAMES = {
        "windows": "rg.exe",
        "darwin": "rg",
        "linux": "rg",
    }
    GITHUB_API_LATEST = "https://api.github.com/repos/BurntSushi/ripgrep/releases/latest"
    INSTALL_ROOT = Path(__file__).resolve().parent / ".bin" / "ripgrep"

    @classmethod
    def _get_platform_bin_name(cls) -> str:
        import platform

        system = platform.system().lower()
        return cls.RG_BIN_NAMES.get(system, "rg")

    @classmethod
    async def filepath(cls) -> str:
        """Locate rg binary, preferring PATH/cache and falling back to download."""
        rg_bin = cls._get_platform_bin_name()
        for candidate in (shutil.which(rg_bin), cls._find_cached_binary()):
            if candidate and cls._is_usable_binary(candidate):
                return candidate

        rg_path = await cls._download_latest_binary()
        if not rg_path or not cls._is_usable_binary(rg_path):
            raise RuntimeError("Failed to locate or install an executable ripgrep binary")
        return rg_path

    @classmethod
    def _is_usable_binary(cls, path: str) -> bool:
        try:
            if not (os.path.isfile(path) and os.access(path, os.X_OK)):
                return False
            # Some binaries on PATH can pass os.access but still fail at process creation.
            probe = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                shell=False,
            )
            return probe.returncode == 0
        except (OSError, subprocess.SubprocessError, PermissionError):
            return False

    @classmethod
    def _find_cached_binary(cls) -> Optional[str]:
        if not cls.INSTALL_ROOT.exists():
            return None

        bin_name = cls._get_platform_bin_name()
        for candidate in cls.INSTALL_ROOT.rglob(bin_name):
            if candidate.is_file() and cls._is_usable_binary(str(candidate)):
                return str(candidate.resolve())
        return None

    @classmethod
    async def _download_latest_binary(cls) -> str:
        release = cls._fetch_latest_release()
        asset = cls._select_asset(release)
        if not asset:
            raise RuntimeError("No compatible ripgrep release asset found for current platform")

        cls.INSTALL_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / asset["name"]
            cls._download_file(asset["browser_download_url"], archive_path)
            cls._extract_archive(archive_path, cls.INSTALL_ROOT)

        binary_path = cls._find_cached_binary()
        if not binary_path:
            raise RuntimeError("ripgrep downloaded but binary not found after extraction")

        try:
            os.chmod(binary_path, 0o755)
        except OSError:
            pass
        return binary_path

    @classmethod
    def _fetch_latest_release(cls) -> Dict[str, Any]:
        req = Request(
            cls.GITHUB_API_LATEST,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "llmautotest-agent",
            },
        )
        with urlopen(req, timeout=20) as resp:
            payload = resp.read().decode("utf-8")

        data = json.loads(payload)
        if not isinstance(data, dict):
            raise RuntimeError("Invalid GitHub release response")
        return data

    @classmethod
    def _asset_suffix(cls) -> str:
        import platform

        system = platform.system().lower()
        machine = platform.machine().lower()
        normalized = {
            "amd64": "x86_64",
            "x64": "x86_64",
            "x86_64": "x86_64",
            "arm64": "aarch64",
            "aarch64": "aarch64",
        }.get(machine, machine)

        if system == "windows":
            if normalized == "aarch64":
                return "aarch64-pc-windows-msvc.zip"
            return "x86_64-pc-windows-msvc.zip"
        if system == "darwin":
            if normalized == "aarch64":
                return "aarch64-apple-darwin.tar.gz"
            return "x86_64-apple-darwin.tar.gz"
        if system == "linux":
            if normalized == "aarch64":
                return "aarch64-unknown-linux-musl.tar.gz"
            return "x86_64-unknown-linux-musl.tar.gz"
        raise RuntimeError(f"Unsupported platform: {system}")

    @classmethod
    def _select_asset(cls, release: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        assets = release.get("assets", [])
        suffix = cls._asset_suffix()
        for asset in assets:
            name = str(asset.get("name", ""))
            if name.endswith(suffix):
                return asset

        if suffix.startswith("aarch64"):
            fallback_suffix = suffix.replace("aarch64", "x86_64")
            for asset in assets:
                name = str(asset.get("name", ""))
                if name.endswith(fallback_suffix):
                    return asset
        return None

    @classmethod
    def _download_file(cls, url: str, destination: Path) -> None:
        req = Request(url, headers={"User-Agent": "llmautotest-agent"})
        with urlopen(req, timeout=120) as resp, open(destination, "wb") as out:
            out.write(resp.read())

    @classmethod
    def _extract_archive(cls, archive_path: Path, extract_to: Path) -> None:
        archive_name = archive_path.name.lower()
        if archive_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(extract_to)
            return
        if archive_name.endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tf:
                tf.extractall(extract_to)
            return
        raise RuntimeError(f"Unsupported ripgrep archive format: {archive_path.name}")
