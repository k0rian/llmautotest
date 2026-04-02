import copy
import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

from tools.core import WorkspaceGuard

from .manager import ensure_workspace_file, get_session, list_sessions, start_session, stop_session


def _normalize_extensions(values: list[str] | tuple[str, ...] | set[str]) -> set[str]:
    result: set[str] = set()
    for item in values:
        text = str(item).strip().lower()
        if not text:
            continue
        if not text.startswith("."):
            text = f".{text}"
        result.add(text)
    return result


def _load_workspace_config(workspace_path: str) -> dict[str, Any]:
    candidates = [
        Path(workspace_path) / "config.yml",
        Path(workspace_path) / "config.yaml",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            with open(path, "r", encoding="utf-8") as file:
                payload = yaml.safe_load(file) or {}
            if isinstance(payload, dict):
                return payload
        except Exception:
            continue
    return {}


@dataclass
class LSPServerProfile:
    name: str
    command: list[str]
    extensions: set[str]
    env: dict[str, str] = field(default_factory=dict)
    initialization: dict[str, Any] = field(default_factory=dict)
    disabled: bool = False
    install_command: list[str] = field(default_factory=list)

    def supports_file(self, file_path: str) -> bool:
        suffix = Path(file_path).suffix.lower()
        return bool(suffix and suffix in self.extensions)

    def executable(self) -> str:
        if not self.command:
            return ""
        return str(self.command[0]).strip()

    def is_available(self) -> bool:
        exe = self.executable()
        if not exe:
            return False
        return shutil.which(exe) is not None


def _builtin_profiles() -> dict[str, LSPServerProfile]:

    profiles = [
        LSPServerProfile(
            name="python",
            command=["pyright-langserver", "--stdio"],
            extensions=_normalize_extensions({".py", ".pyi"}),
            install_command=["npm", "install", "-g", "pyright"],
        ),
        LSPServerProfile(
            name="typescript",
            command=["typescript-language-server", "--stdio"],
            extensions=_normalize_extensions(
                {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts"}
            ),
            install_command=["npm", "install", "-g", "typescript", "typescript-language-server"],
        ),
        LSPServerProfile(
            name="gopls",
            command=["gopls"],
            extensions=_normalize_extensions({".go"}),
            install_command=["go", "install", "golang.org/x/tools/gopls@latest"],
        ),
        LSPServerProfile(
            name="rust",
            command=["rust-analyzer"],
            extensions=_normalize_extensions({".rs"}),
        ),
        LSPServerProfile(
            name="jdtls",
            command=["jdtls"],
            extensions=_normalize_extensions({".java"}),
        ),
        LSPServerProfile(
            name="clangd",
            command=["clangd"],
            extensions=_normalize_extensions(
                {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}
            ),
        ),
        LSPServerProfile(
            name="yaml-ls",
            command=["yaml-language-server", "--stdio"],
            extensions=_normalize_extensions({".yaml", ".yml"}),
            install_command=["npm", "install", "-g", "yaml-language-server"],
        ),
    ]
    return {profile.name: profile for profile in profiles}


def _merge_profile(base: LSPServerProfile, override: dict[str, Any]) -> LSPServerProfile:
    profile = copy.deepcopy(base)
    if "disabled" in override:
        profile.disabled = bool(override.get("disabled"))
    if "command" in override and isinstance(override.get("command"), list):
        profile.command = [str(item).strip() for item in override.get("command", []) if str(item).strip()]
    if "extensions" in override and isinstance(override.get("extensions"), list):
        profile.extensions = _normalize_extensions(override.get("extensions", []))
    if "env" in override and isinstance(override.get("env"), dict):
        profile.env = {str(k): str(v) for k, v in override.get("env", {}).items()}
    if "initialization" in override and isinstance(override.get("initialization"), dict):
        profile.initialization = copy.deepcopy(override.get("initialization", {}))
    if "install" in override and isinstance(override.get("install"), list):
        profile.install_command = [str(item).strip() for item in override.get("install", []) if str(item).strip()]
    return profile


def build_profiles(config: dict[str, Any] | None = None) -> dict[str, LSPServerProfile]:
    profiles = _builtin_profiles()
    if not isinstance(config, dict):
        return profiles
    lsp_section = config.get("lsp")
    if lsp_section is False:
        return {}
    if not isinstance(lsp_section, dict):
        return profiles
    for name, raw in lsp_section.items():
        key = str(name).strip()
        if not key or not isinstance(raw, dict):
            continue
        if key in profiles:
            profiles[key] = _merge_profile(profiles[key], raw)
            continue
        command = raw.get("command", [])
        extensions = raw.get("extensions", [])
        if not isinstance(command, list) or not isinstance(extensions, list):
            continue
        custom = LSPServerProfile(
            name=key,
            command=[str(item).strip() for item in command if str(item).strip()],
            extensions=_normalize_extensions(extensions),
            env={str(k): str(v) for k, v in raw.get("env", {}).items()} if isinstance(raw.get("env"), dict) else {},
            initialization=copy.deepcopy(raw.get("initialization", {}))
            if isinstance(raw.get("initialization"), dict)
            else {},
            disabled=bool(raw.get("disabled", False)),
            install_command=[str(item).strip() for item in raw.get("install", []) if str(item).strip()]
            if isinstance(raw.get("install"), list)
            else [],
        )
        profiles[key] = custom
    return profiles


class LSPServer:
    def __init__(self, workspace_path: str, config: dict[str, Any] | None = None):
        absolute_workspace = os.path.abspath(workspace_path)
        if not os.path.isdir(absolute_workspace):
            raise ValueError(f"invalid workspace path '{workspace_path}'")
        self.workspace_path = absolute_workspace
        effective_config = config if isinstance(config, dict) else _load_workspace_config(absolute_workspace)
        self.profiles = build_profiles(config=effective_config)
        digest = hashlib.sha1(absolute_workspace.encode("utf-8")).hexdigest()[:10]
        self._session_prefix = f"auto-lsp:{digest}:"
        self._install_attempted: set[str] = set()
        self._install_logs: dict[str, dict[str, Any]] = {}

    def available_profiles(self) -> list[dict[str, Any]]:
        data: list[dict[str, Any]] = []
        for profile in self.profiles.values():
            data.append(
                {
                    "name": profile.name,
                    "disabled": profile.disabled,
                    "available": profile.is_available(),
                    "extensions": sorted(profile.extensions),
                    "command": profile.command,
                    "auto_install": bool(profile.install_command),
                    "install_log": self._install_logs.get(profile.name, {}),
                }
            )
        return sorted(data, key=lambda item: item["name"])

    def _ensure_profile_available(self, profile: LSPServerProfile) -> bool:
        if profile.is_available():
            return True
        if not profile.install_command:
            self._install_logs[profile.name] = {"status": "skipped", "reason": "no_install_command"}
            return False
        if profile.name in self._install_attempted:
            return profile.is_available()
        self._install_attempted.add(profile.name)
        installer = str(profile.install_command[0]).strip() if profile.install_command else ""
        if not installer or shutil.which(installer) is None:
            self._install_logs[profile.name] = {
                "status": "failed",
                "reason": f"installer_not_found:{installer or 'unknown'}",
            }
            return False
        try:
            completed = subprocess.run(
                profile.install_command,
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                timeout=300,
                shell=False,
            )
            self._install_logs[profile.name] = {
                "status": "ok" if completed.returncode == 0 else "failed",
                "return_code": completed.returncode,
                "stdout": (completed.stdout or "")[-4000:],
                "stderr": (completed.stderr or "")[-4000:],
                "install_command": profile.install_command,
            }
        except Exception as exc:
            self._install_logs[profile.name] = {
                "status": "failed",
                "reason": str(exc),
                "install_command": profile.install_command,
            }
            return False
        return profile.is_available()

    def profile_for_file(self, file_path: str, preferred_server: str = "") -> LSPServerProfile:
        absolute = os.path.abspath(file_path)
        WorkspaceGuard.ensure_under_workspace(self.workspace_path, absolute)
        preferred = preferred_server.strip()
        if preferred:
            profile = self.profiles.get(preferred)
            if not profile:
                raise ValueError(f"lsp server not found: {preferred}")
            if profile.disabled:
                raise ValueError(f"lsp server disabled: {preferred}")
            if not profile.supports_file(absolute):
                raise ValueError(f"lsp server '{preferred}' does not support file: {absolute}")
            if not self._ensure_profile_available(profile):
                raise ValueError(f"lsp server command unavailable: {profile.executable()}")
            return profile
        supported_profiles = [profile for profile in self.profiles.values() if not profile.disabled and profile.supports_file(absolute)]
        for profile in supported_profiles:
            if profile.disabled:
                continue
            if not profile.is_available():
                continue
            return profile
        for profile in supported_profiles:
            if self._ensure_profile_available(profile):
                return profile
        suffix = Path(absolute).suffix.lower()
        supported = [item.name for item in supported_profiles]
        if not self.profiles:
            raise ValueError("all lsp servers are disabled by config")
        if supported:
            raise ValueError(f"lsp server matched but executable missing: {', '.join(sorted(supported))}")
        raise ValueError(f"no configured lsp server for extension: '{suffix or '<none>'}'")

    def session_id_for_profile(self, profile_name: str) -> str:
        key = profile_name.strip()
        if not key:
            raise ValueError("profile_name cannot be empty")
        return f"{self._session_prefix}{key}"

    def ensure_session(
        self,
        file_path: str,
        preferred_server: str = "",
        trace: str = "off",
    ) -> dict[str, Any]:
        absolute = os.path.abspath(file_path)
        if not os.path.isfile(absolute):
            raise ValueError(f"file not found '{file_path}'")
        profile = self.profile_for_file(absolute, preferred_server=preferred_server)
        session_id = self.session_id_for_profile(profile.name)
        result = start_session(
            session_id=session_id,
            command=profile.command,
            workspace_path=self.workspace_path,
            initialization_options=profile.initialization,
            env=profile.env,
            trace=trace,
        )
        return {
            "session_id": session_id,
            "server": profile.name,
            "workspace_path": self.workspace_path,
            "file_path": absolute,
            "start_result": result,
        }

    def ensure_open_document(
        self,
        file_path: str,
        preferred_server: str = "",
        language_id: str = "",
        trace: str = "off",
    ) -> dict[str, Any]:
        info = self.ensure_session(file_path=file_path, preferred_server=preferred_server, trace=trace)
        sess = get_session(info["session_id"])
        absolute = ensure_workspace_file(sess, file_path)
        opened = sess.did_open(file_path=absolute, language_id=language_id)
        return {**info, "open_result": opened}

    def stop_managed_sessions(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for item in list_sessions():
            sid = str(item.get("session_id", ""))
            if sid.startswith(self._session_prefix):
                result.append(stop_session(sid))
        return result


def ensure_session_for_file(
    workspace_path: str,
    file_path: str,
    preferred_server: str = "",
    config: dict[str, Any] | None = None,
    trace: str = "off",
) -> dict[str, Any]:
    return LSPServer(workspace_path=workspace_path, config=config).ensure_session(
        file_path=file_path,
        preferred_server=preferred_server,
        trace=trace,
    )


def open_document_with_auto_server(
    workspace_path: str,
    file_path: str,
    preferred_server: str = "",
    language_id: str = "",
    config: dict[str, Any] | None = None,
    trace: str = "off",
) -> dict[str, Any]:
    return LSPServer(workspace_path=workspace_path, config=config).ensure_open_document(
        file_path=file_path,
        preferred_server=preferred_server,
        language_id=language_id,
        trace=trace,
    )
