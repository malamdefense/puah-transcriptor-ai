"""Processing pipeline: Audio extraction -> transcription -> speaker identification -> merging."""

from __future__ import annotations

import importlib.util


def _is_absent(module: str) -> bool:
    """Is `module` genuinely not installed? Asked of the import system, never inferred from the
    exception — the exception's shape lies in both directions:

      * `from google import genai` raises a plain ImportError (not ModuleNotFoundError) naming
        'google', because `google` is an implicit namespace package shared with google-auth and
        protobuf. Judged by class, a missing google-genai looks "installed".
      * a missing *sibling* under such a namespace — pyannote.core, google.auth — raises a
        ModuleNotFoundError naming something that is not `module` at all. Judged by prefix, an
        installed pyannote.audio looks "not installed".

    find_spec locates a module without executing it, so a package that is present but unloadable
    still reports as present, which is exactly the distinction we need.
    """
    try:
        return importlib.util.find_spec(module) is None
    except ModuleNotFoundError:
        return True   # a parent package does not exist, so neither does `module`
    except (ImportError, ValueError, AttributeError, TypeError):
        # A parent is present but refused to import: a load failure, not absence. Erring towards
        # "present" is deliberate — wrongly saying "not installed" is the bug this helper exists
        # to kill, and the exception text in the message still names the real failure.
        return False


def import_failure_reason(
    exc: ImportError, module: str, install_hint: str, *, package: str | None = None
) -> str:
    """Explain an ImportError precisely: genuinely absent vs present-but-unloadable.

    A blanket "X is not installed" is wrong, and actively misleading, when the package is in
    fact installed and something underneath it failed to load — a native extension refused by a
    Windows Application Control policy, a missing CUDA/FFmpeg runtime, an ABI mismatch. That
    message sends whoever is on call to reinstall a package that is already there (this cost us
    a diagnosis once: a blocked torch DLL was reported as "pyannote.audio is not installed").

    `module` is the importable name ("pyannote.audio"); `package` is the pip name when it
    differs ("google-genai"). `exc` supplies the reason text; presence is decided by _is_absent.
    """
    name = package or module
    if _is_absent(module):
        return f"{name} is not installed. Run: {install_hint}"
    return (
        f"{name} is installed but could not be loaded: {exc}. "
        "Reinstalling it will not help — fix the failure reported above."
    )
