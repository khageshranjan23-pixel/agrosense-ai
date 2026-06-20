"""
AgroSense AI — Google Earth Engine Authentication Manager
Handles GEE auth via service account (Streamlit Cloud) or interactive (local).
Falls back gracefully to demo/cached mode when credentials unavailable.
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Try importing GEE — mark unavailable if not installed
try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False
    logger.warning("earthengine-api not installed — running in offline mode")

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def _authenticate_service_account(service_account_json: str) -> Tuple[bool, str]:
    """Authenticate GEE using a service account JSON string.

    Args:
        service_account_json: JSON string of service account credentials.

    Returns:
        Tuple of (success: bool, message: str).
    """
    if not GEE_AVAILABLE:
        return False, "earthengine-api not installed"
    try:
        cred_dict = json.loads(service_account_json)
        credentials = ee.ServiceAccountCredentials(
            email=cred_dict["client_email"],
            key_data=service_account_json,
        )
        ee.Initialize(credentials)
        # Quick connectivity test
        _ = ee.Number(1).add(1).getInfo()
        return True, "GEE authenticated via service account"
    except json.JSONDecodeError as exc:
        return False, f"Service account JSON is malformed: {exc}"
    except KeyError as exc:
        return False, f"Service account JSON missing required field: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Service account auth failed: {exc}"


def _authenticate_application_default() -> Tuple[bool, str]:
    """Authenticate GEE using Application Default Credentials (local dev).

    Attempts a silent initialize using ADC already present on the machine.
    Does not attempt interactive ee.Authenticate() to prevent hanging the server thread.

    Returns:
        Tuple of (success: bool, message: str).
    """
    if not GEE_AVAILABLE:
        return False, "earthengine-api not installed"

    # Silent path: ADC already present on the machine
    try:
        ee.Initialize()
        # Quick connectivity test
        _ = ee.Number(1).add(1).getInfo()
        return True, "GEE authenticated via Application Default Credentials"
    except Exception as exc:  # noqa: BLE001
        return False, f"Application Default Credentials auth failed: {exc}"


def initialize_gee() -> Tuple[bool, str]:
    """Initialize Google Earth Engine with the best available auth method.

    Tries authentication sources in priority order:

    1. **Streamlit secrets** — looks for ``gee_service_account`` key inside
       ``st.secrets``; accepts both raw JSON strings and TOML table objects.
    2. **Environment variable** — ``GEE_SERVICE_ACCOUNT_JSON`` containing the
       full service account JSON as a string.
    3. **Application Default Credentials / interactive** — uses
       ``ee.Initialize()`` with locally stored ADC, or falls back to
       ``ee.Authenticate()`` for interactive local development.
    4. **Demo mode** — returns ``(False, …)`` when all paths fail; the caller
       is expected to serve pre-computed cached data instead.

    Returns:
        Tuple of (is_authenticated: bool, status_message: str).
    """
    if not GEE_AVAILABLE:
        return False, "GEE library not available — install earthengine-api"

    # 1. Streamlit secrets ───────────────────────────────────────────────────
    if STREAMLIT_AVAILABLE:
        import os
        # Check if secrets file exists first to prevent Streamlit from raising FileNotFoundError
        secrets_exist = False
        try:
            if hasattr(st.secrets, "_file_paths"):
                for path in st.secrets._file_paths:
                    if os.path.exists(path):
                        secrets_exist = True
                        break
            else:
                # Fallback to standard locations
                for path_str in [
                    ".streamlit/secrets.toml",
                    "../.streamlit/secrets.toml",
                    "agrosense-ai/.streamlit/secrets.toml",
                    os.path.expanduser("~/.streamlit/secrets.toml"),
                ]:
                    if os.path.exists(path_str):
                        secrets_exist = True
                        break
        except Exception:
            pass

        if secrets_exist:
            try:
                secrets = st.secrets
                if "gee_service_account" in secrets:
                    sa_json = secrets["gee_service_account"]
                    # st.secrets may return an AttrDict when loaded from TOML
                    if not isinstance(sa_json, str):
                        sa_json = json.dumps(dict(sa_json))
                    ok, msg = _authenticate_service_account(str(sa_json))
                    if ok:
                        logger.info(msg)
                        return True, msg
                    logger.warning("Streamlit secret auth failed: %s", msg)
            except Exception as exc:  # noqa: BLE001
                logger.debug("No Streamlit secrets available: %s", exc)

    # 2. Environment variable ────────────────────────────────────────────────
    sa_env = os.environ.get("GEE_SERVICE_ACCOUNT_JSON", "").strip()
    if sa_env:
        ok, msg = _authenticate_service_account(sa_env)
        if ok:
            logger.info(msg)
            return True, msg
        logger.warning("Env var auth failed: %s", msg)

    # 3. Application Default Credentials / interactive ───────────────────────
    ok, msg = _authenticate_application_default()
    if ok:
        logger.info(msg)
        return True, msg
    logger.warning("ADC / interactive auth failed: %s", msg)

    # 4. Demo mode ───────────────────────────────────────────────────────────
    return False, "GEE authentication failed — running in demo mode with cached data"


def is_gee_initialized() -> bool:
    """Check whether GEE has been successfully initialized.

    Performs a lightweight ``getInfo`` call rather than relying on internal
    GEE state flags, which vary across ``earthengine-api`` versions.

    Returns:
        True if GEE is ready to use, False otherwise.
    """
    if not GEE_AVAILABLE:
        return False
    try:
        ee.Number(1).getInfo()
        return True
    except Exception:  # noqa: BLE001
        return False


def get_gee_status() -> dict:
    """Return a dict describing the current GEE connectivity status.

    Returns:
        Dict with keys:
        - ``gee_available`` (bool): Whether the ``earthengine-api`` package is installed.
        - ``gee_initialized`` (bool): Whether a live GEE session is active.
        - ``mode`` (str): ``"live"`` or ``"demo"``.
        - ``message`` (str): Human-readable status suitable for display in the UI.
    """
    initialized = is_gee_initialized()
    return {
        "gee_available": GEE_AVAILABLE,
        "gee_initialized": initialized,
        "mode": "live" if initialized else "demo",
        "message": (
            "Connected to Google Earth Engine — live satellite data active"
            if initialized
            else "Demo mode — displaying pre-computed sample data"
        ),
    }
