# -----------------------------------------------------------------------------
# Role: Exports the OpenAI STT block package API.
# File Name: __init__.py
# Author: Alexandre EL
# Email: alex@hackinvent.com
# Created Date: 2024-10-06
# -----------------------------------------------------------------------------

from .block import OpenAiSttBlock, OpenAiSttBlockError

__all__ = ["OpenAiSttBlock", "OpenAiSttBlockError"]
