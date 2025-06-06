import logging


def truncate_payload(payload, max_length=200):
    """Truncate payload for logging purposes"""
    payload_str = str(payload)
    if len(payload_str) > max_length:
        return payload_str[:max_length] + "..."
    return payload_str

logger = logging.getLogger("log_analysis")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)
