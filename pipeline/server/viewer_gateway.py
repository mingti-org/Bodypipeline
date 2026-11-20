def embedded_enabled(settings) -> bool:
    config = getattr(settings, "viewer_gateway", {})
    return isinstance(config, dict) and config.get("embedded") is True
