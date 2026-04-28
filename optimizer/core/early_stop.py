def conditions_enabled(config):
    return config.early_stop_conditions if config.early_stop_enabled else []
