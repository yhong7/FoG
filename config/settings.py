import yaml
import os

def load_config(env="dev"):
    config_path = os.path.join(os.path.dirname(__file__), f'config-{env}.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


CONFIG = load_config()

