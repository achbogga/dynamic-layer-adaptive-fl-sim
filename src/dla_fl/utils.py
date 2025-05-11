import yaml
import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s'
    )


def load_config(path: str = 'configs/config.yaml') -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)