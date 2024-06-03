import logging
from typing import Dict, Any
from datasets import load_dataset
from utils.common import preprocess_dataset
from utils.config import DATASET_NAMES, TEXT_FIELDS

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def exception_handler(func):
    """Decorator to handle exceptions in functions."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

@exception_handler
def load_and_preprocess_datasets(preprocessor: Any) -> Dict[str, Dict[str, Any]]:
    """
    Load and preprocess datasets.

    Args:
        preprocessor (Any): Preprocessing function or class.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary containing preprocessed datasets.
    """
    loaded_datasets = {}

    for name in DATASET_NAMES:
        logger.info(f"Loading dataset: {name}")
        dataset = load_dataset(name)

        for split in dataset.keys():
            logger.info(f"Processing split: {split}")
            logger.info(f"Fields: {dataset[split].column_names}")

            for text_field in TEXT_FIELDS:
                if text_field in dataset[split].column_names:
                    logger.info(f"Preprocessing field: {text_field} in split: {split}")
                    dataset[split] = preprocess_dataset(dataset[split], text_field, preprocessor)

        loaded_datasets[name] = dataset
        logger.info(f"Completed processing dataset: {name}\n")

    return loaded_datasets
