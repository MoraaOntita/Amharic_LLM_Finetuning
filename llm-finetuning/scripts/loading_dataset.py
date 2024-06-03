from datasets import load_dataset
from utils.common import preprocess_dataset
from utils.config import DATASET_NAMES, TEXT_FIELDS

def load_and_preprocess_datasets(preprocessor):
    loaded_datasets = {}
    
    for name in DATASET_NAMES:
        dataset = load_dataset(name)
        print(f"Dataset: {name}")
        
        for split in dataset.keys():
            print(f"  Split: {split}")
            print(f"  Fields: {dataset[split].column_names}")

            for text_field in TEXT_FIELDS:
                if text_field in dataset[split].column_names:
                    print(f"Preprocessing field: {text_field} in split: {split}")
                    dataset[split] = preprocess_dataset(dataset[split], text_field, preprocessor)

        loaded_datasets[name] = dataset
        print()

    return loaded_datasets
