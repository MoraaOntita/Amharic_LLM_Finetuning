import re

def preprocess_dataset(dataset, text_field, preprocessor):
    def preprocess_example(example):
        if text_field in example:
            example[text_field] = preprocessor.preprocess_text(example[text_field])
        return example

    return dataset.map(preprocess_example)