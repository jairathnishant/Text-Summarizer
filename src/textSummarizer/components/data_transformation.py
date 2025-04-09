import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        
    def convert_examples_to_features(self, ex_batch):
        input_enc = self.tokenizer(ex_batch['dialogue'], max_length = 1024, truncation = True)

        with self.tokenizer.as_target_tokenizer():
            label_enc = self.tokenizer(ex_batch['summary'], max_length = 128, truncation = True)

        return {
            'input_ids' : input_enc['input_ids'],
            'attention_mask' : input_enc['attention_mask'],
            'labels' : label_enc['input_ids']
        }
    
    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched = True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))
