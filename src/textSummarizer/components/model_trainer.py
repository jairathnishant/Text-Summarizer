from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import ModelTrainerConfig
import torch
import os

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model = model_pegasus)

        # loading data
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        # Use only first 100 samples from each split
        train_dataset = dataset_samsum_pt["train"].select(range(10))
        eval_dataset = dataset_samsum_pt["test"].select(range(10))

        trainer_args = TrainingArguments(
            output_dir = self.config.root_dir,
            # data_path = self.config.data_path,
            # model_ckpt = self.config.model_ckpt,
            num_train_epochs = self.config.num_train_epochs,
            warmup_steps = self.config.warmup_steps,
            per_device_train_batch_size = self.config.per_device_train_batch_size,
            per_device_eval_batch_size = self.config.per_device_eval_batch_size,
            weight_decay = self.config.weight_decay,
            logging_steps = self.config.logging_steps,
            # evaluation_strategy = self.config.evaluation_strategy,
            eval_steps = self.config.eval_steps,
            save_steps = self.config.save_steps,
            gradient_accumulation_steps = self.config.gradient_accumulation_steps
        )

        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        trainer.train()

        # save  model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum.model"))

        # save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer.model"))
