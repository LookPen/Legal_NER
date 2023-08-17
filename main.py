import json
import os

from src.args import initialize_traing_args
from src.hf_trainer import HFTrainer
from src.model import PromptNER
from src.process import  Processor
from transformers import AutoTokenizer, AutoConfig, default_data_collator

from src.ner_f1 import load_ner_metric
from datasets import load_dataset

if __name__ == "__main__":
    model_path_or_name = r"D:\Code\huggingface\roberta-wwm-ext"
    cache_dir = "./cache_dir"
    output_dir = "./output"
    max_seq_length = 512
    num_train_epochs = 10
    # 0817 evaluation_strategy和save_strategy、eval_steps和save_strategy 设置成一样，调试的时候可以设小点，观察F1是否有值且递增
    # 0817 这里的1个step 即执行完一个batch_size
    # 0817 日志显示一共跑28130步，10 个epoch，相当于1个epoch 要跑2800步，我们把步数设置成5即每个epoch评价5次多
    evaluation_strategy = "epoch"
    eval_steps = 50
    save_strategy = "epoch"
    save_steps = 50

    ent_types = [
        "NATS",
        "NO",
        "NHVI",
        "NHCS",
        "NASI",
        "NCGV",
        "NT",
        "NS",
        "NCSP",
        "NCSM"
    ]

    data_files = {
        "train": "./data/ner/train.json",
        "test": "./data/ner/test.json"
    }

    tokenizer = AutoTokenizer.from_pretrained(model_path_or_name)

    processor = Processor(
        tokenizer,
        ent_types=ent_types,
        max_seq_length=max_seq_length
    )

    config = AutoConfig.from_pretrained(model_path_or_name)
    config.num_ner_labels = len(processor.ent2id)

    model = PromptNER.from_pretrained(model_path_or_name, config=config)

    compute_metrics = load_ner_metric()

    datasets = load_dataset(
        "json",
        data_files=data_files,
        field="data",
        cache_dir=cache_dir
    )

    train_dataset, train_examples = processor.process_dataset(datasets["train"], set_type="train")
    eval_dataset, eval_examples = processor.process_dataset(datasets["test"], set_type="test")

    training_args = initialize_traing_args(
        num_train_epochs=num_train_epochs,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps if evaluation_strategy == "steps" else None,
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else None,
        output_dir=output_dir,
        remove_unused_columns=True
    )

    trainer = HFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        post_processor=processor,
        compute_metrics=compute_metrics
    )

    if not training_args.do_train and training_args.do_eval:
        model_path = f"{training_args.output_dir}/checkpoint-250/pytorch_model.bin"
        trainer.load_checkpoint(model_path)

    if training_args.do_train:
        print("***** Train *****")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_model("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        print("*** Evaluation ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_model("eval", metrics)
