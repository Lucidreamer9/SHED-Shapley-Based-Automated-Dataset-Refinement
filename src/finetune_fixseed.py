import os
import sys
from typing import List
import csv



import json
from tqdm import tqdm

import fire
import torch
import transformers
from datasets import load_dataset

"""

"""

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM

from utils.prompter import Prompter
from transformers import set_seed
set_seed(42)

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set GPU-specific random seed for reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def train(
    # Model/data parameters
    base_model: str = "model-name",  # Base model path
    data_path: str = "path/to/train_data.json",
    dev_data_path: str = 'path/to/dev_data.json',
    output_dir: str = "path/to/output_dir",
    verbose: bool = True,
    count_fine_path: str = 'path/to/count_file.txt',
    # Training hyperparameters
    batch_size: int = 8,
    micro_batch_size: int = 1,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1024,
    val_set_size: int = 0,
    max_new_token: int = 32,
    save_strategy: str = 'no',
    warmup_ratio: float = 0.04,
    # LoRA hyperparameters
    add_lora: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.1,
    lora_target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "gate_proj",
        "up_proj"
    ],
    # LLM hyperparameters
    train_on_inputs: bool = True,  # Whether to train on inputs or mask them in loss
    add_eos_token: bool = True,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "vicuna",  # The prompt template to use, will default to alpaca.
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if 'opt' in base_model:
        load_model =  AutoModelForCausalLM
        load_tokenizer = AutoTokenizer
    else:
        load_model = LlamaForCausalLM
        load_tokenizer = LlamaTokenizer

    model = load_model.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map=device_map,
        cache_dir="path/to/cache",
    )

    tokenizer = load_tokenizer.from_pretrained(base_model,cache_dir="path/to/cache")

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True, return_tensors=None):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=return_tensors,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    if add_lora:
        model = prepare_model_for_int8_training(model)

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path,cache_dir="path/to/cache")
    else:
        data = load_dataset(data_path,cache_dir="path/to/cache")

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")
    if add_lora:
        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="no",
            save_strategy=save_strategy,
            eval_steps=None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False

   # old_state_dict = model.state_dict
   # model.state_dict = (
   #     lambda self, *_, **__: get_peft_model_state_dict(
   #         self, old_state_dict()
   #     )
   # ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    # model.save_pretrained(output_dir)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )
    model.eval()

    sampling = GenerationConfig(
        do_sample=True,
        temperature=0.2,
        top_p=0.6,
        top_k=30,
        num_beams=1,
        max_new_tokens=max_new_token,
        early_stopping=True,
    )
    def eval_usmle(model, dev_data_path, tokenizer, verbose):
        data_class =  ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
        right_count_dict = dict.fromkeys(data_class, 0)
        total_count_dict = dict.fromkeys(data_class, 0)
        acc_count_dict = dict.fromkeys(data_class, 0)
        with open(dev_data_path, 'r') as f:
            test_set = json.load(f)

        # mean_acc_005 = 0
        # mean_acc_010 = 0
        # mean_acc_020 = 0
        count=0
        for data_point in tqdm(test_set):
            count +=1
            target = data_point["output"]
            class_test_set = data_point["class"]
            tgt_ans_idx = target.replace('The answer is: ','').split('. ')[0]
            tgt_ans = target.replace('The answer is: ','').split('. ')[1]

            test_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                'The answer is: ',
            )

            with torch.autocast("cuda"):
                inputs = tokenizer(test_prompt, return_tensors="pt")
                input =inputs["input_ids"].to('cuda')
                with torch.no_grad():
                    generation_output = model.generate(
                        input_ids=input,
                        generation_config=sampling,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=max_new_token
                    )
                generation_output_decoded = tokenizer.decode(generation_output.sequences[0])
                # print(generation_output_decoded)
                split = prompter.template["response_split"]
                ans = generation_output_decoded.split(split)[-1].strip()
                if verbose:
                    print('-------------------')
                    print(test_prompt)
                    print(tgt_ans)
                    print(tgt_ans_idx)
                    print(ans)
                if tgt_ans_idx+'.' in ans or tgt_ans in ans:
                # if tgt_ans_idx in ans or tgt_ans in ans:
                    right_count_dict[class_test_set] += 1
                    # if count %20 ==0:
                    #     mean_acc_005+=1
                    # if count %10 ==0:
                    #     mean_acc_010+=1
                    # if count %5 ==0:
                    #     mean_acc_020+=1
                total_count_dict[class_test_set] += 1
        # print('----------------sampled------------------')
        # print( mean_acc_005)
        # print( mean_acc_010)
        # print( mean_acc_020)

        mean_acc = 0.


        for key in acc_count_dict.keys():
            tmp = right_count_dict[key]/total_count_dict[key]
            mean_acc += tmp
            acc_count_dict[key] = tmp
        mean_acc /= len(acc_count_dict.keys())
        csv_data = [right_count_dict, total_count_dict, acc_count_dict]

        with open(os.path.join('./raw_dict_mmlu',data_path.split('/')[-1].replace('.json','') + '.csv'), 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=right_count_dict.keys())
            writer.writeheader()
            for row in csv_data:
                writer.writerow(row)
        # if verbose:
        print(right_count_dict)
        print(total_count_dict)
        print(acc_count_dict)
        print()
        return mean_acc
            
    score = eval_usmle(model, dev_data_path, tokenizer, verbose=verbose)
    print('========== Accuracy ==========')
    print(score)
    print()
    with open(count_fine_path, "a") as file:
        file.write(str({"dataset_name": data_path.split('/')[-1], "accuracy": score})+'\n')

if __name__ == "__main__":
    fire.Fire(train)
