# https://github.com/ashishjamarkattel/reinforment-learning-with-human-feedback/blob/master/gpt2_sentiment_YT.ipynb
import torch
from tqdm import tqdm
import pandas as pd
import argparse
import yaml
tqdm.pandas()

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler


config = PPOConfig(
    model_name="lvwerra/gpt2-imdb",
    learning_rate=1.41e-5,
)

sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", type=str, help="the directory of the reward data")
    parser.add_argument("--model_path", type=str, help="the model path for training")
    parser.add_argument("--model_type", type=str, help="the model type training")
    parser.add_argument("--train_config",type=str, default=None,help="the filepath of trainig config")
    parser.add_argument("--lora_config",type=str, default=None,help="the filepath of lora config")
    parser.add_argument("--need_parallel", action='store_true', help="whether you need to parallel the training process")
    parser.add_argument("--split_ratio",type=float, default=0.8,help="the ratio to split dataset")
    return parser.parse_args()

def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

dataset = build_dataset(config)


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def main():
    args = parse_arguments()
    
    ## 数据加载部分
    ## ? 
    
    ## 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,trust_remote_code=True)
    ## llam2没指定pad_token,一般设置为2
    if args.model_type == 'llama2':
        tokenizer.pad_token_id = 2
    elif args.model_type == 'llama3':
        tokenizer.pad_token = tokenizer.eos_token
    
    ## 加载模型
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ### 加载reward 模型 
    ref_model= AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

    ## 加载训练配置
       ## 训练配置部分
    train_config_path = args.train_config
    with open(train_config_path, "r") as config_file:
        train_config_content = yaml.load(config_file, Loader=yaml.FullLoader)
        if args.need_parallel and 'deepspeed' not in train_config_content:
            raise ValueError("If parallelism is required, please specify the configuration path of deepspeed_config in training_config.")
    ### 装换为整float型
    ### 将字符串形式的数字转换为浮点型
    train_config_content['learning_rate'] = float(train_config_content['learning_rate'])
    train_config_content['adam_epsilon'] = float(train_config_content['adam_epsilon'])

    ### 创建 TrainingArguments 对象并更新参数
    training_args = TrainingArguments(**train_config_content)

    ## 得到data_module,并且得到ppo_trainer
    data_module = make_rlhf_data_module(tokenizer=tokenizer, tokenized_ds=tokenized_ds,split_ratio=args.split_ratio)

    ppo_trainer = PPOTrainer(training_args , model, ref_model, tokenizer,**data_module)

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)

    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)


    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }


    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        #### Get response from gpt2
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        
    
if __name__=="__main__":
    main()