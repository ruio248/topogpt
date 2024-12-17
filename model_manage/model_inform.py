import os

MODEL_ROOT_DIR = "/path/to/model/root"

NAME_REPATH_TYPE_DICT = {
    "chatglm3_6b": {"path": os.path.join("chatglm3-6b", "chatglm3-6b"), "type": "chatglm"},
    "chatglm4_9b": {"path": "glm-4-9b-chat", "type": "chatglm4"},
    "llama2_newtok_7b": {"path": "/path/to/model/trained_model/continue_pretrain/llama_new_tokenizer_lora/test", "type": "llama2"},
    "llama2_7b": {"path": "Llama-2-7b-chat-hf", "type": "llama2"},
    "Qwen1.5_14B-Chat": {"path": os.path.join("qwen-1.5", "Qwen1.5-14B-Chat"), "type": "qwen1.5"},
    "Qwen1.5_7B-Chat": {"path": os.path.join("qwen-1.5", "Qwen1.5-7B-Chat"), "type": "qwen1.5"},
    "PT_Qwen1.5_7B-Chat": {"path": "/path/to/model/trained_model/continue_pretrain/qwen1.5_7b_1:10/checkpoint-32760", "type": "qwen1.5"},
    "Qwen1.5_72B-Chat": {"path": os.path.join("qwen-1.5", "Qwen1.5-72B-Chat"), "type": "qwen1.5"},
    "Qwen1.5_1.8B-Chat": {"path": os.path.join("qwen-1.5", "Qwen1.5-1.8B-Chat"), "type": "qwen1.5"},
    "phi_2": {"path": os.path.join("phi-2", "AI-ModelScope/phi-2"), "type": "phi2"},
    "yi_34b": {"path": "Yi-34B-chat/01ai/Yi-34B-Chat", "type": "yi"},
    "llama3_8b": {"path": "Llama3-8B-Instruct-HF", "type": "llama3"},
    "llama3_70b": {"path": "Llama3-70B-Instruct-HF", "type": "llama3"}
}

MODEL_TYPE = {'chatglm', 'chatglm4', 'qwen1.5', 'phi2', 'gpt3.5', 'llama2', 'gpt4', 'yi', 'llama3'}

EVAL_PATH_DICT = {
    "physic_eval_csv": "/path/to/evaluation/data/physics.csv",
    "normal_eval_csv": "/path/to/evaluation/data/normal.csv",
    "topo_eval_csv": "/path/to/evaluation/data/topo.csv",
    "physic_eval_json": "/path/to/evaluation/data/physics.json",
    "normal_eval_json": "/path/to/evaluation/data/normal.json",
    "topo_eval_json": "/path/to/evaluation/data/topo.json",
}

NAME_REPATH_TYPE_DICT_CL = {
    "llama2_7b": {"path": "Llama-2-7b-chat-hf", "type": "llama2"},
    "llama2_7b_wash": {"path": "trained_model/llama2_wash", "type": "llama2"},
    "yi_34b": {"path": "Yi-34B-chat/01ai/Yi-34B-Chat", "type": "yi"},
    "llama3_8b": {"path": "Llama3-8B-Instruct-HF", "type": "llama3"},
    "llama3_8b_q": {"path": "trained_model/llama3_8b_trained/answer_format", "type": "llama3"},
    "llama3_70b": {"path": "Llama3-70B-Instruct-HF", "type": "llama3"},
    "chatglm4_9b": {"path": "glm-4-9b-chat", "type": "chatglm4"}
}
