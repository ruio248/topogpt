## Standard AlphaCha instruction template

STANDARD_ALPHACHA = {
    "with_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{Instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "without_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{Instruction}\n\n### Response:"
    ),
}


## 不带任何模版,统一形式 
NORMAL_TEMPL={
    "with_input": (
        "{Instruction}\n{input}\n"
    ),
    "without_input": (      
        "{Instruction}\n"
    ),
}

## 其他模版可以自行定义