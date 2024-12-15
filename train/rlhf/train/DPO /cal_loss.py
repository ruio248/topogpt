## 定义计算dpo损失的过程
import torch.nn.functional as F 
class Cal_loss:
    def cal_dpo_loss(
    model_chosen_logprobs,
    model_rejected_logprobs,
    reference_chosen_logprobs,
    reference_rejected_logprobs,
    beta=0.1,
    ):
    
        policy_logratios = model_chosen_logprobs - model_rejected_logprobs
        reference_logratios = reference_chosen_logprobs - reference_rejected_logprobs
        logits = policy_logratios - reference_logratios
        losses = -F.logsigmoid(beta * logits)
    
    # Optional values to track progress during training
    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()
    
    # .mean() to average over the samples in the batch
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()