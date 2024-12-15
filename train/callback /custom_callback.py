from transformers import TrainerCallback
import os
import torch

class CustomTrainerCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        # 在训练开始时打印信息
        print("Training is starting... Here are the training arguments:")
        print(args)

    def on_epoch_end(self, args, state, control, **kwargs):
        # 在每个epoch结束时打印epoch信息
        print(f"Epoch {int(state.epoch)} has ended. Current step: {state.global_step}")

    def on_save(self, args, state, control, **kwargs):
        # 自定义保存模型的名称, 使用 epoch 作为文件名的一部分
        if state.is_world_process_zero:
            epoch = int(state.epoch)
            output_dir = f"{args.output_dir}/checkpoint-{epoch}"
            print(f"Saving model at {output_dir}")
            kwargs['model'].save_pretrained(output_dir)
            kwargs['tokenizer'].save_pretrained(output_dir)
            if kwargs.get('optimizer'):
                torch.save(kwargs['optimizer'].state_dict(), os.path.join(output_dir, "optimizer.pt"))
            if kwargs.get('scheduler'):
                torch.save(kwargs['scheduler'].state_dict(), os.path.join(output_dir, "scheduler.pt"))

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 打印并记录日志到 TensorBoard
        if logs is not None and state.is_world_process_zero:
            print(f"Logging at step {state.global_step}: {logs}")
            
            # TensorBoard 日志记录 (确保 report_to 中包括 tensorboard)
            if 'loss' in logs:
                # 示例：记录损失
                state.log_history.append({"loss": logs['loss'], "step": state.global_step})
            if 'accuracy' in logs:
                # 示例：记录准确率
                state.log_history.append({"accuracy": logs['accuracy'], "step": state.global_step})
