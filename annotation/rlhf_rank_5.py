## 此脚本用来标注强化学习的训练数据集，5个样本
import gradio as gr
import pandas as pd
import json
import os 
from quality_classify import DataAnnotator

import gradio as gr
import pandas as pd
import json
import os
from quality_classify import DataAnnotator


class RewardAnnotator(DataAnnotator):
    def __init__(self, data_dir, start_index=0):
        super().__init__(data_dir, start_index)

    def evaluate(self, rank_1, rank_2, rank_3, rank_4, rank_5):
        """
        Updates the ranking of the five answers and saves the data.
        """
        def refresh_data(index):
            """
            Updates the display for the next data sample.
            """
            if index >= self.len_data:
                return "All samples have been annotated.", "", "", "", "", "", "", None, None, None, None, None
            data = self.data.iloc[index]
            answer_1 = data['answer_1']
            answer_2 = data['answer_2']
            answer_3 = data['answer_3']
            answer_4 = data['answer_4']
            answer_5 = data['answer_5']
            progress = f"Annotated {self.index}/{self.len_data} samples."
            sample_num = f"### Sample {self.index + 1}"
            return answer_1, answer_2, answer_3, answer_4, answer_5, progress, sample_num, None, None, None, None, None

        # Save the ranking provided by the user
        if len(set([rank_1, rank_2, rank_3, rank_4, rank_5])) != 5:
            return "Please ensure all ranks are unique and valid.", "", "", "", "", "", "", None, None, None, None, None

        if self.index < len(self.data):
            self.data.at[self.index, 'rank_1'] = rank_1
            self.data.at[self.index, 'rank_2'] = rank_2
            self.data.at[self.index, 'rank_3'] = rank_3
            self.data.at[self.index, 'rank_4'] = rank_4
            self.data.at[self.index, 'rank_5'] = rank_5

            self.index += 1  # Move to the next sample
            self.save_data()
            return refresh_data(self.index)
        else:
            return "You have reached the last sample.", "", "", "", "", "", "", None, None, None, None, None

    def launch_interface(self):
        """
        Launches the Gradio interface for ranking five answers.
        """
        data = self.data.iloc[self.index] if len(self.data) > self.index else None
        if data is not None:
            answer_1 = data['answer_1']
            answer_2 = data['answer_2']
            answer_3 = data['answer_3']
            answer_4 = data['answer_4']
            answer_5 = data['answer_5']
        else:
            answer_1 = answer_2 = answer_3 = answer_4 = answer_5 = "No data available."

        progress = f"Annotated {self.index}/{self.len_data} samples."
        sample_num = f"### Sample {self.index + 1}"

        with gr.Blocks() as demo:
            with gr.Row():
                gr.Markdown("## Ranking System for Reinforcement Learning Data")

            with gr.Row():
                title = gr.Markdown(value=sample_num)

            gr.Markdown("Answer 1")
            text_1 = gr.Textbox(label="Answer", value=answer_1, lines=3, interactive=False)
            gr.Markdown("Answer 2")
            text_2 = gr.Textbox(label="Answer", value=answer_2, lines=3, interactive=False)
            gr.Markdown("Answer 3")
            text_3 = gr.Textbox(label="Answer", value=answer_3, lines=3, interactive=False)
            gr.Markdown("Answer 4")
            text_4 = gr.Textbox(label="Answer", value=answer_4, lines=3, interactive=False)
            gr.Markdown("Answer 5")
            text_5 = gr.Textbox(label="Answer", value=answer_5, lines=3, interactive=False)

            # Ranking inputs
            rank_1 = gr.Radio(['answer_1', 'answer_2', 'answer_3', 'answer_4', 'answer_5'], label="Rank 1:")
            rank_2 = gr.Radio(['answer_1', 'answer_2', 'answer_3', 'answer_4', 'answer_5'], label="Rank 2:")
            rank_3 = gr.Radio(['answer_1', 'answer_2', 'answer_3', 'answer_4', 'answer_5'], label="Rank 3:")
            rank_4 = gr.Radio(['answer_1', 'answer_2', 'answer_3', 'answer_4', 'answer_5'], label="Rank 4:")
            rank_5 = gr.Radio(['answer_1', 'answer_2', 'answer_3', 'answer_4', 'answer_5'], label="Rank 5:")

            status_output = gr.Textbox(label="Status", value=progress, interactive=False)

            with gr.Row():
                submit_button = gr.Button("Submit")
                submit_button.click(fn=self.evaluate,
                                    inputs=[rank_1, rank_2, rank_3, rank_4, rank_5],
                                    outputs=[
                                        text_1, text_2, text_3, text_4, text_5,
                                        status_output, title, rank_1, rank_2, rank_3, rank_4, rank_5
                                    ])  # Clear ranking inputs after submit

            demo.launch()


# Example usage
if __name__ == "__main__":
    data_dir = "annotation_system/reward_test_5"  # Ensure this path is correct
    data_annotator = RewardAnnotator(data_dir=data_dir)
    data_annotator.launch_interface()


## 数据的长度需要修复