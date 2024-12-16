import gradio as gr
import pandas as pd
import os
from rlhf_rank_5 import *  # Import everything from rlhf_rank_5

class RewardAnnotator2(RewardAnnotator):
    def __init__(self, data_dir, start_index=0):
        super().__init__(data_dir, start_index)

    def evaluate(self, choice):
        """
        Evaluate the user's choice and save the result.
        """
        def refresh_data(index):
            """
            Update the display for the next data sample.
            """
            if index >= self.len_data:
                return "All samples have been annotated.", "", "", "", "", ""
            data = self.data.iloc[index]
            background = data['text']
            answer_1 = data['answer_1']
            answer_2 = data['answer_2']
            progress = f"Annotated {self.index}/{self.len_data} samples."
            file_progress = f"Currently annotating file: {data['filename']} with index {data['index']}"
            sample_num = f"### Sample {self.index + 1}"
            return background, answer_1, answer_2, file_progress, progress, sample_num

        if self.index < len(self.data):
            # Save the user's choice for the current sample
            self.data.at[self.index, 'choice'] = choice
            self.index += 1  # Move to the next sample
            self.save_data()
            return refresh_data(self.index) + (None,)  # Clear the choice input
        else:
            return "This is the last sample.", "", "", "", "", "", None

    def launch_interface(self):
        """
        Launch the Gradio interface for annotation.
        """
        data = self.data.iloc[self.index] if len(self.data) > self.index else None
        if data is not None:
            background = data['text']
            answer_1 = data['answer_1']
            answer_2 = data['answer_2']
            file_progress = f"Currently annotating file: {data['filename']} with index {data['index']}"
            progress = f"Annotated {self.index}/{self.len_data} samples."
            sample_num = f"### Sample {self.index + 1}"
        else:
            background = answer_1 = answer_2 = file_progress = progress = sample_num = "No data available."

        with gr.Blocks() as demo:
            with gr.Row():
                gr.Markdown("## Reinforcement Learning Data Annotation System")
            
            with gr.Row():
                title = gr.Markdown(value=sample_num)
            
            file_status = gr.Textbox(value=file_progress, interactive=False)
            gr.Markdown("Background")
            back_inform = gr.Textbox(label="Background", value=background, lines=10, interactive=False)
            gr.Markdown("Answer 1")
            text_1 = gr.Textbox(label="Answer", value=answer_1, lines=10, interactive=False)
            gr.Markdown("Answer 2")
            text_2 = gr.Textbox(label="Answer", value=answer_2, lines=10, interactive=False)
    
            choice = gr.Radio(['answer_1', 'answer_2'], label="Choice:")
            
            status_output = gr.Textbox(label="Status", value=progress, interactive=False)
            
            with gr.Row():
                submit_button = gr.Button("Submit")
                submit_button.click(fn=self.evaluate, inputs=[choice],
                                    outputs=[back_inform, text_1, text_2, file_status, status_output, title, choice])  # Clear choice after submission
               
            demo.launch()

# Example usage
if __name__ == "__main__":
    data_dir = "annotation_system/reward_test_2"  # Ensure this path is correct
    data_annotator = RewardAnnotator2(data_dir=data_dir)
    data_annotator.launch_interface()
