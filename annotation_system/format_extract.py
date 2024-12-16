from gc import collect
import gradio as gr
import pandas as pd
import os
import json
from quality_classify import DataAnnotator


class Extract(DataAnnotator):
    def __init__(self, data_dir, save_dir, start_index=0):
        super().__init__(data_dir, start_index)
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_index = start_index  # Record the starting index of saved data

    def extract(self, extract_output, ans_key='score'):
        """
        Extracts specific field content and stores it in memory (saving to file requires clicking the save button).
        """
        def refresh_data(index, ans_key):
            """
            Updates and displays the next data sample.
            """
            if index >= self.len_data:
                return "All data has been annotated!", "", "", ""
            data = self.data.iloc[index]
            output = data[ans_key]
            progress = f"Annotated {self.index}/{self.len_data} data samples."
            sample_num = f"### Sample {self.index + 1}"
            return output, progress, sample_num, ""  # Clear input box

        if self.index < self.len_data:
            # Save extracted result to the current data item
            self.data.at[self.index, f'extract_{ans_key}'] = extract_output

            # Move to the next data sample
            self.index += 1
            return refresh_data(self.index, ans_key)
        else:
            return "All data has been annotated!", "", "", ""  # Clear input box

    def save_extracted_data(self):
        """
        Saves the extracted content to the specified folder's extract.json file.
        """
        # File path
        save_path = os.path.join(self.save_dir, 'extract.json')

        # Collect extracted content including "text", "question", "answer", and extracted fields
        collected_data = []
        for idx in range(self.save_index, self.index):  # From last saved index to current index
            collected_data.append({
                "index": idx,
                "text": self.data.at[idx, 'text'],
                "question": self.data.at[idx, 'question'],
                "answer": self.data.at[idx, 'answer'],
                "extract_score": self.data.at[idx, f'extract_score']
            })

        # If the file already exists, read the previous data and merge
        if os.path.exists(save_path):
            with open(save_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        # Merge data
        existing_data.extend(collected_data)

        # Save the merged data
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

        # Update the saved index
        self.save_index = self.index

        return f"Extracted data has been saved to {save_path}"

    def launch_interface(self, ans_key='score'):
        """
        Launches the Gradio interface for manual extraction and saving of specified field content.
        """
        if len(self.data) > self.index:
            initial_data = self.data.iloc[self.index][ans_key]
        else:
            initial_data = "No data available."

        progress = f"Annotated {self.index}/{self.len_data} data samples."
        sample_num = f"### Sample {self.index + 1}"

        with gr.Blocks() as demo:
            with gr.Row():
                gr.Markdown("## Data Extraction System")
                gr.Markdown("Manually extract data content and save the results.")

            with gr.Row():
                title = gr.Markdown(value=sample_num)

            with gr.Row():
                output = gr.Textbox(label="Current Data", value=initial_data, lines=10, interactive=False)

            with gr.Row():
                extract_output = gr.Textbox(label="Extracted Result", lines=3, interactive=True)

            with gr.Row():
                status_output = gr.Textbox(label="Status", value=progress, interactive=False)

            with gr.Row():
                submit_button = gr.Button("Submit")
                submit_button.click(fn=self.extract,
                                    inputs=[extract_output, gr.Textbox(value=ans_key)],
                                    outputs=[output, status_output, title, extract_output])  # Clear input box after submit

            with gr.Row():
                save_button = gr.Button("Save Extracted Data")
                save_button.click(fn=self.save_extracted_data, inputs=[], outputs=[status_output])

            demo.launch()


# Example usage
if __name__ == "__main__":
    data_dir = 'annotation_system/test/format_extract_test'
    save_dir = 'your save directory'  # Directory to save extracted results
    start_index = 0  # Define starting index
    extractor = Extract(data_dir, save_dir, start_index=start_index)
    extractor.launch_interface()

