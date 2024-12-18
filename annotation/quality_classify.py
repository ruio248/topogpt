import gradio as gr
import pandas as pd
import os
import json

class DataAnnotator:
    def __init__(self, data_dir, start_index=0):
        self.data_dir = data_dir
        self.index = start_index
        self.data = []
        self.load_data()
        self.len_data = len(self.data) - start_index

    def load_data(self):
        """
        Traverse all JSON files in the directory and load the data.
        """
        if not os.path.isdir(self.data_dir):
            raise NotADirectoryError(f"The specified path is not a directory: {self.data_dir}")

        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        item['filename'] = filename  # Save the filename for each data item
                        self.data.append(item)

        if not self.data:
            raise ValueError("No valid JSON data found.")

        self.data = pd.DataFrame(self.data)

    def save_data(self):
        """
        Save the data back to the original JSON files.
        """
        grouped = self.data.groupby('filename')
        for filename, group in grouped:
            file_path = os.path.join(self.data_dir, filename)
            data_to_save = group.drop(columns=['filename']).to_dict(orient='records')
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=4)

    def evaluate(self, quality, ans_key='instruction'):
        """
        Evaluate the quality of the current data item and save the result.
        """
        def refresh_data(index, ans_key):
            """
            Update and display the next data sample.
            """
            data = self.data.iloc[index]
            output = data[ans_key]
            progress = f"Annotated {self.index}/{self.len_data} samples."
            sample_num = f"### Sample {self.index + 1}"
            return output, progress, sample_num

        if self.index < len(self.data):
            # Add the quality label
            self.data.at[self.index, 'quality'] = quality

            # Move to the next data item
            self.index += 1
            self.save_data()
            return refresh_data(self.index, ans_key)
        else:
            return "All data has been annotated!"

    def launch_interface(self, ans_key='instruction'):
        """
        Launch the Gradio interface to display and annotate data.
        """
        initial_data = self.data.iloc[self.index][ans_key] if len(self.data) > self.index else "No data available."
        progress = f"Annotated {self.index}/{self.len_data} samples."
        sample_num = f"### Sample {self.index + 1}"

        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Data Quality Annotation System")
                    gr.Markdown("Please select the quality of the data (True or False) and click submit to save the evaluation and load the next sample.")
                    title = gr.Markdown(value=sample_num)

                    quality_radio = gr.Radio(['True', 'False'], label="Data Quality", show_label=True)
                    output = gr.Textbox(label=ans_key.capitalize(), value=initial_data, lines=10, interactive=False)
                    status_output = gr.Textbox(label="Status", value=progress, interactive=False)

                    with gr.Row():
                        submit_button = gr.Button("Submit")
                        submit_button.click(fn=self.evaluate, inputs=[quality_radio], outputs=[output, status_output, title])

            demo.launch()


# Example usage
if __name__ == "__main__":
    start_index = 0  # Customize the starting index if needed
    annotator = DataAnnotator('annotation_system/test/quality_test', start_index=start_index)
    annotator.launch_interface()
