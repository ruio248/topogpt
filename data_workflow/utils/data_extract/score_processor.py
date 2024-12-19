from data_processor import *
import json
import re
from tqdm import tqdm

class Score_Processor(DataProcessor):
   
    def process_all_files(self):
        total_entries = 0
        successful_extractions = 0
        score_dict = {"Relevance": [], "Accuracy": [], "Depth": [], "Total": []}

        # Get all the JSON files first to show progress bar
        files = []
        for folder_path, _, file_names in os.walk(self.root_folder):
            for file_name in file_names:
                if self.file_filter(file_name):  # Filter files based on some criterion
                    file_path = os.path.join(folder_path, file_name)
                    files.append(file_path)

        # Initialize the progress bar using tqdm
        for file_path in tqdm(files, desc="Processing Files", unit="file"):
            entries, extractions, score_list = self.process_data(file_path)
            total_entries += entries
            successful_extractions += extractions

            for score in score_list:
                score_dict["Relevance"].append(score["Topological Relevance"])
                score_dict["Accuracy"].append(score["Accuracy"])
                score_dict["Depth"].append(score["Depth"])
                score_dict["Total"].append(score["Total"])

        return total_entries, successful_extractions, score_dict
 

    def process_data(self, file_path):
        total_entries = 0
        successful_extractions = 0
        score_list = []  
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading file {file_path}: {e}")
            return total_entries, successful_extractions

        required_keys = {"Topological Relevance", "Accuracy", "Depth", "Clarity", "Total"}
        for entry in data:
            total_entries += 1
            raw_output = entry.get("raw_score", "")
            if raw_output:
                score = self.extract_data(raw_output)
                if score and set(score.keys()) == required_keys:
                    score_list.append(score)
                    entry["score"] = score
                    successful_extractions += 1
                else:
                    print(f"Validation error: Missing required score keys in entry")

        with open(file_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Updated: {file_path}")

        return total_entries, successful_extractions, score_list 

    def extract_data(self, raw_output):
        try:
            pattern = r'(?<=\n\{)(.*?)(?=\}\n)'
            json_matches = re.findall(pattern, raw_output, re.DOTALL)
            if json_matches:
                score = json.loads('{' + json_matches[0] + '}')
                return score
        except (IndexError, json.JSONDecodeError):
            return None

    def run(self, save_path=None):
        total_entries, successful_extractions, score_dict = self.process_all_files()
    
        if total_entries > 0:
            success_rate = (successful_extractions / total_entries) * 100
        else:
            success_rate = 0.0

        print("\n--- Statistics ---")
        print(f"Total Entries Processed: {total_entries}")
        print(f"Successful Extractions: {successful_extractions}")
        print(f"Success Rate: {success_rate:.2f}%")
        
        if save_path: 
            with open(save_path, "w") as f:
                json.dump(score_dict, f, indent=2)
            print(f"Score statistics saved to '{save_path}'.")

if __name__ == "__main__": 
    root_dir = "/path/to/data/score_result"
    processor = Score_Processor(root_dir)  
    processor.run(save_path="/path/to/output/score_statistics.json")

