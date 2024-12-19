import json
from data_processor import*

class QAProcessor(DataProcessor):
    def process_data(self, file_path):
        total_entries = 0
        successful_extractions = 0
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading file {file_path}: {e}")
            return total_entries, successful_extractions

        for entry in data:
            total_entries += 1
            raw_output = entry.get("raw_qa", "")
            if raw_output:
                qa_pair = self.extract_data(raw_output)
                if qa_pair:
                    entry["Question"] = qa_pair["Question"]
                    entry["Answer"] = qa_pair["Answer"]
                    successful_extractions += 1

        with open(file_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Updated: {file_path}")

        return total_entries, successful_extractions

    def extract_data(self, raw_output):
        try:
            json_part = raw_output.split('{\n', 1)[1]
            json_part = '{' + json_part
            qa_data = json.loads(json_part)
            if "Question" in qa_data and "Answer" in qa_data:
                return {"Question": qa_data["Question"], "Answer": qa_data["Answer"]}
        except (IndexError, json.JSONDecodeError):
            return None
