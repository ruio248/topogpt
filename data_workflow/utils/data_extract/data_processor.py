import os
from tqdm import tqdm

class DataProcessor:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def process_all_files(self):
        total_entries = 0
        successful_extractions = 0
        # Get all the JSON files first to show progress bar
        files = []
        for folder_path, _, file_names in os.walk(self.root_folder):
            for file_name in file_names:
                if self.file_filter(file_name):  # Filter files based on some criterion
                    file_path = os.path.join(folder_path, file_name)
                    files.append(file_path)
        
        # Initialize the progress bar using tqdm
        for file_path in tqdm(files, desc="Processing Files", unit="file"):
            entries, extractions = self.process_data(file_path)
            total_entries += entries
            successful_extractions += extractions
        
        return total_entries, successful_extractions

    def process_data(self, file_path):
        raise NotImplementedError("This method should be implemented by subclasses")

    def file_filter(self, file_name):
        return file_name.endswith(".json")  # Specific to JSON files

    def run(self):
        total_entries, successful_extractions = self.process_all_files()
    
        # Calculate and print the success probability
        if total_entries > 0:
            success_rate = (successful_extractions / total_entries) * 100
        else:
            success_rate = 0.0  # Avoid division by zero if no entries are processed
    
        print("\n--- Statistics ---")
        print(f"Total Entries Processed: {total_entries}")
        print(f"Successful Extractions: {successful_extractions}")
        print(f"Success Rate: {success_rate:.2f}%")
