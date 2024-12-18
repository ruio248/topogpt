import os
import time
import sentencepiece as spm

class SentencePieceTrainer:
    """
    A utility class for training a SentencePiece tokenizer model with user-defined parameters.
    This class includes methods to find input files, train the tokenizer, and measure training time.
    """
    def __init__(self, input_dir, model_prefix, vocab_size=30000, character_coverage=1.0, model_type="unigram"):
        """
        Initialize the SentencePieceTrainer with necessary parameters.

        Args:
            input_dir (str): Directory containing the input text files.
            model_prefix (str): Prefix for the trained model files.
            vocab_size (int): Desired vocabulary size (default is 30,000).
            character_coverage (float): Character coverage for the training corpus (default is 1.0).
            model_type (str): SentencePiece model type ("unigram", "bpe", "char", "word").
        """
        self.input_dir = input_dir
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.model_type = model_type
        self.file_paths = []

    def find_files(self):
        """
        Find and list all files within the specified input directory.
        """
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self.file_paths.append(file_path)

        print(f"Total files found: {len(self.file_paths)}")

    def train_model(self, input_file):
        """
        Train the SentencePiece tokenizer model using the specified input file.

        Args:
            input_file (str): Path to the single input text file for training.
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file '{input_file}' not found.")

        # Start timer
        start_time = time.time()
        print("Starting tokenizer training...")

        # Train the model
        spm.SentencePieceTrainer.train(
            input=input_file, input_format="text",
            model_prefix=self.model_prefix, vocab_size=self.vocab_size,
            character_coverage=self.character_coverage, model_type=self.model_type
        )

        # End timer
        training_time = time.time() - start_time
        print(f"Tokenizer training completed in {training_time:.2f} seconds.")

    def save_model_info(self):
        """
        Provide information on the generated model files.
        """
        print(f"Model saved with prefix: {self.model_prefix}")
        print(f"Generated files: {self.model_prefix}.model and {self.model_prefix}.vocab")


if __name__ == "__main__":
    
    INPUT_DIR = "/path/to/data/instruct_data"
    MODEL_PREFIX = "/path/to/model/tokenizer/physic_instruct_2"
    VOCAB_SIZE = 30000
    CHARACTER_COVERAGE = 1.0
    MODEL_TYPE = "unigram"
    INPUT_FILE = "/path/to/data/instruct_data/train_tokenizer.txt"  # Path to single input file


    # Initialize the trainer
    trainer = SentencePieceTrainer(
        input_dir=INPUT_DIR, model_prefix=MODEL_PREFIX, vocab_size=VOCAB_SIZE,
        character_coverage=CHARACTER_COVERAGE, model_type=MODEL_TYPE
    )

    # Step 1: Find all input files
    trainer.find_files()

    # Step 2: Train the SentencePiece tokenizer
    trainer.train_model(input_file=INPUT_FILE)

    # Step 3: Save model info
    trainer.save_model_info()
