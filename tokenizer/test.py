import sentencepiece as spm

class VocabTester:
    """
    VocabTester is a utility class to:
    - Load a SentencePiece tokenizer model.
    - Extract and test the vocabulary from the tokenizer.
    - Save the vocabulary to a text file for further analysis.
    """

    def __init__(self, model_path, output_path):
        """
        Initialize the VocabTester class.

        Args:
            model_path (str): Path to the SentencePiece tokenizer model.
            output_path (str): Path to save the extracted vocabulary.
        """
        self.model_path = model_path
        self.output_path = output_path

    def extract_and_save_vocab(self):
        """
        Load the tokenizer model, extract its vocabulary, and save it to a text file.
        """
        # Load the SentencePiece model
        sp = spm.SentencePieceProcessor()
        sp.Load(self.model_path)

        # Get the vocabulary size
        vocab_size = sp.get_piece_size()
        print(f"Vocabulary size: {vocab_size}")

        # Extract and save the vocabulary to the specified file
        with open(self.output_path, "w", encoding="utf-8") as f:
            for i in range(vocab_size):
                piece = sp.id_to_piece(i)
                f.write(piece + "\n")

        print(f"Vocabulary saved to {self.output_path}")

if __name__ == "__main__":
    model_path = "path/to/your/tokenizer.model"  # Path to the SentencePiece model
    output_path = "path/to/save/vocab.txt"      # Path to save the vocabulary

    tester = VocabTester(model_path, output_path)
    tester.extract_and_save_vocab()

