import os
import argparse
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm


class TokenizerMerger:
    """
    A utility to merge tokens from a vertical SentencePiece model into an existing LLaMA tokenizer.
    This is useful for customizing a tokenizer to include domain-specific or vertical tokens.
    """

    def __init__(self, base_tokenizer_path, vertical_sp_model_path, output_dir):
        """
        Initialize the TokenizerMerger.

        Args:
            base_tokenizer_path (str): Path to the base LLaMA tokenizer.
            vertical_sp_model_path (str): Path to the SentencePiece model to merge tokens from.
            output_dir (str): Directory to save the updated tokenizer.
        """
        self.base_tokenizer_path = base_tokenizer_path
        self.vertical_sp_model_path = vertical_sp_model_path
        self.output_dir = output_dir

        self.base_tokenizer = None
        self.vertical_sp_model = None
        self.llama_spm = None
        self.vertical_spm = None

    def load_tokenizers(self):
        """
        Load the base LLaMA tokenizer and the vertical SentencePiece model.
        """
        print("Loading tokenizers...")
        self.base_tokenizer = LlamaTokenizer.from_pretrained(self.base_tokenizer_path)
        self.vertical_sp_model = spm.SentencePieceProcessor()
        self.vertical_sp_model.Load(self.vertical_sp_model_path)

        # Deserialize both models
        self.llama_spm = sp_pb2_model.ModelProto()
        self.llama_spm.ParseFromString(self.base_tokenizer.sp_model.serialized_model_proto())

        self.vertical_spm = sp_pb2_model.ModelProto()
        self.vertical_spm.ParseFromString(self.vertical_sp_model.serialized_model_proto())
        print("Tokenizers loaded successfully.")

    def merge_tokens(self):
        """
        Merge tokens from the vertical SentencePiece model into the base LLaMA tokenizer.
        """
        print("Merging tokens...")
        llama_tokens_set = set(p.piece for p in self.llama_spm.pieces)
        print(f"Original LLaMA tokenizer size: {len(llama_tokens_set)}")

        for piece in self.vertical_spm.pieces:
            if piece.piece not in llama_tokens_set:
                new_piece = sp_pb2_model.ModelProto().SentencePiece()
                new_piece.piece = piece.piece
                new_piece.score = 0  # Assign zero score to avoid introducing biases
                self.llama_spm.pieces.append(new_piece)

        print(f"New tokenizer size after merging: {len(self.llama_spm.pieces)}")

    def save_tokenizer(self):
        """
        Save the merged tokenizer in both SentencePiece and HuggingFace formats.

        Output:
            - SentencePiece model: `vertical_llama.model`
            - HuggingFace tokenizer: `vertical_llama`
        """
        sp_output_path = os.path.join(self.output_dir, "vertical_llama.model")
        hf_output_path = os.path.join(self.output_dir, "vertical_llama")

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Save the merged SentencePiece model
        with open(sp_output_path, 'wb') as f:
            f.write(self.llama_spm.SerializeToString())
        print(f"Saved merged SentencePiece model to: {sp_output_path}")

        # Convert and save as HuggingFace tokenizer
        tokenizer = LlamaTokenizer(vocab_file=sp_output_path)
        tokenizer.save_pretrained(hf_output_path)
        print(f"Saved HuggingFace tokenizer to: {hf_output_path}")

    def test_tokenizers(self, text):
        """
        Compare the tokenization results of the original and merged tokenizers.

        Args:
            text (str): Input text to test the tokenizers.
        """
        print("\nTesting tokenizer differences...")
        original_tokenizer = LlamaTokenizer.from_pretrained(self.base_tokenizer_path)
        merged_tokenizer = LlamaTokenizer.from_pretrained(os.path.join(self.output_dir, "vertical_llama"))

        original_tokens = original_tokenizer.tokenize(text)
        merged_tokens = merged_tokenizer.tokenize(text)

        print(f"Original tokenizer output ({len(original_tokens)} tokens): {original_tokens}")
        print(f"Merged tokenizer output ({len(merged_tokens)} tokens): {merged_tokens}")


def main():
    """
    Main function to merge a SentencePiece model into an existing LLaMA tokenizer and test the results.
    """
    parser = argparse.ArgumentParser(description="Merge vertical SentencePiece tokens into LLaMA tokenizer.")
    parser.add_argument("--base_tokenizer", type=str, required=True, help="Path to the base LLaMA tokenizer.")
    parser.add_argument("--vertical_sp_model", type=str, required=True, help="Path to the vertical SentencePiece model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the updated tokenizer.")
    args = parser.parse_args()

    # Initialize and execute the pipeline
    merger = TokenizerMerger(
        base_tokenizer_path=args.base_tokenizer,
        vertical_sp_model_path=args.vertical_sp_model,
        output_dir=args.output_dir
    )

    merger.load_tokenizers()
    merger.merge_tokens()
    merger.save_tokenizer()

    # Example input for testing tokenizers
    test_text = """
    Modern materials engineering is increasingly dominated by the nanoscale, 
    where surface area trumps volume. The discovery of interface properties, such as superconductivity,
    has become a pivotal area of research.
    """
    merger.test_tokenizers(test_text)


if __name__ == "__main__":
    main()
