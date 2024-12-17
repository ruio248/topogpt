## Test Tokenizer Encoding Efficiency
import time
import random
import string
import os
from transformers import BertTokenizer, AutoTokenizer

def generate_random_string(length):
    """Generate a random string of a given length."""
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

def load_corpus(file_path):
    """Load text data from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def test_tokenizer_efficiency(tokenizer, test_data):
    """
    Measure the efficiency of a tokenizer.

    Args:
        tokenizer: The tokenizer to test.
        test_data: A list of strings to tokenize.

    Outputs:
        Prints the tokenization time, compression ratio, and efficiency statistics.
    """
    start_time = time.time()
    total_original_chars = sum(len(data) for data in test_data)

    # Tokenization process
    encoded_data = [tokenizer.encode(data, add_special_tokens=True, truncation=True, max_length=512) for data in test_data]
    encode_time = time.time() - start_time
    total_encoded_tokens = sum(len(tokens) for tokens in encoded_data)

    print(f"Tokenized {len(test_data)} strings, total time taken: {encode_time:.5f} seconds")
    print(f"Average encoding time per string: {encode_time / len(test_data):.5f} seconds")
    print(f"Total number of original characters: {total_original_chars}")
    print(f"Total number of encoded tokens: {total_encoded_tokens}")
    print(f"Average compression ratio (characters/token): {total_original_chars / total_encoded_tokens:.2f}")


tokenizer = AutoTokenizer.from_pretrained("/path/to/your/tokenizer", trust_remote_code=True)

# Test with randomly generated text data
random_test_size = 200  # Number of test strings
random_text_length = 300  # Length of each random string
random_test_data = [generate_random_string(random_text_length) for _ in range(random_test_size)]

print("Test results with randomly generated data:")
test_tokenizer_efficiency(tokenizer, random_test_data)
