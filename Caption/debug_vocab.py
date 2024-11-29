import json

# Load the word_to_idx and idx_to_word mappings
with open('deep_learning_model/word_to_idx.json', 'r') as f:
    word_to_idx = json.load(f)

with open('deep_learning_model/idx_to_word.json', 'r') as f:
    idx_to_word = json.load(f)

# Print out the contents for debugging
print("word_to_idx contents:", word_to_idx)
print("idx_to_word contents:", idx_to_word)
print("Number of entries in idx_to_word:", len(idx_to_word))

# Test lookup for some indices
test_indices = range(0, 20)  # Adjust the range as needed
for idx in test_indices:
    print(f"Index {idx}: {idx_to_word.get(idx, '<unk>')}")
