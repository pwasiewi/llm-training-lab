from datasets import load_dataset
import nltk
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import pandas as pd
import random
import re
from tqdm import tqdm

# Load IMDb dataset
dataset = load_dataset("imdb")

# Download required NLTK resources
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')


def get_wordnet_pos(treebank_tag):
    """
    Convert Penn Treebank POS tags to WordNet POS tags
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('N') and treebank_tag != 'NNP':
        return wordnet.NOUN
    return None


def find_word_positions(text):
    """
    Find positions of words in text and their POS tags.
    Returns list of (word, start_pos, end_pos, pos_tag)
    """
    words_with_positions = []
    tokens = word_tokenize(text)
    tagged_words = pos_tag(tokens)

    current_pos = 0
    for word, tag in tagged_words:
        # Find the actual position of the word in original text
        word_pos = text.find(word, current_pos)
        if word_pos != -1:
            words_with_positions.append((
                word,
                word_pos,
                word_pos + len(word),
                tag
            ))
            current_pos = word_pos + len(word)

    return words_with_positions


def get_synonym(word, pos):
    """
    Get a single-word synonym for a word with specific part of speech
    """
    wordnet_pos = get_wordnet_pos(pos)
    if not wordnet_pos or len(word) <= 2:
        return None

    synonyms = wordnet.synsets(word, pos=wordnet_pos)
    if not synonyms:
        return None

    # Only collect single-word synonyms
    synonym_words = set()
    for syn in synonyms:
        for lemma in syn.lemmas():
            # Only add if it's a single word (no spaces or underscores)
            synonym = lemma.name()
            if " " not in synonym and "_" not in synonym:
                synonym_words.add(synonym)

    synonym_words.discard(word)
    return random.choice(list(synonym_words)) if synonym_words else None


def replace_with_synonyms(text, num_replacements=200):
    """
    Replace words with single-word synonyms while preserving original text structure.
    Returns a tuple of (modified_text, number_of_replacements_made)
    """
    # Get words with their positions and POS tags
    words_with_positions = find_word_positions(text)

    # Randomly select words to replace
    replaceable_words = [
        (word, start, end, pos)
        for word, start, end, pos in words_with_positions
        if get_wordnet_pos(pos) and len(word) > 2  # Only consider nouns and adjectives
    ]

    if not replaceable_words:
        return text, 0

    # Randomly select words to replace
    num_to_replace = min(num_replacements, len(replaceable_words))
    words_to_replace = random.sample(replaceable_words, num_to_replace)

    # Sort by position in reverse order to replace from end to start
    words_to_replace.sort(key=lambda x: x[1], reverse=True)

    # Make replacements
    result = text
    replacements_made = 0
    for word, start, end, pos in words_to_replace:
        synonym = get_synonym(word, pos)
        if synonym:
            result = result[:start] + synonym + result[end:]
            replacements_made += 1

    return (result, replacements_made)  # Explicitly return as tuple


# Convert dataset to DataFrame
imdb_train = dataset["train"].to_pandas()

# Create new data through synonym replacement
augmented_data = []
total_replacements = 0

# Create progress bar
progress_bar = tqdm(total=len(imdb_train), desc="Processing reviews")

for index, row in imdb_train.iterrows():
    augmented_text, replacements = replace_with_synonyms(row['text'])
    augmented_data.append({"text": augmented_text, "label": row['label']})

    total_replacements += replacements
    avg_replacements = total_replacements / (index + 1)

    # Update progress bar with average replacements
    progress_bar.update(1)
    progress_bar.set_postfix({
        "avg_replacements": f"{avg_replacements:.2f}"
    })

progress_bar.close()

# Add new data to existing dataset
augmented_df = pd.DataFrame(augmented_data)
print("\nAugmented data sample:")
print(augmented_df.head())

# imdb_train_extended = pd.concat([imdb_train, augmented_df], ignore_index=True)

# Save the extended dataset
augmented_df.to_csv("imdb_train_augmented.csv", index=False)

print("Dataset augmented with single-word synonyms and saved.")