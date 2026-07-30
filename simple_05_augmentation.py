"""
LESSON 5 — Data augmentation for text: making more data out of data.

Small labeled datasets overfit: the lesson-2 encoder memorizes 2000 IMDB
reviews long before it learns "sentiment". Augmentation manufactures new
LABEL-PRESERVING variants of existing examples — the text changes, the
sentiment does not — which acts as regularization at the data level.

Three operations, from smartest to dumbest (EDA, Wei & Zou 2019):

  synonym  Replace nouns/adjectives with WordNet synonyms, guided by
           part-of-speech tagging so "novel" (noun) doesn't become "new"
           (adjective) mid-sentence. Needs nltk; skipped gracefully if
           missing.
  swap     Randomly swap two words, n times. Grammar breaks slightly;
           a bag-of-words-ish classifier barely notices, and the noise
           teaches word-order robustness.
  delete   Drop each word with probability p. Simulates missing evidence —
           the model can no longer rely on any single keyword.

Why this preserves labels (usually): sentiment lives in content words and
their rough co-occurrence, not exact syntax. The same trick would be UNSAFE
for tasks where syntax IS the label (grammar checking, NLI).

The per-operation stats printed at the end are the actual lesson: how much
did each op really change the text? An augmentation that changes nothing is
a no-op; one that changes too much flips labels.

Run:  python simple_05_augmentation.py                # 300 reviews, all ops
      python simple_05_augmentation.py --num-reviews 2000 --ops synonym,swap
Output: outputs/imdb_train_augmented.csv (original + augmented rows).
"""

import argparse
import random
import os

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Graceful degradation: WordNet synonym replacement is the only op that
# needs nltk. Without it the script still runs swap/delete and says why.
try:
    import nltk
    from nltk.corpus import wordnet
    from nltk.tag import pos_tag
    from nltk.tokenize import word_tokenize
    HAVE_NLTK = True
except ImportError:
    HAVE_NLTK = False


def ensure_nltk_data():
    """First run downloads WordNet + tokenizer + POS models (~40 MB)."""
    for resource in ("wordnet", "omw-1.4", "punkt", "punkt_tab",
                     "averaged_perceptron_tagger",
                     "averaged_perceptron_tagger_eng"):
        nltk.download(resource, quiet=True)


# ---------------------------------------------------------------- synonym --

def get_wordnet_pos(treebank_tag):
    """
    Penn Treebank tag -> WordNet POS. Only adjectives (J*) and common nouns
    are considered: verbs conjugate ("running" -> "operate" breaks tense)
    and proper nouns (NNP) are names — "Tarantino" has no synonym.
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("N") and treebank_tag != "NNP":
        return wordnet.NOUN
    return None


def get_synonym(word, treebank_tag):
    """A single-word WordNet synonym with matching POS, or None."""
    pos = get_wordnet_pos(treebank_tag)
    if not pos or len(word) <= 2:
        return None
    candidates = set()
    for synset in wordnet.synsets(word, pos=pos):
        for lemma in synset.lemmas():
            name = lemma.name()
            if " " not in name and "_" not in name:
                candidates.add(name)
    candidates.discard(word)
    return random.choice(sorted(candidates)) if candidates else None


def synonym_replace(text, max_replacements=20):
    """
    Replace up to max_replacements nouns/adjectives, editing the ORIGINAL
    string back-to-front so earlier character offsets stay valid — the
    surrounding punctuation and casing survive untouched.
    Returns (new_text, n_replaced).
    """
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    # Locate each token's character span in the original text.
    spans = []
    cursor = 0
    for word, tag in tagged:
        start = text.find(word, cursor)
        if start != -1:
            spans.append((word, start, start + len(word), tag))
            cursor = start + len(word)

    replaceable = [s for s in spans if get_wordnet_pos(s[3]) and len(s[0]) > 2]
    if not replaceable:
        return text, 0

    chosen = random.sample(replaceable, min(max_replacements, len(replaceable)))
    chosen.sort(key=lambda s: s[1], reverse=True)   # back-to-front

    result = text
    replaced = 0
    for word, start, end, tag in chosen:
        synonym = get_synonym(word, tag)
        if synonym:
            result = result[:start] + synonym + result[end:]
            replaced += 1
    return result, replaced


# ------------------------------------------------------------ swap/delete --

def random_swap(text, n_swaps=3):
    """Swap two random word positions, n times. Returns (text, n_swaps_done)."""
    words = text.split()
    if len(words) < 2:
        return text, 0
    for _ in range(n_swaps):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words), n_swaps


def random_delete(text, p_delete=0.1):
    """
    Drop each word independently with probability p. Keeps at least one
    word (an empty review would be an undefined label). Returns
    (text, n_deleted).
    """
    words = text.split()
    kept = [w for w in words if random.random() > p_delete]
    if not kept:
        kept = [random.choice(words)]
    return " ".join(kept), len(words) - len(kept)


# ------------------------------------------------------------------- main --

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--num-reviews", type=int, default=300,
                        help="how many training reviews to augment")
    parser.add_argument("--ops", default="synonym,swap,delete",
                        help="comma-separated subset of: synonym,swap,delete")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    ops = [op.strip() for op in args.ops.split(",") if op.strip()]
    if "synonym" in ops and not HAVE_NLTK:
        print("nltk is not installed — skipping the synonym op "
              "(swap/delete still run).")
        ops.remove("synonym")
    if "synonym" in ops:
        print("Checking nltk data (downloads on first run)...")
        ensure_nltk_data()
    if not ops:
        raise SystemExit("No operations left to run.")

    print("Loading IMDB training split...")
    dataset = load_dataset("imdb")
    train_df = dataset["train"].to_pandas().sample(
        n=args.num_reviews, random_state=args.seed).reset_index(drop=True)

    # Each op produces one augmented copy per review => len(ops) x more data.
    rows = []
    stats = {op: {"changed_units": 0, "reviews": 0} for op in ops}

    for _, row in tqdm(train_df.iterrows(), total=len(train_df),
                       desc=f"Augmenting ({'+'.join(ops)})"):
        rows.append({"text": row["text"], "label": row["label"],
                     "source": "original"})
        for op in ops:
            if op == "synonym":
                new_text, changed = synonym_replace(row["text"])
            elif op == "swap":
                new_text, changed = random_swap(row["text"])
            else:
                new_text, changed = random_delete(row["text"])
            rows.append({"text": new_text, "label": row["label"], "source": op})
            stats[op]["changed_units"] += changed
            stats[op]["reviews"] += 1

    out_df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/imdb_train_augmented.csv"
    out_df.to_csv(out_path, index=False)

    print(f"\nWrote {len(out_df)} rows ({len(train_df)} original + "
          f"{len(out_df)-len(train_df)} augmented) -> {out_path}")
    print("\nPer-operation stats (how much did the text actually change?):")
    unit = {"synonym": "words replaced", "swap": "swaps made",
            "delete": "words deleted"}
    for op in ops:
        avg = stats[op]["changed_units"] / max(stats[op]["reviews"], 1)
        print(f"  {op:<8} avg {avg:5.1f} {unit[op]} per review")

    print("\nSample (same review, per op):")
    sample = out_df[out_df["source"] == "original"].iloc[0]["text"][:200]
    print(f"  original: {sample}...")
    for op in ops:
        aug = out_df[out_df["source"] == op].iloc[0]["text"][:200]
        print(f"  {op:<8}: {aug}...")

    print("\nThings to try: train lesson 2 on this CSV vs the raw split and")
    print("compare test accuracy; raise --num-reviews; crank p_delete to 0.5")
    print("and watch the samples stop being label-preserving.")


if __name__ == "__main__":
    main()
