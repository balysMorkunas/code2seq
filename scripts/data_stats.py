import json
from tqdm import tqdm
import os
import nltk
import statistics
from pathlib import Path

dirs = Path("data/java/training").glob("*")
file_paths = []

examples = 0
code_lengths = 0
comment_lengths = 0
code_tokens = 0
comment_tokens = 0
inline_comment_count = 0

code_tokens_list = []
comment_tokens_list = []

stop_words = {
    "I",
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "let's",
    "me",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours 	ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


# Get all file paths
for dir in dirs:
    for filename in os.scandir(dir):
        if filename.is_file():
            file_paths.append(filename.path)


# go over each file and collect its json
for path in tqdm(file_paths):
    with open(path, "r") as file:
        lines = file.readlines()

        for line in lines:
            examples += 1
            data = json.loads(line)
            code = str(data["original_string"])
            comment = str(data["docstring"])

            # Count inline comments
            if "//" in code or "/*" in code:
                inline_comment_count += 1
                continue

            # Count code lengths and tokens
            code_lengths += code.count("\n")
            tokenized = set(nltk.word_tokenize(code))
            tokenized = tokenized - stop_words

            code_tokens += len(tokenized)
            code_tokens_list.append(len(tokenized))

            # Count comment lengths and tokens
            comment_lengths += comment.count("\n")
            tokenized = nltk.word_tokenize(comment)
            comment_tokens += len(tokenized)
            comment_tokens_list.append(len(tokenized))


print("Total examples:", examples)
print("Avg code len:", code_lengths / examples)
print("Avg code tokens:", code_tokens / examples)
print("Avg comment len:", comment_lengths / examples)
print("Avg comment tokens:", comment_tokens / examples)
print("Inline comment count:", inline_comment_count)
print("Code len mode:", statistics.mode(code_tokens_list))
print("Code len median:", statistics.median(code_tokens_list))

print("Comment len mode:", statistics.mode(comment_tokens_list))
print("Comment len median:", statistics.median(comment_tokens_list))

print("token count no inline comments:", code_tokens)
