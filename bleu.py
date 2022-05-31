import nltk
import os
import subprocess
import sys

model_dirname = "./models/java-comments-gpu"

ref_file_name = "models/nc/ref.txt"
predicted_file_name = "models/nc/pred.txt"


def compute_bleu(ref_file_name, predicted_file_name):
    with open(predicted_file_name) as predicted_file:
        pipe = subprocess.Popen(
            ["perl", "scripts/multi-bleu.perl", ref_file_name],
            stdin=predicted_file,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )


compute_bleu(ref_file_name, predicted_file_name)
