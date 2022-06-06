import subprocess
import sys

model_dir = "../../models/no_com/"

reference_file = model_dir + "ref.txt"
prediction_file = model_dir + "pred.txt"

with open(prediction_file) as predictions:
    pipe = subprocess.Popen(
        ["perl", "multi-bleu.perl", reference_file],
        stdin=predictions,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
