from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import scipy.stats as stats
from bisect import bisect_left
from typing import List
from cliffs_delta import cliffs_delta

no_com_jac = []
com_jac = []
no_com_bleu = []
com_bleu = []

references = open("ref.txt").readlines()
predictions_com = open("pred_com.txt").readlines()
predictions_no_com = open("pred_no_com.txt").readlines()

for i, ref in enumerate(references):
    ref_tokens = set(word_tokenize(ref))
    com_pred_tokens = set(word_tokenize(predictions_com[i]))
    no_com_pred_tokens = set(word_tokenize(predictions_no_com[i]))

    # Jaccard distance
    jac = 1 - (len(ref_tokens & com_pred_tokens) / len(ref_tokens | com_pred_tokens))
    jac_no = 1 - (
        len(ref_tokens & no_com_pred_tokens) / len(ref_tokens | no_com_pred_tokens)
    )
    com_jac.append(jac)
    no_com_jac.append(jac_no)

    # BLEU score
    bleu = sentence_bleu(
        ref_tokens, com_pred_tokens, smoothing_function=SmoothingFunction().method1
    )
    bleu_no = sentence_bleu(
        ref_tokens, no_com_pred_tokens, smoothing_function=SmoothingFunction().method1
    )
    com_bleu.append(bleu)
    no_com_bleu.append(bleu_no)

print(
    "Rank Sum with BLEU:\n", stats.ranksums(com_bleu, no_com_bleu, alternative="less")
)
print(
    "Rank Sum  with Jaccard Distance:\n",
    stats.ranksums(com_jac, no_com_jac, alternative="less"),
)


print("BLEU Cliff Delta:", cliffs_delta(com_bleu, no_com_bleu))
print("Jaccard Cliff Delta:", cliffs_delta(com_jac, no_com_jac))
