from nltk.tokenize import word_tokenize
import scipy.stats as stats

no_com = []
com = []

references = open("ref.txt").readlines()
predictions_com = open("pred_com.txt").readlines()
predictions_no_com = open("pred_no_com.txt").readlines()

# com_out = open("com.out", "x")
# no_com_out = open("no_com.out", "x")


for i, ref in enumerate(references):
    ref_tokens = set(word_tokenize(ref))
    com_pred_tokens = set(word_tokenize(predictions_com[i]))
    no_com_pred_tokens = set(word_tokenize(predictions_no_com[i]))

    jac = 1 - (len(ref_tokens & com_pred_tokens) / len(ref_tokens | com_pred_tokens))
    jac_no = 1 - (
        len(ref_tokens & no_com_pred_tokens) / len(ref_tokens | no_com_pred_tokens)
    )
    # com_out.write(str(jac) + "\n")
    # no_com_out.write(str(jac_no) + "\n")
    com.append(jac)
    no_com.append(jac_no)

print(len(com), len(no_com))
print(stats.ranksums(com, no_com, alternative="less"))
