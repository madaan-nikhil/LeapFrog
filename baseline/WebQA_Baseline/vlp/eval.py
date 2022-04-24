import json
import re
import string
import sys
import time
from collections import Counter, defaultdict
import numpy as np
import spacy
sys.path.append('/home/ubuntu')
from BARTScore.bart_score import BARTScorer
from word2number import w2n
import pandas as pd

color_set= {'orangebrown', 'spot', 'yellow', 'blue', 'rainbow', 'ivory', 'brown', 'gray', 'teal', 'bluewhite', 'orangepurple', 'black', 'white', 'gold', 'redorange', 'pink', 'blonde', 'tan', 'turquoise', 'grey', 'beige', 'golden', 'orange', 'bronze', 'maroon', 'purple', 'bluere', 'red', 'rust', 'violet', 'transparent', 'yes', 'silver', 'chrome', 'green', 'aqua'}
shape_set = {'globular', 'octogon', 'ring', 'hoop', 'octagon', 'concave', 'flat', 'wavy', 'shamrock', 'cross', 'cylinder', 'cylindrical', 'pentagon', 'point', 'pyramidal', 'crescent', 'rectangular', 'hook', 'tube', 'cone', 'bell', 'spiral', 'ball', 'convex', 'square', 'arch', 'h', 'cuboid', 'step', 'rectangle', 'dot', 'oval', 'circle', 'star', 'crosse', 'crest', 'octagonal', 'cube', 'triangle', 'semicircle', 'domeshape', 'obelisk', 'corkscrew', 'curve', 'circular', 'xs', 'slope', 'pyramid', 'round', 'bow', 'straight', 'triangular', 'heart', 'fork', 'teardrop', 'fold', 'curl', 'spherical', 'diamond', 'keyhole', 'conical', 'dome', 'sphere', 'bellshaped', 'rounded', 'hexagon', 'flower', 'globe', 'torus'}
yesno_set = {'yes', 'no'}

def get_bart_scorer():
    TABLE = str.maketrans(dict.fromkeys(string.punctuation))
    bart_scorer_ParaBank = BARTScorer(
        device="cuda:0", checkpoint="facebook/bart-large-cnn"
    )
    def normalize_text_for_bart(x,):
        """Light text normalization for WebQA eval: white space fix + punctuation removal"""
        return " ".join(x.translate(TABLE).split())
    def compute_bartscore_ParaBank(c, a):
        t = time.time()
        c_removepunc = [normalize_text_for_bart(x) for x in c]
        a_removepunc = [normalize_text_for_bart(x) for x in a]
        score = np.exp(
            bart_scorer_ParaBank.score(a_removepunc, c_removepunc, batch_size=4)
        )
        print(f"Computed BARTScore in {time.time() - t} seconds")
        return score
    return compute_bartscore_ParaBank
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat", "parser"])
except OSError:
    print("Please run `python -m spacy download en_core_web_sm` first")
    exit()
def detectNum(l):
    result = []
    for w in l:
        try:
            result.append(str(int(w)))
        except:
            pass
    return result
def toNum(word):
    if word == "point":
        return word
    try:
        return w2n.word_to_num(word)
    except:
        return word
def normalize_text(s):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)
    def white_space_fix(text):  # additional: converting numbers to digit form
        return " ".join([str(toNum(w)) for w in text.split()])
    def remove_punc(text):
        exclude = set(string.punctuation) - set(["."])
        text1 = "".join(ch for ch in text if ch not in exclude)
        return re.sub(r"\.(?!\d)", "", text1)  # remove '.' if it's not a decimal point
    def lower(text):
        return text.lower()
    def lemmatization(text):
        return " ".join([token.lemma_ for token in nlp(text)])
    if len(s.strip()) == 1:
        # accept article and punc if input is a single char
        return white_space_fix(lower(s))
    elif len(s.strip().split()) == 1:
        # accept article if input is a single word
        return lemmatization(white_space_fix(remove_punc(lower(s))))
    return lemmatization(white_space_fix(remove_articles(remove_punc(lower(s)))))

def _webqa_acc_approx(predction, ground_truth, domain=None):
    """VQA Eval (SQuAD style EM, F1)"""
    bow_pred = normalize_text(predction).split()
    bow_target = normalize_text(ground_truth).split()
    if domain == {"NUMBER"}:
        bow_pred = detectNum(bow_pred)
        bow_target = detectNum(bow_target)
    elif domain is not None:
        bow_pred = list(domain.intersection(bow_pred))
        bow_target = list(domain.intersection(bow_target))
    else:
        # TODO: fine-grained evaluation (e.g., content words) for text question types
        bow_pred = bow_pred
        bow_target = bow_target
    common = Counter(bow_target) & Counter(bow_pred)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = num_same / len(bow_pred)
    recall = num_same / len(bow_target)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, recall, precision

def webqa_metrics_approx(prediction, ground_truth, Qcate="text"):
    f1, recall, precision = _webqa_acc_approx(
        prediction,
        ground_truth,
        domain={
            "color": color_set,
            "shape": shape_set,
            "YesNo": yesno_set,
            "number": {"NUMBER"},
            "text": None,
            "Others": None,
            "choose": None,
        }[Qcate],
    )
    if Qcate in ["color", "shape", "number", "YesNo"]:
        accuracy = f1
    else:
        accuracy = recall
    return {"acc_approx": accuracy}

if __name__ == "__main__":
    pred_file = sys.argv[1]
    log_metrics = defaultdict(list)
    
    df = pd.read_csv(pred_file, delimiter='\t')
    predictions = []
    #df = df.loc[df['Qcate']=='Others']
    for index, row in df.iterrows():
        #print(json.loads(row['Output']))
        #print(json.loads(row['Output'])[np.argmax(row['Output_conf'])])
        #print(json.loads(row['A']))
        #print(json.loads(row['Output_conf']))
        #print(np.argmax(json.loads(row['Output_conf'])))
        acc = webqa_metrics_approx(json.loads(row['Output'])[np.argmax(json.loads(row['Output_conf']))], json.loads(row['A'])[0],row['Qcate'])
        log_metrics["f1"].append(list(acc.values())[0])
        predictions.append(json.loads(row['Output'])[np.argmax(json.loads(row['Output_conf']))])
        #break
    
    #print(log_metrics["f1"])
    #exit(0)
    log_metrics = {f"val/{k}": np.mean(v) for k, v in log_metrics.items()}
    bart_scorer = get_bart_scorer()



    log_metrics["val/BARTScore"] = np.mean(
        np.array(bart_scorer(
            list(df['A']),
            predictions
        ))/np.array(bart_scorer(list(df['A']), list(df['A'])))
    )
    print(f'log_metrics {log_metrics}')
