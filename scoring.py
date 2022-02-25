import argparse
import sys
import json
from sklearn.metrics import precision_recall_fscore_support

#parameters
debug = False

###########
#functions#
###########

######
#main#
######

def main(args):

    groundtruth = json.load(open(args.groundtruth))
    predictions = json.load(open(args.predictions))

    y_true, y_pred = [], []
    for k, v in groundtruth.items():
        if k in predictions:
            y_pred.append(int(predictions[k]["label"]))
        #if ID isn't in predictions, assume incorrect predictions
        else:
            y_pred.append(int(not bool(v["label"])))
        y_true.append(int(v["label"]))
        
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, pos_label=1, average="binary")

    print("Performance on the positive class (documents with misinformation):")
    print("Precision =", p)
    print("Recall    =", r)
    print("F1        =", f)
        

if __name__ == "__main__":

    #parser arguments
    desc = "Computes precision, recall and F-score of positive class ('1')"
    parser = argparse.ArgumentParser(description=desc)

    #arguments
    parser.add_argument("--predictions", required=True, help="json file containing system predictions")
    parser.add_argument("--groundtruth", required=True, help="json file containing ground truth labels")
    args = parser.parse_args()

    main(args)
