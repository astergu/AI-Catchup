# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import utils

EVAL_CORPUS_PATH = "birth_dev.tsv"
NUM_PREDS = 500

if __name__ == '__main__':
    predictions = ["London"] * NUM_PREDS
    total, correct = utils.evaluate_places(EVAL_CORPUS_PATH, predictions)

    print('Correct: {} out of {}: {}%'.format(correct, total, correct/total*100))