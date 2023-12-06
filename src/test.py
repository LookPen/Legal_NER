import datetime

from seqeval.metrics import classification_report

if __name__ == '__main__':
    y_true = [['B-PER', 'I-PER', 'O', 'B-LOC', 'O'],
              ['B-ORG', 'I-ORG', 'O', 'O', 'B-PER'],
              ['O', 'O', 'O', 'B-LOC', 'O']]

    y_pred = [['B-PER', 'I-PER', 'O', 'B-LOC', 'O'],
              ['B-ORG', 'I-ORG', 'O', 'O', 'B-PER'],
              ['O', 'O', 'O', 'B-LOC', 'O']]

    print(datetime.datetime.now(), "####\n", classification_report(y_true, y_pred))
