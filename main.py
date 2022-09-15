import pandas as pd
from text_extractor import TextExtractor
from text_classifier import TextClassifier

OTHER_LABEL_LIMIT = 0.3

df = pd.read_csv('restricted_dataset_public_bucket - Sheet1.csv')

# remove problematic labeling
df.label = df.label.str.replace('Payment Processing\n', 'Payment Processing')

# zero shot classification will not use 'other'
labels = list(df.label.unique())
labels.remove('Other')

# create empty df
res_df = pd.DataFrame(columns=['id', 'actual_label', 'predicted_label'])

classifier = TextClassifier()

for index, out_dir in df.ouput_dir.iteritems():

    processed_text = (TextExtractor.extract_from_dir('/Users/stavsalomon/Desktop/tryout/full' + out_dir[44:]))
    if processed_text is None or processed_text == '':
        continue

    res = classifier.classify(input_text=processed_text, candidate_labels=labels)
    if res['scores'][0] < OTHER_LABEL_LIMIT:
        pred_label = 'Other'
    else:
        pred_label = res['labels'][0]

    if not res_df.empty:
        res_df = pd.concat([res_df, pd.DataFrame({'id': df.job_id[index],
                                                  'actual_label': df.label[index],
                                                  'predicted_label': pred_label}, index=[0])],
                           ignore_index=True)
    else:
        res_df = pd.DataFrame({'id': df.job_id[index],
                               'actual_label': df.label[index],
                               'predicted_label': pred_label},
                              index=[0])

res_df.to_csv('red_df.csv')
