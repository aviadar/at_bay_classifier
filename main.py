import os
import pandas as pd
import boto3
import json

from text_extractor import TextExtractor
from text_classifier import TextClassifier


OTHER_LABEL_LIMIT = 0.3


def main():
    s3_resource = boto3.resource('s3')
    my_bucket = s3_resource.Bucket('crawling-idc')
    objects = my_bucket.objects.filter(Prefix='year=2022/month=9/')
    object_depth = {}
    for obj in objects:
        path, filename = os.path.split(obj.key)
        os.makedirs(path, exist_ok=True)
        my_bucket.download_file(obj.key, os.path.join(path, filename))
        object_depth[obj.key] = json.loads(s3_resource.Object('crawling-idc', obj.key).metadata['metaflow-user-attributes'])['task']['depth']

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
        processed_text = (TextExtractor.extract_from_dir(out_dir[27:], object_depth))
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


if __name__ == "__main__":
    main()