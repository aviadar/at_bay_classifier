import os
import pandas as pd
import boto3
import json
from os.path import exists
import pickle
import logging
from tqdm import tqdm

from text_extractor import TextExtractor
from text_classifier import TextClassifier


OTHER_LABEL_LIMIT = 0.3


def main():
    logging.basicConfig(filename='run.log', encoding='utf-8', level=logging.ERROR)
    logging.error('main started')
    s3_resource = boto3.resource('s3')
    my_bucket = s3_resource.Bucket('crawling-idc')
    objects = my_bucket.objects.filter(Prefix='year=2022/month=9/')
    obj_depth_path = 'obj_depth.pckl'
    is_obj_depth_new = True
    if exists(obj_depth_path):
        with open(obj_depth_path, 'rb') as fid:
            object_depth = pickle.load(fid)
        is_obj_depth_new = False
    else:
        object_depth = {}
    # i=0
    logging.error('downloading bucket')
    for obj in tqdm(objects):
        # i+=1
        path, filename = os.path.split(obj.key)
        os.makedirs(path, exist_ok=True)
        my_bucket.download_file(obj.key, os.path.join(path, filename))
        if is_obj_depth_new:
            object_depth[obj.key] = json.loads(s3_resource.Object('crawling-idc', obj.key).metadata['metaflow-user-attributes'])['task']['depth']
        # if i > 10:
        #     break

    with open(obj_depth_path, 'wb') as fid:
        pickle.dump(object_depth, fid)

    df = pd.read_csv('restricted_dataset_public_bucket - Sheet1.csv')

    # remove problematic labeling
    df.label = df.label.str.replace('Payment Processing\n', 'Payment Processing')

    # zero shot classification will not use 'other'
    labels = list(df.label.unique())
    labels.remove('Other')

    # create empty df
    res_df = pd.DataFrame(columns=['id', 'actual_label', 'predicted_label'])

    classifier = TextClassifier()

    logging.error('processing text')
    # print('processing text')
    for index, row in df.iterrows():
        processed_text = (TextExtractor.extract_from_dir(row.ouput_dir[27:], object_depth))
        if processed_text is None or processed_text == '':
            continue

        print(index)
        logging.error('classifying ' + str(index) + ' of ' + str(len(df.ouput_dir)))
        # print('classifying')
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

    res_df.to_csv('res_df.csv')


if __name__ == "__main__":
    main()
