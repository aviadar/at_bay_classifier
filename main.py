import os
import pandas as pd
import boto3
import json
from os.path import exists
import pickle
import logging
from tqdm import tqdm

from text_extractor import TextExtractor
from text_classifier import TextClassifier, GpuUsage

OTHER_LABEL_LIMIT = 0.132
DOWNLOAD_BUCKET = False


def main():
    logging.basicConfig(filename='run.log', encoding='utf-8', level=logging.FATAL)
    logging.error('main started')
    s3_resource = boto3.resource('s3')
    my_bucket = s3_resource.Bucket('crawling-idc')
    # objects = my_bucket.objects.filter(Prefix='year=2022/month=9/')
    obj_depth_path = 'obj_depth.pckl'
    is_obj_depth_new = True
    if exists(obj_depth_path):
        with open(obj_depth_path, 'rb') as fid:
            object_depth = pickle.load(fid)
        is_obj_depth_new = False
    else:
        object_depth = {}

    df = pd.read_csv('restricted_dataset_public_bucket - Sheet1.csv')

    if DOWNLOAD_BUCKET:
        logging.error('downloading bucket')
        for _, output_dir in tqdm(df.ouput_dir.iteritems()):
            # if output_dir[27:] != 'year=2022/month=9/day=12/hour=11/minutes=27/d63dff3f-e904-4b6e-803b-1e98994e9927':
            #     continue
            objects = my_bucket.objects.filter(Prefix=output_dir[27:])
            for obj in objects:
                path, filename = os.path.split(obj.key)
                os.makedirs(path, exist_ok=True)
                my_bucket.download_file(obj.key, os.path.join(path, filename))
                if is_obj_depth_new:
                    object_depth[obj.key] = \
                    json.loads(s3_resource.Object('crawling-idc', obj.key).metadata['metaflow-user-attributes'])['task'][
                        'depth']

        with open(obj_depth_path, 'wb') as fid:
            pickle.dump(object_depth, fid)

    # remove problematic labeling
    df.label = df.label.str.replace('Payment Processing\n', 'Credit')
    df.label = df.label.str.replace('Adult Content', 'Porn')
    df.label = df.label.str.replace('Debt Collection Agency', 'Debt')
    df.label = df.label.str.replace('Educational Services', 'Education')


    # zero shot classification will not use 'other'
    labels = list(df.label.unique())
    labels.remove('Other')

    # create empty df
    res_df = pd.DataFrame(columns=['id', 'actual_label', 'predicted_label', 'result_labels', 'result_scores'])

    classifier = TextClassifier(gpu=GpuUsage.Off)

    logging.error('processing text')
    for index, row in tqdm(df.iterrows()):
        if row.ouput_dir[27:] != 'year=2022/month=9/day=12/hour=11/minutes=27/d63dff3f-e904-4b6e-803b-1e98994e9927':
            continue
        # if index < 95:
        #     continue
        #
        # if index > 200:
        #     break
        try:
            processed_text = (TextExtractor.extract_from_dir(row.ouput_dir[27:], object_depth))
            if processed_text is None or processed_text == '':
                continue

            logging.error('classifying ' + str(index) + ' of ' + str(len(df.ouput_dir)))
            res = classifier.classify(input_text=processed_text, candidate_labels=labels)
            if res is None:
                pred_label = 'Unknown_lang'
            else:
                if res['scores'][0] < OTHER_LABEL_LIMIT:
                    pred_label = 'Other'
                else:
                    pred_label = res['labels'][0]
        except Exception as e:
            print(e)
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            logging.error(template.format(type(e).__name__, e.args))
            pred_label = 'Unknown'

        if not res_df.empty:
            if pred_label != 'Unknown' and pred_label != 'Unknown_lang':
                res_df = pd.concat([res_df, pd.DataFrame({'id': df.job_id[index],
                                                          'actual_label': df.label[index],
                                                          'predicted_label': pred_label,
                                                          'result_labels_0': res['labels'][0],
                                                          'result_scores_0': res['scores'][0],
                                                          'result_labels_1': res['labels'][1],
                                                          'result_scores_1': res['scores'][1],
                                                          'result_labels_2': res['labels'][2],
                                                          'result_scores_2': res['scores'][2]
                                                          }, index=[0])],
                                   ignore_index=True)
            else:
                res_df = pd.concat([res_df, pd.DataFrame({'id': df.job_id[index],
                                                          'actual_label': df.label[index],
                                                          'predicted_label': pred_label,
                                                          'result_labels_0': 'NaN',
                                                          'result_scores_0': 0,
                                                          'result_labels_1': 'NaN',
                                                          'result_scores_1': 0,
                                                          'result_labels_2': 'NaN',
                                                          'result_scores_2': 0
                                                          }, index=[0])],
                                   ignore_index=True)
        else:
            if pred_label != 'Unknown':
                res_df = pd.DataFrame({'id': df.job_id[index],
                                       'actual_label': df.label[index],
                                       'predicted_label': pred_label,
                                       'result_labels_0': res['labels'][0],
                                       'result_scores_0': res['scores'][0],
                                       'result_labels_1': res['labels'][1],
                                       'result_scores_1': res['scores'][1],
                                       'result_labels_2': res['labels'][2],
                                       'result_scores_2': res['scores'][2]
                                       },
                                      index=[0])
            else:
                res_df = pd.DataFrame({'id': df.job_id[index],
                                       'actual_label': df.label[index],
                                       'predicted_label': pred_label,
                                       'result_labels_0': 'NaN',
                                       'result_scores_0': 0,
                                       'result_labels_1': 'NaN',
                                       'result_scores_1': 0,
                                       'result_labels_2': 'NaN',
                                       'result_scores_2': 0
                                       },
                                      index=[0])

    res_df.to_csv('res_df.csv')


if __name__ == "__main__":
    main()
