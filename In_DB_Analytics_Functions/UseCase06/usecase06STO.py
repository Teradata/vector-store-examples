# Import the necessary libraries
import sys
import numpy as np
import pandas as pd
import subprocess
import sys, os

from contextlib import contextmanager
import logging
import datetime
# from datetime import date
from dateutil.relativedelta import relativedelta

logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')

import argparse
import timeit

import numpy as np
import pandas as pd

from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as Imb_Pipeline
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_random_state = 0xDEADBEEF

label_column = ""

feature_columns = [
    "smart_5_raw",
    "smart_10_raw",
    "smart_184_raw",
    "smart_188_raw",
    "smart_197_raw",
    "smart_198_raw",
]

###
### Read input
###
delimiter = '\t'
inputData = []

for line in sys.stdin.read().splitlines():
    line = line.split(delimiter)
    inputData.append(line)

###
### If no data received, gracefully exit rather than producing an error later.
###

if not inputData:
    sys.exit()

###
### Set up input DataFrame according to input schema
###

# Know your data: You must know in advance the number of incoming columns from the database!
columns = ['id','failure_Date','serial_Number','model', 'failure', 'smart_5_raw', 'smart_10_raw','smart_184_raw', 'smart_188_raw','smart_197_raw', 'smart_198_raw']

raw_data = pd.DataFrame(inputData, columns=columns).copy()

del inputData

# type cast
raw_data['failure'] = pd.to_numeric(raw_data['failure'])

# ----------------------------------------------------------------------
def pre_process(data: pd.DataFrame, failures_only=True) -> (np.array, pd.DataFrame):
    if "failure" in data.columns:
        if failures_only:
            # only get the hdd's that fail eventually
            failures_raw = data.groupby(by=["serial_Number", "model"]).filter(lambda x: int(float(x["failure"].sum())) > 0)
        else:
            failures_raw = data
        failures_raw["ttf"] = pre_label(failures_raw).fillna(pd.Timedelta.max)
        failures_raw = failures_raw[failures_raw["ttf"] >= pd.Timedelta("0 days")]

        training_data = failures_raw[feature_columns]
        labels = failures_raw.ttf.apply(
            lambda x: label(x, thresholds=pd.Timedelta("1 days"))
        )

        return labels, training_data[feature_columns]
    else:
        return None, data[feature_columns]


def train(training_data: pd.DataFrame, labels: np.array):
    clf_pipeline = Imb_Pipeline(
        [
            ("upsample_random", ADASYN(random_state=_random_state)),
            ("std_scale", StandardScaler()),
            (
                "svc_rbf",
                SVC(kernel="rbf", class_weight="balanced", random_state=_random_state),
            ),
        ]
    )

    model = clf_pipeline.fit(training_data, labels)
    return model


def serve(model, data):
    predictions = model.predict(data)
    return predictions


def score(model, data, labels):
    predictions = serve(model, data)

    f_score = f1_score(labels, predictions, average="weighted")

    tn, fn, fp, tp = confusion_matrix(labels, predictions).ravel()

    # print(confusion_matrix(labels, predictions))
    # print(classification_report(labels, predictions))

    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    false_positive_rate = fp / (fp + tn)
    return {"f1": f_score, "fpr": false_positive_rate, "auc": auc}


def pre_label(
    df,
    absolute_time="failure_Date",
    failure_indicator="failure",
    grouping_key=["model", "serial_Number"],
):
    tmp = df.copy()
    tmp["last"] = df.apply(
        lambda x: x[absolute_time] if x[failure_indicator] == 1 else np.NaN,
        axis="columns",
    )

    # convert datatypes
    tmp['last'] = pd.to_datetime(tmp['last'])
    tmp['failure_Date'] = pd.to_datetime(tmp['failure_Date'])
    tmp["last"] = tmp.groupby(grouping_key)["last"].transform(lambda x: np.max(x))

    # needs to extract values first 
    tmp['last2'] = tmp["last"].values
    tmp['failure_Date2'] = tmp["failure_Date"].values
    return tmp["last2"] - tmp["failure_Date2"]


def label(x, thresholds=pd.Timedelta("1 days")):
    if x <= thresholds:
        return 1
    else:
        return 0

# ----------------------------------------------------------------------

# step1: pre_process
# failures_only: True - while training
failures_only = True
(labels, data) = pre_process(raw_data, failures_only)


# step2: training
model = train(data, labels)

# step3: serving
# before serving, we have to call pre_process with failures_only set to false
failures_only = False
(labels, data) = pre_process(raw_data, failures_only)
# serving
predictions = serve(model, data)

# step3: prepare output data
out_data = pd.DataFrame(
    {
        "model": raw_data["model"],
        "serial_number": raw_data["serial_Number"],
        "failure_Date": raw_data["failure_Date"],
        "failure": predictions,
    }
)

# scoring
# scores = score(model, data, labels)

#step4: Export results to Advanced SQL Engine through standard output in expected format.
for index, row in out_data.iterrows():
    print(row['model'], delimiter, row['serial_number'], delimiter, str(row['failure_Date']),  delimiter, row['failure'])
