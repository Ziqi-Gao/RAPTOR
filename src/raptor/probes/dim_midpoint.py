"""Raw-space difference-in-means classification with closed-form midpoint.

No logistic fitting, GCAV probability, strength, epsilon or threshold search
belongs to this readout. A zero score always predicts the positive class.
"""
from __future__ import annotations

from typing import Any
import numpy as np

METHOD = "dim_midpoint"
DESCRIPTION = "Difference-in-means with midpoint readout"
READOUT_VERSION = "raw_euclidean_midpoint_ge_zero_v1"
ZERO_DIRECTION_POLICY = "score_zero_predict_all_positive_and_explicit_degenerate_flag"


def _indices(hidden, indices):
    rows = np.asarray(indices)
    if rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows):
        raise ValueError("indices must be a nonempty integer vector")
    rows = rows.astype(np.int64, copy=False)
    if len(np.unique(rows)) != len(rows) or rows.min() < 0 or rows.max() >= len(hidden):
        raise ValueError("indices must be unique and within the activation rows")
    return rows


def _blocks(indices, rows=512):
    for start in range(0,len(indices),rows):
        yield indices[start:start+rows]


def from_means(mean_negative, mean_positive, fit_counts, *, coordinate_space="raw") -> dict[str,Any]:
    if coordinate_space != "raw":
        raise ValueError("midpoint requires verified raw activation coordinates; recover an actual matching scaler first")
    negative=np.asarray(mean_negative,dtype=np.float64);positive=np.asarray(mean_positive,dtype=np.float64)
    if negative.ndim!=1 or negative.shape!=positive.shape or not len(negative):
        raise ValueError("class means must be nonempty matching feature vectors")
    if not np.isfinite(negative).all() or not np.isfinite(positive).all():
        raise ValueError("nonfinite class mean")
    if (set(fit_counts)!={"negative","positive","total"} or not all(type(fit_counts[k]) is int for k in fit_counts)
            or fit_counts['negative']<=0 or fit_counts['positive']<=0
            or fit_counts['total']!=fit_counts['negative']+fit_counts['positive']):
        raise ValueError("both actual class sample counts must be positive and sum to fit count")
    difference=positive-negative
    midpoint=.5*positive+.5*negative
    bias=-float(np.dot(difference,midpoint))
    if not np.isfinite(difference).all() or not np.isfinite(midpoint).all() or not np.isfinite(bias):
        raise ValueError("nonfinite closed-form direction/midpoint/intercept")
    degenerate=bool(np.all(difference==0))
    return {'mean_negative':negative.copy(),'mean_positive':positive.copy(),'mean_difference':difference,
            'midpoint':midpoint,'W':difference.copy(),'b':bias,'fit_counts':dict(fit_counts),
            'method':METHOD,'readout_version':READOUT_VERSION,'coordinate_space':'raw',
            'score_rule':'(raw_h-midpoint) dot (mean_positive-mean_negative)',
            'classification_rule':'score>=0 predicts1; exact zero predicts1',
            'degenerate':degenerate,'zero_direction_policy':ZERO_DIRECTION_POLICY}


def fit_midpoint(hidden,labels,fit_indices) -> dict[str,Any]:
    hidden=np.asarray(hidden);labels=np.asarray(labels)
    if hidden.ndim!=2 or not hidden.shape[1] or labels.shape!=(len(hidden),):
        raise ValueError("activation/label dimensions do not match")
    indices=_indices(hidden,fit_indices)
    if set(np.unique(labels[indices]))!={0,1}:
        raise ValueError("both and only binary classes are required in fit partition")
    means=[];counts=[]
    for label in (0,1):
        selected=indices[labels[indices]==label]
        total=np.zeros(hidden.shape[1],dtype=np.float64)
        for rows in _blocks(selected):
            block=np.asarray(hidden[rows],dtype=np.float64)
            if not np.isfinite(block).all():raise ValueError("nonfinite activation in fit partition")
            total+=np.sum(block,axis=0,dtype=np.float64)
        means.append(total/len(selected));counts.append(int(len(selected)))
    return from_means(means[0],means[1],{'negative':counts[0],'positive':counts[1],'total':sum(counts)})


def scores(hidden,indices,readout):
    hidden=np.asarray(hidden)
    if hidden.ndim!=2 or hidden.shape[1]!=len(readout['mean_difference']):
        raise ValueError("activation/readout feature dimensions do not match")
    rows=_indices(hidden,indices)
    output=[]
    for chunk in _blocks(rows):
        block=np.asarray(hidden[chunk],dtype=np.float64)
        if not np.isfinite(block).all():raise ValueError("nonfinite activation in scoring partition")
        value=(block-readout['midpoint'])@readout['mean_difference']
        if not np.isfinite(value).all():raise ValueError("nonfinite midpoint score")
        output.append(value)
    return np.concatenate(output)


def predict(hidden,indices,readout):
    return (scores(hidden,indices,readout)>=0).astype(np.int64)


def accuracy(hidden,labels,indices,readout):
    labels=np.asarray(labels)
    if labels.shape!=(len(hidden),):raise ValueError("scoring labels/activation row counts do not match")
    rows=_indices(hidden,indices)
    if not set(np.unique(labels[rows])) <= {0,1}:raise ValueError("scoring labels must be binary")
    predicted=predict(hidden,rows,readout)
    return float(np.mean(predicted==labels[rows]))
