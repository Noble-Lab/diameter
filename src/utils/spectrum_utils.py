import os, sys, re, zlib, base64;
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from enum import Enum
import numpy as np
from sklearn import linear_model


# mass of a hydrogen atom
HYDROGEN_MASS = 1.00782503207

DTYPE = np.dtype([('m/z array', np.float64), ('intensity array', np.float64)]).newbyteorder('>')

class IntTransformType(Enum):
    Origin = 1,
    Sqrt = 2,
    BasePeakNorm = 3,
    RankTransform = 4,
    SegmentedBasePeakNorm = 5


def mz_to_binidx(mz, bin_width=0.02, offset=0.0):
	return int( (mz-offset)/bin_width );


def get_uniq_binned_mz_intensity(mz_arr, intensity_arr, bin_width=0.02, offset=0.0, max_mzbin=1e5):
    mzbin_arr = np.asarray([mz_to_binidx(mz, bin_width, offset) for mz in mz_arr]);
    uniq_mzbin_arr, inverse_mzbin_indices = np.unique(mzbin_arr, return_inverse=True);
    uniq_intensity_arr = np.bincount(inverse_mzbin_indices, weights=intensity_arr);

    validIndices = np.where(uniq_mzbin_arr <= max_mzbin)[0];
    return uniq_mzbin_arr[validIndices], uniq_intensity_arr[validIndices];


def encode_mzintensity_arr(mz_arr, intensity_arr):
    assert len(mz_arr) == len(intensity_arr);
    assert len(intensity_arr) > 0;

    peak_data = [];
    for idx in range(len(mz_arr)):
        peak_data.append((mz_arr[idx], intensity_arr[idx]))

    if hasattr(np, 'getbuffer'):
        peak_buffer = np.getbuffer(np.asarray(peak_data, dtype=DTYPE));
    else:
        peak_buffer = memoryview(np.asarray(peak_data, dtype=DTYPE));

    peak_buffer = zlib.compress(peak_buffer)
    encoded_source = base64.b64encode(peak_buffer)
    return encoded_source;


def decode_mzintensity_arr(mz_intensity_str):

    if hasattr(mz_intensity_str, 'encode'):
        decoded_source = base64.b64decode(mz_intensity_str.encode('ascii'));
    else:
        decoded_source = base64.b64decode(mz_intensity_str);

    decoded_source = zlib.decompress(decoded_source)
    peak_data = np.frombuffer(decoded_source, dtype=DTYPE)

    mz_arr = []; intensity_arr = [];
    for idx in range(len(peak_data)):
        mz_arr.append(peak_data[idx][0])
        intensity_arr.append(peak_data[idx][1])
    return np.asarray(mz_arr), np.asarray(intensity_arr);


def calc_ppm(valueA, valueB):
    return abs(valueA - valueB) * 1e6 / max(valueA, valueB);


def calc_signed_ppm(valueA, valueB):
    return (valueA - valueB) * 1e6 / max(valueA, valueB);


def calc_masstol_by_ppm(mass, ppm):
    return ppm * mass * 1e-6;


def calc_masstol_range_by_ppm(mass, ppm):
    mass_tol = calc_masstol_by_ppm(mass, ppm);
    return [mass-mass_tol, mass+mass_tol];


def calc_mztol_range_by_ppm(mz, ppm):
    return (mz-HYDROGEN_MASS)*(1-ppm*1e-6)+HYDROGEN_MASS, (mz-HYDROGEN_MASS)*(1+ppm*1e-6)+HYDROGEN_MASS


def calc_mz_from_neutralmass_charge(neutral_mass, charge):
    return neutral_mass / (charge * 1.0) + HYDROGEN_MASS;

def calc_neutralmass_from_mz_charge(mz, charge):
    return (mz - HYDROGEN_MASS) * (charge * 1.0);


def calc_normalized_innerproduct(arr1, arr2):
    assert len(arr1) == len(arr2);
    if len(arr1) <= 0: return 0;
    return np.dot(arr1, arr2) / np.sqrt(np.dot(arr1, arr1) * np.dot(arr2, arr2) + np.finfo(np.float).eps);


def get_transformed_intensity(mz_arr, intensity_arr, int_transform_type):
    assert len(mz_arr) == len(intensity_arr);
    if len(intensity_arr) <= 0: return intensity_arr;

    intensity_arr = np.asarray(intensity_arr);
    if int_transform_type == IntTransformType.Sqrt:
        return np.sqrt(intensity_arr);
    elif int_transform_type == IntTransformType.BasePeakNorm:
        return intensity_arr / np.max(intensity_arr);
    elif int_transform_type == IntTransformType.RankTransform:
        rank_arr = np.zeros(len(intensity_arr));
        rank_arr[np.argsort(intensity_arr)] = list(range(len(intensity_arr)));
        rank_arr = rank_arr / np.sqrt(np.dot(rank_arr, rank_arr));
        return rank_arr;
    elif int_transform_type == IntTransformType.SegmentedBasePeakNorm:
        seg_num = 10;
        smallest_nonzero_mz = np.min(mz_arr); largest_nonzero_mz = np.max(mz_arr); assert smallest_nonzero_mz > 0;
        window_mz = [smallest_nonzero_mz + int(i * (largest_nonzero_mz - smallest_nonzero_mz) / (seg_num*1.0)) for i in range(seg_num)];
        window_mz.append(largest_nonzero_mz + 1);

        for segIdx in range(seg_num):
            segIndices = np.where((mz_arr >= window_mz[segIdx])&(mz_arr <= window_mz[segIdx+1]))[0];
            if len(segIndices) > 0: intensity_arr[segIndices] /= np.max(intensity_arr[segIndices]);
        return intensity_arr;
    else: return intensity_arr;


def get_grassfiltered_mz_intensity(mz_arr, intensity_arr, thres=0.05):
    assert len(mz_arr) == len(intensity_arr);
    if len(intensity_arr) <= 0: return mz_arr, intensity_arr, np.asarray([]);

    max_intensity = np.max(intensity_arr);
    filterThres = thres*max_intensity;

    validIndices = np.where(intensity_arr >= filterThres)[0];
    return mz_arr[validIndices], intensity_arr[validIndices];


def get_debiased_intensity(intensity_arr, percentile=90):
    log_intensity_arr = np.asarray(sorted(np.log(intensity_arr), reverse=True));
    log_rank_arr = np.log(np.asarray(range(len(intensity_arr))) + 1) - np.log(len(log_intensity_arr) + 1);

    cutoff_log_intensity = np.percentile(log_intensity_arr, percentile);
    valid_indices = np.where(log_intensity_arr >= cutoff_log_intensity)[0];

    regr = linear_model.LinearRegression();
    regr.fit(log_intensity_arr[valid_indices].reshape(-1, 1), log_rank_arr[valid_indices].reshape(-1, 1));
    slope = regr.coef_[0]; intercept = regr.intercept_;

    log_rank_refit = log_intensity_arr * slope + intercept;
    # log_intensity_refit = (log_rank_arr - intercept) / (slope + np.finfo(np.float).eps);

    log_intensity_uniq_mean = np.zeros(len(log_intensity_arr)); log_intensity_uniq_min = np.zeros(len(log_intensity_arr)); log_intensity_uniq_max = np.zeros(len(log_intensity_arr));
    log_intensity_uniq_arr, uniq_inverse_arr = np.unique(log_intensity_arr, return_inverse=True);
    for inverse_idx, log_intensity_uniq in enumerate(log_intensity_uniq_arr):
        valid_indices = np.where(uniq_inverse_arr == inverse_idx)[0];

        log_intensity_uniq_mean[valid_indices] = (np.mean(log_rank_arr[valid_indices]) - intercept) / (slope + np.finfo(np.float).eps);
        log_intensity_uniq_min[valid_indices] = (np.min(log_rank_arr[valid_indices]) - intercept) / (slope + np.finfo(np.float).eps);
        log_intensity_uniq_max[valid_indices] = (np.max(log_rank_arr[valid_indices]) - intercept) / (slope + np.finfo(np.float).eps);

    debiased_score1 = np.maximum(-log_rank_refit, 0);
    debiased_score3 = np.maximum(log_intensity_uniq_mean, 0);

    return debiased_score1, debiased_score3;


def estimate_noiselevel(intensity_arr, percent=10):
    lower_thres = np.percentile(intensity_arr, percent); upper_thres = np.percentile(intensity_arr, 100-percent);
    intensity_arr1 = intensity_arr[np.where((intensity_arr > lower_thres) & (intensity_arr < upper_thres))];

    noise_level = 0;
    if len(intensity_arr1) > 0:
        intensity_baseline = np.median(intensity_arr1);
        noise_level = np.sqrt(np.sum(np.square(intensity_arr1 - intensity_baseline)) / len(intensity_arr1));
    return noise_level;