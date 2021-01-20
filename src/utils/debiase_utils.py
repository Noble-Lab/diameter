import os, sys, re;
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from enum import Enum
import numpy as np
import networkx as nx
from bisect import bisect
from scipy.stats import norm


def quantile_normalize(key_group_map, ignore_lower_quantile=0.25, ignore_upper_quantile=0.75):
    if len(key_group_map) <= 0: return key_group_map;

    emsembled_value_arr = np.asarray([], dtype=float); key_quant_map = {};
    for key in key_group_map:
        value_arr = np.asarray(key_group_map[key]);
        value_lower = np.quantile(value_arr, ignore_lower_quantile);
        value_upper = np.quantile(value_arr, ignore_upper_quantile);
        value_median = np.quantile(value_arr, 0.5);

        value_arr = value_arr[(value_arr >= value_lower) & (value_arr <= value_upper)];
        emsembled_value_arr = np.concatenate((emsembled_value_arr, value_arr));
        key_quant_map[key] = (value_lower, value_upper, value_median);

    emsembled_value_lower = np.quantile(emsembled_value_arr, ignore_lower_quantile);
    emsembled_value_upper = np.quantile(emsembled_value_arr, ignore_upper_quantile);
    emsembled_value_median = np.quantile(emsembled_value_arr, 0.5);
    emsembled_value_gap = emsembled_value_upper - emsembled_value_lower;

    calibrated_key_group_map = {};
    for key in key_group_map:
        value_lower, value_upper, value_median = key_quant_map[key]; value_gap = value_upper - value_lower;
        value_arr = np.asarray(key_group_map[key]);
        # logger.info('key={}\temsembled_value_gap={}\tvalue_gap={}\tmedian={}'.format(key, emsembled_value_gap, value_gap, value_median));

        if value_gap > 0:
            value_arr *= (emsembled_value_gap / value_gap);
            value_arr += (emsembled_value_lower - np.quantile(value_arr, ignore_lower_quantile));
        else:
            value_arr *= (emsembled_value_median / value_median);
            value_arr += (emsembled_value_median - np.quantile(value_arr, 0.5));

        calibrated_key_group_map[key] = value_arr;

    return calibrated_key_group_map;


def quantile_normalize_v2(key_group_map):
    if len(key_group_map) <= 0: return key_group_map;

    emsembled_rank_arr = np.asarray([], dtype=float); emsembled_value_arr = np.asarray([], dtype=float); group_size_list = [];
    for key in key_group_map:
        group_arr = np.asarray(key_group_map[key]);
        rank_arr = np.zeros_like(group_arr);
        rank_arr[np.argsort(group_arr)] = list(range(1, 1+len(group_arr)));
        rank_arr /= len(rank_arr);

        group_size_list.append(len(group_arr));
        emsembled_rank_arr = np.concatenate((emsembled_rank_arr, rank_arr ));
        emsembled_value_arr = np.concatenate((emsembled_value_arr, group_arr));
    median_group_size = int(np.median(group_size_list));

    # train the mapping
    rank_partition = np.linspace(0, 1, median_group_size); mean_value_list = [];
    for rank_start, rank_end in zip(rank_partition[:-1], rank_partition[1:]):
        valid_indices = np.where((emsembled_rank_arr>=rank_start) & (emsembled_rank_arr<=rank_end))[0];
        mean_value_list.append(np.mean(emsembled_value_arr[valid_indices]));
    mean_value_list = np.asarray(mean_value_list);
    logger.info('rank_partition={}\tmean_value_list={}'.format(len(rank_partition), len(mean_value_list)));

    # use the mapping
    calibrated_key_group_map = {};
    for key in key_group_map:
        group_arr = np.asarray(key_group_map[key]);
        rank_arr = np.zeros_like(group_arr);
        rank_arr[np.argsort(group_arr)] = list(range(1, 1+len(group_arr)));
        rank_arr /= len(rank_arr);

        bin_indices = np.digitize(rank_arr, bins=rank_partition ) - 1;
        bin_indices = np.maximum(0, bin_indices-1);
        new_group_arr = mean_value_list[bin_indices];
        calibrated_key_group_map[key] = new_group_arr;

    return calibrated_key_group_map;


def quantile_normalize_v3(key_group_map, ignore_lower_quantile=0.10, ignore_upper_quantile=0.90):
    emsembled_rank_arr = np.asarray([], dtype=float); emsembled_value_arr = np.asarray([], dtype=float); key_quant_map = {}; group_size_list = [];
    for key in key_group_map:
        value_arr = np.asarray(key_group_map[key]);
        value_lower = np.quantile(value_arr, ignore_lower_quantile);
        value_upper = np.quantile(value_arr, ignore_upper_quantile);

        value_arr = value_arr[(value_arr >= value_lower) & (value_arr <= value_upper)];
        rank_arr = np.zeros_like(value_arr, dtype=float);
        rank_arr[np.argsort(value_arr)] = list(range(1, 1 + len(value_arr)));
        rank_arr /= len(rank_arr);

        group_size_list.append(len(value_arr));
        key_quant_map[key] = (value_lower, value_upper);
        emsembled_value_arr = np.concatenate((emsembled_value_arr, value_arr));
        emsembled_rank_arr = np.concatenate((emsembled_rank_arr, rank_arr));

    # train the mapping
    rank_partition = np.linspace(0, 1, int(np.median(group_size_list))); mean_value_list = [];
    for rank_start, rank_end in zip(rank_partition[:-1], rank_partition[1:]):
        valid_indices = np.where((emsembled_rank_arr>=rank_start) & (emsembled_rank_arr<=rank_end))[0];
        mean_value_list.append(np.mean(emsembled_value_arr[valid_indices]));
    mean_value_list = np.asarray(mean_value_list);

    # use the mapping
    calibrated_key_group_map = {};
    for key in key_group_map:
        value_lower, value_upper = key_quant_map[key]; # logger.info('value_lower={}, value_upper={}'.format(value_lower, value_upper));
        group_arr = np.asarray(key_group_map[key]); new_group_arr = np.zeros_like(group_arr, dtype=float);

        if np.fabs(value_lower- value_upper) <= 1e-3:
            value_arr = group_arr;

            rank_arr = np.zeros_like(value_arr, dtype=float);
            rank_arr[np.argsort(value_arr)] = list(range(1, 1 + len(value_arr)));
            rank_arr /= len(rank_arr);

            bin_indices = np.digitize(rank_arr, bins=rank_partition) - 1;
            bin_indices = np.maximum(0, bin_indices - 1);
            new_group_arr = mean_value_list[bin_indices];

        else:
            within_indices = np.where((group_arr >= value_lower) & (group_arr <= value_upper))[0];
            without_indices = np.where((group_arr < value_lower) | (group_arr > value_upper))[0];

            value_arr = group_arr[within_indices];
            rank_arr = np.zeros_like(value_arr, dtype=float);
            rank_arr[np.argsort(value_arr)] = list(range(1, 1 + len(value_arr)));
            rank_arr /= len(rank_arr);

            bin_indices = np.digitize(rank_arr, bins=rank_partition) - 1;
            bin_indices = np.maximum(0, bin_indices-1);
            new_group_arr[within_indices] = mean_value_list[bin_indices];

            if len(without_indices) > 0 and len(bin_indices) > 0:
                new_value_lower = np.min(mean_value_list[bin_indices]); new_value_upper = np.max(mean_value_list[bin_indices]);
                new_group_arr[without_indices] = (group_arr[without_indices] - value_upper)*(new_value_lower - new_value_upper)/(value_lower - value_upper) + new_value_upper;

        assert len(group_arr) == len(new_group_arr);
        calibrated_key_group_map[key] = new_group_arr;
    return calibrated_key_group_map;


# def quantile_normalize_v2(key_group_map, ignore_quantile=0.001):
#     emsembled_value_arr = np.asarray([], dtype=float);
#     for key in key_group_map: emsembled_value_arr = np.concatenate((emsembled_value_arr, key_group_map[key]));
#     emsembled_value_arr.sort();
#
#     logger.info('before emsembled_value_arr\tlen={}\tmin={}\tmax={}'.format(len(emsembled_value_arr), np.min(emsembled_value_arr), np.max(emsembled_value_arr) ));
#
#     value_lower = np.quantile(emsembled_value_arr, ignore_quantile); value_upper = np.quantile(emsembled_value_arr, 1.0-ignore_quantile);
#     emsembled_value_arr = emsembled_value_arr[(emsembled_value_arr>=value_lower) & (emsembled_value_arr<=value_upper)]; emsembled_size = len(emsembled_value_arr);
#     logger.info('after emsembled_value_arr\tlen={}\tmin={}\tmax={}'.format(len(emsembled_value_arr), np.min(emsembled_value_arr), np.max(emsembled_value_arr) ));
#
#     calibrated_key_group_map = {};
#     for key in key_group_map:
#         group_arr = np.asarray(key_group_map[key]);
#         rank_arr = np.zeros_like(group_arr);
#         rank_arr[np.argsort(group_arr)] = list(range(len(group_arr)));
#         new_group_arr = np.asarray([emsembled_value_arr[min(int(rank * 1.0 * emsembled_size / len(rank_arr)), emsembled_size - 1)] for rank in rank_arr]);
#         calibrated_key_group_map[key] = new_group_arr;
#
#     return calibrated_key_group_map;


# def quantile_normalize_v3(key_group_map):
#     def twoSampZ(data1, data2):
#         mu1 = np.mean(data1); std1 = np.std(data1); n1 = len(data1);
#         mu2 = np.mean(data2); std2 = np.std(data2); n2 = len(data2);
#
#         pooledSE = np.sqrt(std1 ** 2 / n1 + std2 ** 2 / n2);
#         z = (mu1 - mu2) / (pooledSE + np.finfo(np.float).eps);
#         pval = 2*(1 - norm.cdf(np.abs(z)));
#         return np.round(z, 3), np.round(pval, 4);
#
#
#     key_list = list(key_group_map.keys());
#     size_list = [len(key_group_map[key]) for key in key_list];
#     max_size = np.max(size_list); max_multikey = [np.argmax(size_list)];
#
#     key_graph = nx.Graph();
#     for key1_idx in range(len(key_list)):
#         key1 = key_list[key1_idx];
#         group_arr1 = np.asarray(key_group_map[key1]);
#
#         for key2_idx in range(key1_idx+1, len(key_list)):
#             key2 = key_list[key2_idx];
#             group_arr2 = np.asarray(key_group_map[key2]);
#
#             z, p = twoSampZ(group_arr1, group_arr2);
#             if p > 0.05: key_graph.add_edge(key1, key2); # logger.info('key1={}\tkey2={}\tz={}\tp={}'.format(key1, key2, z, p));
#
#     multikey_list = [list(c) for c in nx.connected_components(key_graph) if len(c) > 1];
#     for multikey in multikey_list:
#         multi_size = np.sum([len(key_group_map[key]) for key in multikey]);
#         if multi_size > max_size: max_size = multi_size; max_multikey = multikey;
#     logger.info('max_size={}\tmax_multikey={}'.format(max_size, max_multikey));
#
#
#     ref_arr = [];
#     for key in max_multikey: ref_arr = np.concatenate((ref_arr, key_group_map[key] ));
#     ref_arr = np.sort(ref_arr); ref_size = len(ref_arr);
#
#     calibrated_key_group_map = {};
#     for key in key_group_map:
#         group_arr = np.asarray(key_group_map[key]);
#         rank_arr = np.zeros_like(group_arr);
#         rank_arr[np.argsort(group_arr)] = list(range(len(group_arr)));
#
#         new_group_arr = np.asarray([ref_arr[min(int(rank * 1.0 * ref_size / len(rank_arr)), ref_size - 1)] for rank in rank_arr]);
#         calibrated_key_group_map[key] = new_group_arr;
#
