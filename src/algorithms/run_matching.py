import os, sys, random, csv, math, subprocess, shutil
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pyteomics import mzxml,mgf
import numpy as np;
import networkx as nx;

if (sys.version_info > (3, 0)):
    from queue import PriorityQueue
else:
    from Queue import PriorityQueue

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns;

from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import operator
from scipy import sparse
from scipy.stats import pearsonr, spearmanr;
from scipy.special import logsumexp
import pandas as pd

import convertMzxml;
import utils.mzxml_utils as mzxml_utils;
import utils.spectrum_utils as spectrum_utils;
import utils.debiase_utils as debiase_utils;
import crux.read_xcorr_results as read_xcorr_results;


def getMatchingParamDict(alphaPreP, alphaMS2P, alphaInt, alphaRT, alphaTCC, betaTCC, lamdaTCC, ppm, rtdiff_thres, evalue_mode, metric_type_list=[], bias_type_list=[], inputFile='', xcorrFile='', cruxExecFile='', fastaFile=''):
    matchingParaDict = {};


    matchingParaDict["alphaPreP"] = alphaPreP;
    matchingParaDict["alphaMS2P"] = alphaMS2P;
    matchingParaDict["alphaInt"] = alphaInt;
    matchingParaDict["alphaRT"] = alphaRT;
    matchingParaDict["alphaTCC"] = alphaTCC;

    matchingParaDict["betaTCC"] = betaTCC;
    matchingParaDict["lamdaTCC"] = lamdaTCC;

    matchingParaDict["ppm"] = ppm;
    matchingParaDict["rtdiff_thres"] = rtdiff_thres;
    matchingParaDict["evalue_mode"] = evalue_mode;

    matchingParaDict["metric_type_list"] = metric_type_list;
    matchingParaDict["metrics_str"] = ','.join([str(x) for x in metric_type_list]);

    matchingParaDict["bias_type_list"] = bias_type_list;
    matchingParaDict["bias_str"] = str(bias_type_list[0]);

    matchingParaDict["inputFile"] = inputFile;
    matchingParaDict["xcorrFile"] = xcorrFile;
    matchingParaDict["cruxExecFile"] = cruxExecFile;
    matchingParaDict["fastaFile"] = fastaFile;
    return matchingParaDict;


def getMatchingParamStr(matchingParaDict, modular=False):
    if modular:
        return 'met{}_bis{}_aPreP{}_aMS2P{}_aInt{}_aRT{}_aTCC{}_ppm{}_rtd{}_filter{}'.format(
        matchingParaDict["metrics_str"], matchingParaDict["bias_str"],
        matchingParaDict["alphaPreP"], matchingParaDict["alphaMS2P"], matchingParaDict["alphaInt"], matchingParaDict["alphaRT"], matchingParaDict["alphaTCC"],
        matchingParaDict["ppm"], matchingParaDict["rtdiff_thres"], matchingParaDict["evalue_mode"]);
    else:
        return 'met{}_bis{}_aPreP{}_aMS2P{}_aInt{}_aRT{}_aTCC{}_bTCC{}_lTCC{}_ppm{}_rtd{}_filter{}'.format(
        matchingParaDict["metrics_str"], matchingParaDict["bias_str"],
        matchingParaDict["alphaPreP"], matchingParaDict["alphaMS2P"], matchingParaDict["alphaInt"], matchingParaDict["alphaRT"], matchingParaDict["alphaTCC"], matchingParaDict["betaTCC"], matchingParaDict["lamdaTCC"],
        matchingParaDict["ppm"], matchingParaDict["rtdiff_thres"], matchingParaDict["evalue_mode"]);


def get_debiased_metrics():
    return ['Rtdiff1', 'IntensityIsoCorr1', 'IntensityIsoCorr2', 'IntensityMS1MS2Corr' ];  # 'MS2Pvalue1', 'IntensityIsoCorr1', 'IntensityIsoCorr2'

def get_type_metrics_mapping():
    return {0: ['lnEvalue', ],
            1: ['MS2Pvalue1' ],
            2: ['MS2Pvalue2' ],
            3: ['Rtdiff1'],
            4: ['Intensity'],
            5: ['IntensityM1', 'IntensityM2'],
            6: ['lnIntensityRank1', 'lnIntensityRank2'],
            7: ['IntensityIsoCorr1', 'IntensityIsoCorr2'],          # good in Orbitrap, bad in TTOF
            8: ['IntensityMS1MS2Corr', 'IntensityMS2Corr'],         # good in Orbitrap, bad in TTOF
            9: ['CC', 'TC'],                                        # bad
            10: ['Xcorr', 'XcorrRank'],
            11: ['lnPvalue' ],                                      # bad in Orbitrap, good in TTOF
            12: ['lnPvalueMean' ],                                  # bad in Orbitrap, good in TTOF
            13: ['lnEvalueAccpep', 'lnEvalueCoop'],
            14: ['Tailor'],
            };


def get_debiased_evidence(evidences_url, debiase_types, qn_type=1):
    assert qn_type in [1, 2];

    evidences_url_sortbyscan = evidences_url.replace('.txt', '_sorted.txt');
    if not os.path.isfile(evidences_url_sortbyscan):
        header_url = evidences_url.replace('.txt', '_tmp1.txt');
        body_url = evidences_url.replace('.txt', '_tmp2.txt');

        command_line = "head -1 {} > {}".format(evidences_url, header_url);
        print(subprocess.check_output(command_line, shell=True));

        command_line = "tail -n+2 {} | sort -k7n -k10n -k19nr -o {}".format(evidences_url, body_url);
        print(subprocess.check_output(command_line, shell=True));

        command_line = "cat {} {} > {}".format(header_url, body_url, evidences_url_sortbyscan);
        print(subprocess.check_output(command_line, shell=True));
        os.remove(header_url); os.remove(body_url);

    if len(debiase_types) <= 0: return evidences_url_sortbyscan;
    metric_list = get_debiased_metrics();

    # Calibrate by Quantile normalization
    def calibrate_by_qn(debiase_type, evidence_url):
        metric_key_values_map = {};
        for metric_type in metric_list: metric_key_values_map[metric_type] = {};

        with open(evidence_url) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                if debiase_type == 1: bias_key = int(row['MS1Scan']);
                elif debiase_type == 2: bias_key = int(row['Scanwin']);
                elif debiase_type == 3: bias_key = int(row['Peplen']);
                elif debiase_type == 4: bias_key = int(row['Charge']);
                else: assert False;

                for metric_type in metric_list:
                    if bias_key not in metric_key_values_map[metric_type]: metric_key_values_map[metric_type][ bias_key] = [];
                    value = float(row[metric_type]);
                    if not np.isinf(value) and not np.isnan(value): metric_key_values_map[metric_type][bias_key].append(value);

        for metric_type in metric_list:
            for bias_key in metric_key_values_map[metric_type].keys():
                metric_key_values_map[metric_type][bias_key] = np.asarray(metric_key_values_map[metric_type][bias_key]);

        return metric_key_values_map;

    final_debiased_evidence_url = evidences_url_sortbyscan.replace('pepid_evidence', 'pepid_debias{}_qn{}_evidence'.format( ','.join([str(x) for x in debiase_types]), qn_type )); logger.info(final_debiased_evidence_url);
    if os.path.isfile(final_debiased_evidence_url): return final_debiased_evidence_url;

    curr_debiased_evidence_url = evidences_url_sortbyscan;
    for debiase_idx, debiase_type in enumerate(debiase_types):
        new_debiased_evidence_url = evidences_url_sortbyscan.replace('pepid_evidence', 'pepid_debias{}_qn{}_evidence'.format( ','.join([str(x) for x in debiase_types[:debiase_idx+1]]), qn_type )); logger.info(new_debiased_evidence_url);
        if not os.path.isfile(new_debiased_evidence_url):
            metric_key_values_map = calibrate_by_qn(debiase_type, curr_debiased_evidence_url); calibrated_metric_key_values_map = {};
            for metric_type in metric_list:
                logger.info('metric_type={}\tdebiase_type={}\tqn_type={}'.format(metric_type, debiase_type, qn_type));
                if qn_type == 1: calibrated_metric_key_values_map[metric_type] = debiase_utils.quantile_normalize_v3( metric_key_values_map[metric_type], ignore_lower_quantile=0.10, ignore_upper_quantile=0.90);
                elif qn_type == 2: calibrated_metric_key_values_map[metric_type] = debiase_utils.quantile_normalize_v3( metric_key_values_map[metric_type], ignore_lower_quantile=0.25, ignore_upper_quantile=0.75);
                else: assert False;

            input_csvfile = open(curr_debiased_evidence_url);
            reader = csv.DictReader(input_csvfile, delimiter='\t');

            output_csvfile = open(new_debiased_evidence_url, 'w');
            writer = csv.DictWriter(output_csvfile, fieldnames=reader.fieldnames, delimiter='\t');
            writer.writeheader();

            for row in reader:
                if debiase_type == 1: bias_key = int(row['MS1Scan']);
                elif debiase_type == 2: bias_key = int(row['Scanwin']);
                elif debiase_type == 3: bias_key = int(row['Peplen']);
                elif debiase_type == 4: bias_key = int(row['Charge']);
                else: assert False;

                for metric_type in metric_list:
                    cal_value_list = [];
                    for value in [float(x) for x in row[metric_type].split(',')]:
                        if not np.isinf(value) and not np.isnan(value):
                            hit_idx = np.argmin(np.fabs(metric_key_values_map[metric_type][bias_key] - value));
                            cal_value = calibrated_metric_key_values_map[metric_type][bias_key][hit_idx];
                            cal_value_list.append(cal_value);
                        else: cal_value_list.append(value);
                    row[metric_type] = ','.join([str(x) for x in cal_value_list]);
                writer.writerow(row);

            output_csvfile.close(); input_csvfile.close();
        curr_debiased_evidence_url = new_debiased_evidence_url;
    return curr_debiased_evidence_url;


def evidence_tdc_marginal(evidences_url, matchingParaDict, train_fdr=0.01, test_fdr=0.01):
    tdc_filter_str = 'aPreP{}_aMS2P{}_aInt{}_aRT{}_aCorr{}.pin'.format(matchingParaDict["alphaPreP"], matchingParaDict["alphaMS2P"], matchingParaDict["alphaInt"], matchingParaDict["alphaRT"], matchingParaDict["alphaTCC"]);
    logger.info('tdc_filter_str={}'.format(tdc_filter_str));

    percolator_evidence_url_list = [];
    for file in os.listdir(os.path.dirname(evidences_url)):
        if 'percolator_met' in file and tdc_filter_str in file: percolator_evidence_url_list.append(os.path.join(os.path.dirname(evidences_url), file));
    logger.info('percolator_evidence_url_list={}'.format(percolator_evidence_url_list));

    for percolator_url in percolator_evidence_url_list:
        percolator_marginal_url_list = [];

        percolator_noPvalue_url = percolator_url.replace('.pin', '_noPvalue.pin');
        percolator_noPvalue_dir = percolator_noPvalue_url.replace('percolator_', 'results_percolator_').replace('.pin', '');
        if not os.path.isfile(os.path.join(percolator_noPvalue_dir, 'eval_new.txt')):
            percolator_noPvalue_cmd =  "cut -f13-14 --complement {} > {}".format(percolator_url, percolator_noPvalue_url);
            if not os.path.isfile(percolator_noPvalue_url): print(subprocess.check_output(percolator_noPvalue_cmd, shell=True));
            percolator_marginal_url_list.append(percolator_noPvalue_url);

        percolator_noRT_url = percolator_url.replace('.pin', '_noRT.pin');
        percolator_noRT_dir = percolator_noRT_url.replace('percolator_', 'results_percolator_').replace('.pin', '');
        if not os.path.isfile(os.path.join(percolator_noRT_dir, 'eval_new.txt')):
            percolator_noRT_cmd =  "cut -f15 --complement {} > {}".format(percolator_url, percolator_noRT_url);
            if not os.path.isfile(percolator_noRT_url): print(subprocess.check_output(percolator_noRT_cmd, shell=True));
            percolator_marginal_url_list.append(percolator_noRT_url);

        percolator_noInt_url = percolator_url.replace('.pin', '_noInt.pin');
        percolator_noInt_dir = percolator_noInt_url.replace('percolator_', 'results_percolator_').replace('.pin', '');
        if not os.path.isfile(os.path.join(percolator_noInt_dir, 'eval_new.txt')):
            percolator_noInt_cmd =  "cut -f16-17 --complement {} > {}".format(percolator_url, percolator_noInt_url);
            if not os.path.isfile(percolator_noInt_url): print(subprocess.check_output(percolator_noInt_cmd, shell=True));
            percolator_marginal_url_list.append(percolator_noInt_url);

        percolator_noElution_url = percolator_url.replace('.pin', '_noElution.pin');
        percolator_noElution_dir = percolator_noElution_url.replace('percolator_', 'results_percolator_').replace('.pin', '');
        if not os.path.isfile(os.path.join(percolator_noElution_dir, 'eval_new.txt')):
            percolator_noElution_cmd =  "cut -f18-21 --complement {} > {}".format(percolator_url, percolator_noElution_url);
            if not os.path.isfile(percolator_noElution_url): print(subprocess.check_output(percolator_noElution_cmd, shell=True));
            percolator_marginal_url_list.append(percolator_noElution_url);

        percolator_noXcorr_url = percolator_url.replace('.pin', '_noXcorr.pin');
        percolator_noXcorr_dir = percolator_noXcorr_url.replace('percolator_', 'results_percolator_').replace('.pin', '');
        if not os.path.isfile(os.path.join(percolator_noXcorr_dir, 'eval_new.txt')):
            percolator_noXcorr_cmd =  "cut -f22 --complement {} > {}".format(percolator_url, percolator_noXcorr_url);
            if not os.path.isfile(percolator_noXcorr_url): print(subprocess.check_output(percolator_noXcorr_cmd, shell=True));
            percolator_marginal_url_list.append(percolator_noXcorr_url);

        # percolator_onlyPvalue_url = percolator_url.replace('.pin', '_onlyPvalue.pin');
        # percolator_onlyPvalue_dir = percolator_onlyPvalue_url.replace('percolator_', 'results_percolator_').replace('.pin', '');
        # if not os.path.isfile(os.path.join(percolator_onlyPvalue_dir, 'eval_new.txt')):
        #     percolator_onlyPvalue_cmd =  "cut -f15-22 --complement {} > {}".format(percolator_url, percolator_onlyPvalue_url);
        #     if not os.path.isfile(percolator_onlyPvalue_url): print(subprocess.check_output(percolator_onlyPvalue_cmd, shell=True));
        #     percolator_marginal_url_list.append(percolator_onlyPvalue_url);
        #
        # percolator_onlyRT_url = percolator_url.replace('.pin', '_onlyRT.pin');
        # percolator_onlyRT_dir = percolator_onlyRT_url.replace('percolator_', 'results_percolator_').replace('.pin', '');
        # if not os.path.isfile(os.path.join(percolator_onlyRT_dir, 'eval_new.txt')):
        #     percolator_onlyRT_cmd =  "cut -f13-14,16-22 --complement {} > {}".format(percolator_url, percolator_onlyRT_url);
        #     if not os.path.isfile(percolator_onlyRT_url): print(subprocess.check_output(percolator_onlyRT_cmd, shell=True));
        #     percolator_marginal_url_list.append(percolator_onlyRT_url);
        #
        # percolator_onlyInt_url = percolator_url.replace('.pin', '_onlyInt.pin');
        # percolator_onlyInt_dir = percolator_onlyInt_url.replace('percolator_', 'results_percolator_').replace('.pin', '');
        # if not os.path.isfile(os.path.join(percolator_onlyInt_dir, 'eval_new.txt')):
        #     percolator_onlyInt_cmd =  "cut -f13-15,18-22 --complement {} > {}".format(percolator_url, percolator_onlyInt_url);
        #     if not os.path.isfile(percolator_onlyInt_url): print(subprocess.check_output(percolator_onlyInt_cmd, shell=True));
        #     percolator_marginal_url_list.append(percolator_onlyInt_url);
        #
        # percolator_onlyElution_url = percolator_url.replace('.pin', '_onlyElution.pin');
        # percolator_onlyElution_dir = percolator_onlyElution_url.replace('percolator_', 'results_percolator_').replace('.pin', '');
        # if not os.path.isfile(os.path.join(percolator_onlyElution_dir, 'eval_new.txt')):
        #     percolator_onlyElution_cmd =  "cut -f13-17,22 --complement {} > {}".format(percolator_url, percolator_onlyElution_url);
        #     if not os.path.isfile(percolator_onlyElution_url): print(subprocess.check_output(percolator_onlyElution_cmd, shell=True));
        #     percolator_marginal_url_list.append(percolator_onlyElution_url);
        #
        # percolator_onlyXcorr_url = percolator_url.replace('.pin', '_onlyXcorr.pin');
        # percolator_onlyXcorr_dir = percolator_onlyXcorr_url.replace('percolator_', 'results_percolator_').replace('.pin', '');
        # if not os.path.isfile(os.path.join(percolator_onlyXcorr_dir, 'eval_new.txt')):
        #     percolator_onlyXcorr_cmd =  "cut -f13-21 --complement {} > {}".format(percolator_url, percolator_onlyXcorr_url);
        #     if not os.path.isfile(percolator_onlyXcorr_url): print(subprocess.check_output(percolator_onlyXcorr_cmd, shell=True));
        #     percolator_marginal_url_list.append(percolator_onlyXcorr_url);

        for percolator_marginal_url in percolator_marginal_url_list:
            logger.info('percolator_marginal_url={}'.format(percolator_marginal_url));
            percolator_marginal_output_dir = percolator_marginal_url.replace('percolator_', 'results_percolator_').replace('.pin', '');
            if not os.path.isfile(os.path.join(percolator_marginal_output_dir, 'eval_new.txt')):
                command_line = "sh {}/{} {} {} {} {} {} {}".format(os.path.dirname(os.path.realpath(__file__)), 'run_percolator.sh', percolator_marginal_url, matchingParaDict["fastaFile"], percolator_marginal_output_dir, matchingParaDict["cruxExecFile"], train_fdr, test_fdr);
                print(command_line);
                print(subprocess.check_output(command_line, shell=True));
                plot_qvalue_via_pseudolabel_diasearch(percolator_marginal_output_dir, matchingParaDict);

            # os.remove(os.path.join(percolator_marginal_output_dir, 'percolator.target.peptides.txt'));
            # os.remove(os.path.join(percolator_marginal_output_dir, 'percolator.decoy.peptides.txt'));
            # os.remove(os.path.join(percolator_marginal_output_dir, 'percolator.decoy.psms.txt'));
            # os.remove(os.path.join(percolator_marginal_output_dir, 'percolator.target.psms.txt'));
            os.remove(percolator_marginal_url);


def evidence_tdc_marginal2(evidences_url, matchingParaDict, minmax_norm = True, train_fdr=0.01, test_fdr=0.01 ):
    assert 'sorted' in evidences_url;
    exclude_fieldnames = ['Edgeid', 'Pepid', 'MS1Scan', 'MS2Scan', 'Scanwin', 'Charge', 'NgbrEdgeids' ];
    metric_str = ','.join([str(x) for x in matchingParaDict["metric_type_list"]]);

    type_metrics_map = get_type_metrics_mapping(); all_metric_list = [];
    for metric_type in type_metrics_map: all_metric_list = np.concatenate((all_metric_list, type_metrics_map[metric_type]));
    all_metric_list = np.concatenate((all_metric_list, exclude_fieldnames));


    onlyPvalue_fieldnames = ['lnPvalue', 'lnPvalueMean'];
    onlyRT_fieldnames = ['Rtdiff1'];
    onlyInt_fieldnames = ['lnIntensityRank1', 'lnIntensityRank2'];
    onlyElution_fieldnames = ['IntensityIsoCorr1', 'IntensityIsoCorr2', 'IntensityMS1MS2Corr', 'IntensityMS2Corr'];
    onlyXcorr_fieldnames = ['Tailor'];

    for marginal_label in ['noPvalue', 'noRT', 'noInt', 'noElution', 'noXcorr', 'onlyPvalue', 'onlyRT', 'onlyInt', 'onlyElution', 'onlyXcorr']:
        if marginal_label == 'noPvalue':
            accept_metric_list = np.concatenate((onlyRT_fieldnames, onlyInt_fieldnames, onlyElution_fieldnames, onlyXcorr_fieldnames ));
            tdc_filter_str = 'aInt{}_aRT{}_aCorr{}_{}'.format(matchingParaDict["alphaInt"], matchingParaDict["alphaRT"], matchingParaDict["alphaTCC"], marginal_label);
        elif marginal_label == 'noRT':
            accept_metric_list = np.concatenate((onlyPvalue_fieldnames, onlyInt_fieldnames, onlyElution_fieldnames, onlyXcorr_fieldnames ));
            tdc_filter_str = 'aMS2P{}_aInt{}_aCorr{}_{}'.format(matchingParaDict["alphaMS2P"], matchingParaDict["alphaInt"], matchingParaDict["alphaTCC"], marginal_label);
        elif marginal_label == 'noInt':
            accept_metric_list = np.concatenate((onlyPvalue_fieldnames, onlyRT_fieldnames, onlyElution_fieldnames, onlyXcorr_fieldnames ));
            tdc_filter_str = 'aMS2P{}_aRT{}_aCorr{}_{}'.format(matchingParaDict["alphaMS2P"], matchingParaDict["alphaRT"], matchingParaDict["alphaTCC"], marginal_label);
        elif marginal_label == 'noElution':
            accept_metric_list = np.concatenate((onlyPvalue_fieldnames, onlyRT_fieldnames, onlyInt_fieldnames, onlyXcorr_fieldnames ));
            tdc_filter_str = 'aMS2P{}_aInt{}_aRT{}_{}'.format(matchingParaDict["alphaMS2P"], matchingParaDict["alphaInt"], matchingParaDict["alphaRT"], marginal_label);
        elif marginal_label == 'noXcorr':
            accept_metric_list = np.concatenate((onlyPvalue_fieldnames, onlyRT_fieldnames, onlyInt_fieldnames, onlyElution_fieldnames ));
            tdc_filter_str = 'aMS2P{}_aInt{}_aRT{}_aCorr{}_{}'.format(matchingParaDict["alphaMS2P"], matchingParaDict["alphaInt"], matchingParaDict["alphaRT"], matchingParaDict["alphaTCC"], marginal_label);
        elif marginal_label == 'onlyPvalue':
            accept_metric_list = onlyPvalue_fieldnames;
            tdc_filter_str = 'aMS2P{}_{}'.format(matchingParaDict["alphaMS2P"], marginal_label);
        elif marginal_label == 'onlyRT':
            accept_metric_list = onlyRT_fieldnames;
            tdc_filter_str = 'aRT{}_{}'.format(matchingParaDict["alphaRT"], marginal_label);
        elif marginal_label == 'onlyInt':
            accept_metric_list = onlyInt_fieldnames;
            tdc_filter_str = 'aInt{}_{}'.format(matchingParaDict["alphaInt"], marginal_label);
        elif marginal_label == 'onlyElution':
            accept_metric_list = onlyElution_fieldnames;
            tdc_filter_str = 'aCorr{}_{}'.format(matchingParaDict["alphaTCC"], marginal_label);
        elif marginal_label == 'onlyXcorr':
            accept_metric_list = onlyXcorr_fieldnames;
            tdc_filter_str = '{}'.format(marginal_label);
        else: assert False;


        tdc_evidence_url = evidences_url.replace('.txt', '_trainfdr{}_testfdr{}_{}.txt'.format(train_fdr, test_fdr, tdc_filter_str));
        tdc_percolator_evidence_url = tdc_evidence_url.replace('pepid_', 'percolator_met{}_'.format(metric_str)).replace('.txt', '.pin');
        tdc_percolator_output_dir = tdc_percolator_evidence_url.replace('percolator_', 'results_percolator_').replace('.pin', '');
        if os.path.isfile(os.path.join(tdc_percolator_output_dir, 'eval_new.txt')):  continue;

        if minmax_norm:
            minmax_norm_shift_map = get_minmax_norm_shift(evidences_url, matchingParaDict);
            MS2Pvalue1_lower, MS2Pvalue1_upper = minmax_norm_shift_map['MS2Pvalue1'];
            lnIntensityRank2_lower, lnIntensityRank2_upper = minmax_norm_shift_map['lnIntensityRank2'];


        if not os.path.isfile(tdc_percolator_evidence_url):
            input_csvfile = open(evidences_url);
            reader = csv.DictReader(input_csvfile, delimiter='\t');

            combined_header_list = [];
            for curr_fieldname in reader.fieldnames:
                if curr_fieldname not in all_metric_list or curr_fieldname in accept_metric_list: combined_header_list.append(curr_fieldname);

            output_csvfile = open(tdc_percolator_evidence_url, 'w');
            writer = csv.DictWriter(output_csvfile, fieldnames=combined_header_list, delimiter='\t');
            writer.writeheader();

            curr_ms2scan_charge = -1; record_list = [];
            def deal_with_ms2scan_charge(curr_ms2scan_charge, record_list):

                record_list.sort(key=lambda x: -x[0]);
                ref_ensemble = record_list[0][1] - 1e-8;
                record_list.sort(key=lambda x: -x[1]);  # rank by ensemble_score

                whitelist_edgeid_set = set();
                for xcorr, ensemble_score, label, edgeid, row in record_list:
                    if ref_ensemble <= ensemble_score: whitelist_edgeid_set.add(edgeid);

                for xcorr, ensemble_score, label, edgeid, row in record_list:
                    if edgeid not in whitelist_edgeid_set: continue;
                    data = {key: value for key, value in row.items() if key in combined_header_list};
                    writer.writerow(data);

            for row in reader:
                label = int(row['Label']); edgeid = int(row['Edgeid']); Xcorr = float(row['Xcorr']);
                ms2scan = int(row['MS2Scan']); charge = int(row['Charge']); ms2scan_charge = '{}_{}'.format(ms2scan, charge);

                Tailor = float(row['Tailor']);
                MS2Pvalue1 = float(row['MS2Pvalue1']);
                lnIntensityRank2 = float(row['lnIntensityRank2']);
                Rtdiff1 = float(row['Rtdiff1']);
                IntensityIsoCorr1 = float(row['IntensityIsoCorr1']); IntensityMS1MS2Corr = float(row['IntensityMS1MS2Corr']);

                if minmax_norm:
                    MS2Pvalue1 = (MS2Pvalue1 - MS2Pvalue1_lower) / (MS2Pvalue1_upper - MS2Pvalue1_lower);
                    lnIntensityRank2 = (lnIntensityRank2 - lnIntensityRank2_lower) / (lnIntensityRank2_upper - lnIntensityRank2_lower);

                if marginal_label == 'noPvalue': ensemble_score = Tailor - matchingParaDict["alphaInt"] * lnIntensityRank2  - matchingParaDict["alphaRT"] * Rtdiff1 + matchingParaDict["alphaTCC"] * (IntensityIsoCorr1 + IntensityMS1MS2Corr);
                elif marginal_label == 'noRT': ensemble_score = Tailor - matchingParaDict["alphaMS2P"] * MS2Pvalue1 - matchingParaDict["alphaInt"] * lnIntensityRank2  + matchingParaDict["alphaTCC"] * (IntensityIsoCorr1 + IntensityMS1MS2Corr);
                elif marginal_label == 'noInt': ensemble_score = Tailor - matchingParaDict["alphaMS2P"] * MS2Pvalue1 - matchingParaDict["alphaRT"] * Rtdiff1 + matchingParaDict["alphaTCC"] * (IntensityIsoCorr1 + IntensityMS1MS2Corr);
                elif marginal_label == 'noElution': ensemble_score = Tailor - matchingParaDict["alphaMS2P"] * MS2Pvalue1 - matchingParaDict["alphaInt"] * lnIntensityRank2  - matchingParaDict["alphaRT"] * Rtdiff1;
                elif marginal_label == 'noXcorr': ensemble_score = - matchingParaDict["alphaMS2P"] * MS2Pvalue1 - matchingParaDict["alphaInt"] * lnIntensityRank2  - matchingParaDict["alphaRT"] * Rtdiff1 + matchingParaDict["alphaTCC"] * (IntensityIsoCorr1 + IntensityMS1MS2Corr);
                elif marginal_label == 'onlyPvalue': ensemble_score = - matchingParaDict["alphaMS2P"] * MS2Pvalue1 ;
                elif marginal_label == 'onlyRT': ensemble_score = - matchingParaDict["alphaRT"] * Rtdiff1 ;
                elif marginal_label == 'onlyInt': ensemble_score = - matchingParaDict["alphaInt"] * lnIntensityRank2 ;
                elif marginal_label == 'onlyElution': ensemble_score = + matchingParaDict["alphaTCC"] * (IntensityIsoCorr1 + IntensityMS1MS2Corr);
                elif marginal_label == 'onlyXcorr': ensemble_score = Tailor ;
                else: assert False;

                if curr_ms2scan_charge != ms2scan_charge:
                    if len(record_list) > 0: deal_with_ms2scan_charge(curr_ms2scan_charge, record_list);
                    curr_ms2scan_charge = ms2scan_charge; record_list = [];
                record_list.append((Xcorr, ensemble_score, label, edgeid, row));
            if len(record_list) > 0: deal_with_ms2scan_charge(curr_ms2scan_charge, record_list);

            output_csvfile.close();  input_csvfile.close();


            if not os.path.isfile(os.path.join(tdc_percolator_output_dir, 'eval_new.txt')):
                logger.info('tdc_percolator_evidence_url={}'.format(tdc_percolator_evidence_url));

                command_line = "sh {}/{} {} {} {} {} {} {}".format(os.path.dirname(os.path.realpath(__file__)), 'run_percolator.sh', tdc_percolator_evidence_url, matchingParaDict["fastaFile"], tdc_percolator_output_dir, matchingParaDict["cruxExecFile"], train_fdr, test_fdr );
                print(command_line);
                print(subprocess.check_output(command_line, shell=True));

                plot_qvalue_via_pseudolabel_diasearch(tdc_percolator_output_dir, matchingParaDict);
                # os.remove(tdc_percolator_evidence_url);



def evidence_tdc_by_fix_weights(evidences_url, matchingParaDict, neighbor_labels = [], filter_type=1, minmax_norm = True, train_fdr=0.01, test_fdr=0.01 ):
    assert 'sorted' in evidences_url;

    type_metrics_map = get_type_metrics_mapping();
    metric_type_list = matchingParaDict["metric_type_list"]; accept_metric_list = []; all_metric_list = [];
    exclude_fieldnames = ['Edgeid', 'Pepid', 'MS1Scan', 'MS2Scan', 'Scanwin', 'Charge', 'NgbrEdgeids' ];

    for metric_type in type_metrics_map: all_metric_list = np.concatenate((all_metric_list, type_metrics_map[metric_type]));
    for metric_type in metric_type_list: accept_metric_list = np.concatenate((accept_metric_list, type_metrics_map[metric_type]));
    metric_str = ','.join([str(x) for x in metric_type_list]); rej_metric_list = np.setdiff1d(all_metric_list, accept_metric_list);
    logger.info('all_metric_list={}\taccept_metric_list={}\trej_metric_list={}'.format(len(all_metric_list), len(accept_metric_list), len(rej_metric_list) ));

    tdc_filter_str = 'aPreP{}_aMS2P{}_aInt{}_aRT{}_aCorr{}'.format(matchingParaDict["alphaPreP"], matchingParaDict["alphaMS2P"], matchingParaDict["alphaInt"], matchingParaDict["alphaRT"], matchingParaDict["alphaTCC"]);
    tdc_evidence_url = evidences_url.replace('.txt', '_filter{}_minmax{}_trainfdr{}_testfdr{}_{}.txt'.format(int(filter_type), int(minmax_norm), train_fdr, test_fdr, tdc_filter_str));
    tdc_percolator_evidence_url = tdc_evidence_url.replace('pepid_', 'percolator_met{}_'.format(metric_str)).replace('.txt', '.pin');
    tdc_percolator_output_dir = tdc_percolator_evidence_url.replace('percolator_', 'results_percolator_').replace('.pin', '');
    if os.path.isfile(os.path.join(tdc_percolator_output_dir, 'eval_new.txt')):
        # plot_qvalue_via_pseudolabel_diasearch(tdc_percolator_output_dir, matchingParaDict);
        return;


    if minmax_norm:
        minmax_norm_shift_map = get_minmax_norm_shift(evidences_url, matchingParaDict);
        MS2Pvalue1_lower, MS2Pvalue1_upper = minmax_norm_shift_map['MS2Pvalue1'];
        lnIntensityRank2_lower, lnIntensityRank2_upper = minmax_norm_shift_map['lnIntensityRank2'];
        # Rtdiff1_lower, Rtdiff1_upper = minmax_norm_shift_map['Rtdiff1'];
        # IntensityIsoCorr1_lower, IntensityIsoCorr1_upper = minmax_norm_shift_map['IntensityIsoCorr1'];
        # IntensityMS1MS2Corr_lower, IntensityMS1MS2Corr_upper = minmax_norm_shift_map['IntensityMS1MS2Corr'];


    if not os.path.isfile(tdc_percolator_evidence_url):
        input_csvfile = open(evidences_url);
        reader = csv.DictReader(input_csvfile, delimiter='\t');

        remain_fieldnames = [];
        for curr_fieldname in reader.fieldnames:
            if curr_fieldname not in exclude_fieldnames and curr_fieldname not in all_metric_list: remain_fieldnames.append(curr_fieldname);
        combined_header_list = np.concatenate((remain_fieldnames[:-2], accept_metric_list, remain_fieldnames[-2:]));

        output_csvfile = open(tdc_percolator_evidence_url, 'w');
        writer = csv.DictWriter(output_csvfile, fieldnames=combined_header_list, delimiter='\t');
        writer.writeheader();

        curr_ms2scan_charge = -1; record_list = [];
        def deal_with_ms2scan_charge(curr_ms2scan_charge, record_list):
            curr_ms2scan, curr_charge = [int(x) for x in curr_ms2scan_charge.split('_')];
            # new_ms2scan = curr_ms2scan*100+curr_charge;

            record_list.sort(key=lambda x: -x[0]);
            ref_xcorr = record_list[0][0] - 1e-8;
            ref_ensemble = record_list[0][1] - 1e-8;
            # ref_ensemble2 = 0.5* (record_list[0][1] + record_list[1][1]) ;

            if record_list[0][1] <= record_list[1][1]: ref_ensemble2 = record_list[0][1] - 1e-8;
            else: ref_ensemble2 = record_list[1][1] - 1e-8;

            if record_list[0][1] <= record_list[1][1]: ref_ensemble3 = record_list[0][1] - 1e-8;
            else: ref_ensemble3 = record_list[1][1] + 1e-8;


            record_list.sort(key=lambda x: -x[1]);  # rank by ensemble_score

            whitelist_edgeid_set = set();
            for xcorr, ensemble_score, label, edgeid, row in record_list:

                if filter_type == 1 and ref_ensemble <= ensemble_score: whitelist_edgeid_set.add(edgeid);
                elif filter_type == 2 and ref_ensemble2 <= ensemble_score: whitelist_edgeid_set.add(edgeid);
                elif filter_type == 3 and ref_ensemble3 <= ensemble_score: whitelist_edgeid_set.add(edgeid);
                elif filter_type == 4: whitelist_edgeid_set.add(edgeid);
                elif filter_type == 5:
                    ref_ensemble3 = ref_ensemble * xcorr / ref_xcorr;
                    if ensemble_score >= ref_ensemble3: whitelist_edgeid_set.add(edgeid);


            for xcorr, ensemble_score, label, edgeid, row in record_list:
                if edgeid not in whitelist_edgeid_set: continue;
                data = {key: value for key, value in row.items() if key in combined_header_list};
                # data['ScanNr'] = whitelist_edgeid_scan_map[edgeid]
                writer.writerow(data);

        for row in reader:
            label = int(row['Label']); edgeid = int(row['Edgeid']); Xcorr = float(row['Xcorr']);
            ms2scan = int(row['MS2Scan']); charge = int(row['Charge']); ms2scan_charge = '{}_{}'.format(ms2scan, charge);
            # lnEvalue = float(row['lnEvalue']);
            # lnEvalueAccpep = float(row['lnEvalueAccpep']);
            # lnPvalue = float(row['lnPvalue']); lnPvalueMean = float(row['lnPvalueMean']);

            Tailor = float(row['Tailor']);
            MS2Pvalue1 = float(row['MS2Pvalue1']);
            lnIntensityRank2 = float(row['lnIntensityRank2']);
            Rtdiff1 = float(row['Rtdiff1']);
            IntensityIsoCorr1 = float(row['IntensityIsoCorr1']); IntensityMS1MS2Corr = float(row['IntensityMS1MS2Corr']);

            if minmax_norm:
                # lnEvalue = (lnEvalue - lnEvalue_lower) / (lnEvalue_upper - lnEvalue_lower);
                # lnEvalueAccpep = (lnEvalueAccpep - lnEvalueAccpep_lower) / (lnEvalueAccpep_upper - lnEvalueAccpep_lower);
                # lnPvalue = (lnPvalue - lnPvalue_lower) / (lnPvalue_upper - lnPvalue_lower);
                # lnPvalueMean = (lnPvalueMean - lnPvalueMean_lower) / (lnPvalueMean_upper - lnPvalueMean_lower);

                MS2Pvalue1 = (MS2Pvalue1 - MS2Pvalue1_lower) / (MS2Pvalue1_upper - MS2Pvalue1_lower);
                lnIntensityRank2 = (lnIntensityRank2 - lnIntensityRank2_lower) / (lnIntensityRank2_upper - lnIntensityRank2_lower);
                # Rtdiff1 = (Rtdiff1 - Rtdiff1_lower) / (Rtdiff1_upper - Rtdiff1_lower);
                # IntensityIsoCorr1 = (IntensityIsoCorr1 - IntensityIsoCorr1_lower) / (IntensityIsoCorr1_upper - IntensityIsoCorr1_lower);
                # IntensityMS1MS2Corr = (IntensityMS1MS2Corr - IntensityMS1MS2Corr_lower) / (IntensityMS1MS2Corr_upper - IntensityMS1MS2Corr_lower);

            # -(lnEvalue+lnEvalueAccpep) - - matchingParaDict["alphaPreP"] * lnPvalueMean
            ensemble_score = + Tailor \
                             - matchingParaDict["alphaMS2P"] * MS2Pvalue1 \
                             - matchingParaDict["alphaInt"] * lnIntensityRank2 \
                             - matchingParaDict["alphaRT"] * Rtdiff1 \
                             + matchingParaDict["alphaTCC"] * (IntensityIsoCorr1 + IntensityMS1MS2Corr);

            if curr_ms2scan_charge != ms2scan_charge:
                if len(record_list) > 0: deal_with_ms2scan_charge(curr_ms2scan_charge, record_list);
                curr_ms2scan_charge = ms2scan_charge; record_list = [];
            record_list.append((Xcorr, ensemble_score, label, edgeid, row));
        if len(record_list) > 0: deal_with_ms2scan_charge(curr_ms2scan_charge, record_list);

        output_csvfile.close();  input_csvfile.close();


    if not os.path.isfile(os.path.join(tdc_percolator_output_dir, 'eval_new.txt')):
        logger.info('tdc_percolator_evidence_url={}'.format(tdc_percolator_evidence_url));

        if len(neighbor_labels) <= 1:
            command_line = "sh {}/{} {} {} {} {} {} {}".format(os.path.dirname(os.path.realpath(__file__)), 'run_percolator.sh', tdc_percolator_evidence_url, matchingParaDict["fastaFile"], tdc_percolator_output_dir, matchingParaDict["cruxExecFile"], train_fdr, test_fdr );
        elif len(neighbor_labels) == 2:
            assert neighbor_labels[0] in tdc_percolator_evidence_url;
            tdc_percolator_evidence_url2 = tdc_percolator_evidence_url.replace(neighbor_labels[0], neighbor_labels[1]); logger.info('tdc_percolator_evidence_url2={}'.format(tdc_percolator_evidence_url2)); assert os.path.isfile(tdc_percolator_evidence_url2);
            command_line = "sh {}/{} {} {} {} {} {} {} {}".format(os.path.dirname(os.path.realpath(__file__)), 'run_percolator2.sh', tdc_percolator_evidence_url, matchingParaDict["fastaFile"], tdc_percolator_output_dir, matchingParaDict["cruxExecFile"], train_fdr, test_fdr, tdc_percolator_evidence_url2 );
        elif len(neighbor_labels) == 4:
            assert neighbor_labels[0] in tdc_percolator_evidence_url;
            tdc_percolator_evidence_url2 = tdc_percolator_evidence_url.replace(neighbor_labels[0], neighbor_labels[1]); logger.info('tdc_percolator_evidence_url2={}'.format(tdc_percolator_evidence_url2)); assert os.path.isfile(tdc_percolator_evidence_url2);
            tdc_percolator_evidence_url3 = tdc_percolator_evidence_url.replace(neighbor_labels[0], neighbor_labels[2]); logger.info('tdc_percolator_evidence_url3={}'.format(tdc_percolator_evidence_url3)); assert os.path.isfile(tdc_percolator_evidence_url3);
            tdc_percolator_evidence_url4 = tdc_percolator_evidence_url.replace(neighbor_labels[0], neighbor_labels[3]); logger.info('tdc_percolator_evidence_url4={}'.format(tdc_percolator_evidence_url4)); assert os.path.isfile(tdc_percolator_evidence_url4);
            command_line = "sh {}/{} {} {} {} {} {} {} {} {} {}".format(os.path.dirname(os.path.realpath(__file__)), 'run_percolator4.sh', tdc_percolator_evidence_url, matchingParaDict["fastaFile"], tdc_percolator_output_dir, matchingParaDict["cruxExecFile"], train_fdr, test_fdr, tdc_percolator_evidence_url2, tdc_percolator_evidence_url3, tdc_percolator_evidence_url4 );


        # command_line = "sh {}/{} {} {} {} {} {} {}".format(os.path.dirname(os.path.realpath(__file__)), 'run_percolator.sh', tdc_percolator_evidence_url, matchingParaDict["fastaFile"], tdc_percolator_output_dir, matchingParaDict["cruxExecFile"], train_fdr, test_fdr );
        print(command_line);
        print(subprocess.check_output(command_line, shell=True));

        plot_qvalue_via_pseudolabel_diasearch(tdc_percolator_output_dir, matchingParaDict);
        os.remove(os.path.join(tdc_percolator_output_dir, 'percolator.target.peptides.txt'));
        os.remove(os.path.join(tdc_percolator_output_dir, 'percolator.decoy.peptides.txt'));
        os.remove(os.path.join(tdc_percolator_output_dir, 'percolator.decoy.psms.txt'));
        # os.remove(os.path.join(tdc_percolator_output_dir, 'percolator.target.psms.txt'));
        # os.remove(tdc_percolator_evidence_url);



def postprecess_walnut_prosit_features(matchingParaDict, tag, train_fdr=0.01, test_fdr=0.01):
    inputFilename = matchingParaDict["inputFile"]; fastaFilename = matchingParaDict["fastaFile"];

    file_name, file_extension = os.path.splitext(os.path.basename(inputFilename));
    feature_url = os.path.join(os.path.dirname(inputFilename), tag, '{}.mzML.features.txt'.format(file_name)); assert os.path.isfile(feature_url);
    new_feature_url = feature_url.replace('features.txt', 'features_new.pin');
    if not os.path.isfile(new_feature_url):

        peptide_pepid_map = read_xcorr_results.get_peptide_pepid_map(fastaFilename);
        pepid_label_map = read_xcorr_results.get_pepid_labels(fastaFilename);

        input_csvfile = open(feature_url);
        reader = csv.DictReader(input_csvfile, delimiter='\t');

        old2new_header_map = {'id':'SpecId', 'TD':'Label', 'charge2':'Charge2', 'charge3':'Charge3', 'sequence':'Peptide', 'protein':'Proteins'};
        old_headers = reader.fieldnames; new_headers = [];
        for curr_fieldname in old_headers:
            if curr_fieldname in old2new_header_map: new_headers.append(old2new_header_map[curr_fieldname]);
            else: new_headers.append(curr_fieldname);
        logger.info('new_headers={}'.format(new_headers));

        output_csvfile = open(new_feature_url, 'w');
        writer = csv.DictWriter(output_csvfile, fieldnames=new_headers, delimiter='\t');
        writer.writeheader();

        for row in reader:
            if row['TD'] is None: continue;
            if row['sequence'] is None: continue;
            start_idx = row['sequence'].find('.'); end_idx = row['sequence'].rfind('.');
            peptide = row['sequence'][start_idx+1:end_idx];

            peptide = peptide.replace('*', '');
            peptide = peptide.replace('M(ox)', 'M[15.99]').replace('S(ph)', 'S[79.97]').replace('T(ph)', 'T[79.97]').replace('Y(ph)', 'Y[79.97]');
            peptide = peptide.replace('C[+57.0214635]', 'C');

            peptide = peptide.replace(')xo(', '').replace('xo(', '').replace(')', '');
            if '(' in peptide or ')' in peptide or '*' in peptide: logger.info(peptide);
            row['sequence'] = '-.{}.-'.format(peptide);

            if peptide in peptide_pepid_map:
                pepid = peptide_pepid_map[peptide]; real_label = pepid_label_map[pepid];
                if real_label >= 0: row['TD'] = str(1);
                else: row['TD'] = str(-1);
            else: row['TD'] = str(-1);

            data = {};
            for key, value in row.items():
                if key in old2new_header_map: data[old2new_header_map[key]] = value;
                else: data[key] = value;
            writer.writerow(data);

        input_csvfile.close(); output_csvfile.close();

    percolator_dir = os.path.join(os.path.dirname(inputFilename), tag, 'results_percolator');

    command_line = "sh {}/{} {} {} {} {} {} {}".format(os.path.dirname(os.path.realpath(__file__)), 'run_percolator.sh',
                                                       new_feature_url, fastaFilename,
                                                       percolator_dir, matchingParaDict["cruxExecFile"],
                                                       train_fdr, test_fdr);
    print(subprocess.check_output(command_line, shell=True));

    return new_feature_url;


def get_targets_via_pseudolabel(percolator_output_dir, matchingParaDict):


    pepid_label_map = read_xcorr_results.get_pepid_labels(matchingParaDict["fastaFile"]);
    peptide_pepid_map = read_xcorr_results.get_peptide_pepid_map(matchingParaDict["fastaFile"]);

    eval_url = os.path.join(percolator_output_dir, 'eval_new.txt');
    # if os.path.isfile(eval_url): return;
    output = open(eval_url, 'w');
    output.write('percolator_output_dir={}\n'.format(percolator_output_dir));

    qvalue_url = os.path.join(percolator_output_dir, 'qvalue_new.txt');
    output1 = open(qvalue_url, 'w');

    pepid_label_qvalue_list = [];

    target_psm_url = os.path.join(percolator_output_dir, 'percolator.target.psms.txt');
    if os.path.isfile(target_psm_url):
        with open(target_psm_url, 'rU') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                # logger.info(row);
                peptide = row['sequence'];
                if peptide not in peptide_pepid_map: logger.info('Missing peptide as key: {}'.format(peptide)); continue;
                pepid = peptide_pepid_map[peptide]; real_label = pepid_label_map[pepid]; qvalue = float(row['percolator q-value']);

                if real_label > 0: pepid_label_qvalue_list.append((pepid, 1, qvalue));
                elif real_label == 0: pepid_label_qvalue_list.append((pepid, -1, qvalue));

    decoy_psm_url = os.path.join(percolator_output_dir, 'percolator.decoy.psms.txt');
    if os.path.isfile(decoy_psm_url):
        with open(decoy_psm_url, 'rU') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                # logger.info(row);
                peptide = row['sequence'];
                if row['percolator q-value'] is None: continue;
                qvalue = float(row['percolator q-value']);

                if peptide not in peptide_pepid_map: pepid_label_qvalue_list.append((peptide, -1, qvalue));
                else:
                    pepid = peptide_pepid_map[peptide]; real_label = pepid_label_map[pepid];
                    if real_label > 0: pepid_label_qvalue_list.append((pepid, 1, qvalue));
                    elif real_label == 0: pepid_label_qvalue_list.append((pepid, -1, qvalue));

    pepid_label_qvalue_list.sort(key=lambda x: x[2]);

    target_cnt = 0; decoy_cnt = 0; fdr_list = []; decoy_target_map = {}; pepid_set = set();
    for pepid, label, qvalue in pepid_label_qvalue_list:
        if pepid in pepid_set: continue;
        pepid_set.add(pepid);

        if label > 0:
            target_cnt += 1;
            fdr = decoy_cnt * 1.0 / target_cnt;
            fdr_list.append((fdr, target_cnt, pepid));
        else: decoy_cnt += 1;
        decoy_target_map[decoy_cnt] = target_cnt;
    # logger.info('pepid_set={}'.format(len(pepid_set)));

    for idx in range(len(fdr_list) - 1, 0, -1):
        fdr, target_cnt, pepid = fdr_list[idx - 1];
        fdr_list[idx - 1] = (min(fdr, fdr_list[idx][0]), target_cnt, pepid);


    qvalue_list = np.asarray([x[0] for x in fdr_list]);
    for qvalue_level in [0.01, 0.05, 0.1, 0.2]: output.write('level=peptide\tpepid={}\tfdr={}\ttarget_cnt={}\n'.format(pepid, qvalue_level, len(np.where(qvalue_list <= qvalue_level)[0]) ));
    for fdr, target_cnt, pepid in fdr_list: output1.write('{}\t{}\t{}\n'.format(fdr, target_cnt, pepid));

    output1.close();
    output.close();


def plot_qvalue_via_pseudolabel_diasearch(tdc_evidence_url, matchingParaDict):
    metric_str = ','.join([str(x) for x in matchingParaDict["metric_type_list"]]);

    diasearch_percolator_dir = tdc_evidence_url.replace('pepid_', 'results_percolator_met{}_'.format(metric_str)).replace('.txt', '');
    logger.info(diasearch_percolator_dir); assert os.path.exists(diasearch_percolator_dir);
    # plot_qvalue_via_pseudolabel(diasearch_percolator_dir, matchingParaDict);
    get_targets_via_pseudolabel(diasearch_percolator_dir, matchingParaDict)


def plot_qvalue_via_pseudolabel_encyclopedia(matchingParaDict, tag):

    for file in os.listdir(os.path.join(os.path.dirname(matchingParaDict["inputFile"]), tag)):
        if 'results_percolator' in file:
            encyclopedia_percolator_dir = os.path.join(os.path.dirname(matchingParaDict["inputFile"]), tag, file);
            # plot_qvalue_via_pseudolabel(encyclopedia_percolator_dir, matchingParaDict);
            get_targets_via_pseudolabel(encyclopedia_percolator_dir, matchingParaDict);


def plot_qvalue_via_pseudolabel_diaumpire(matchingParaDict):
    logger.info('input_dir={}'.format(os.path.dirname(matchingParaDict["inputFile"])));

    for file in os.listdir(os.path.join(os.path.dirname(matchingParaDict["inputFile"]), 'results')):
        if 'umpire_percolator_' in file:
            umpire_percolator_dir = os.path.join(os.path.dirname(matchingParaDict["inputFile"]), 'results', file);
            logger.info('umpire_percolator_dir={}'.format(umpire_percolator_dir));
            # plot_qvalue_via_pseudolabel(umpire_percolator_dir, matchingParaDict);
            get_targets_via_pseudolabel(umpire_percolator_dir, matchingParaDict);


def plot_qvalue_via_pseudolabel_topDiameter(evidences_url, matchingParaDict):
    for file in os.listdir(os.path.dirname(evidences_url)):
        if 'results_percolator_met' in file:
            diameter_dir = os.path.join(os.path.dirname(evidences_url), file);
            logger.info('diameter_dir={}'.format(diameter_dir));
            get_targets_via_pseudolabel(diameter_dir, matchingParaDict);


def plot_qvalue_via_pseudolabel(percolator_output_dir, matchingParaDict):

    pepid_set = read_xcorr_results.get_whitelist_pepids(matchingParaDict["xcorrFile"]);
    peptide_pepid_map = read_xcorr_results.get_peptide_pepid_map(matchingParaDict["fastaFile"], whitelist_pepids=pepid_set);
    pepid_label_map = read_xcorr_results.get_pepid_labels(matchingParaDict["fastaFile"], whitelist_pepids=pepid_set);

    figure = plt.figure(figsize=(20, 10));
    for level_idx, level in enumerate(['psms', 'peptides']):
        target_psm_url = os.path.join(percolator_output_dir, 'percolator.target.{}.txt'.format(level)); assert target_psm_url;
        decoy_psm_url = os.path.join(percolator_output_dir, 'percolator.decoy.{}.txt'.format(level)); assert decoy_psm_url;

        pepid_label_qvalue_list = [];
        with open(target_psm_url) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                peptide = row['sequence']; pepid = peptide_pepid_map[peptide]; real_label = pepid_label_map[pepid]; qvalue = float(row['percolator q-value']);
                if real_label > 0: pepid_label_qvalue_list.append((pepid, 1, qvalue));
                elif real_label == 0: pepid_label_qvalue_list.append((pepid, -1, qvalue));

        with open(decoy_psm_url) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                peptide = row['sequence']; pepid = peptide_pepid_map[peptide]; real_label = pepid_label_map[pepid]; qvalue = float(row['percolator q-value']);
                if real_label > 0: pepid_label_qvalue_list.append((pepid, 1, qvalue));
                elif real_label == 0: pepid_label_qvalue_list.append((pepid, -1, qvalue));

        pepid_label_qvalue_list.sort(key=lambda x: x[2]);

        target_cnt = 0; decoy_cnt = 0; fdr_list = []; qvalue_list = [];
        for pepid, label, qvalue in pepid_label_qvalue_list:
            if label > 0:
                target_cnt += 1;
                qvalue_list.append(qvalue);
                fdr_list.append(decoy_cnt * 1.0 / target_cnt);
            else: decoy_cnt += 1;

        for idx in range(len(fdr_list) - 1, 0, -1): fdr_list[idx - 1] = min(fdr_list[idx - 1], fdr_list[idx]);
        qvalue_list1 = np.asarray(qvalue_list); qvalue_list2 = np.asarray(fdr_list * 3);

        ax1 = figure.add_subplot(1, 2, level_idx + 1);
        ax1.autoscale();

        max_idx = max(np.max(np.where(qvalue_list1 <= 0.01)[0]), np.max(np.where(qvalue_list2 <= 0.01)[0])) + 1;
        ax1.plot(qvalue_list1[:max_idx], qvalue_list2[:max_idx], 's', marker='.', alpha=0.3);
        ax1.plot([0, 0.01], [0, 0.01], linestyle='--', color='black');

        ax1.set_xlabel('qvalue by percolator', fontsize=16);
        ax1.set_ylabel('qvalue by camouflaged decoys', fontsize=16);
        ax1.set_title('Level={}'.format(level), fontsize=20);

    plt.savefig(os.path.join(percolator_output_dir, 'percolator.png'),  dpi=100);
    plt.close(figure);



def plot_debiased_evidence(calibrated_values_url, matchingParaDict):
    values_list = [];
    for evidence_row in read_debiased_evidence(calibrated_values_url, matchingParaDict):
        real_label = evidence_row['real_label'];

        ms1scan = evidence_row['ms1scan'];
        scanwin = evidence_row['scanwin'];
        charge = evidence_row['charge'];
        peplen = evidence_row['peplen'];

        Xcorr = evidence_row['Xcorr'];
        lnPvalue = evidence_row['lnPvalue'];
        lnEvalue = evidence_row['lnEvalue'];
        lnEvalueAccpep = evidence_row['lnEvalueAccpep'];
        lnEvalueCoop = evidence_row['lnEvalueCoop'];

        Rtdiff1 = evidence_row['Rtdiff1'];
        lnIntensityRank1 = evidence_row['lnIntensityRank1'];
        lnIntensityRank2 = evidence_row['lnIntensityRank2'];

        IntensityIsoCorr1 = evidence_row['IntensityIsoCorr1'];
        IntensityIsoCorr2 = evidence_row['IntensityIsoCorr2'];
        IntensityMS1MS2Corr = evidence_row['IntensityMS1MS2Corr'];
        IntensityMS2Corr = evidence_row['IntensityMS2Corr'];

        MS2Pvalue1 = evidence_row['MS2Pvalue1'];
        MS2Pvalue2 = evidence_row['MS2Pvalue2'];

        values_list.append((real_label, ms1scan, scanwin, charge, peplen, Xcorr, lnPvalue, lnEvalue, lnEvalueAccpep, lnEvalueCoop,
                            Rtdiff1,lnIntensityRank1, lnIntensityRank2, IntensityIsoCorr1, IntensityIsoCorr2, IntensityMS1MS2Corr,
                            IntensityMS2Corr, MS2Pvalue1, MS2Pvalue2));

    # plotting scatter plot
    key_indices = [1]; key_names = ['ms1scan'];
    metric_name_indices = [(5, 'Xcorr'), (6, 'lnPvalue'), (7, 'lnEvalue'), (8, 'lnEvalueAccpep'), (9, 'lnEvalueCoop'),
                           (10, 'Rtdiff1'), (11, 'lnIntensityRank1'), (12, 'lnIntensityRank2'),
                            (13, 'IntensityIsoCorr1'), (14, 'IntensityIsoCorr2'), (15, 'IntensityMS1MS2Corr'),
                            (16, 'IntensityMS2Corr'), (17, 'MS2Pvalue1'), (18, 'MS2Pvalue2')];
    for metric_idx, metric_name in metric_name_indices:
        for key_idx, key_name in zip(key_indices, key_names):

            key_value_map = {};
            for values_record in values_list:
                is_target = values_record[0];
                key = values_record[key_idx];
                value = values_record[metric_idx];

                if key not in key_value_map: key_value_map[key] = [];
                key_value_map[key].append((value, is_target));

            key_list = np.sort(np.asarray(list(key_value_map.keys()), dtype=int));
            target_value_list = []; decoy_value_list = [];
            for key in key_list:
                target_values = [x[0] for x in key_value_map[key] if x[1] > 0];
                decoy_values = [x[0] for x in key_value_map[key] if x[1] <= 0];

                if len(target_values) > 0: target_value_list.append(np.mean(target_values));
                else: target_value_list.append(0);

                if len(decoy_values) > 0: decoy_value_list.append(np.mean(decoy_values));
                else: decoy_value_list.append(0);

            figure = plt.figure(figsize=(20, 10));
            ax = figure.add_subplot(1, 1, 1);
            ax.autoscale();
            ax.plot(key_list, target_value_list, 's', color='green', marker='o', alpha=0.5, label='target');
            ax.plot(key_list, decoy_value_list, 's', color='red', marker='o', alpha=0.5, label='decoy');
            ax.legend(prop={'size': 25});
            ax.tick_params(axis='both', which='major', labelsize=20);

            plt.savefig(calibrated_values_url.replace('.txt', '.png').replace('.pin', '.png').replace('pepid_', 'means_{}_{}_'.format(metric_name, key_name)), dpi=100);
            plt.close(figure);


def get_minmax_norm_shift(calibrated_values_url, matchingParaDict, lower_quantile=0.01, upper_quantile=0.99):
    minmax_norm_shift_url = os.path.join(os.path.dirname(calibrated_values_url), 'minmax_norm_shift_lowq{}_highq{}.npy'.format(lower_quantile, upper_quantile));
    if os.path.isfile(minmax_norm_shift_url):  minmax_norm_shift_map = np.load(minmax_norm_shift_url, allow_pickle=True).item();
    else:
        metric_name_values_map = {};
        metric_name_list = ['lnEvalue', 'lnEvalueAccpep', 'lnPvalue', 'lnPvalueMean', 'MS2Pvalue1','MS2Pvalue2', 'lnIntensityRank1', 'lnIntensityRank2'];
        for metric_name in metric_name_list: metric_name_values_map[metric_name] = [];

        for evidence_row in read_debiased_evidence(calibrated_values_url, matchingParaDict):
            for metric_name in metric_name_list:
                metric_value = evidence_row[metric_name];
                metric_name_values_map[metric_name].append(metric_value);

        minmax_norm_shift_map = {};
        for metric_name in metric_name_list:
            origin_values = np.asarray(metric_name_values_map[metric_name], dtype=float);

            if 'Evalue' in metric_name:
                # value_lower = np.quantile(origin_values[origin_values <= 0], lower_quantile);
                # value_upper = np.quantile(origin_values[origin_values <= 0], upper_quantile);
                value_lower = np.quantile(origin_values[origin_values <= 0], lower_quantile);
                value_upper = 0;
            else:
                value_lower = np.quantile(origin_values, lower_quantile);
                value_upper = np.quantile(origin_values, upper_quantile);
            minmax_norm_shift_map[metric_name] = (value_lower, value_upper);
            logger.info('metric={}\tvalue_lower={}\tvalue_upper={}'.format(metric_name, value_lower, value_upper));

            # normed_values = (origin_values - value_lower) / (value_upper - value_lower);
            # figure = plt.figure(figsize=(30, 15));
            # ax = figure.add_subplot(1, 1, 1);
            # ax.autoscale();
            # ax.hist(normed_values, bins=200, log=True, alpha=0.8);
            # ax.set_xlabel(metric_name, fontsize=15);
            # ax.set_ylabel('# of matches', fontsize=15);
            # plt.savefig(os.path.join(os.path.dirname(calibrated_values_url), 'hist_metric_{}_normed.png'.format(metric_name)), dpi=100);
            # plt.close(figure);

        for metric_name in [ 'Rtdiff1','IntensityIsoCorr1', 'IntensityIsoCorr2', 'IntensityMS1MS2Corr']:
            minmax_norm_shift_map[metric_name] = (0, 1);

        np.save(minmax_norm_shift_url, minmax_norm_shift_map);
    logger.info('minmax_norm_shift_map={}'.format(minmax_norm_shift_map));
    return minmax_norm_shift_map;


def get_evidence_scan_rt_map(input_url, evidences_url):
    saved_evidence_scan_rt_url = os.path.join(os.path.dirname(evidences_url), 'pepid_evidence_scan_rt.npy');
    if os.path.isfile(saved_evidence_scan_rt_url): evidence_scan_rt_map = np.load(saved_evidence_scan_rt_url, allow_pickle=True).item();
    else:
        evidence_scan_rt_map = {};
        scannum_arr, rt_arr = mzxml_utils.get_scannum_rt_pair(input_url);
        scannum_rt_map = dict(zip(scannum_arr, rt_arr));

        with open(evidences_url) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                scan = int(row['ScanNr']);
                ms2scan = int(row['MS2Scan']);
                rt = scannum_rt_map[ms2scan];
                evidence_scan_rt_map[scan] = rt; logger.info('scan={}\trt={}'.format(scan, rt));

        np.save(saved_evidence_scan_rt_url, evidence_scan_rt_map);
    logger.info('evidence_scan_rt_map={}'.format(len(evidence_scan_rt_map)));
    return evidence_scan_rt_map;


def read_debiased_evidence(calibrated_evidences_url, matchingParaDict):
    pepid_set = read_xcorr_results.get_whitelist_pepids(matchingParaDict["xcorrFile"]);
    pepid_label_map = read_xcorr_results.get_pepid_labels(matchingParaDict["fastaFile"], whitelist_pepids=pepid_set);
    with open(calibrated_evidences_url) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t');
        for row in reader:
            label = int(row['Label']);
            edgeid = int(row['Edgeid']);
            pepid = int(row['Pepid']);

            real_label = pepid_label_map[pepid];
            if real_label > 0: real_label = 1;
            else: real_label = -1;


            ms1scan = int(row['MS1Scan']);
            ms2scan = int(row['MS2Scan']);
            scanwin = int(row['Scanwin']);
            peplen = int(row['Peplen']);
            charge = int(row['Charge']);

            Xcorr = float(row['Xcorr']);
            Tailor = float(row['Tailor']);

            lnPvalue = float(row['lnPvalue']);
            lnPvalueMean = float(row['lnPvalueMean']);

            lnEvalue = float(row['lnEvalue']);
            lnEvalueAccpep = float(row['lnEvalueAccpep']);
            lnEvalueCoop = float(row['lnEvalueCoop']);

            Rtdiff1 = float(row['Rtdiff1']);
            Intensity = float(row['Intensity']);
            IntensityM1 = float(row['IntensityM1']);
            IntensityM2 = float(row['IntensityM2']);
            lnIntensityRank1 = float(row['lnIntensityRank1']);
            lnIntensityRank2 = float(row['lnIntensityRank2']);
            IntensityIsoCorr1 = float(row['IntensityIsoCorr1']);
            IntensityIsoCorr2 = float(row['IntensityIsoCorr2']);
            IntensityMS1MS2Corr = float(row['IntensityMS1MS2Corr']);
            IntensityMS2Corr = float(row['IntensityMS2Corr']);
            MS2Pvalue1 = float(row['MS2Pvalue1']);
            MS2Pvalue2 = float(row['MS2Pvalue2']);
            CC = float(row['CC']);
            TC = float(row['TC']);

            ngbrEdgeids = np.asarray([int(x) for x in row['NgbrEdgeids'].split(',')], dtype=int);

            yield {'edgeid': edgeid, 'pepid': pepid, 'label': label, 'real_label': real_label,
                   'ms1scan': ms1scan, 'ms2scan': ms2scan, 'scanwin': scanwin, 'peplen': peplen, 'charge': charge,
                   'Xcorr': Xcorr, 'Tailor':Tailor, 'lnPvalue': lnPvalue, 'lnPvalueMean': lnPvalueMean, 'lnEvalue': lnEvalue, 'Rtdiff1': Rtdiff1,
                   'lnEvalueAccpep': lnEvalueAccpep, 'lnEvalueCoop': lnEvalueCoop,
                   'Intensity': Intensity, 'IntensityM1': IntensityM1, 'IntensityM2': IntensityM2,
                   'lnIntensityRank1': lnIntensityRank1, 'lnIntensityRank2': lnIntensityRank2,
                   'IntensityIsoCorr1': IntensityIsoCorr1, 'IntensityIsoCorr2': IntensityIsoCorr2,
                   'IntensityMS1MS2Corr': IntensityMS1MS2Corr, 'IntensityMS2Corr': IntensityMS2Corr,
                   'MS2Pvalue1': MS2Pvalue1, 'MS2Pvalue2': MS2Pvalue2,
                   'CC': CC, 'TC':TC, 'ngbrEdgeids': ngbrEdgeids,
                   };


def get_metric_debiased_values(debiased_evidence_row, metric_type, matchingParaDict):
    # ms1/ms2 pvalue/evalue related
    if metric_type == 0: metric_val = -debiased_evidence_row['evalue_log']; metric_pow = 1.0;
    elif metric_type == 1: metric_val = -debiased_evidence_row['pvalue_log']; metric_pow = matchingParaDict["alphaPreP"];
    elif metric_type == 2: metric_val = -debiased_evidence_row['ms2_pvalue']; metric_pow = matchingParaDict["alphaMS2P"];

    # rtdiff related
    elif metric_type == 3: metric_val = -debiased_evidence_row['rtdiff']; metric_pow = matchingParaDict["alphaRT"];

    # intensity related
    elif metric_type == 5: metric_val = -np.sum(debiased_evidence_row['intensity_pvalues']); metric_pow = matchingParaDict["alphaInt"];
    elif metric_type == 6: metric_val = np.sum(debiased_evidence_row['intensities']); metric_pow = matchingParaDict["alphaInt"];
    elif metric_type == 7: metric_val = -np.sum(debiased_evidence_row['intensity_exp_pvalues']); metric_pow = matchingParaDict["alphaInt"];

    # multi related
    elif metric_type == 10: metric_val = np.log(len(debiased_evidence_row['multi_charge'])*min(10, len(debiased_evidence_row['multi_scan']))); metric_pow = matchingParaDict["alphaTCC"];
    elif metric_type == 11: metric_val = np.log(len(debiased_evidence_row['multi_charge'])*min(1 + 2*matchingParaDict["rtdiff_thres"], len(debiased_evidence_row['multi_scan']))); metric_pow = matchingParaDict["alphaTCC"];
    elif metric_type == 12: metric_val = np.log(len(debiased_evidence_row['neighbor_edgeids'])); metric_pow = matchingParaDict["alphaTCC"];

    else: assert False;
    return metric_val, metric_pow;


def get_debiased_edgeid_info(calibrated_evidences_url, matchingParaDict):
    edgeid_info_map = {};
    for debiased_evidence_row in read_debiased_evidence(calibrated_evidences_url, matchingParaDict):
        component_scores = [];
        for metric_type in matchingParaDict["metric_type_list"]:
            metric_val, metric_pow = get_metric_debiased_values(debiased_evidence_row, metric_type, matchingParaDict);
            component_scores.append(metric_pow * metric_val);

        edgeid = debiased_evidence_row['edgeid']; assert edgeid not in edgeid_info_map;
        edgeid_info_map[edgeid] = (np.sum(component_scores), debiased_evidence_row['is_target'], debiased_evidence_row['charge'], debiased_evidence_row['ms1_scannum'], debiased_evidence_row['scanwin'], debiased_evidence_row['peplen'], debiased_evidence_row['pepid'], debiased_evidence_row['neighbor_edgeids'], component_scores );

    return edgeid_info_map;


def matching_modular(calibrated_evidences_url, matchingParaDict):
    output_results = os.path.join(os.path.dirname(calibrated_evidences_url), 'results');
    if not os.path.exists(output_results): os.makedirs(output_results);
    outputFile = os.path.join(output_results, 'rst_mod_{}.txt'.format(getMatchingParamStr(matchingParaDict, modular=True) ));

    eval_url = outputFile.replace('.txt', '_eval.txt');
    if os.path.isfile(outputFile) or os.path.isfile(eval_url): return outputFile;
    edgeid_info_map = get_debiased_edgeid_info(calibrated_evidences_url, matchingParaDict); logger.info('edgeid_info_map={}'.format(len(edgeid_info_map)));

    final_edge_scores = [];
    for edgeid in edgeid_info_map:
        score, is_target, charge, ms1_scannum, scanwin, peplen, pepid, neighbor_edgeids, component_scores = edgeid_info_map[edgeid];
        final_edge_scores.append((is_target, pepid, charge, score, component_scores ));

    final_edge_scores.sort(key=lambda x: -x[3]);

#     write_results(outputFile, final_edge_scores, matchingParaDict, write_eval=True, remove_output=True);
#
#
# def write_results(outputFile, results, matchingParaDict, write_eval=True, remove_output=False, overwrite=False):
#     eval_url = outputFile.replace('.txt', '_eval.txt');
#     if os.path.isfile(eval_url) and not overwrite: return eval_url;
#
#     if not os.path.isfile(outputFile) or overwrite:
#         selected_cpepid_set = set();
#         metric_type_list = matchingParaDict["metric_type_list"];
#
#         f = open(outputFile, 'w');
#         f.write("td\tpeptide\tcharge\tscore\t{}\n".format('\t'.join(['metric_{}'.format(x) for x in metric_type_list]) ));
#         for s in results:
#             pepid = int(s[1]); charge = int(s[2]); cpepid = '{}_{}'.format(pepid, charge);
#             if cpepid in selected_cpepid_set: continue;
#
#             label = 't';
#             if s[0] <= 0: label = 'd';
#             f.write("{}\t{}\t{}\t{}\t{}\n".format(label, s[1], s[2], s[3], '\t'.join([str(x) for x in s[4]]) ));
#         f.close();
#
#     if write_eval:
#         target_precursor = eval_topmatch.comp_target_precursor_by_fdr(outputFile);
#         target_peptide = eval_topmatch.comp_target_peptide_by_fdr(outputFile);
#
#         f = open(eval_url, 'w');
#         f.write("target_precursor:\t{}\n".format(target_precursor));
#         f.write("target_peptide:\t{}\n".format(target_peptide));
#         f.close();
#         if remove_output: os.remove(outputFile);
#     return eval_url;
#

