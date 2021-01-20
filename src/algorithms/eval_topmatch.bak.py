import os, sys, random, csv, math, re
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np;
import networkx as nx;
import matplotlib
import matplotlib.pyplot as plt

import plotly;
import plotly.express as px;
import plotly.graph_objects as go;
from plotly.subplots import make_subplots;
import seaborn as sns


from scipy import interpolate
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from sklearn import linear_model, preprocessing
import itertools
import pandas as pd;

from pyteomics import mgf, mzxml;

import utils.viz_utils as viz_utils;
import utils.mzxml_utils as mzxml_utils;
import utils.peptide_utils as peptide_utils;
import utils.spectrum_utils as spectrum_utils;
import utils.diaumpire_utils as diaumpire_utils;
import utils.debiase_utils as debiase_utils;
import crux.read_xcorr_results as read_xcorr_results;
import run_matching;


def comp_target_by_decoy_thres(result_url, decoy_thres_list=[100,200,300,400,500,600,700,800,900,1000]):
    peptide_set = set(); td_map_allprecursor = {}; td_map_precursor = {}; td_map_peptide = {};
    t_precursor, t_peptide, d_precursor, all_precursor = 0, 0, 0, 0; max_decoy = max(decoy_thres_list)+10;

    with open(result_url) as input:
        for row in csv.DictReader(input, delimiter='\t'):
            peptide = row['peptide']; is_new_peptide = False;
            if peptide not in peptide_set: is_new_peptide = True; peptide_set.add(peptide);

            all_precursor += 1;
            if 't' == row['td']:
                t_precursor += 1;
                if is_new_peptide: t_peptide += 1;
            elif 'd' == row['td']:
                d_precursor += 1;
                if d_precursor > max_decoy: break;

            td_map_precursor[d_precursor] = t_precursor;
            td_map_peptide[d_precursor] = t_peptide;
            td_map_allprecursor[d_precursor] = all_precursor;

    target_arr_precursor = [td_map_precursor[decoy] for decoy in np.sort(list(td_map_precursor.keys()))];
    target_arr_peptide = [td_map_peptide[decoy] for decoy in np.sort(list(td_map_peptide.keys()))];
    target_arr_allprecursor = [td_map_allprecursor[decoy] for decoy in np.sort(list(td_map_allprecursor.keys()))];


    autdc_arr_precursor = [np.sum(target_arr_precursor[:(decoy+1)]) for decoy in np.sort(list(td_map_precursor.keys()))];
    autdc_arr_peptide = [np.sum(target_arr_peptide[:(decoy + 1)]) for decoy in np.sort(list(td_map_peptide.keys()))];

    return target_arr_precursor, target_arr_peptide, target_arr_allprecursor, autdc_arr_precursor, autdc_arr_peptide;


def comp_target_precursor_by_fdr(result_url, fdr=0.01):
    cpep_set = set();

    target = 0.0; decoy = 0.0;
    with open(result_url) as input:
        for row in csv.DictReader(input, delimiter='\t'):
            pep = row['peptide']; charge = row['charge']; cpep = '{}_{}'.format(pep, charge);
            if cpep in cpep_set: continue;

            cpep_set.add(cpep);
            # if pep in target_decoy_pepid_map: cpep_set.add('{}_{}'.format(target_decoy_pepid_map[pep], charge));

            if 't' == row['td']: target += 1;
            elif 'd' == row['td']: decoy += 1;
            if decoy / (target+1) >= fdr: break;
    return target;

def comp_target_peptide_by_fdr(result_url, fdr=0.01):
    pep_set = set();

    target = 0.0; decoy = 0.0;
    with open(result_url) as input:
        for row in csv.DictReader(input, delimiter='\t'):
            pep = row['peptide'];
            if pep in pep_set: continue;

            pep_set.add(pep);
            # if pep in target_decoy_pepid_map: pep_set.add(target_decoy_pepid_map[pep]);

            if 't' == row['td']: target += 1;
            elif 'd' == row['td']: decoy += 1;
            if decoy / (target+1) >= fdr: break;
    return target;



def plot_background_lognoise(matchingParaDict):
    ms1_lognoise_map = mzxml_utils.get_ms1_lognoise(matchingParaDict["inputFile"]);
    scanwin_lognoise_map = mzxml_utils.get_scanwin_lognoise(matchingParaDict["inputFile"]);
    ms1_scanwin_lognoise_map = mzxml_utils.get_ms1_scanwin_lognoise(matchingParaDict["inputFile"]);


    sep_lognoise_list = []; joint_lognoise_list = [];
    for ms1_scanwin in ms1_scanwin_lognoise_map:
        ms1scannum = int(ms1_scanwin.split('_')[0]); ms1_lognoise = ms1_lognoise_map[ms1scannum];
        scanwin = int(ms1_scanwin.split('_')[1]); scanwin_lognoise = scanwin_lognoise_map[scanwin];

        sep_lognoise_list.append(ms1_lognoise+scanwin_lognoise);
        joint_lognoise_list.append(ms1_scanwin_lognoise_map[ms1_scanwin]);

    figure = plt.figure();
    ax1 = figure.add_subplot(1, 1, 1);
    ax1.autoscale();
    ax1.plot(sep_lognoise_list, joint_lognoise_list, 's', marker='.');
    ax1.set_xlabel('sep_lognoise', fontsize=20);
    ax1.set_ylabel('joint_lognoise', fontsize=20);
    plt.show();


def comp_umpire_tdc(evidences_url, matchingParaDict, level='peptides'):
    ### compare percolator results between diaumpire and diasearch. There are four cases:
    ### 1. A high-confident (q-value<=0.01) diaumpire peptide/psm matches a high-confident (q-value<=0.01) diasearch peptide/psm
    ### 2. A high-confident (q-value<=0.01) diaumpire peptide/psm matches a relatively high-confident (0.01<q-value<=0.05) diasearch peptide/psm
    ### 3. A high-confident (low q-value) diaumpire peptide/psm matches a low-confident (q-value>0.05) diasearch peptide/psm
    ### 3. A high-confident (low q-value) diaumpire peptide/psm fails to match diasearch peptide/psm because the target fails the TDC
    ### 4. A high-confident (low q-value) diaumpire peptide/psm fails to match diasearch peptide/psm because it's not even selected for TDC

    assert level in ['peptides', 'psms'];
    key_headers = ['MS1Scan', 'Scanwin', 'Peplen'];  # 'Charge'
    metric_headers = ['Xcorr', 'lnEvalue', 'lnEvalueRank2', 'lnPvalue', 'MS2Pvalue1', 'MS2Pvalue2', 'Rtdiff2', 'lnIntensityRank1', 'lnIntensityRank2', 'IntensityIsoCorr1', 'IntensityMS1MS2Corr', 'IntensityMS2Corr'];
    accept_headers = np.concatenate((key_headers, metric_headers));

    output_url = os.path.join(os.path.dirname(evidences_url), 'diaumpire_diasearch_comp_{}.txt'.format(level));
    if not os.path.isfile(output_url):
        output = open(output_url, 'w');

        peptide_pepid_map = read_xcorr_results.get_peptide_pepid_map(matchingParaDict["fastaFile"]);
        pepid_peptide_map = read_xcorr_results.get_pepid_peptide_map(matchingParaDict["fastaFile"]);
        pepid_label_map = read_xcorr_results.get_pepid_labels(matchingParaDict["fastaFile"]);

        def load_percolator_results(percolator_output_dir, qvalue_thres=0.01):
            target_psm_url = os.path.join(percolator_output_dir, 'percolator.target.{}.txt'.format(level)); assert target_psm_url;
            decoy_psm_url = os.path.join(percolator_output_dir, 'percolator.decoy.{}.txt'.format(level)); assert decoy_psm_url;

            pepid_label_qvalue_list = [];
            for psm_url in [target_psm_url, decoy_psm_url]:
                with open(psm_url) as csvfile:
                    reader = csv.DictReader(csvfile, delimiter='\t');
                    for row in reader:
                        peptide = row['sequence']; qvalue = float(row['percolator q-value']); scan=int(row['scan']); charge=int(row['charge']);
                        pepid = peptide_pepid_map[peptide]; real_label = pepid_label_map[pepid];
                        if real_label > 0:  pepid_label_qvalue_list.append((pepid, 1, qvalue, scan, charge));
                        elif real_label == 0: pepid_label_qvalue_list.append((pepid, -1, qvalue, scan, charge));

            pepid_label_qvalue_list.sort(key=lambda x: x[2]);

            target_cnt = 0; decoy_cnt = 0; fdr_list = []; pepid_list = []; scan_list = []; charge_list = [];
            for pepid, label, qvalue, scan, charge in pepid_label_qvalue_list:
                if label > 0: target_cnt += 1; fdr_list.append(decoy_cnt * 1.0 / target_cnt); pepid_list.append(pepid); scan_list.append(scan); charge_list.append(charge);
                else: decoy_cnt += 1;

            for idx in range(len(fdr_list) - 1, 0, -1): fdr_list[idx - 1] = min(fdr_list[idx - 1], fdr_list[idx]);
            pepid_list = np.asarray(pepid_list); fdr_list = np.asarray(fdr_list); scan_list = np.asarray(scan_list); charge_list = np.asarray(charge_list);
            if qvalue_thres > 0:
                valid_indices = np.where(fdr_list <= qvalue_thres);
                pepid_list = pepid_list[valid_indices]; fdr_list = fdr_list[valid_indices]; scan_list = scan_list[valid_indices]; charge_list = charge_list[valid_indices];

            logger.info('pepid_info_list={}'.format(len(pepid_list)));
            return zip(pepid_list, fdr_list, scan_list, charge_list);

        umpire_percolator_dir = '';
        for file in os.listdir(os.path.join(os.path.dirname(matchingParaDict["inputFile"]), 'results')):
            if 'umpire_percolator_' in file: umpire_percolator_dir = os.path.join(os.path.dirname(matchingParaDict["inputFile"]), 'results', file); break;
        # logger.info('umpire_percolator_dir={}'.format(umpire_percolator_dir));
        umpire_pepid_info_list = load_percolator_results(umpire_percolator_dir);
        percolator_peptide_info_map = diaumpire_utils.load_percolator_info(matchingParaDict);

        umpire_pepid_set = set();
        for umpire_pepid_info in umpire_pepid_info_list: umpire_pepid_set.add(umpire_pepid_info[0]);


        diasearch_percolator_dir = '';
        for file in os.listdir(os.path.dirname(evidences_url)):
            if 'results_percolator' in file and 'auto_iter1' in file: diasearch_percolator_dir = os.path.join(os.path.dirname(evidences_url), file); break;
        # logger.info('diasearch_percolator_dir={}'.format(diasearch_percolator_dir));
        diasearch_pepid_info_list = load_percolator_results(diasearch_percolator_dir, qvalue_thres=-1);

        diasearch_scan_rt_map = {}; diasearch_pepid_evidence_map = {}; diasearch_pepid_set = set();
        scannum_list, rt_list = mzxml_utils.get_scannum_rt_pair(matchingParaDict["inputFile"]);
        scannum_rt_map = dict(zip(scannum_list, rt_list));
        with open(evidences_url) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                scan = int(row['ScanNr']); ms2scan = int(row['MS2Scan']); rt = scannum_rt_map[ms2scan];
                pepid = int(row['Pepid']); diasearch_pepid_set.add(pepid);

                assert scan not in diasearch_scan_rt_map;
                diasearch_scan_rt_map[scan] = rt;

                if pepid not in umpire_pepid_set: continue;
                if pepid not in diasearch_pepid_evidence_map: diasearch_pepid_evidence_map[pepid] = [];
                diasearch_pepid_evidence_map[pepid].append(row);

        diasearch_pepid_info_map = {};
        for diasearch_pepid, diasearch_qvalue, diasearch_scan, diasearch_charge in diasearch_pepid_info_list:
            diasearch_rt = diasearch_scan_rt_map[diasearch_scan];
            if diasearch_pepid not in diasearch_pepid_info_map: diasearch_pepid_info_map[diasearch_pepid] = [];
            diasearch_pepid_info_map[diasearch_pepid].append((diasearch_qvalue, diasearch_charge, diasearch_rt));

        for umpire_idx, umpire_pepid_info in enumerate(umpire_pepid_info_list):
            umpire_pepid, umpire_qvalue, umpire_scan, umpire_charge = umpire_pepid_info;
            umpire_peptide = pepid_peptide_map[umpire_pepid];

            umpire_startrt = -1; umpire_endrt = -1; umpire_apexrt = -1;
            for _, mgf_idx, start_rt, end_rt, apex_rt in percolator_peptide_info_map[umpire_peptide]:
                if mgf_idx == umpire_scan: umpire_startrt = start_rt; umpire_endrt = end_rt; umpire_apexrt = apex_rt; break;
            output.write('DIAUmpire\trank={}\tpepid={}\tpeptide={}\tqvalue{}\tcharge={}\tstart_rt={}\tend_rt={}\tapex_rt={}\n'.format(umpire_idx+1, umpire_pepid, umpire_peptide, umpire_qvalue, umpire_charge, umpire_startrt, umpire_endrt, umpire_apexrt ));


            min_diasearch_qvalue = 100; min_diasearch_rt = -1; min_diasearch_charge = -1;
            if umpire_pepid in diasearch_pepid_info_map:
                for diasearch_qvalue, diasearch_charge, diasearch_rt in diasearch_pepid_info_map[umpire_pepid]:
                    if diasearch_charge != umpire_charge: continue;
                    if umpire_startrt > 0 and (diasearch_rt < umpire_startrt-0.5 or diasearch_rt > umpire_endrt+0.5) : continue;
                    if diasearch_qvalue < min_diasearch_qvalue:
                        min_diasearch_qvalue = diasearch_qvalue;
                        min_diasearch_rt = diasearch_rt;
                        min_diasearch_charge = diasearch_charge;

                if min_diasearch_rt >= 0:
                    if min_diasearch_qvalue <= 0.01: output.write('type1 match\n');
                    elif min_diasearch_qvalue <= 0.05: output.write('type2 match\n');
                    else: output.write('type3 match\n');
                else:
                    for diasearch_qvalue, diasearch_charge, diasearch_rt in diasearch_pepid_info_map[umpire_pepid]:
                        if diasearch_qvalue < min_diasearch_qvalue:
                            min_diasearch_qvalue = diasearch_qvalue;
                            min_diasearch_rt = diasearch_rt;
                            min_diasearch_charge = diasearch_charge;
                    assert min_diasearch_rt >= 0;

                    if min_diasearch_qvalue <= 0.01: output.write('type1_nonperfect match\n');
                    elif min_diasearch_qvalue <= 0.05: output.write('type2_nonperfect match\n');
                    else: output.write('type3_nonperfect match\n');

                output.write('DIASearch\tqvalue={}\tcharge={}\trt={}\n'.format(min_diasearch_qvalue, min_diasearch_charge, min_diasearch_rt));

            elif umpire_pepid in diasearch_pepid_set: output.write('type4 match\n');
            else: output.write('type5 match\n');


            if umpire_pepid in diasearch_pepid_evidence_map :
                if min_diasearch_rt < 0: min_diasearch_rt = umpire_apexrt;
                if min_diasearch_rt >= 0:
                    apex_diff_list = []; row_list = [];
                    for row in diasearch_pepid_evidence_map[umpire_pepid]:
                        diasearch_charge = int(row['Charge']); ms2scan = int(row['MS2Scan']); diasearch_rt = scannum_rt_map[ms2scan];
                        if diasearch_charge != umpire_charge: continue;
                        apex_diff_list.append(np.fabs(diasearch_rt-min_diasearch_rt)); row_list.append(row);

                    if len(apex_diff_list) <= 0:
                        ms2scan = int(row['MS2Scan']); diasearch_rt = scannum_rt_map[ms2scan];
                        apex_diff_list.append(np.fabs(diasearch_rt-min_diasearch_rt)); row_list.append(row);

                    if len(apex_diff_list) > 0:
                        row = row_list[np.argmin(apex_diff_list)];
                        diasearch_charge = int(row['Charge']); ms2scan = int(row['MS2Scan']); diasearch_rt = scannum_rt_map[ms2scan];
                        output.write('rt={}\t{}\n'.format(diasearch_rt, '\t'.join( ['{}={}'.format(key, row[key]) for key in accept_headers])));
                else:
                    row = diasearch_pepid_evidence_map[umpire_pepid][0]
                    diasearch_charge = int(row['Charge']); ms2scan = int(row['MS2Scan']); diasearch_rt = scannum_rt_map[ms2scan];
                    output.write('charge={}\trt={}\t{}\n'.format(diasearch_charge, diasearch_rt, '\t'.join( ['{}={}'.format(key, row[key]) for key in accept_headers])));

        output.close();


    target_type_list = ['1', '2', '3', '4'];
    def parse_output(output_url):
        type_metrics_map = {};  curr_type='';
        with open(output_url) as fp:
            for cnt, line in enumerate(fp):
                line = line.rstrip();
                if line.startswith('type'): curr_type = line.split(' ')[0].replace('type', '');
                elif line.startswith('rt'):
                    metric_map = {};
                    for key_value_pair in line.split('\t'):
                        key, value = key_value_pair.split('=');
                        if re.compile('^\s*\d+\s*$').search(value): value = int(value);
                        elif re.compile('^\s*(\d*\.\d+)|(\d+\.\d*)\s*$').search(value): value = float(value);
                        else: assert False;
                        metric_map[key] = value;

                    if curr_type not in type_metrics_map: type_metrics_map[curr_type] = [];
                    type_metrics_map[curr_type].append(metric_map);
        return type_metrics_map;
    type_metrics_map = parse_output(output_url);


    def load_evidence_backgroud(evidences_url, key):
        evidence_key_background_url = evidences_url.replace('.txt', '_{}.npy'.format(key));
        if os.path.isfile(evidence_key_background_url): evidence_key_values_map = np.load(evidence_key_background_url, allow_pickle=True).item();
        else:
            pepid_label_map = read_xcorr_results.get_pepid_labels(matchingParaDict["fastaFile"]);

            evidence_key_values_map = {1:{}, -1:{}};
            with open(evidences_url) as csvfile:
                reader = csv.DictReader(csvfile, delimiter='\t');
                for row in reader:
                    pepid = int(row['Pepid']); key_metric = int(row[key]); real_label = pepid_label_map[pepid];
                    if real_label <= 0 and np.random.rand() > 0.25: continue;

                    if real_label > 0: real_label = 1;
                    else: real_label = -1;

                    if key_metric not in evidence_key_values_map[real_label]: evidence_key_values_map[real_label][key_metric] = [];
                    evidence_key_values_map[real_label][key_metric].append(tuple([float(row[value_metric]) for value_metric in metric_headers]));

            np.save(evidence_key_background_url, evidence_key_values_map);
        logger.info('evidence_key_values_map={}'.format(len(evidence_key_values_map)));
        return evidence_key_values_map;
    # for key in key_headers: evidence_key_values_map = load_evidence_backgroud(evidences_url, key);
    # evidence_ms1scan_values_map = load_evidence_backgroud(evidences_url, 'MS1Scan');

    type_rts_map = {}; type_charges_map = {}; type_peplens_map = {};
    for target_type in target_type_list:
        type_rts_map[target_type] = [metric_map['rt'] for metric_map in type_metrics_map[target_type]];
        type_charges_map[target_type] = [metric_map['Charge'] for metric_map in type_metrics_map[target_type]];
        type_peplens_map[target_type] = [metric_map['Peplen'] for metric_map in type_metrics_map[target_type]];


    figure = plt.figure(figsize=(20, 10));
    ax = figure.add_subplot(1, 1, 1);
    ax.autoscale();

    min_value = np.min([np.min(type_rts_map[type]) for type in type_rts_map]);
    max_value = np.max([np.max(type_rts_map[type]) for type in type_rts_map]);
    for type in type_rts_map:  ax.hist(type_rts_map[type], bins=100, range=(min_value, max_value), alpha=0.2, label='type{}'.format(type));
    ax.set_xlabel('Retention time', fontsize=25);

    # for type in type_rts_map:  ax.hist(type_charges_map[type], bins=5, range=(0, 5.1), alpha=0.2, label='type{}'.format(type));
    # ax.set_xlabel('Charge', fontsize=25);

    ax.set_ylabel('# of peptides', fontsize=25);
    ax.legend(prop={'size': 25});
    plt.show();

    # for metric_idx, metric_name in enumerate(metric_headers):
    #     type_arr = [];  value_arr = [];
    #     for target_type in target_type_list:
    #         metric_values = [metric_map[metric_name] for metric_map in type_metrics_map[target_type] if ];
    #         type_arr = np.concatenate((type_arr, ['type{}'.format(target_type)] * len(metric_values)));
    #         value_arr = np.concatenate((value_arr, metric_values ));
    #
    #     values_pd = pd.DataFrame({'level': [level]*len(value_arr), 'type': type_arr, 'value': value_arr, });
    #     figure = plt.figure(figsize=(30, 15));
    #     ax = sns.violinplot(x="level", y="value", hue="type", data=values_pd, palette="Set1", split=False);
    #     ax.set_xlabel('');
    #     ax.set_ylabel('# of detection (1% FDP)', fontsize=30);
    #     ax.tick_params(axis='both', which='major', labelsize=25)
    #     ax.legend(fontsize=25);
    #     ax.set_title('metric={}'.format(metric_name), fontsize=40);
    #     # plt.show();
    #
    #     plt.savefig( os.path.join(os.path.dirname(evidences_url), 'comp_umpire_tdc_{}.png'.format(metric_name)), dpi=100);
    #     plt.close(figure);


def get_evidence_summary_stat(xcorrFile, matchingParaDict, charge_states=[1,2,3,4,5]):

    # ms2scan_peakcnt_map = {};
    # with mzxml.read(matchingParaDict["inputFile"]) as reader:
    #     for spectrum in reader:
    #         scanNum = int(spectrum['num']); msLevel = int(spectrum['msLevel']); logger.info('scan={}'.format(scanNum) );
    #         if msLevel <= 1: continue;
    #
    #         mz_arr = spectrum['m/z array']; intensity_arr = spectrum['intensity array']; peak_cnt_beforefiltering = len(mz_arr);
    #         if len(mz_arr) <= 1: continue;
    #
    #         intensity_arr = spectrum_utils.get_transformed_intensity(mz_arr, intensity_arr, spectrum_utils.IntTransformType.Sqrt);
    #         mz_arr, intensity_arr = spectrum_utils.get_grassfiltered_mz_intensity(mz_arr, intensity_arr);
    #         # intensity_arr = spectrum_utils.get_transformed_intensity(mz_arr, intensity_arr, spectrum_utils.IntTransformType.SegmentedBasePeakNorm);
    #         peak_cnt_afterfiltering = len(mz_arr);
    #         ms2scan_peakcnt_map[scanNum] = (peak_cnt_beforefiltering, peak_cnt_afterfiltering);
    #
    # figure = plt.figure(figsize=(30, 15));
    # ax = figure.add_subplot(1, 1, 1);
    # ax.autoscale();
    #
    # peak_cnt_beforefiltering_list = [ms2scan_peakcnt_map[ms2scan][0] for ms2scan in ms2scan_peakcnt_map];
    # peak_cnt_afterfiltering_list = [ms2scan_peakcnt_map[ms2scan][1] for ms2scan in ms2scan_peakcnt_map];
    #
    # min_value = min(np.min(peak_cnt_beforefiltering_list), np.min(peak_cnt_afterfiltering_list));
    # max_value = max(np.max(peak_cnt_beforefiltering_list), np.max(peak_cnt_afterfiltering_list));
    #
    # ax.hist(peak_cnt_beforefiltering_list, bins=200, range=(min_value, max_value), log=True, alpha=0.4, label='Before grassfiltering');
    # ax.hist(peak_cnt_afterfiltering_list, bins=200, range=(min_value, max_value), log=True, alpha=0.4, label='After grassfiltering');
    #
    # ax.set_xlabel('Peak count', fontsize=15);
    # ax.set_ylabel('# of MS2 scans', fontsize=15);
    # ax.legend(prop={'size': 25});
    # plt.show();


    pepid_label_map = read_xcorr_results.get_pepid_labels(matchingParaDict["fastaFile"]);
    charge_ms2scan_xcorr_map = { }; ms2scan_evalue_map = { };
    for charge in charge_states: charge_ms2scan_xcorr_map[charge] = {};
    with open(xcorrFile) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t');
        for row in reader:
            pepid = int(row['pepid']);
            real_label = pepid_label_map[pepid];
            if real_label > 0: real_label = 1;
            else: real_label = -1;

            ms2scan = int(row['scan']); charge = int(row['charge']);
            xcorr = float(row['xcorr']); lnEvalue = float(row['evalue_log']);

            if ms2scan not in charge_ms2scan_xcorr_map[charge]: charge_ms2scan_xcorr_map[charge][ms2scan] = (xcorr, real_label);
            elif charge_ms2scan_xcorr_map[charge][ms2scan][0] < xcorr:
                charge_ms2scan_xcorr_map[charge][ms2scan] = (xcorr, real_label);

            if ms2scan not in ms2scan_evalue_map: ms2scan_evalue_map[ms2scan] = (lnEvalue, real_label);
            elif ms2scan_evalue_map[ms2scan][0] > lnEvalue:
                ms2scan_evalue_map[ms2scan] = (lnEvalue, real_label);

    for charge in charge_states:
        pos_cnt = 0; neg_cnt = 0;
        for ms2scan in charge_ms2scan_xcorr_map[charge]:
            xcorr, real_label = charge_ms2scan_xcorr_map[charge][ms2scan];
            if real_label > 0: pos_cnt += 1;
            elif real_label < 0: neg_cnt += 1;
            else: assert False;
        logger.info('xcorr\tcharge={}\tpos_cnt={}\tneg_cnt={}\tTMP={}'.format(charge, pos_cnt, neg_cnt, 100.0*pos_cnt/(pos_cnt+neg_cnt)));

    pos_cnt = 0; neg_cnt = 0;
    for ms2scan in ms2scan_evalue_map:
        xcorr, real_label = ms2scan_evalue_map[ms2scan];
        if real_label > 0: pos_cnt += 1;
        elif real_label < 0: neg_cnt += 1;
        else: assert False;
    logger.info('evalue\tpos_cnt={}\tneg_cnt={}\tTMP={}'.format(pos_cnt, neg_cnt, 100.0*pos_cnt/(pos_cnt+neg_cnt)));


def comp_cc_vs_pp(evalParaDict):
    result_dir = os.path.abspath(os.path.join(evalParaDict["outputFolder"], 'results')); logger.info(result_dir);
    if not os.path.exists(result_dir): os.makedirs(result_dir);

    modular_param_list_combo = [evalParaDict["alphaInt_list"], evalParaDict["alphaRT_list"], evalParaDict["alphaCC_list"], evalParaDict["alphaTC_list"]];
    bp_param_list_combo = [evalParaDict["betaCC_list"], evalParaDict["lamdaCC_list"], evalParaDict["betaTC_list"], evalParaDict["lamdaTC_list"]];


    for selmethod_idx, selmethod in enumerate(evalParaDict["selmethod_list"]):
        for alphaInt, alphaRT, alphaCC, alphaTC in itertools.product(*modular_param_list_combo):
            matchingParaDict1 = run_matching.getMatchingParamDict(alphaInt, alphaRT, alphaCC, alphaTC, 0, 0, 0, 0, evalParaDict["ppm"], evalParaDict["rtdiff_thres"], evalParaDict["evalue_mode"], evalParaDict["td_balance"], metric_type_list=evalParaDict["metric_type_list"], debiase_type_list=evalParaDict["debiase_type_list"], scale_type_list=evalParaDict["scale_type_list"]);

            target_decoy = 100;
            cached_param_metrics_url = os.path.join(result_dir, 'cached_param_metrics_{}_{}.npy'.format(selmethod, run_matching.getMatchingParamStr(matchingParaDict1, modular=True) ));
            if os.path.isfile(cached_param_metrics_url): param_metrics_map = np.load(cached_param_metrics_url, allow_pickle=True).item();
            else:
                param_metrics_map = {};
                for betaCC, lamdaCC, betaTC, lamdaTC in itertools.product( *bp_param_list_combo):
                    matchingParaDict = run_matching.getMatchingParamDict(alphaInt, alphaRT, alphaCC, alphaTC, betaCC, betaTC, lamdaCC, lamdaTC, evalParaDict["ppm"], evalParaDict["rtdiff_thres"],  evalParaDict["evalue_mode"], evalParaDict["td_balance"], metric_type_list=evalParaDict["metric_type_list"], debiase_type_list=evalParaDict["debiase_type_list"], scale_type_list=evalParaDict["scale_type_list"]);

                    result_url = os.path.join(evalParaDict["outputFolder"], 'rst_{}_{}.txt'.format(selmethod, run_matching.getMatchingParamStr(matchingParaDict)));
                    if not os.path.isfile(result_url): continue;
                    logger.info(result_url);


                    param = 'bCC{}_lCC{};bPP{}_lPP{}'.format(betaCC, lamdaCC, betaTC, lamdaTC);
                    tgtcnt_precursor_precthres, tgtcnt_peptide_precthres, _, _ = comp_target_decoy_by_precursor_thres(result_url, select_thres_list=[6000]); # logger.info(result_url);
                    tgtcnt_precursor_dcythres, tgtcnt_peptide_dcythres, allcntprecursor_dcythres, _, _, _, _ = comp_target_by_decoy_thres(result_url, decoy_thres_list=[target_decoy]);

                    param_metrics_map[param] = (tgtcnt_precursor_precthres, tgtcnt_peptide_precthres, tgtcnt_precursor_dcythres, tgtcnt_peptide_dcythres, allcntprecursor_dcythres);
                np.save(cached_param_metrics_url, param_metrics_map);
            logger.info('selmethod={}\tparam_metrics_map={}'.format(selmethod, len(param_metrics_map)));


            param_values_map = {};
            for param in param_metrics_map:
                tgtcnt_precursor_precthres, tgtcnt_peptide_precthres, tgtcnt_precursor_dcythres, tgtcnt_peptide_dcythres, allcntprecursor_dcythres = param_metrics_map[param];

                cc_param = param.split(';')[0]; bcc_param = cc_param.split('_')[0]; lcc_param = cc_param.split('_')[1];
                tc_param = param.split(';')[1]; btc_param = tc_param.split('_')[0]; ltc_param = tc_param.split('_')[1];

                bcc_param = float(bcc_param.replace('bCC','')); lcc_param = (lcc_param.replace('lCC',''));
                btc_param = float(btc_param.replace('bPP','')); ltc_param = (ltc_param.replace('lPP',''));

                target_precursor = allcntprecursor_dcythres[target_decoy];
                tgtcnt_precursor = tgtcnt_precursor_precthres[target_precursor];
                tgtcnt_peptide = tgtcnt_peptide_precthres[target_precursor];
                param_values_map[param] = (target_precursor, tgtcnt_precursor, tgtcnt_peptide, bcc_param, lcc_param, btc_param, ltc_param );

            param_list = np.sort(list(param_values_map.keys()));
            values_pd = pd.DataFrame({'Precursor_selected': [param_values_map[param][0] for param in param_list],
                                    'Target_precursor': [param_values_map[param][1] for param in param_list],
                                    'Target_peptide': [param_values_map[param][2] for param in param_list],
                                    'betaCC': [param_values_map[param][3] for param in param_list],
                                    'lambdaCC': [param_values_map[param][4] for param in param_list],
                                    'betaPP': [param_values_map[param][5] for param in param_list],
                                    'lambdaPP': [param_values_map[param][6] for param in param_list]
                                    });

            fig = px.scatter(values_pd, x="Target_precursor", y="Target_peptide", color="betaCC", hover_data=['betaCC','lambdaCC','betaPP','lambdaPP']);
            plotly.offline.plot(fig, filename=os.path.join(result_dir, 'html_betaCC_{}_{}.html'.format(selmethod, run_matching.getMatchingParamStr(matchingParaDict1, modular=True))), auto_open=False)

            fig = px.scatter(values_pd, x="Target_precursor", y="Target_peptide", color="lambdaCC", hover_data=['betaCC','lambdaCC','betaPP','lambdaPP']);
            plotly.offline.plot(fig, filename=os.path.join(result_dir, 'html_lambdaCC_{}_{}.html'.format(selmethod, run_matching.getMatchingParamStr(matchingParaDict1, modular=True))), auto_open=False)

            fig = px.scatter(values_pd, x="Target_precursor", y="Target_peptide", color="betaPP", hover_data=['betaCC','lambdaCC','betaPP','lambdaPP']);
            plotly.offline.plot(fig, filename=os.path.join(result_dir, 'html_betaPP_{}_{}.html'.format(selmethod, run_matching.getMatchingParamStr(matchingParaDict1, modular=True))), auto_open=False)

            fig = px.scatter(values_pd, x="Target_precursor", y="Target_peptide", color="lambdaPP", hover_data=['betaCC','lambdaCC','betaPP','lambdaPP']);
            plotly.offline.plot(fig, filename=os.path.join(result_dir, 'html_lambdaPP_{}_{}.html'.format(selmethod, run_matching.getMatchingParamStr(matchingParaDict1, modular=True))), auto_open=False)


def plot_ms2_intensity(matchingParaDict):

    with mzxml.read(matchingParaDict["inputFile"]) as reader:
        for spectrum in reader:
            scanNum = int(spectrum['num']); msLevel = int(spectrum['msLevel']); logger.info('scan={}'.format(scanNum) );
            if msLevel <= 1: continue;

            intensity_arr = spectrum['intensity array'];
            intensity_arr = np.log(intensity_arr);
            intensity_arr = np.asarray(sorted(intensity_arr, reverse=True));

            if len(intensity_arr) <= 1: continue;
            if np.random.rand()>0.02: continue;

            ranks = np.log(np.asarray(range(len(intensity_arr))) + 1);

            ignore_top = 25; retain_cnt = min(1000, int((len(intensity_arr) - ignore_top) * 0.2));
            slope, intercept, x_values, y_values = read_xcorr_results.fit_log_linear(intensity_arr, ignore_top=ignore_top, ignore_bottom=max(len(intensity_arr) - retain_cnt - ignore_top, 200));

            figure = plt.figure();
            ax1 = figure.add_subplot(1, 1, 1);
            ax1.autoscale();
            ax1.plot(intensity_arr, ranks, 's', marker='.');

            min_intensity = 2.5;
            ax1.plot([min_intensity, -intercept / slope], [min_intensity * slope + intercept, 0], linestyle='-', color='red');

            min_logintensity = np.min(intensity_arr);
            for iter in range(3):
                curr_logintensity = (np.log(len(intensity_arr)) - intercept) / slope;
                if curr_logintensity > min_logintensity:
                    intensity_arr = intensity_arr[intensity_arr >= curr_logintensity];
                    min_logintensity = curr_logintensity;
                else: break;
            ax1.plot([min_logintensity, min_logintensity], [0, intercept], linestyle=(0, (1, 10)), color='green');


            ax1.set_xlabel('logrithm of fragments intensity', fontsize=20);
            ax1.set_ylabel('logrithm of ranking', fontsize=20);
            ax1.xaxis.set_tick_params(labelsize=15);
            ax1.yaxis.set_tick_params(labelsize=15);
            plt.show();


def plot_perf_hyperparam():

    result_url1 = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_tailor/e01306-top1000-chargeAll-prewindow10-mzbinwidth0.02-mzbinoffset0/results_top5/results_final.txt';
    result_url2 = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01307/results_tailor/e01307-top1000-chargeAll-prewindow5-mzbinwidth0.02-mzbinoffset0/results_top5/results_final.txt';
    result_url3 = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01309/results_tailor/e01309-top1000-chargeAll-prewindow2.5-mzbinwidth0.02-mzbinoffset0/results_top5/results_final.txt';

    result_url4 = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2/results_tailor_denoised2/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2-top1000-chargeAll-prewindow12.5-mzbinwidth0.02-mzbinoffset0/results_top5/results_final.txt';
    result_url5 = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF5600_64var_007_cwt2/results/HYE124_TTOF5600_64var_007_cwt2-top1000-chargeAll-prewindow-1-mzbinwidth0.02-mzbinoffset0/results_top5/results_final.txt';
    result_url6 = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2/results_tailor/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2-top1000-chargeAll-prewindow12.5-mzbinwidth0.02-mzbinoffset0/results_top5/results_final.txt';
    result_url7 = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF6600_64var_008_cwt2/results/HYE124_TTOF6600_64var_008_cwt2-top1000-chargeAll-prewindow-1-mzbinwidth0.02-mzbinoffset0/results_top5/results_final.txt';

    result_url8 ='/media/yanglu/TOSHIBA1/data/dia_search/Rosenberger/plasma/yanliu_L130104_007_cwt2/results_denoised2/yanliu_L130104_007_cwt2-top1000-chargeAll-prewindow12.5-mzbinwidth0.02-mzbinoffset0/results_top5/results_final.txt';


    def parse_result(result_url):
        key_value_map = {};
        with open(result_url) as fp:
            for cnt, line in enumerate(fp):
                line = line.rstrip();
                if line.startswith('percolator'):
                    line = line[line.find('results_percolator_'):];  # logger.info(line);

                    key_part1 = line[:line.find('evidence')].replace('results_percolator_', '');
                    key_part2 = line[line.find('aPreP'):];
                    # key = '{}{}'.format(key_part1, key_part2);
                    if key_part1 not in key_value_map: key_value_map[key_part1] = {};

                elif line.startswith('level=peptide'):
                    value = int(line.split('\t')[-1].split('=')[-1]);
                    # key_value_map[key] = value;
                    key_value_map[key_part1][key_part2] = value;

        # logger.info('key_value_map={}'.format(len(key_value_map)));
        return key_value_map;

    result_url_list = [result_url1, result_url2,  result_url4, result_url5, result_url6, result_url7, result_url8 ];
    for result_idx1 in range(len(result_url_list)):
        key_value_map1 = parse_result(result_url_list[result_idx1]);
        # logger.info('result_idx1={}\tkey_value_map={}'.format(result_idx1, len(key_value_map1)));

        key_overlap_values_map = {};
        for result_idx2 in range(len(result_url_list)):
            if result_idx1 == result_idx2: continue;

            key_value_map2 = parse_result(result_url_list[result_idx2]);
            for key_part1 in key_value_map2:
                if key_part1 not in key_value_map1: continue;
                # if ',11,' in key_part1: continue;
                if 'met1,2,3,6,7,8,14_' not in key_part1: continue;

                if key_part1 not in key_overlap_values_map: key_overlap_values_map[key_part1] = {};
                for key_part2 in key_value_map2[key_part1]:
                    if key_part2 in key_value_map1[key_part1]:

                        if key_part2 not in key_overlap_values_map[key_part1]: key_overlap_values_map[key_part1][key_part2] = [];
                        key_overlap_values_map[key_part1][key_part2].append((key_value_map1[key_part1][key_part2], key_value_map2[key_part1][key_part2]))


        figure = plt.figure(figsize=(20, 15));
        ax1 = figure.add_subplot(1, 1, 1);
        ax1.autoscale();

        key_idx = 0;
        for key_part1 in key_overlap_values_map:
            x_list = []; y_list = [];
            for key_part2 in key_overlap_values_map[key_part1]:
                x_list.append(np.mean([x[0] for x in key_overlap_values_map[key_part1][key_part2] ]));
                y_list.append(np.sum([x[1] for x in key_overlap_values_map[key_part1][key_part2] ]));

            ax1.plot(x_list, y_list, 's', marker='o', color=viz_utils.COLORS[key_idx], alpha=0.8, label=key_part1);
            key_idx += 1;

            max_idx = np.argmax(y_list);
            logger.info('max_x={}\tmax_y={}'.format(x_list[max_idx], y_list[max_idx] ));


        ax1.set_xlabel('result_url{}'.format(result_idx1 + 1), fontsize=20);
        ax1.set_ylabel('rest', fontsize=20);
        ax1.legend(fontsize=25);
        plt.show();


def plot_comp_instrument():

    result_url1 = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_tailor/e01306-top1000-chargeAll-prewindow10-mzbinwidth0.02-mzbinoffset0/results_top5/results_final.txt';
    result_url2 = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01307/results_tailor/e01307-top1000-chargeAll-prewindow5-mzbinwidth0.02-mzbinoffset0/results_top5/results_final.txt';
    result_url3 = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01309/results_tailor/e01309-top1000-chargeAll-prewindow2.5-mzbinwidth0.02-mzbinoffset0/results_top5/results_final.txt';

    result_url4 = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2/results_tailor_denoised2/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2-top1000-chargeAll-prewindow12.5-mzbinwidth0.02-mzbinoffset0/results_top5/results_final.txt';
    result_url5 = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF5600_64var_007_cwt2/results/HYE124_TTOF5600_64var_007_cwt2-top1000-chargeAll-prewindow-1-mzbinwidth0.02-mzbinoffset0/results_top5/results_final.txt';
    result_url6 = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2/results_tailor/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2-top1000-chargeAll-prewindow12.5-mzbinwidth0.02-mzbinoffset0/results_top5/results_final.txt';
    result_url7 = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF6600_64var_008_cwt2/results/HYE124_TTOF6600_64var_008_cwt2-top1000-chargeAll-prewindow-1-mzbinwidth0.02-mzbinoffset0/results_top5/results_final.txt';

    result_url8 ='/media/yanglu/TOSHIBA1/data/dia_search/Rosenberger/plasma/yanliu_L130104_007_cwt2/results_denoised2/yanliu_L130104_007_cwt2-top1000-chargeAll-prewindow12.5-mzbinwidth0.02-mzbinoffset0/results_top5/results_final.txt';

    def parse_result(result_url):
        key_value_map = {};
        with open(result_url) as fp:
            for cnt, line in enumerate(fp):
                line = line.rstrip();
                if line.startswith('percolator'):
                    line = line[line.find('results_percolator_'):]; # logger.info(line);

                    key_part1 = line[:line.find('evidence')].replace('results_percolator_', '');
                    key_part2 = line[line.find('aPreP'):];
                    # key = '{}{}'.format(key_part1, key_part2);
                    if key_part1 not in key_value_map: key_value_map[key_part1] = {};

                elif line.startswith('level=peptide'):
                    value = int(line.split('\t')[-1].split('=')[-1]);
                    # key_value_map[key] = value;
                    key_value_map[key_part1][key_part2] = value;

        # logger.info('key_value_map={}'.format(len(key_value_map)));
        return key_value_map;

    # result_url_list = [(result_url4, 'Navarro_TTOF5600_32fix'), (result_url5, 'Navarro_TTOF5600_64var'), (result_url6, 'Navarro_TTOF6600_32fix'), (result_url7, 'Navarro_TTOF6600_64var')];
    result_url_list = [(result_url1, 'MetOxYeast_Orbitrap_20mz'), (result_url2, 'MetOxYeast_Orbitrap_10mz'), (result_url4, 'Navarro_TTOF5600_32fix'), (result_url6, 'Navarro_TTOF6600_32fix')];

    for result_idx1 in range(len(result_url_list)):
        key_value_map1 = parse_result(result_url_list[result_idx1][0]); result_label1 = result_url_list[result_idx1][1];

        for result_idx2 in range(result_idx1 + 1, len(result_url_list)):
            key_value_map2 = parse_result(result_url_list[result_idx2][0]); result_label2 = result_url_list[result_idx2][1];

            key_overlap_values_map = {};
            for key_part1 in key_value_map2:
                if key_part1 not in key_value_map1: continue;
                if ',11,' in key_part1: continue;
                key_overlap_values_map[key_part1] = [];

                for key_part2 in key_value_map2[key_part1]:
                    if key_part2 in key_value_map1[key_part1]:
                        key_overlap_values_map[key_part1].append((key_value_map1[key_part1][key_part2], key_value_map2[key_part1][key_part2] ));
            logger.info('result_idx1={}\tresult_idx2={}\tkey_size1={}\tkey_size2={}\toverlap={}'.format(result_idx1, result_idx2, len(key_value_map1), len(key_value_map2), np.sum([len(key_overlap_values_map[key]) for key in key_overlap_values_map]) ));

            figure = plt.figure(figsize=(10, 15));
            ax1 = figure.add_subplot(1, 1, 1);
            ax1.autoscale();

            key_idx = 0;
            for key in key_overlap_values_map:
                ax1.plot([x[0] for x in key_overlap_values_map[key]], [x[1] for x in key_overlap_values_map[key]], 's', marker='o', color='blue', alpha=0.8, label=key);
                key_idx += 1;

            ax1.set_xlabel(result_label1, fontsize=20);
            ax1.set_ylabel(result_label2, fontsize=20);
            ax1.legend(fontsize=25);
            # plt.show();

            from matplotlib.backends.backend_pdf import PdfPages
            pp = PdfPages(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')), 'fig_hyperparam_{}_{}.pdf'.format(result_label1, result_label2)));
            pp.savefig(figure, bbox_inches='tight');
            pp.close();
            plt.close(figure);


def plot_orbitrap():

    result_url1 = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_tailor/e01306-top1000-chargeAll-prewindow10-mzbinwidth0.02-mzbinoffset0/results_top5/results_new.txt';
    result_url2 = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01307/results_tailor/e01307-top1000-chargeAll-prewindow5-mzbinwidth0.02-mzbinoffset0/results_top5/results_new.txt';
    result_url3 = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01309/results_tailor/e01309-top1000-chargeAll-prewindow2.5-mzbinwidth0.02-mzbinoffset0/results_top5/results_new.txt';
    result_url4 = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_tailor/e01306-top1000-chargeAll-prewindow10-mzbinwidth0.02-mzbinoffset0/results_top20/results_new.txt';

    fdr_list = ['0.01', '0.05', '0.1', '0.2'];
    use_normalize = True;

    def parse_result(result_url):
        key_value_map = {};


        with open(result_url) as fp:
            for cnt, line in enumerate(fp):
                line = line.rstrip();
                if line.startswith('percolator'):
                    line = line[line.find('results_percolator_'):];  # logger.info(line);

                    key_part1 = line[:line.find('evidence')].replace('results_percolator_', '');
                    key_part2 = line[line.find('aPreP'):];
                    # key = '{}{}'.format(key_part1, key_part2);
                    if key_part1 not in key_value_map: key_value_map[key_part1] = {};
                    if key_part2 not in key_value_map[key_part1]: key_value_map[key_part1][key_part2] = {};

                elif line.startswith('level=peptide'):
                    arr = line.split('\t');
                    fdr = arr[-2].split('=')[-1];
                    target_cnt = int(arr[-1].split('=')[-1]);
                    # logger.info('fdr={}\ttarget_cnt={}'.format(fdr, target_cnt));
                    key_value_map[key_part1][key_part2][fdr] = target_cnt;

        # logger.info('key_value_map={}'.format(len(key_value_map)));
        return key_value_map;

    def key_value_quantile(key_value_map):
        key_value_quantile_map = {};
        for key_part1 in key_value_map1:
            key_value_quantile_map[key_part1] = {};

            for fdr in fdr_list:
                value_list = [];
                for key_part2 in key_value_map[key_part1]:
                    if fdr in key_value_map[key_part1][key_part2]: value_list.append(key_value_map[key_part1][key_part2][fdr]);
                key_value_quantile_map[key_part1][fdr] = np.quantile(value_list, 0.95);
        return key_value_quantile_map;


    logger.info('---------------------------------------------------------------------------------\nuse_normalize={}'.format(use_normalize));
    result_url_list = [result_url1, result_url2, result_url3];
    for result_idx1 in range(len(result_url_list)):
        key_value_map1 = parse_result(result_url_list[result_idx1]);
        logger.info('result_idx1={}\tkey_value_map={}'.format(result_idx1, len(key_value_map1)));

        key_overlap_values_map = {}; overlap_size = len(result_url_list) - 1;
        for key_part1 in key_value_map1:
            key_overlap_values_map[key_part1] = {};
            for key_part2 in key_value_map1[key_part1]:
                key_overlap_values_map[key_part1][key_part2] = {};
                for fdr in fdr_list: key_overlap_values_map[key_part1][key_part2][fdr] = [];

        for result_idx2 in range(len(result_url_list)):
            if result_idx1 == result_idx2: continue;
            key_value_map2 = parse_result(result_url_list[result_idx2]);
            key_value_quantile_map = key_value_quantile(key_value_map2)

            for key_part1 in key_value_map2:
                if key_part1 not in key_overlap_values_map: continue;

                for key_part2 in key_value_map2[key_part1]:
                    if key_part2 not in key_overlap_values_map[key_part1]: continue;

                    for fdr in key_value_map2[key_part1][key_part2]:
                        if use_normalize: key_overlap_values_map[key_part1][key_part2][fdr].append(key_value_map2[key_part1][key_part2][fdr] / key_value_quantile_map[key_part1][fdr] );
                        else: key_overlap_values_map[key_part1][key_part2][fdr].append(key_value_map2[key_part1][key_part2][fdr]);


        for key_part1 in key_overlap_values_map:
            figure = plt.figure(figsize=(20, 20));

            for fdr_idx, fdr in enumerate(fdr_list):
                ax1 = figure.add_subplot(2, 2, fdr_idx+1);
                ax1.autoscale();

                x_list = []; y_list = []; label_list = [];
                for key_part2 in key_overlap_values_map[key_part1]:
                    if key_part2 not in key_value_map1[key_part1]: continue;
                    if fdr not in key_value_map1[key_part1][key_part2]: continue;

                    if len(key_overlap_values_map[key_part1][key_part2][fdr]) >= overlap_size:

                        x_list.append(key_value_map1[key_part1][key_part2][fdr]);
                        y_list.append(np.mean(key_overlap_values_map[key_part1][key_part2][fdr]));
                        label_list.append(key_part2);

                ax1.plot(x_list, y_list, 's', marker='o', color=viz_utils.COLORS[fdr_idx], alpha=0.8, label=fdr);

                max_idx = np.argmax(y_list);
                logger.info('fdr={}\tmax_x={}\tmax_y={}\tmax_possible_x={}'.format(fdr, x_list[max_idx], y_list[max_idx], np.max(x_list)));
                logger.info('max_label={}'.format(label_list[max_idx]));

                ax1.set_xlabel('result_url={}'.format(result_idx1 + 1), fontsize=20);
                ax1.set_ylabel('rest', fontsize=20);
                ax1.legend(fontsize=25);
                ax1.set_title('fdr={}'.format(fdr));

        # mixture of fdr
        sub_fdr_list = ['0.01', '0.05']; sub_weight_list = [1.0, 1.0]; x_list = [];  y_list = []; label_list = [];
        for key_part2 in key_overlap_values_map[key_part1]:
            if key_part2 not in key_value_map1[key_part1]: continue;

            sub_y_list = [];
            for sub_fdr in sub_fdr_list:
                if sub_fdr not in key_value_map1[key_part1][key_part2]: continue;
                if len(key_overlap_values_map[key_part1][key_part2][sub_fdr]) >= overlap_size: sub_y_list.append(np.mean(key_overlap_values_map[key_part1][key_part2][sub_fdr]));

            if len(sub_y_list) == len(sub_fdr_list):
                x_list.append(np.inner([key_value_map1[key_part1][key_part2][sub_fdr] for sub_fdr in sub_fdr_list], sub_weight_list));
                y_list.append(np.inner(sub_y_list, sub_weight_list));
                label_list.append(key_part2);

        max_idx = np.argmax(y_list);
        logger.info('fdr={}\tmax_x={}\tmax_y={}\tmax_possible_x={}'.format(sub_fdr_list, x_list[max_idx], y_list[max_idx], np.max(x_list)));
        logger.info('max_label={}'.format(label_list[max_idx]));

        plt.show();


def plot_orbitrap_sibling_comp():

    input_url = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/e01306.mzXML';
    fasta_url = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/fasta/cerevisiae_orf_trans_all_PlusDecoy.fasta';
    evidences_url = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_tailor/e01306-top1000-chargeAll-prewindow10-mzbinwidth0.02-mzbinoffset0/results_top5/pepid_evidence_scangap2.0_ppm10.txt';
    diameter_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_tailor/e01306-top1000-chargeAll-prewindow10-mzbinwidth0.02-mzbinoffset0/results_top5/results_percolator_met1,2,3,6,7,8,14_evidence_scangap2.0_ppm10_sorted_filter1_minmax1_trainfdr0.01_testfdr0.01_aPreP0.0_aMS2P0.0_aInt25.6_aRT0.0_aCorr0.0';
    diameter_psm_url = os.path.join(diameter_output_dir, 'percolator.target.psms.txt');
    umpire_output_dir = os.path.join(os.path.dirname(input_url), 'results', 'new_umpire_percolator_tdcF_fdr0.01');
    umpire_psm_url = os.path.join(umpire_output_dir, 'percolator.target.psms.txt');
    prosit_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_prosit_PlusDecoy/results_percolator';
    prosit_psm_url = os.path.join(prosit_output_dir, 'percolator.target.psms.txt');
    pecan_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_walnut_PlusDecoy/results_percolator';
    pecan_psm_url = os.path.join(pecan_output_dir, 'percolator.target.psms.txt');


    saved_evidence_scan_ms2scan_url = os.path.join(os.path.dirname(evidences_url), 'pepid_evidence_scan_ms2scan.npy');
    if os.path.isfile(saved_evidence_scan_ms2scan_url): evidence_scan_ms2scan_map = np.load(saved_evidence_scan_ms2scan_url, allow_pickle=True).item();
    else:
        evidence_scan_ms2scan_map = {};
        with open(evidences_url) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                scan = int(row['ScanNr']);  ms2scan = int(row['MS2Scan']);
                evidence_scan_ms2scan_map[scan] = ms2scan;
        np.save(saved_evidence_scan_ms2scan_url, evidence_scan_ms2scan_map);
    logger.info('evidence_scan_ms2scan_map={}'.format(len(evidence_scan_ms2scan_map)));


    peptide_pepid_map = read_xcorr_results.get_peptide_pepid_map(fasta_url);
    pepid_label_map = read_xcorr_results.get_pepid_labels(fasta_url);


    def parse_psm_result(psm_url, is_diameter=True):
        peptide_modlist_map = {}; pep_set = set(); target_cnt = 0; decoy_cnt = 0;
        with open(psm_url, 'rU') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                peptide = row['sequence'];
                scan = int(row['scan']);

                if is_diameter: ms2scan = evidence_scan_ms2scan_map[scan];
                else: ms2scan = scan;

                qvalue = float(row['percolator q-value']);
                pepid = peptide_pepid_map[peptide]; real_label = pepid_label_map[pepid];

                if real_label > 0:
                    target_cnt += 1;
                    fdr = decoy_cnt * 1.0 / target_cnt;
                    # if fdr >= 0.01: break;
                    if qvalue >= 0.01: break;
                else:
                    decoy_cnt += 1;
                    continue;

                if peptide in pep_set: continue;
                pep_set.add(peptide);

                peptide_unmod, mod_list = peptide_utils.get_unmod_peptide(peptide);
                # uniq_mod_list, mod_cnt_list = np.unique([x[1] for x in mod_list], return_counts=True);
                # assert len(uniq_mod_list) == 1;
                if peptide_unmod not in peptide_modlist_map: peptide_modlist_map[peptide_unmod] = [];
                peptide_modlist_map[peptide_unmod].append((ms2scan, mod_list));

        logger.info('peptide_modlist_map={}'.format(len(peptide_modlist_map)));
        return peptide_modlist_map;


    # peptide_modlist_map = parse_psm_result(diameter_psm_url, is_diameter=True); data_tag= 'diameter';
    # peptide_modlist_map = parse_psm_result(umpire_psm_url, is_diameter=False); data_tag= 'umpire';
    # peptide_modlist_map = parse_psm_result(prosit_psm_url, is_diameter=False); data_tag= 'prosit';
    peptide_modlist_map = parse_psm_result(pecan_psm_url, is_diameter=False); data_tag= 'pecan';



    sibling_scandiff_list = [];
    for peptide_unmod in peptide_modlist_map:
        if len(peptide_modlist_map[peptide_unmod]) <= 1: continue;

        peptide_mod_list = [(ms2scan, len(mod_list)) for ms2scan, mod_list in peptide_modlist_map[peptide_unmod]];
        logger.info('peptide_unmod={}\t{}'.format(peptide_unmod, peptide_mod_list));
        if len(peptide_mod_list) > 2: continue;

        mod1_ms2scan, mod1_cnt = peptide_mod_list[0];
        mod2_ms2scan, mod2_cnt = peptide_mod_list[1];
        if min(mod1_cnt, mod2_cnt) > 0: continue;
        if abs(mod1_cnt-mod2_cnt) > 1: continue;
        sibling_scandiff_list.append(abs(mod1_ms2scan - mod2_ms2scan));
    logger.info('sibling_scandiff_list={}'.format(len(sibling_scandiff_list)));

    figure = plt.figure(figsize=(16, 9));
    ax1 = figure.add_subplot(1, 1, 1);
    ax1.hist(sibling_scandiff_list, bins=500, alpha=0.8);

    ax1.set_xlim(0, 5000);
    ax1.set_xlabel('MS2scan difference between sibling peptide pair (1% peptide-level FDR)', fontsize=25);
    ax1.set_ylabel('Number of peptide pairs', fontsize=25);
    ax1.xaxis.set_tick_params(labelsize=18);
    ax1.yaxis.set_tick_params(labelsize=18);
    ax1.grid(linestyle=':');

    # plt.show();
    from matplotlib.backends.backend_pdf import PdfPages
    output_url = os.path.join(os.path.dirname(evidences_url), 'fig_orbitrap_sibling_comp_{}.pdf'.format(data_tag));
    pp = PdfPages(output_url);
    pp.savefig(figure, bbox_inches='tight');
    pp.close();
    plt.close(figure);
    del figure;




def plot_comp_qvalues(qvalue_thres=0.1, psm_level=True):
    # input_url = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/e01306.mzXML';
    # fasta_url = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/fasta/cerevisiae_orf_trans_all_PlusDecoy.fasta';
    # evidences_url = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_tailor/e01306-top1000-chargeAll-prewindow10-mzbinwidth0.02-mzbinoffset0/results_top5/pepid_evidence_scangap2.0_ppm10.txt';
    # # diameter_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_tailor/e01306-top1000-chargeAll-prewindow10-mzbinwidth0.02-mzbinoffset0/results_top5/results_percolator_met1,2,3,6,7,8,14_evidence_scangap2.0_ppm10_sorted_filter1_minmax1_trainfdr0.01_testfdr0.01_aPreP0.0_aMS2P0.0_aInt25.6_aRT0.0_aCorr0.0';
    # diameter_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_tailor/e01306-top1000-chargeAll-prewindow10-mzbinwidth0.02-mzbinoffset0/results_top5/results_percolator_met1,2,3,6,7,8,14_evidence_scangap2.0_ppm10_sorted_filter1_minmax1_trainfdr0.01_testfdr0.01_aPreP0.0_aMS2P0.2_aInt0.4_aRT0.0_aCorr0.8_noPvalue';
    # diameter_psm_url = os.path.join(diameter_output_dir, 'percolator.target.psms.txt');
    # umpire_output_dir = os.path.join(os.path.dirname(input_url), 'results', 'new_umpire_percolator_tdcF_fdr0.01');
    # umpire_psm_url = os.path.join(umpire_output_dir, 'percolator.target.psms.txt');
    # prosit_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_prosit_PlusDecoy/results_percolator';
    # prosit_psm_url = os.path.join(prosit_output_dir, 'percolator.target.psms.txt');
    # pecan_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_walnut_PlusDecoy/results_percolator';
    # pecan_psm_url = os.path.join(pecan_output_dir, 'percolator.target.psms.txt');
    # data_tag='noPvalue';


    # input_url = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2.mzXML';
    # fasta_url = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/fasta/napedro_3mixed_human_yeast_ecoli_20140403_iRT_PlusDecoy.fasta';
    # evidences_url = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2/results_tailor_denoised2/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2-top1000-chargeAll-prewindow12.5-mzbinwidth0.02-mzbinoffset0/results_top5/pepid_evidence_scangap2.0_ppm30.txt';
    # diameter_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2/results_tailor_denoised2/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2-top1000-chargeAll-prewindow12.5-mzbinwidth0.02-mzbinoffset0/results_top5/results_percolator_met1,2,3,6,7,8,14_evidence_scangap2.0_ppm30_sorted_filter1_minmax1_trainfdr0.01_testfdr0.01_aPreP0.0_aMS2P0.2_aInt0.2_aRT0.0_aCorr0.0_noPvalue';
    # diameter_psm_url = os.path.join(diameter_output_dir, 'percolator.target.psms.txt');
    # umpire_output_dir = os.path.join(os.path.dirname(input_url), 'results', 'new_umpire_percolator_tdcF_fdr0.01');
    # umpire_psm_url = os.path.join(umpire_output_dir, 'percolator.target.psms.txt');
    # prosit_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2/results_prosit_PlusDecoy/results_percolator/';
    # prosit_psm_url = os.path.join(prosit_output_dir, 'percolator.target.psms.txt');
    # pecan_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2/results_walnut_PlusDecoy/results_percolator/';
    # pecan_psm_url = os.path.join(pecan_output_dir, 'percolator.target.psms.txt');
    # data_tag = 'noPvalue';


    input_url = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2.mzXML';
    fasta_url = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/fasta/napedro_3mixed_human_yeast_ecoli_20140403_iRT_PlusDecoy.fasta';
    evidences_url = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2/results_tailor/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2-top1000-chargeAll-prewindow12.5-mzbinwidth0.02-mzbinoffset0/results_top5/pepid_evidence_scangap2.0_ppm30.txt';
    diameter_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2/results_tailor/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2-top1000-chargeAll-prewindow12.5-mzbinwidth0.02-mzbinoffset0/results_top5/results_percolator_met1,2,3,6,7,8,14_evidence_scangap2.0_ppm30_sorted_filter1_minmax1_trainfdr0.01_testfdr0.01_aPreP0.0_aMS2P0.2_aInt0.2_aRT0.0_aCorr0.0_noPvalue';
    diameter_psm_url = os.path.join(diameter_output_dir, 'percolator.target.psms.txt');
    umpire_output_dir = os.path.join(os.path.dirname(input_url), 'results', 'new_umpire_percolator_tdcF_fdr0.01');
    umpire_psm_url = os.path.join(umpire_output_dir, 'percolator.target.psms.txt');
    prosit_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2/results_prosit_PlusDecoy/results_percolator/';
    prosit_psm_url = os.path.join(prosit_output_dir, 'percolator.target.psms.txt');
    pecan_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2/results_walnut_PlusDecoy/results_percolator';
    pecan_psm_url = os.path.join(pecan_output_dir, 'percolator.target.psms.txt');
    data_tag = 'noPvalue';


    peptide_pepid_map = read_xcorr_results.get_peptide_pepid_map(fasta_url);
    pepid_label_map = read_xcorr_results.get_pepid_labels(fasta_url);

    def parse_psm_result(psm_url, psm_level):
        if not psm_level: psm_url = psm_url.replace('.psms.txt', '.peptides.txt')

        pepid_list = []; fdr_list = []; qvalue_list = []; target_cnt = 0; decoy_cnt = 0; pep_set = set();
        with open(psm_url, 'rU') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                if row['charge'] == '': charge = 1;
                else: charge = int(row['charge']);

                peptide = row['sequence']; qvalue = float(row['percolator q-value']);
                pepid = peptide_pepid_map[peptide]; real_label = pepid_label_map[pepid];
                # precursor = '{}_{}'.format(pepid, charge);
                # if (not psm_level) and precursor in pep_set: continue;
                # pep_set.add(precursor);

                if real_label > 0:
                    target_cnt += 1;
                    pepid_list.append(pepid);
                    fdr_list.append(decoy_cnt * 1.0 / target_cnt);
                    qvalue_list.append(qvalue);
                else: decoy_cnt += 1;

        for idx in range(len(fdr_list) - 1, 0, -1): fdr_list[idx - 1] = min(fdr_list[idx - 1], fdr_list[idx]);
        pepid_list = np.asarray(pepid_list, dtype=int); fdr_list = np.asarray(fdr_list, dtype=float); qvalue_list = np.asarray(qvalue_list, dtype=float);

        valid_indices = np.where(fdr_list <= qvalue_thres);
        pepid_list = pepid_list[valid_indices]; fdr_list = fdr_list[valid_indices]; qvalue_list = qvalue_list[valid_indices];
        return zip(pepid_list, fdr_list, qvalue_list);


    def parse_peptide_result(psm_url):
        assert '.psms.txt' in psm_url;

        pepid_list = []; fdr_list = []; qvalue_list = []; target_cnt = 0; decoy_cnt = 0; pep_set = set();
        with open(psm_url, 'rU') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                if row['charge'] == '': charge = 1;
                else: charge = int(row['charge']);

                peptide = row['sequence']; qvalue = float(row['percolator q-value']);
                pepid = peptide_pepid_map[peptide]; real_label = pepid_label_map[pepid];
                precursor = '{}_{}'.format(pepid, charge);
                if precursor in pep_set: continue;
                pep_set.add(precursor);

                if real_label > 0:
                    target_cnt += 1;
                    pepid_list.append(pepid);
                    fdr_list.append(decoy_cnt * 1.0 / target_cnt);
                    qvalue_list.append(qvalue);
                else: decoy_cnt += 1;

        for idx in range(len(fdr_list) - 1, 0, -1): fdr_list[idx - 1] = min(fdr_list[idx - 1], fdr_list[idx]);
        pepid_list = np.asarray(pepid_list, dtype=int); fdr_list = np.asarray(fdr_list, dtype=float); qvalue_list = np.asarray(qvalue_list, dtype=float);

        valid_indices = np.where(fdr_list <= qvalue_thres);
        pepid_list = pepid_list[valid_indices]; fdr_list = fdr_list[valid_indices]; qvalue_list = qvalue_list[valid_indices];
        return zip(pepid_list, fdr_list, qvalue_list);


    diameter_qvalueinfo_list = parse_psm_result(diameter_psm_url, psm_level=psm_level); logger.info('diameter_qvalueinfo_list={}'.format(len(diameter_qvalueinfo_list)));
    umpire_qvalueinfo_list = parse_psm_result(umpire_psm_url, psm_level=psm_level); logger.info('umpire_qvalueinfo_list={}'.format(len(umpire_qvalueinfo_list)));

    prosit_qvalueinfo_list = parse_psm_result(prosit_psm_url, psm_level=psm_level); logger.info('prosit_qvalueinfo_list={}'.format(len(prosit_qvalueinfo_list)));
    if (not psm_level) and len(prosit_qvalueinfo_list) <= 0: prosit_qvalueinfo_list = parse_peptide_result(prosit_psm_url);

    pecan_qvalueinfo_list = parse_psm_result(pecan_psm_url, psm_level=psm_level); logger.info('pecan_qvalueinfo_list={}'.format(len(pecan_qvalueinfo_list)));
    if (not psm_level) and len(pecan_qvalueinfo_list) <= 0: pecan_qvalueinfo_list = parse_peptide_result(pecan_psm_url);

    color_list = ['blue', 'red', 'orange', 'green'];
    label_list = ['DIAmeter', 'DIA-Umpire', 'PECAN', 'Prosit+EncyclopeDIA'];
    qvalueinfo_all_list = [diameter_qvalueinfo_list, umpire_qvalueinfo_list, pecan_qvalueinfo_list,  prosit_qvalueinfo_list];

    figure = plt.figure(figsize=(15, 15));
    ax1 = figure.add_subplot(1, 1, 1); ax1.autoscale(); max_detect=-1;
    for color_name, label_name, qvalueinfo_list in zip(color_list, label_list, qvalueinfo_all_list):
        if len(qvalueinfo_list) <= 0: continue; # qvalueinfo_list=[(0,0,0)];
        ax1.plot([x[2] for x in qvalueinfo_list], [x[1] for x in qvalueinfo_list], linestyle='-', color=color_name, linewidth=1.5, label=label_name);
        max_detect = max(max_detect, max([x[1] for x in qvalueinfo_list]), max([x[2] for x in qvalueinfo_list]) );

    ax1.set_xlim(0, qvalue_thres);
    ax1.set_ylim(0, qvalue_thres);
    ax1.plot([0.01, 0.01], [0, qvalue_thres], linestyle='--', linewidth=3, color='black');
    ax1.plot([0, qvalue_thres], [0, qvalue_thres], linestyle='--', linewidth=3, color='black');

    ax1.set_xlabel('Reported q-value by each method', fontsize=25);
    ax1.set_ylabel('Estimated q-value by pseudo-targets', fontsize=25);
    ax1.xaxis.set_tick_params(labelsize=18);
    ax1.yaxis.set_tick_params(labelsize=18);
    ax1.legend(fontsize=28);
    ax1.grid(linestyle=':');

    # plt.show();
    from matplotlib.backends.backend_pdf import PdfPages
    if psm_level: output_url = os.path.join(os.path.dirname(evidences_url), 'fig_comp_qvalues_psm_{}.pdf'.format(data_tag));
    else: output_url = os.path.join(os.path.dirname(evidences_url), 'fig_comp_qvalues_peptide_{}.pdf'.format(data_tag));
    pp = PdfPages(output_url);
    pp.savefig(figure, bbox_inches='tight');
    pp.close();
    plt.close(figure);
    del figure;




def plot_percolator_nonMs1Int_chromatogram(neighbor_scan=5):
    input_url = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/e01306.mzXML';
    fasta_url = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/fasta/cerevisiae_orf_trans_all_PlusDecoy.fasta';
    evidences_url = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_tailor/e01306-top1000-chargeAll-prewindow10-mzbinwidth0.02-mzbinoffset0/results_top5/pepid_evidence_scangap2.0_ppm10.txt';
    percolator_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_tailor/e01306-top1000-chargeAll-prewindow10-mzbinwidth0.02-mzbinoffset0/results_top5/results_percolator_met1,2,3,6,7,8,14_evidence_scangap2.0_ppm10_sorted_filter1_minmax1_trainfdr0.01_testfdr0.01_aPreP0.0_aMS2P0.0_aInt25.6_aRT0.0_aCorr0.0';
    percolator_psm_url = os.path.join(percolator_output_dir, 'percolator.target.psms.txt');
    umpire_percolator_dir = os.path.join(os.path.dirname(input_url), 'results', 'new_umpire_percolator_tdcF_fdr0.01');
    ppm_thres = 10;


    # input_url = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2.mzXML';
    # fasta_url = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/fasta/napedro_3mixed_human_yeast_ecoli_20140403_iRT_PlusDecoy.fasta';
    # evidences_url = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2/results_tailor_denoised2/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2-top1000-chargeAll-prewindow12.5-mzbinwidth0.02-mzbinoffset0/results_top5/pepid_evidence_scangap2.0_ppm30.txt';
    # percolator_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2/results_tailor_denoised2/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2-top1000-chargeAll-prewindow12.5-mzbinwidth0.02-mzbinoffset0/results_top5/results_percolator_met1,2,3,6,7,8,14_evidence_scangap2.0_ppm30_sorted_filter1_minmax1_aPreP0.0_aMS2P0.0_aInt25.6_aRT0.0_aCorr0.0';
    # percolator_psm_url = os.path.join(percolator_output_dir, 'percolator.target.psms.txt');
    # umpire_percolator_dir = os.path.join(os.path.dirname(input_url), 'results', 'new_umpire_percolator_tdcF_fdr0.01');
    # ppm_thres = 30;


    # input_url = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2.mzXML';
    # fasta_url = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/fasta/napedro_3mixed_human_yeast_ecoli_20140403_iRT_PlusDecoy.fasta';
    # evidences_url = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2/results_tailor/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2-top1000-chargeAll-prewindow12.5-mzbinwidth0.02-mzbinoffset0/results_top5/pepid_evidence_scangap2.0_ppm30.txt';
    # percolator_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2/results_tailor/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2-top1000-chargeAll-prewindow12.5-mzbinwidth0.02-mzbinoffset0/results_top5/results_percolator_met1,2,3,6,7,8,14_evidence_scangap2.0_ppm30_sorted_filter1_minmax1_aPreP0.0_aMS2P0.0_aInt25.6_aRT0.0_aCorr0.0';
    # percolator_psm_url = os.path.join(percolator_output_dir, 'percolator.target.psms.txt');
    # umpire_percolator_dir = os.path.join(os.path.dirname(input_url), 'results', 'new_umpire_percolator_tdcF_fdr0.01');
    # ppm_thres = 30;


    peptide_percolator_info_map = diaumpire_utils.load_tidesearch_percolator_info(umpire_percolator_dir, input_url);
    evidence_scan_rt_map = run_matching.get_evidence_scan_rt_map(input_url, evidences_url);

    scannum_arr, rt_arr = mzxml_utils.get_scannum_rt_pair(input_url);
    scannum_rt_map = dict(zip(scannum_arr, rt_arr));

    ms1scan_arr = mzxml_utils.get_ms1_scannums_from_mzxml(input_url);
    ms1rt_arr = np.asarray([scannum_rt_map[x] for x in ms1scan_arr]);

    ms1_scanwins_arr = mzxml_utils.get_ms1_scanwins(input_url);
    scanwin_ms2scan_map = mzxml_utils.get_scanwin_ms2scannum_map_from_mzxml(input_url);
    scanwin_ms2_scan_rt_map = {};
    for scanwin in scanwin_ms2scan_map:
        scanwin_ms2_scan_rt_map[scanwin] = (np.asarray(scanwin_ms2scan_map[scanwin]), np.asarray([scannum_rt_map[x] for x in scanwin_ms2scan_map[scanwin]]));


    percolator_scan_info_map = {}; precursor_set = set(); aa_mass_dict = peptide_utils.getAnimoMassDict(); scan_gap = mzxml_utils.get_scan_gap(input_url);
    with open(percolator_psm_url, 'rU') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t');
        for row in reader:
            scan = int(row['scan']); charge = int(row['charge']); peptide = row['sequence']; qvalue = float(row['percolator q-value']);
            precursor = '{}_{}'.format(peptide, charge); rt = evidence_scan_rt_map[scan];

            if precursor in precursor_set: is_top1 = False; continue;
            else: is_top1 = True;
            precursor_set.add(precursor);

            calc_mz = spectrum_utils.calc_mz_from_neutralmass_charge(peptide_utils.neutral_mass(peptide_utils.getTransformPeptide(peptide), aminoMass=aa_mass_dict), charge);
            scanwin = mzxml_utils.get_ms1_scanwinindices_by_mz(ms1_scanwins_arr, calc_mz)[0];
            # logger.info('calc_mz={}\tscanwin={}'.format(calc_mz, scanwin));

            umpire_qvalue = -1; umpire_apex_ms1scan = -1; umpire_apex_ms2scan = -1;
            if peptide in peptide_percolator_info_map:
                min_qvalue = 100; min_apex_rt = -1;
                for percolator_info in peptide_percolator_info_map[peptide]:
                    charge_se, start_rt_se, end_rt_se, apex_rt_se, qvalue_se, Q_se = percolator_info;
                    if charge == charge_se:
                        if rt >= start_rt_se and rt <= end_rt_se: umpire_qvalue = qvalue_se;
                        if qvalue_se < min_qvalue: min_qvalue = qvalue_se; min_apex_rt = apex_rt_se;

                if min_apex_rt > 0:
                    umpire_apex_ms1scan = ms1scan_arr[np.argmin(np.fabs(ms1rt_arr - min_apex_rt))];
                    ms2scan_arr, ms2rt_arr = scanwin_ms2_scan_rt_map[scanwin];
                    umpire_apex_ms2scan = ms2scan_arr[np.argmin(np.fabs(ms2rt_arr - min_apex_rt))];
                    umpire_qvalue = min_qvalue;
                    # logger.info('min_apex_rt={}\tumpire_apex_ms1scan={}\tumpire_apex_ms2scan={}'.format(min_apex_rt, umpire_apex_ms1scan, umpire_apex_ms2scan));


            if qvalue > 0.01: break;
            assert scan not in percolator_scan_info_map;

            percolator_scan_info_map[scan] = (charge, peptide, qvalue, calc_mz, umpire_qvalue, umpire_apex_ms1scan, umpire_apex_ms2scan);

    logger.info('percolator_scan_info_map={}'.format(len(percolator_scan_info_map)));


    ms1scan_mzs_map = {}; nonMs1Int_info_list = []; pepid_peptide_map = read_xcorr_results.get_pepid_peptide_map(fasta_url);
    with open(evidences_url) as input_csvfile:
        reader = csv.DictReader(input_csvfile, delimiter='\t');
        for rowIdx, row in enumerate(reader):
            scan = int(row['ScanNr']);
            if scan not in percolator_scan_info_map: continue;

            percolator_charge, percolator_peptide, qvalue, calc_mz, umpire_qvalue, umpire_apex_ms1scan, umpire_apex_ms2scan = percolator_scan_info_map[scan];
            charge = int(row['Charge']); pepid = int(row['Pepid']); peptide = pepid_peptide_map[pepid];
            ms1scan = int(row['MS1Scan']); ms2scan = int(row['MS2Scan']); label = int(row['Label']);
            Intensity = float(row['Intensity']); # IntensityM1 = float(row['IntensityM1']); IntensityM2 = float(row['IntensityM2']);
            # logger.info('scan={}\tms1scan={}\tms2scan={}'.format(scan, ms1scan, ms2scan));


            candidate_ms1scan_list = list(range(ms1scan - neighbor_scan * scan_gap, ms1scan + neighbor_scan * scan_gap + 1, scan_gap));
            if umpire_apex_ms1scan > 0:
                alter_ms1scan_list = list( range(umpire_apex_ms1scan - neighbor_scan * scan_gap, umpire_apex_ms1scan + neighbor_scan * scan_gap + 1, scan_gap));
                candidate_ms1scan_list = np.concatenate((candidate_ms1scan_list, alter_ms1scan_list));
            candidate_ms1scan_list = np.unique(candidate_ms1scan_list);

            for candidate_ms1scan in candidate_ms1scan_list:
                if candidate_ms1scan not in ms1scan_mzs_map: ms1scan_mzs_map[candidate_ms1scan] = [];
                ms1scan_mzs_map[candidate_ms1scan].append(calc_mz);


            assert percolator_charge == charge; assert percolator_peptide == peptide;
            if Intensity > 1: continue;
            if label <= 0: continue;

            nonMs1Int_info_list.append((qvalue, peptide, charge, ms1scan, ms2scan, calc_mz, umpire_qvalue, umpire_apex_ms1scan, umpire_apex_ms2scan));
    logger.info('nonMs1Int_info_list={}'.format(len(nonMs1Int_info_list)));
    nonMs1Int_info_list.sort(key=lambda x: x[0]);


    ms2scan_candidate_mzbins_map = {}; peptide_ion_mzbins_map = {};
    for nonMs1Int_info in nonMs1Int_info_list:
        qvalue, peptide, charge, ms1scan, ms2scan, calc_mz, umpire_qvalue, umpire_apex_ms1scan, umpire_apex_ms2scan = nonMs1Int_info; # logger.info('peptide={}'.format(peptide));
        peptide1 = peptide_utils.getTransformPeptide(peptide); assert '[' not in peptide1;

        candidate_ms2scan_list = list(range(ms2scan-neighbor_scan*scan_gap, ms2scan+neighbor_scan*scan_gap+1, scan_gap));
        if umpire_apex_ms2scan > 0:
            alter_ms2scan_list = list(range(umpire_apex_ms2scan - neighbor_scan * scan_gap, umpire_apex_ms2scan + neighbor_scan * scan_gap + 1, scan_gap));
            candidate_ms2scan_list = np.concatenate((candidate_ms2scan_list, alter_ms2scan_list));
        candidate_ms2scan_list = np.unique(candidate_ms2scan_list);

        for candidate_ms2scan in candidate_ms2scan_list:
            if candidate_ms2scan not in ms2scan_candidate_mzbins_map: ms2scan_candidate_mzbins_map[candidate_ms2scan] = [];

        bion_map, yion_map = peptide_utils.fragmentIon(peptide1, charge, aminoMass=aa_mass_dict);
        ion_mzbin_arr = []; ion_label_arr = [];
        for frag_charge in [1, 2]:
            if frag_charge in bion_map:
                bion_mzbin_arr = np.asarray([spectrum_utils.mz_to_binidx(mz) for mz in bion_map[frag_charge]], dtype=int);
                bion_label_arr = ['b{}{}'.format(x+1, '+'*frag_charge) for x in range(len(bion_mzbin_arr))];
                ion_mzbin_arr = np.concatenate((ion_mzbin_arr, bion_mzbin_arr));
                ion_label_arr = np.concatenate((ion_label_arr, bion_label_arr));

            if frag_charge in yion_map:
                yion_mzbin_arr = np.asarray([spectrum_utils.mz_to_binidx(mz) for mz in yion_map[frag_charge]], dtype=int);
                yion_label_arr = ['y{}{}'.format(x+1, '+'*frag_charge) for x in range(len(yion_mzbin_arr))];
                ion_mzbin_arr = np.concatenate((ion_mzbin_arr, yion_mzbin_arr));
                ion_label_arr = np.concatenate((ion_label_arr, yion_label_arr));

        ion_mzbin_arr = np.asarray(ion_mzbin_arr, dtype=int);
        for candidate_ms2scan in candidate_ms2scan_list: ms2scan_candidate_mzbins_map[candidate_ms2scan] = np.concatenate((ms2scan_candidate_mzbins_map[candidate_ms2scan], np.unique(ion_mzbin_arr) ));

        if peptide not in peptide_ion_mzbins_map:  peptide_ion_mzbins_map[peptide] = (ion_mzbin_arr, ion_label_arr, candidate_ms2scan_list) ;
    logger.info('ms2scan_candidate_mzbins_map={}'.format(len(ms2scan_candidate_mzbins_map)));


    ms1scan_mz_intensity_map_url = os.path.join(percolator_output_dir, 'ms1scan_mz_intensity_map.npy');
    ms2scan_mzbin_intensity_map_url = os.path.join(percolator_output_dir, 'ms2scan_mzbin_intensity_map.npy');
    ms2scan_mzbin_intensity_tuple_url = os.path.join(percolator_output_dir, 'ms2scan_mzbin_intensity_tuple.npy');
    if os.path.isfile(ms1scan_mz_intensity_map_url) and os.path.isfile(ms2scan_mzbin_intensity_map_url) and os.path.isfile(ms2scan_mzbin_intensity_tuple_url):
        ms1scan_mz_intensity_map = np.load(ms1scan_mz_intensity_map_url, allow_pickle=True).item();
        ms2scan_mzbin_intensity_map = np.load(ms2scan_mzbin_intensity_map_url, allow_pickle=True).item();
        ms2scan_mzbin_intensity_tuple = np.load(ms2scan_mzbin_intensity_tuple_url, allow_pickle=True).item();
    else:
        ms1scan_mz_intensity_map = {}; ms2scan_mzbin_intensity_map = {}; ms2scan_mzbin_intensity_tuple = {};
        with mzxml.read(input_url) as reader:
            for spectrum in reader:
                scannum = int(spectrum['num']); msLevel = int(spectrum['msLevel']);

                if msLevel == 1 and scannum not in ms1scan_mzs_map: continue;
                if msLevel == 2 and scannum not in ms2scan_candidate_mzbins_map: continue;

                mz_arr = np.asarray(spectrum['m/z array']); intensity_arr = np.asarray(spectrum['intensity array']);
                if len(mz_arr) <= 0: continue;

                if msLevel == 1:

                    ms1scan_mz_intensity_map[scannum] = [];
                    for calc_mz in ms1scan_mzs_map[scannum]:
                        min_idx = np.argmin(np.fabs(mz_arr - calc_mz));
                        ppm_closest = spectrum_utils.calc_ppm(mz_arr[min_idx], calc_mz);
                        # logger.info('scannum={}\tcalc_mz={}\tppm_closest={}'.format(scannum, calc_mz, ppm_closest))
                        if ppm_closest <= ppm_thres: ms1scan_mz_intensity_map[scannum].append((calc_mz, intensity_arr[min_idx]) );

                elif msLevel == 2:

                    mzbin_arr = np.asarray([spectrum_utils.mz_to_binidx(mz) for mz in mz_arr], dtype=int); uniq_mzbin_arr = np.unique(mzbin_arr);
                    if len(uniq_mzbin_arr) != len(mz_arr):
                        indices = np.argsort(intensity_arr);
                        mzbin_intensity_map = dict(zip(mzbin_arr[indices], intensity_arr[indices]));
                        mzbin_arr = uniq_mzbin_arr;
                        intensity_arr = np.asarray([mzbin_intensity_map[mzbin] for mzbin in uniq_mzbin_arr]);

                    ms2scan_mzbin_intensity_map[scannum] = {};
                    candidate_mzbins = np.unique(np.asarray(ms2scan_candidate_mzbins_map[scannum], dtype=int));
                    hit_mzbins, hit_indices, hit_candidate_indices =np.intersect1d(mzbin_arr, candidate_mzbins, return_indices=True);
                    for hit_mzbin, hit_intensity in zip(mzbin_arr[hit_indices], intensity_arr[hit_indices]): ms2scan_mzbin_intensity_map[scannum][hit_mzbin] = hit_intensity;
                    ms2scan_mzbin_intensity_tuple[scannum] = (mzbin_arr, intensity_arr);

        np.save(ms1scan_mz_intensity_map_url, ms1scan_mz_intensity_map);
        np.save(ms2scan_mzbin_intensity_map_url, ms2scan_mzbin_intensity_map);
        np.save(ms2scan_mzbin_intensity_tuple_url, ms2scan_mzbin_intensity_tuple);
    logger.info('ms1scan_mz_intensity_map={}\tms2scan_mzbin_intensity_map={}\tms2scan_mzbin_intensity_tuple={}'.format(len(ms1scan_mz_intensity_map), len(ms2scan_mzbin_intensity_map), len(ms2scan_mzbin_intensity_tuple)));


    fig_idx = 0; diameter_top1_diaumpire_empty = 0; diameter_top1_diaumpire_mismatch = 0;
    for nonMs1Int_info in nonMs1Int_info_list:
        # logger.info('------------------------------------------------------------------------');
        qvalue, peptide, charge, ms1scan, ms2scan, calc_mz, umpire_qvalue, umpire_apex_ms1scan, umpire_apex_ms2scan = nonMs1Int_info;  # logger.info('peptide={}'.format(peptide));
        ion_mzbin_arr, ion_label_arr, _ = peptide_ion_mzbins_map[peptide];
        if ms2scan not in ms2scan_mzbin_intensity_tuple: continue;
        obv_mzbin_arr, obv_intensity_arr = ms2scan_mzbin_intensity_tuple[ms2scan];

        ms1_chrom_list = []; alter_ms1_chrom_list = []; show_umpire=False;
        candidate_ms1scan_list = np.asarray(list(range(ms1scan - neighbor_scan * scan_gap, ms1scan + neighbor_scan * scan_gap + 1, scan_gap)), dtype=int);
        if len(np.where(candidate_ms1scan_list <=0)[0]) > 0: continue;


        for candidate_ms1scan in candidate_ms1scan_list:
            tmp_mz_list = np.asarray([x[0] for x in ms1scan_mz_intensity_map[candidate_ms1scan]]);
            tmp_intensity_list = np.asarray([np.sqrt(x[1]) for x in ms1scan_mz_intensity_map[candidate_ms1scan]]);

            if len(ms1scan_mz_intensity_map[candidate_ms1scan]) <= 0: ms1_chrom_list.append(0);
            else:
                min_idx = np.argmin(np.fabs(tmp_mz_list - calc_mz));
                if np.fabs(tmp_mz_list[min_idx] - calc_mz) < 1e-6: ms1_chrom_list.append(tmp_intensity_list[min_idx]);
                else: ms1_chrom_list.append(0);
        # logger.info('ms1_chrom_list={}'.format(ms1_chrom_list));

        apex_intensity_evidence = ms1_chrom_list[neighbor_scan-1]+ms1_chrom_list[neighbor_scan]+ms1_chrom_list[neighbor_scan+1];
        if apex_intensity_evidence > 0: continue;


        fig_idx += 1;
        if umpire_apex_ms1scan > 0:
            alter_ms1scan_list = np.asarray(list(range(umpire_apex_ms1scan - neighbor_scan * scan_gap, umpire_apex_ms1scan + neighbor_scan * scan_gap + 1,scan_gap)), dtype=int);
            if len(np.where(alter_ms1scan_list <= 0)[0]) > 0: continue;


            for candidate_ms1scan in alter_ms1scan_list:
                tmp_mz_list = np.asarray([x[0] for x in ms1scan_mz_intensity_map[candidate_ms1scan]]);
                tmp_intensity_list = np.asarray([np.sqrt(x[1]) for x in ms1scan_mz_intensity_map[candidate_ms1scan]]);

                if len(ms1scan_mz_intensity_map[candidate_ms1scan]) <= 0: alter_ms1_chrom_list.append(0);
                else:
                    min_idx = np.argmin(np.fabs(tmp_mz_list - calc_mz));
                    if np.fabs(tmp_mz_list[min_idx] - calc_mz) < 1e-6: alter_ms1_chrom_list.append(tmp_intensity_list[min_idx]);
                    else: alter_ms1_chrom_list.append(0);
            logger.info('alter_ms1_chrom_list={}'.format(alter_ms1_chrom_list));


            if np.abs(ms1scan - umpire_apex_ms1scan) > 2* neighbor_scan * scan_gap:
                logger.info('candidate_ms1scan_list={}\talter_ms1scan_list={}'.format(candidate_ms1scan_list, alter_ms1scan_list));
                diameter_top1_diaumpire_mismatch += 1; show_umpire = True;

        else: diameter_top1_diaumpire_empty += 1;


        if show_umpire:
            ### show diameter
            candidate_ms2scan_list = list(range(ms2scan - neighbor_scan * scan_gap, ms2scan + neighbor_scan * scan_gap + 1, scan_gap));

            ion_chrom_map = {};
            for ion_label in ion_label_arr: ion_chrom_map[ion_label] = [];
            for candidate_ms2scan in candidate_ms2scan_list:
                for ion_mbin, ion_label in zip(ion_mzbin_arr, ion_label_arr):
                    if candidate_ms2scan not in ms2scan_mzbin_intensity_map: ion_chrom_map[ion_label].append(0);
                    else: ion_chrom_map[ion_label].append(np.sqrt(ms2scan_mzbin_intensity_map[candidate_ms2scan].get(ion_mbin, 0)));

            fig_spectrum_url = os.path.join(percolator_output_dir, 'fig_spectrum_{}.pdf'.format(fig_idx));
            fig_elution_url = os.path.join(percolator_output_dir, 'fig_elution_{}.pdf'.format(fig_idx));

            hit_ion_labels = viz_utils.plot_binned_spectrum_with_theoretical(obv_mzbin_arr, np.sqrt(obv_intensity_arr), ion_mzbin_arr, ion_label_arr, title='peptide:{}, charge:{}, scan:{}, qvalue:{}'.format(peptide, charge, ms2scan, qvalue), save_url=fig_spectrum_url);
            viz_utils.plot_ion_elution_chromatogram(ms1_chrom_list, candidate_ms1scan_list, ion_chrom_map, candidate_ms2scan_list, hit_ion_labels=hit_ion_labels, title='peptide:{}, charge:{}, scan:{}, qvalue:{}'.format(peptide, charge, ms2scan, qvalue), save_url=fig_elution_url);


            ### show diaumpire
            alter_ms2scan_list = list(range(umpire_apex_ms2scan - neighbor_scan * scan_gap, umpire_apex_ms2scan + neighbor_scan * scan_gap + 1, scan_gap));

            ion_chrom_map = {};
            for ion_label in ion_label_arr: ion_chrom_map[ion_label] = [];
            for candidate_ms2scan in alter_ms2scan_list:
                for ion_mbin, ion_label in zip(ion_mzbin_arr, ion_label_arr):
                    if candidate_ms2scan not in ms2scan_mzbin_intensity_map: ion_chrom_map[ion_label].append(0);
                    else: ion_chrom_map[ion_label].append(np.sqrt(ms2scan_mzbin_intensity_map[candidate_ms2scan].get(ion_mbin, 0)));

            tag = '_umpire_qvalue_{}'.format(umpire_qvalue);
            fig_spectrum_url = os.path.join(percolator_output_dir, 'fig_spectrum_{}{}.pdf'.format(fig_idx, tag));
            fig_elution_url = os.path.join(percolator_output_dir, 'fig_elution_{}{}.pdf'.format(fig_idx, tag));

            obv_mzbin_arr, obv_intensity_arr = ms2scan_mzbin_intensity_tuple[umpire_apex_ms2scan];
            hit_ion_labels = viz_utils.plot_binned_spectrum_with_theoretical(obv_mzbin_arr, np.sqrt(obv_intensity_arr), ion_mzbin_arr, ion_label_arr, title='peptide:{}, charge:{}, scan:{}, qvalue:{}'.format(peptide, charge, ms2scan, qvalue), save_url=fig_spectrum_url);
            viz_utils.plot_ion_elution_chromatogram(alter_ms1_chrom_list, alter_ms1scan_list, ion_chrom_map, alter_ms2scan_list, hit_ion_labels=hit_ion_labels, title='peptide:{}, charge:{}, scan:{}, qvalue:{}'.format(peptide, charge, ms2scan, qvalue), save_url=fig_elution_url);



    logger.info('nonMs1IntCnt={}'.format(fig_idx));
    logger.info('diameter_top1_diaumpire_empty={}'.format(diameter_top1_diaumpire_empty));
    logger.info('diameter_top1_diaumpire_mismatch={}'.format(diameter_top1_diaumpire_mismatch));


def plot_prosit_nonconf_versus_diameter():

    input_url = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/e01306.mzXML';
    fasta_url = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/fasta/cerevisiae_orf_trans_all_PlusDecoy.fasta';
    all_evidences_url = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_tailor/e01306-top1000-chargeAll-prewindow10-mzbinwidth0.02-mzbinoffset0/results_top5/pepid_evidence_scangap2.0_ppm10.txt';

    diameter_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_tailor/e01306-top1000-chargeAll-prewindow10-mzbinwidth0.02-mzbinoffset0/results_top5/results_percolator_met1,2,3,6,7,8,14_evidence_scangap2.0_ppm10_sorted_filter1_minmax1_trainfdr0.01_testfdr0.01_aPreP0.0_aMS2P0.0_aInt25.6_aRT0.0_aCorr0.0';
    diameter_qvalues_url = os.path.join(diameter_output_dir, 'qvalue_new.txt');

    umpire_output_dir = os.path.join(os.path.dirname(input_url), 'results', 'new_umpire_percolator_tdcF_fdr0.01');
    umpire_qvalues_url = os.path.join(umpire_output_dir, 'qvalue_new.txt');

    prosit_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_prosit_PlusDecoy/results_percolator';
    prosit_qvalues_url = os.path.join(prosit_output_dir, 'qvalue_new.txt');

    pecan_output_dir = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_walnut_PlusDecoy/results_percolator';
    pecan_qvalues_url = os.path.join(pecan_output_dir, 'qvalue_new.txt');

    pepid_label_map = read_xcorr_results.get_pepid_labels(fasta_url);
    metric_headers = ['MS2Pvalue1', 'MS2Pvalue2', 'Rtdiff1', 'lnIntensityRank1', 'lnIntensityRank2', 'IntensityIsoCorr1', 'IntensityIsoCorr2', 'IntensityMS1MS2Corr', 'IntensityMS2Corr', 'Tailor'];

    def parse_qvalue_result(qvalues_url):
        qvalue_list = []; pepid_list = [];
        with open(qvalues_url) as fp:
            for cnt, line in enumerate(fp):
                arr = line.rstrip().split('\t');
                qvalue = float(arr[0]);
                pepid = int(arr[-1]);
                if pepid_label_map[pepid] > 0:
                    qvalue_list.append(qvalue);
                    pepid_list.append(pepid);

        qvalue_list[-1] = min(qvalue_list[-1], 1.0);
        for idx in range(len(qvalue_list) - 1, 0, -1): qvalue_list[idx - 1] = min(qvalue_list[idx-1], qvalue_list[idx] );
        logger.info('qvalue_list={}\tpepid_list={}'.format(len(qvalue_list), len(pepid_list)));
        logger.info('max_qvalue={}'.format(np.max(qvalue_list)));
        return np.asarray(qvalue_list, dtype=float), np.asarray(pepid_list, dtype=int);

    def parse_evidences(evidences_url):
        pepid_evidences_map = {};
        with open(evidences_url) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                pepid = int(row['Pepid']);
                metric_values = tuple([float(row[x]) for x in metric_headers]);
                if pepid not in pepid_evidences_map: pepid_evidences_map[pepid] = [];
                pepid_evidences_map[pepid].append(metric_values);
        return pepid_evidences_map;


    pepid_evidences_map_url = os.path.join(os.path.dirname(all_evidences_url), 'pepid_evidences_map.npy');
    if os.path.isfile(pepid_evidences_map_url): pepid_evidences_map = np.load(pepid_evidences_map_url, allow_pickle=True).item();
    else:
        pepid_evidences_map = parse_evidences(all_evidences_url);
        np.save(pepid_evidences_map_url, pepid_evidences_map);
    logger.info('pepid_evidences_map={}'.format(len(pepid_evidences_map)));


    diameter_qvalue_arr, diameter_pepid_arr = parse_qvalue_result(diameter_qvalues_url);
    diameter_indict_arr = [(pepid in pepid_evidences_map) for pepid in diameter_pepid_arr];
    diameter_qvalue_arr = diameter_qvalue_arr[diameter_indict_arr]; diameter_pepid_arr = diameter_pepid_arr[diameter_indict_arr];
    diameter_pepid_filtered_list1 = diameter_pepid_arr[np.where(diameter_qvalue_arr <= 0.01)];
    diameter_pepid_filtered_list2 = diameter_pepid_arr[np.where((diameter_qvalue_arr >= 0.02)&(diameter_qvalue_arr <= 0.2))];
    logger.info('diameter_pepid_filtered_list1={}\tdiameter_pepid_filtered_list2={}'.format(len(diameter_pepid_filtered_list1), len(diameter_pepid_filtered_list2)));

    prosit_qvalue_arr, prosit_pepid_arr = parse_qvalue_result(prosit_qvalues_url);
    prosit_indict_arr = [(pepid in pepid_evidences_map) for pepid in prosit_pepid_arr];
    prosit_qvalue_arr = prosit_qvalue_arr[prosit_indict_arr]; prosit_pepid_arr = prosit_pepid_arr[prosit_indict_arr];
    prosit_pepid_filtered_list1 = prosit_pepid_arr[np.where(prosit_qvalue_arr <= 0.01)];
    prosit_pepid_filtered_list2 = prosit_pepid_arr[np.where((prosit_qvalue_arr >= 0.02)&(prosit_qvalue_arr <= 0.2))];
    logger.info('prosit_pepid_filtered_list1={}\tprosit_pepid_filtered_list2={}'.format(len(prosit_pepid_filtered_list1), len(prosit_pepid_filtered_list2)));

    pecan_qvalue_arr, pecan_pepid_arr = parse_qvalue_result(pecan_qvalues_url);
    pecan_indict_arr = [(pepid in pepid_evidences_map) for pepid in pecan_pepid_arr];
    pecan_qvalue_arr = pecan_qvalue_arr[pecan_indict_arr]; pecan_pepid_arr = pecan_pepid_arr[pecan_indict_arr];
    pecan_pepid_filtered_list1 = pecan_pepid_arr[np.where(pecan_qvalue_arr <= 0.01)];
    pecan_pepid_filtered_list2 = pecan_pepid_arr[np.where((pecan_qvalue_arr >= 0.02)&(pecan_qvalue_arr <= 0.2))];
    logger.info('pecan_pepid_filtered_list1={}\tpecan_pepid_filtered_list2={}'.format(len(pecan_pepid_filtered_list1), len(pecan_pepid_filtered_list2)));


    for metric_idx, metric_name in enumerate(metric_headers):

        diameter_metric_values1 = []; diameter_metric_values2 = [];
        for pepid in diameter_pepid_filtered_list1:
            for pepid_evidence in pepid_evidences_map.get(pepid, []): diameter_metric_values1.append(pepid_evidence[metric_idx]);
        for pepid in diameter_pepid_filtered_list2:
            for pepid_evidence in pepid_evidences_map.get(pepid, []): diameter_metric_values2.append(pepid_evidence[metric_idx]);

        prosit_metric_values1 = []; prosit_metric_values2 = [];
        for pepid in prosit_pepid_filtered_list1:
            for pepid_evidence in pepid_evidences_map.get(pepid, []): prosit_metric_values1.append(pepid_evidence[metric_idx]);
        for pepid in prosit_pepid_filtered_list2:
            for pepid_evidence in pepid_evidences_map.get(pepid, []): prosit_metric_values2.append(pepid_evidence[metric_idx]);

        pecan_metric_values1 = []; pecan_metric_values2 = [];
        for pepid in pecan_pepid_filtered_list1:
            for pepid_evidence in pepid_evidences_map.get(pepid, []): pecan_metric_values1.append(pepid_evidence[metric_idx]);
        for pepid in pecan_pepid_filtered_list2:
            for pepid_evidence in pepid_evidences_map.get(pepid, []): pecan_metric_values2.append(pepid_evidence[metric_idx]);

        min_value = np.min([np.min(diameter_metric_values1), np.min(prosit_metric_values1), np.min(pecan_metric_values1), np.min(diameter_metric_values2), np.min(prosit_metric_values2), np.min(pecan_metric_values2) ]);
        max_value = np.max([np.max(diameter_metric_values1), np.max(prosit_metric_values1), np.max(pecan_metric_values1), np.max(diameter_metric_values2), np.max(prosit_metric_values2), np.max(pecan_metric_values2) ]);

        figure = plt.figure(figsize=(16, 9));

        ax1 = figure.add_subplot(2, 3, 1);
        ax1.hist(diameter_metric_values1, bins=100, range=(min_value, max_value), alpha=0.5, label='diameter_0.01');
        ax1.hist(diameter_metric_values2, bins=100, range=(min_value, max_value), alpha=0.5, label='diameter_0.2');
        ax1.legend(fontsize=20);

        ax1 = figure.add_subplot(2, 3, 2);
        ax1.hist(prosit_metric_values1, bins=100, range=(min_value, max_value), alpha=0.5, label='prosit_0.01');
        ax1.hist(prosit_metric_values2, bins=100, range=(min_value, max_value), alpha=0.5, label='prosit_0.2');
        ax1.legend(fontsize=20);

        ax1 = figure.add_subplot(2, 3, 3);
        ax1.hist(pecan_metric_values1, bins=100, range=(min_value, max_value), alpha=0.5, label='pecan_0.01');
        ax1.hist(pecan_metric_values2, bins=100, range=(min_value, max_value), alpha=0.5, label='pecan_0.2');
        ax1.legend(fontsize=20);

        ax1 = figure.add_subplot(2, 3, 4);
        ax1.hist(diameter_metric_values2, bins=100, range=(min_value, max_value), alpha=0.5, label='diameter_0.2');
        ax1.hist(prosit_metric_values2, bins=100, range=(min_value, max_value), alpha=0.5, label='prosit_0.2');
        ax1.legend(fontsize=20);

        ax1 = figure.add_subplot(2, 3, 5);
        ax1.hist(diameter_metric_values2, bins=100, range=(min_value, max_value), alpha=0.5, label='diameter_0.2');
        ax1.hist(pecan_metric_values2, bins=100, range=(min_value, max_value), alpha=0.5, label='pecan_0.2');
        ax1.legend(fontsize=20);

        ax1 = figure.add_subplot(2, 3, 6);
        ax1.hist(prosit_metric_values2, bins=100, range=(min_value, max_value), alpha=0.5, label='prosit_0.2');
        ax1.hist(pecan_metric_values2, bins=100, range=(min_value, max_value), alpha=0.5, label='pecan_0.2');
        ax1.legend(fontsize=20);

        from matplotlib.backends.backend_pdf import PdfPages
        output_url = os.path.join(os.path.dirname(all_evidences_url), 'fig_diameter_versus_prosit_{}.pdf'.format(metric_name))
        pp = PdfPages(output_url);
        pp.savefig(figure, bbox_inches='tight');
        pp.close();
        plt.close(figure);
        del figure;


def plot_marginal():
    result_url1 = '/media/yanglu/TOSHIBA1/data/dia_search/MetOxYeast/e01306/results_tailor/e01306-top1000-chargeAll-prewindow10-mzbinwidth0.02-mzbinoffset0/results_top5/results_marginal.txt';
    result_url2 = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2/results_tailor_denoised2/HYE124_TTOF5600_32fix_lgillet_L150206_001_cwt2-top1000-chargeAll-prewindow12.5-mzbinwidth0.02-mzbinoffset0/results_top5/results_marginal.txt';
    result_url3 = '/media/yanglu/TOSHIBA1/data/dia_search/Navarro/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2/results_tailor/HYE124_TTOF6600_32fix_lgillet_I150211_002_cwt2-top1000-chargeAll-prewindow12.5-mzbinwidth0.02-mzbinoffset0/results_top5/results_marginal.txt';

    fdr_list = np.asarray(['0.01', '0.05', '0.1', '0.2']);
    dataset_list = ['MetOxYeast_Orbitrap', 'LFQbench_TripTOF5600', 'LFQbench_TripTOF6600']
    marginal_list = np.asarray(['all', 'noXcorr', 'noPvalue', 'noInt', 'noRT', 'noElution', 'onlyXcorr', 'onlyPvalue', 'onlyInt', 'onlyRT', 'onlyElution']);

    def parse_result(result_url):
        key_value_map = {};
        with open(result_url) as fp:
            for cnt, line in enumerate(fp):
                line = line.rstrip();
                if line.startswith('results_percolator'):
                    key_part1 = line[:line.find('evidence')].replace('results_percolator_', '');
                    key_part2 = 'all';
                    for marginal in marginal_list:
                        if marginal in line: key_part2 = marginal; break;

                    if key_part1 not in key_value_map: key_value_map[key_part1] = {};
                    if key_part2 not in key_value_map[key_part1]: key_value_map[key_part1][key_part2] = [];

                elif line.startswith('level=peptide'):
                    arr = line.split('\t');
                    fdr = arr[-2].split('=')[-1];
                    target_cnt = int(arr[-1].split('=')[-1]);
                    if fdr == '0.01': key_value_map[key_part1][key_part2].append(target_cnt);
                else: assert False;

        return key_value_map;

    key_value_map1 = parse_result(result_url1);
    key_value_map2 = parse_result(result_url2);
    key_value_map3 = parse_result(result_url3);


    for key_part1 in key_value_map1:
        type_arr = []; value_arr = []; source_arr = [];

        # dataset_idx = 0;
        # for key_part2 in key_value_map1[key_part1]:
        #     logger.info('dataset={}\tkey_part2={}\tsize={}'.format(dataset_list[dataset_idx], key_part2, len(key_value_map1[key_part1][key_part2]) ));
        #     type_arr = np.concatenate((type_arr, [key_part2] * len(key_value_map1[key_part1][key_part2])));
        #     value_arr = np.concatenate((value_arr, key_value_map1[key_part1][key_part2] ));
        #     source_arr = np.concatenate((source_arr, [dataset_list[dataset_idx]] * len(key_value_map1[key_part1][key_part2])));

        # dataset_idx = 1;
        # for key_part2 in key_value_map2[key_part1]:
        #     logger.info('dataset={}\tkey_part2={}\tsize={}'.format(dataset_list[dataset_idx], key_part2, len(key_value_map2[key_part1][key_part2]) ));
        #     type_arr = np.concatenate((type_arr, [key_part2] * len(key_value_map2[key_part1][key_part2])));
        #     value_arr = np.concatenate((value_arr, key_value_map2[key_part1][key_part2] ));
        #     source_arr = np.concatenate((source_arr, [dataset_list[dataset_idx]] * len(key_value_map2[key_part1][key_part2])));
        #
        dataset_idx = 2;
        for key_part2 in key_value_map3[key_part1]:
            logger.info('dataset={}\tkey_part2={}\tsize={}'.format(dataset_list[dataset_idx], key_part2, len(key_value_map3[key_part1][key_part2]) ));
            type_arr = np.concatenate((type_arr, [key_part2] * len(key_value_map3[key_part1][key_part2])));
            value_arr = np.concatenate((value_arr, key_value_map3[key_part1][key_part2] ));
            source_arr = np.concatenate((source_arr, [dataset_list[dataset_idx]] * len(key_value_map3[key_part1][key_part2])));


        values_pd = pd.DataFrame({'type': type_arr, 'value': value_arr, 'source': source_arr, });
        figure = plt.figure(figsize=(10, 8));
        ax = sns.violinplot(x="source", y="value", hue="type", data=values_pd, palette="Set1", split=False, hue_order=marginal_list);
        ax.set_xlabel('');
        ax.set_ylabel('Detected peptides (1% peptide-level FDR)', fontsize=20);
        # ax.tick_params(axis='both', which='major', labelsize=25);
        ax.xaxis.set_tick_params(labelsize=18);
        ax.yaxis.set_tick_params(labelsize=18);
        ax.legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5) );
        ax.grid(linestyle=':');
        # plt.show();

        from matplotlib.backends.backend_pdf import PdfPages

        result_dir = '/media/yanglu/OS/Users/isaac/Documents/2017_wenruo_dia-matching/doc/paper_jpr/data/';
        output_fig_url = os.path.join(result_dir, 'fig_margin_{}.pdf'.format(dataset_list[dataset_idx]));
        pp = PdfPages(output_fig_url);
        pp.savefig(figure, bbox_inches='tight');
        pp.close();
        plt.close(figure);
        del figure;
