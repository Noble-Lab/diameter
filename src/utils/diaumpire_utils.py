import os, sys, random, csv, math
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from enum import Enum
import numpy as np
import networkx as nx;
import matplotlib
import matplotlib.pyplot as plt

from pyteomics import mgf, mzxml;
import mzxml_utils, spectrum_utils;
import utils.viz_utils as viz_utils;


def get_diaumpire_result_url(mzxml_url):
    umpire_result_url = '';
    for file in os.listdir(os.path.join(os.path.dirname(mzxml_url), 'results')):
        if 'umpire_ppm' in file: umpire_result_url = os.path.join(os.path.dirname(mzxml_url), 'results', file); break;
    logger.info('umpire_result_url={}'.format(umpire_result_url));
    return umpire_result_url;


def load_peakclust_info(mzxml_url):
    peakClustURL = mzxml_url.replace('.mzXML', '_PeakCluster.csv'); assert os.path.isfile(peakClustURL);
    saved_peakclustID_info_url = os.path.join(os.path.dirname(mzxml_url), 'cache', 'peakclustID_info_rtinmin.npy');
    if os.path.isfile(saved_peakclustID_info_url): peakclustID_info_map = np.load(saved_peakclustID_info_url, allow_pickle=True).item();
    else:
        peakclustID_info_map = {};
        with open(peakClustURL) as input:
            reader = csv.DictReader(input, delimiter=',');
            for row in reader:
                peakclustID = int(row['Cluster_Index']); startRTinmin = float(row['StartRT']); endRTinmin = float(row['EndRT']); mz1 = float(row['mz1']); charge = int(row['Charge']);
                peakclustID_info_map[peakclustID] = (startRTinmin, endRTinmin, mz1, charge);
        # np.save(saved_peakclustID_info_url, peakclustID_info_map);
    logger.info('peakclustID_info_map={}'.format(len(peakclustID_info_map)));
    return peakclustID_info_map;


def load_se_info(mzxml_url, Q_states = [1, 2, 3]):
    saved_peptide_se_url = os.path.join(os.path.dirname(mzxml_url), 'results', 'umpire_peptide_se_info.npy');
    if os.path.isfile(saved_peptide_se_url): peptide_se_info_map = np.load(saved_peptide_se_url, allow_pickle=True).item();
    else:
        peakclustID_info_map = load_peakclust_info(mzxml_url); peptide_se_info_map = {};

        scan_gap = mzxml_utils.get_scan_gap(mzxml_url);
        scannum_arr, rt_arr = mzxml_utils.get_scannum_rt_pair(mzxml_url);
        rt_radius = (rt_arr[scan_gap]-rt_arr[0]) * 5;

        for Q in Q_states:
            file_name, file_extension = os.path.splitext(mzxml_url);
            mgfFilename = '{}_Q{}.mgf'.format(file_name, Q); logger.info(mgfFilename); assert os.path.isfile(mgfFilename);

            if Q in [1,2]:
                qualMapURL = mzxml_url.replace('.mzXML', '.ScanClusterMapping_Q{}'.format(Q)); assert os.path.isfile(qualMapURL);
                mgf_idx_list = [int(line.rstrip('\n').split('_')[0]) for line in open(qualMapURL)];
                peakclust_ID_list = [int(line.rstrip('\n').split('_')[1]) for line in open(qualMapURL)];
                mgfIdx_peakclustID_map = dict(zip(mgf_idx_list, peakclust_ID_list));


            umpire_dirname = '';
            for file in os.listdir(os.path.join(os.path.dirname(mzxml_url), 'results')):
                if 'umpireQ{}'.format(Q) in file: umpire_dirname = file; break;
            if len(umpire_dirname) <= 0: continue;

            mgfidx_peptide_info_map = {};
            crux_umpire_file = os.path.join(os.path.dirname(mzxml_url), 'results', umpire_dirname, "tide-search.txt"); logger.info(crux_umpire_file); assert os.path.isfile(crux_umpire_file);
            with open(crux_umpire_file) as csvfile:
                reader = csv.DictReader(csvfile, delimiter='\t');
                for row in reader:
                    mgf_idx = int(row["scan"]); charge = int(row["charge"]); peptide = row["sequence"]; xcorr = float(row['xcorr score']); mz = float(row['spectrum precursor m/z']);
                    assert mgf_idx not in mgfidx_peptide_info_map;
                    mgfidx_peptide_info_map[mgf_idx] = (peptide, mz, charge, xcorr);


            mgfIdx = 0;
            for row in mgf.read(mgfFilename):
                apex_rtinmin = float(row['params']['rtinseconds']) / 60.0; peptide_mz = float(row['params']['pepmass'][0]); mgfIdx += 1;
                if mgfIdx not in mgfidx_peptide_info_map: continue;

                if Q in [1, 2]:
                    assert mgfIdx in mgfIdx_peakclustID_map;
                    start_rtinmin = float(peakclustID_info_map[mgfIdx_peakclustID_map[mgfIdx]][0]);
                    end_rtinmin = float(peakclustID_info_map[mgfIdx_peakclustID_map[mgfIdx]][1]);
                    mz1 = float(peakclustID_info_map[mgfIdx_peakclustID_map[mgfIdx]][2]);
                    # logger.info('start_rtinmin={}\tend_rtinmin={}\tapex_rtinmin={}'.format(start_rtinmin, end_rtinmin, apex_rtinmin));
                else:
                    start_rtinmin = apex_rtinmin - rt_radius;
                    end_rtinmin = apex_rtinmin + rt_radius;

                peptide, mz, charge, xcorr = mgfidx_peptide_info_map[mgfIdx];
                # logger.info('mz={}\tmz1={}'.format(mz, mz1));

                if peptide not in peptide_se_info_map: peptide_se_info_map[peptide] = [];
                peptide_se_info_map[peptide].append((peptide_mz, charge, start_rtinmin, end_rtinmin, apex_rtinmin, mgfIdx, Q ));

        np.save(saved_peptide_se_url, peptide_se_info_map);
    logger.info('peptide_se_info_map={}'.format(len(peptide_se_info_map)));
    return peptide_se_info_map;


def load_percolator_info(umpire_percolator_dir):
    target_psm_url = os.path.join(umpire_percolator_dir, 'percolator.target.psms.txt'); assert target_psm_url;
    decoy_psm_url = os.path.join(umpire_percolator_dir, 'percolator.decoy.psms.txt'); assert decoy_psm_url;

    qvalue_peptide_info_list = [];
    for psm_url in [target_psm_url, decoy_psm_url]:
        with open(psm_url) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                peptide = row['sequence']; qvalue = float(row['percolator q-value']); scan = int(row['scan']); charge = int(row['charge']);
                qvalue_peptide_info_list.append((qvalue, peptide, charge, scan));
    qvalue_peptide_info_list.sort(key=lambda x: x[0]);
    logger.info('qvalue_peptide_info_list={}'.format(len(qvalue_peptide_info_list)));
    return qvalue_peptide_info_list;


def load_tidesearch_percolator_info(umpire_percolator_dir, mzxml_url):
    saved_peptide_percolator_url = os.path.join(os.path.dirname(mzxml_url), 'results', 'umpire_peptide_percolator_info1.npy');
    if os.path.isfile(saved_peptide_percolator_url): peptide_percolator_info_map = np.load(saved_peptide_percolator_url, allow_pickle=True).item();
    else:
        qvalue_peptide_info_list = load_percolator_info(umpire_percolator_dir);
        peptide_se_info_map = load_se_info(mzxml_url);
        peptide_percolator_info_map = {};

        for qvalue_peptide_info in qvalue_peptide_info_list:
            qvalue, peptide, charge, mgfIdx = qvalue_peptide_info;
            peptide_percolator_info_map[peptide] = [];

            if peptide in peptide_se_info_map:
                for se_info in peptide_se_info_map[peptide]:
                    peptide_mz, charge_se, start_rtinmin, end_rtinmin, apex_rtinmin, mgfIdx_se, Q = se_info;
                    if charge == charge_se and mgfIdx == mgfIdx_se:
                        peptide_percolator_info_map[peptide].append((charge_se, start_rtinmin, end_rtinmin, apex_rtinmin, qvalue, Q, peptide_mz ));

        np.save(saved_peptide_percolator_url, peptide_percolator_info_map);
    logger.info('peptide_percolator_info_map={}'.format(len(peptide_percolator_info_map)));
    return peptide_percolator_info_map;

