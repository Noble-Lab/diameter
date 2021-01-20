import os, sys, random, csv, math, subprocess
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pyteomics import mzxml,mgf,mass
import numpy as np;
import networkx as nx;

import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, binom;
from scipy.special import gammaln, logsumexp;

import utils.mzxml_utils as mzxml_utils;
import utils.spectrum_utils as spectrum_utils;
import utils.viz_utils as viz_utils;
import utils.debiase_utils as debiase_utils;
import utils.peptide_utils as peptide_utils;

import crux.read_xcorr_results as read_xcorr_results;
import read_deeprt_results;


def get_bipartite_info(xcorr_evalues_url, matchingParaDict, top_match, pepid_set  ):
    inputFilename = matchingParaDict["inputFile"];
    assert 'sorted' in xcorr_evalues_url;

    result_dir = os.path.join(os.path.dirname(xcorr_evalues_url), 'results_top{}'.format(top_match));
    if not os.path.exists(result_dir): os.makedirs(result_dir);

    topmatch_bipartite_info_url = os.path.join(result_dir, 'bipartite_info.txt');
    if os.path.isfile(topmatch_bipartite_info_url): return topmatch_bipartite_info_url;

    ms2scannum_scanwin_map = mzxml_utils.get_ms2scannum_scanwin_map_from_mzxml(inputFilename);
    ms2_to_ms1_scannum_map = mzxml_utils.get_ms2_to_ms1_scannum_map(inputFilename);

    pepid_peptide_map = read_xcorr_results.get_pepid_peptide_map(matchingParaDict["fastaFile"], whitelist_pepids=pepid_set);
    pepid_label_map = read_xcorr_results.get_pepid_labels(matchingParaDict["fastaFile"], whitelist_pepids=pepid_set);

    peptide_deeprt_map = read_deeprt_results.read_peptide_deeprt(matchingParaDict["fastaFile"]);

    obv_rt_arr = np.sort(list(ms2_to_ms1_scannum_map.keys()));
    deeprt_list = np.unique(list(peptide_deeprt_map.values())); pred_bins = 200; pred_rt_arr = np.linspace(np.min(deeprt_list), np.max(deeprt_list), pred_bins); del deeprt_list;
    obv_rt_quant_map = dict(zip(obv_rt_arr,  np.asarray(list(range(len(obv_rt_arr))), dtype=float) / len(obv_rt_arr)));

    output = open(topmatch_bipartite_info_url, 'w');
    curr_ms2scannum_charge = -1; xcorr_line_list = []; ms2scannum_charge_set = set();
    def deal_with_ms2scannum_charge(curr_ms2scannum_charge, xcorr_line_list):
        assert curr_ms2scannum_charge not in ms2scannum_charge_set;
        ms2scannum_charge_set.add(curr_ms2scannum_charge);

        xcorr_line_list.sort(key=lambda x: -x[0]);
        xcorr_list = np.asarray([x[0] for x in xcorr_line_list]);

        for xcorr, curr_label, pepid, line in xcorr_line_list[:top_match]:
            arr = line.split('\t');
            # if pepid not in whitelist_pepid_set: continue;

            ms2_scannum = int(arr[0]); peptide = pepid_peptide_map[pepid];
            peplen = len(peptide_utils.getTransformPeptide(peptide)); # peplen = pepid_len_map[pepid];
            ms1_scannum = ms2_to_ms1_scannum_map[ms2_scannum]; scanwin = ms2scannum_scanwin_map[ms2_scannum];

            # if peptide not in peptide_deeprt_map: logger.info('not find deeprt! {}'.format(peptide)); continue;
            if 'U' in peptide or 'O' in peptide: continue;
            obv_rt_quant = obv_rt_quant_map[ms2_scannum];
            pred_rt_quant = np.digitize(peptide_deeprt_map[peptide], bins=pred_rt_arr) * 1.0 / pred_bins;
            rtdiff = np.fabs(obv_rt_quant - pred_rt_quant);

            xcorr_rank = np.argmin(np.fabs(xcorr_list - xcorr));

            output.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(line, ms1_scannum, scanwin, peplen, rtdiff, xcorr_rank ));

    with open(xcorr_evalues_url) as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip(); arr = line.split('\t');
            if cnt <= 0: output.write('{}\tms1scan\tscanwin\tpeplen\trtdiff\txcorr_rank\n'.format(line)); continue;

            ms2_scannum = int(arr[0]); charge = int(arr[2]); xcorr = float(arr[4]);

            pepid = int(arr[1]); real_label = pepid_label_map[pepid];
            if real_label >= 0: curr_label = 1;
            else: curr_label = -1;
            arr[3] = str(curr_label);
            line = '\t'.join(arr);

            ms2scannum_charge = '{}_{}'.format(ms2_scannum, charge);
            if curr_ms2scannum_charge != ms2scannum_charge:
                if len(xcorr_line_list) > 0: deal_with_ms2scannum_charge(curr_ms2scannum_charge, xcorr_line_list);
                curr_ms2scannum_charge = ms2scannum_charge; xcorr_line_list = [];
            xcorr_line_list.append((xcorr, curr_label, pepid, line));
        if len(xcorr_line_list) > 0: deal_with_ms2scannum_charge(curr_ms2scannum_charge, xcorr_line_list);

    output.close();
    del peptide_deeprt_map; del pepid_label_map; del pepid_peptide_map;

    return topmatch_bipartite_info_url;


def yield_bipartite_info(topmatch_bipartite_info_url):
    topmatch_bipartite_info_url_sortbypepid = topmatch_bipartite_info_url.replace('.txt', '_sortbypepid.txt');
    if not os.path.isfile(topmatch_bipartite_info_url_sortbypepid):
        header_url = topmatch_bipartite_info_url.replace('.txt', '_tmp1.txt');
        body_url = topmatch_bipartite_info_url.replace('.txt', '_tmp2.txt');

        command_line = "head -1 {} > {}".format(topmatch_bipartite_info_url, header_url);
        print(subprocess.check_output(command_line, shell=True));

        command_line = "tail -n+2 {} | sort -k2n -k1n -o {}".format(topmatch_bipartite_info_url, body_url);
        print(subprocess.check_output(command_line, shell=True));

        command_line = "cat {} {} > {}".format(header_url, body_url, topmatch_bipartite_info_url_sortbypepid);
        print(subprocess.check_output(command_line, shell=True));
        os.remove(header_url); os.remove(body_url);


    ms2scanlnPvalues_map = {};  ms2scan_mean_lnPvalue_map = {};
    with open(topmatch_bipartite_info_url_sortbypepid) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t');
        for row in reader:
            ms2scan = int(row["scan"]); pvalue_log = float(row["pvalue_log"]);

            if ms2scan not in ms2scanlnPvalues_map: ms2scanlnPvalues_map[ms2scan] = [];
            ms2scanlnPvalues_map[ms2scan].append(pvalue_log);

    for ms2scan in ms2scanlnPvalues_map: ms2scan_mean_lnPvalue_map[ms2scan] = np.mean(ms2scanlnPvalues_map[ms2scan]);


    with open(topmatch_bipartite_info_url_sortbypepid) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t');
        for row in reader:
            ms1scannum = int(row["ms1scan"]);
            ms2scannum = int(row["scan"]);
            scanwin = int(row["scanwin"]);
            peplen = int(row["peplen"]);
            rtdiff = float(row["rtdiff"]);
            xcorr_rank = int(row["xcorr_rank"]);

            pepid = int(row["pepid"]);
            charge = int(row["charge"]);
            is_target = int(row["is_target"]);
            xcorr = float(row["xcorr"]);
            tailor = float(row["tailor"]);

            pvalue_log = float(row["pvalue_log"]);
            pvalue_log_mean = ms2scan_mean_lnPvalue_map[ms2scannum];

            evalue_log_str = row["evalue_log"];
            if evalue_log_str == 'NA': evalue_log = 10;
            else: evalue_log = float(evalue_log_str);

            DeltCn = float(row["DeltCn"]);
            DeltLcn = float(row["DeltLcn"]);
            NumSP = int(row["NumSP"]);
            FlankAA = row["FlankAA"];
            Proteins = row["Proteins"];

            yield {'is_target':is_target, 'pepid':pepid, 'charge':charge, 'ms1scannum': ms1scannum, 'ms2scannum':ms2scannum,
                   'scanwin':scanwin, 'peplen':peplen, 'rtdiff':rtdiff, 'xcorr_rank':xcorr_rank,
                   'xcorr':xcorr, 'tailor':tailor, 'pvalue_log':pvalue_log, 'pvalue_log_mean':pvalue_log_mean, 'evalue_log':evalue_log,
                   'DeltCn':DeltCn, 'DeltLcn':DeltLcn, 'NumSP':NumSP, 'FlankAA':FlankAA, 'Proteins':Proteins,
            };


edgeid = 0;
aa_mass_dict = peptide_utils.getAnimoMassDict();

def construct_bipartite(matchingParaDict, top_match):
    inputFilename = matchingParaDict["inputFile"]; xcorrFilename = matchingParaDict["xcorrFile"]; fastaFilename = matchingParaDict["fastaFile"];

    result_dir = os.path.join(os.path.dirname(xcorrFilename), 'results_top{}'.format(top_match));
    if not os.path.exists(result_dir): os.makedirs(result_dir);


    evidences_url = os.path.join(result_dir, 'pepid_evidence_scangap{}_ppm{}.txt'.format(matchingParaDict["rtdiff_thres"], matchingParaDict["ppm"] ));
    if os.path.isfile(evidences_url): return evidences_url;


    pepid_set = read_xcorr_results.get_whitelist_pepids(xcorrFilename);
    topmatch_bipartite_info_url = get_bipartite_info(xcorrFilename, matchingParaDict, top_match, pepid_set);

    scan_gap = mzxml_utils.get_scan_gap(inputFilename);
    pepid_peptide_map = read_xcorr_results.get_pepid_peptide_map(fastaFilename, whitelist_pepids=pepid_set);

    ms1_slope_intercept_map = mzxml_utils.get_ms1_slope_intercept(inputFilename);
    ms1_mz_logintensity_map = mzxml_utils.get_ms1_mz_intensity(inputFilename);

    ms2_mzbin_map = mzxml_utils.get_ms2_mzbin(inputFilename, top=10);
    # ms2_mzbin_map2 = mzxml_utils.get_ms2_mzbin(inputFilename, top=50);
    ms2_mzbin_sqrtintensity_map = mzxml_utils.get_ms2_mzbin_intensity(inputFilename);


    ### in case for imputation
    avg_ms1_peakbum = np.mean([len(ms1_mz_logintensity_map[ms1scan][1]) for ms1scan in ms1_mz_logintensity_map]);
    avg_ms1_intercept = np.mean([ms1_slope_intercept_map[ms1scan][1] for ms1scan in ms1_slope_intercept_map]);
    logger.info('avg_ms1_peakbum={}\tavg_ms1_intercept={}'.format(avg_ms1_peakbum, avg_ms1_intercept));


    output = open(evidences_url, 'w');
    output.write('SpecId\t'                                 # 0. SpecId
                 'Label\t'                                  # 1. Label
                 'ScanNr\t'                                 # 2. ScanNr
                 'Edgeid\t'                                 # 3. Edgeid
                 'Pepid\t'                                  # 4. Pepid
                 'MS1Scan\t'                                # 5. MS1Scan
                 'MS2Scan\t'                                # 6. MS2Scan
                 'Scanwin\t'                                # 7. Scanwin
                 'Peplen\t'                                 # 8. Peplen
                 'Charge\t'                                 # 9. Charge
                 'Charge1\t'                                # 10. Charge1
                 'Charge2\t'                                # 11. Charge2
                 'Charge3\t'                                # 12. Charge3
                 'Charge4\t'                                # 13. Charge4
                 'Charge5\t'                                # 14. Charge5
                 'DeltLcn\t'                                # 15. DeltLcn
                 'DeltCn\t'                                 # 16. DeltCn
                 'lnNumSP\t'                                # 17. lnNumSP
                 'Xcorr\t'                                  # 18. Xcorr
                 'XcorrRank\t'                              # 19. XcorrRank
                 'Tailor\t'                                 # 20. Tailor
                 'lnPvalue\t'                               # 21. lnPvalue
                 'lnPvalueMean\t'                           # 22. lnPvalueMean
                 'lnEvalue\t'                               # 23. lnEvalue
                 'lnEvalueAccpep\t'                         # 24. lnEvalueAccpep
                 'lnEvalueCoop\t'                           # 25. lnEvalueCoop
                 'Rtdiff1\t'                                # 26. Rtdiff
                 'Intensity\t'                              # 27. Intensity
                 'IntensityM1\t'                            # 28. IntensityM1
                 'IntensityM2\t'                            # 29 IntensityM2
                 'lnIntensityRank1\t'                       # 30. lnIntensityRank1
                 'lnIntensityRank2\t'                       # 31. lnIntensityRank2
                 'IntensityIsoCorr1\t'                      # 32. IntensityIsoCorr1
                 'IntensityIsoCorr2\t'                      # 33. IntensityIsoCorr2
                 'IntensityMS1MS2Corr\t'                    # 34. IntensityMS1MS2Corr
                 'IntensityMS2Corr\t'                       # 35. IntensityMS2Corr
                 'MS2Pvalue1\t'                             # 36. MS2Pvalue1
                 'MS2Pvalue2\t'                             # 37. MS2Pvalue2
                 'CC\t'                                     # 38. CC
                 'TC\t'                                     # 39. TC
                 'NgbrEdgeids\t'                            # 40. NgbrEdgeids
                 'Peptide\t'                                # 41. Peptide
                 'Proteins\n'                               # 42. Proteins
                 );


    def deal_with_multi(multiscan_row_list):
        global edgeid;

        multi_charge = np.asarray([row['charge'] for row in multiscan_row_list]);
        multi_scan = np.asarray([row['ms1scannum'] for row in multiscan_row_list]);
        neighbor_lnEvalues = np.asarray([row['evalue_log'] for row in multiscan_row_list]);

        neighbor_edgeids = list(range(edgeid, edgeid + len(multiscan_row_list)));
        edgeid += len(multiscan_row_list);

        mean_lnEvalue = np.mean(neighbor_lnEvalues);

        for neighbor_edgeid, row in zip(neighbor_edgeids, multiscan_row_list):
            # cpepid = '{}_{}'.format(row['pepid'], row['charge']);
            ms2scan_charge = '{}_{}'.format(row['ms2scannum'], row['charge']);

            peptide = pepid_peptide_map[row['pepid']];
            peptide1 = peptide_utils.getTransformPeptide(peptide);
            assert '[' not in peptide1;

            calc_mz = spectrum_utils.calc_mz_from_neutralmass_charge(peptide_utils.neutral_mass(peptide1, aminoMass=aa_mass_dict), row['charge']);
            isotope_mzs = [calc_mz, calc_mz+1.0/row['charge'], calc_mz+2.0/row['charge'] ];

            candidate_ms1scannum_list = [row['ms1scannum'], row['ms1scannum']+scan_gap, row['ms1scannum']-scan_gap, row['ms1scannum']+2*scan_gap, row['ms1scannum']-2*scan_gap];
            candidate_ms2scannum_list = [row['ms2scannum'], row['ms2scannum']+scan_gap, row['ms2scannum']-scan_gap, row['ms2scannum']+2*scan_gap, row['ms2scannum']-2*scan_gap];

            candidate_logintensities = []; candidate_sqrtintensities = []; candidate_intensity_evalues = []; candidate_intensity_pvalues = [];
            for candidate_ms1scan in candidate_ms1scannum_list:
                if (candidate_ms1scan not in ms1_mz_logintensity_map) or (candidate_ms1scan not in ms1_slope_intercept_map):
                    candidate_logintensities.append((0,0,0));
                    candidate_sqrtintensities.append((0,0,0));
                    candidate_intensity_evalues.append(tuple([avg_ms1_intercept - np.log(40000)]*3));
                    candidate_intensity_pvalues.append(tuple([np.log(max(1, avg_ms1_peakbum)) - np.log(40000)]*3));
                else:
                    ms1_mz_arr, ms1_logintensity_arr = ms1_mz_logintensity_map[candidate_ms1scan];
                    slope, intercept = ms1_slope_intercept_map[candidate_ms1scan];

                    log_intensities = []; sqrt_intensities = [];
                    for isotope_mz in isotope_mzs:
                        min_idx = np.argmin(np.fabs(ms1_mz_arr - isotope_mz)); curr_logintensity = 0; curr_sqrtintensity = 0;
                        ppm_closest = spectrum_utils.calc_ppm(ms1_mz_arr[min_idx], isotope_mz);
                        if ppm_closest <= matchingParaDict["ppm"]: curr_logintensity = ms1_logintensity_arr[min_idx]; curr_sqrtintensity = np.sqrt(np.exp(curr_logintensity));
                        log_intensities.append(curr_logintensity); sqrt_intensities.append(curr_sqrtintensity);

                    candidate_logintensities.append(tuple(log_intensities)); candidate_sqrtintensities.append(tuple(sqrt_intensities));

                    intensity_evalues = np.asarray([(slope * x + intercept) for x in log_intensities]) - np.log(40000);
                    intensity_pvalues = np.asarray([np.log(max(1, len(np.where(x <= ms1_logintensity_arr)[0]))) for x in log_intensities]) - np.log(40000);
                    candidate_intensity_evalues.append(tuple(intensity_evalues));
                    candidate_intensity_pvalues.append(tuple(intensity_pvalues));


            bion_map, yion_map = peptide_utils.fragmentIon(peptide1, row['charge'], aminoMass=aa_mass_dict);
            frag_mz_arr = [];
            for frag_charge in [1]:
                if frag_charge in bion_map: frag_mz_arr = np.concatenate((frag_mz_arr, bion_map[frag_charge]));
                if frag_charge in yion_map: frag_mz_arr = np.concatenate((frag_mz_arr, yion_map[frag_charge]));
            frag_mzbin_arr = np.unique(np.asarray([spectrum_utils.mz_to_binidx(mz) for mz in frag_mz_arr], dtype=int));

            ms2_mzbin_arr = ms2_mzbin_map[row['ms2scannum']];
            ms2_coverage = 1.0 * len(ms2_mzbin_arr) / (np.max(ms2_mzbin_arr) - np.min(ms2_mzbin_arr) + 1);
            overlap_mzbin_arr = np.intersect1d(frag_mzbin_arr, ms2_mzbin_arr);
            ms2_pvalue = np.log( 1.0 - np.exp(binom.logcdf(len(overlap_mzbin_arr) - 1, len(frag_mzbin_arr), ms2_coverage)));

            # ms2_mzbin_arr2 = ms2_mzbin_map2[row['ms2scannum']];
            # ms2_coverage2 = 1.0 * len(ms2_mzbin_arr2) / (np.max(ms2_mzbin_arr2) - np.min(ms2_mzbin_arr2) + 1);
            # overlap_mzbin_arr2 = np.intersect1d(frag_mzbin_arr, ms2_mzbin_arr2);
            # ms2_pvalue2 = np.log( 1.0 - np.exp(binom.logcdf(len(overlap_mzbin_arr2) - 1, len(frag_mzbin_arr), ms2_coverage2)));

            ms2_mzbin_arr, ms2_sqrtintensity_arr = ms2_mzbin_sqrtintensity_map[row['ms2scannum']]; ms2_mzbin_arr = np.asarray(ms2_mzbin_arr, dtype=int);
            ms2_pvalue2 = 0;
            if 1 in bion_map:
                mzbin_arr = np.unique(np.asarray([spectrum_utils.mz_to_binidx(mz) for mz in bion_map[1]], dtype=int));
                overlap_mzbins, indices1, indices2 = np.intersect1d(mzbin_arr, ms2_mzbin_arr, return_indices=True);
                if len(indices2) > 0:
                    ms2_pvalue2 += gammaln(len(indices2)+1);
                    ms2_pvalue2 += np.log(np.sum(ms2_sqrtintensity_arr[indices2]));
            if 1 in yion_map:
                mzbin_arr = np.unique(np.asarray([spectrum_utils.mz_to_binidx(mz) for mz in yion_map[1]], dtype=int));
                overlap_mzbins, indices1, indices2 = np.intersect1d(mzbin_arr, ms2_mzbin_arr, return_indices=True);
                if len(indices2) > 0:
                    ms2_pvalue2 += gammaln(len(indices2)+1);
                    ms2_pvalue2 += np.log(np.sum(ms2_sqrtintensity_arr[indices2]));


            candidate_ms2_sqrtintensities = [];
            for candidate_ms2scan in candidate_ms2scannum_list:
                frag_intensities = np.zeros_like(frag_mzbin_arr);
                if candidate_ms2scan in ms2_mzbin_sqrtintensity_map:
                    ms2_mzbin_arr, ms2_sqrtintensity_arr = ms2_mzbin_sqrtintensity_map[candidate_ms2scan]; ms2_mzbin_arr = np.asarray(ms2_mzbin_arr, dtype=int);
                    overlap_mzbins, frag_mzbin_indices, ms2_mzbin_indices =  np.intersect1d(frag_mzbin_arr, ms2_mzbin_arr, return_indices=True);
                    if len(overlap_mzbins) > 0: frag_intensities[frag_mzbin_indices] = ms2_sqrtintensity_arr[ms2_mzbin_indices];
                candidate_ms2_sqrtintensities.append(frag_intensities);

            ms1_sqrtintensity_M0 = [x[0] for x in candidate_sqrtintensities];
            ms1_sqrtintensity_M1 = [x[1] for x in candidate_sqrtintensities];
            ms1_sqrtintensity_M2 = [x[2] for x in candidate_sqrtintensities];
            intensityIsoCorr1 = spectrum_utils.calc_normalized_innerproduct(ms1_sqrtintensity_M0, ms1_sqrtintensity_M1);
            intensityIsoCorr2 = spectrum_utils.calc_normalized_innerproduct(ms1_sqrtintensity_M0, ms1_sqrtintensity_M2);

            ms2_sqrtintensity = [np.sum(x) for x in candidate_ms2_sqrtintensities];
            ms1ms2_corr = spectrum_utils.calc_normalized_innerproduct(ms1_sqrtintensity_M0, ms2_sqrtintensity);

            ms2_corr_list = [spectrum_utils.calc_normalized_innerproduct(candidate_ms2_sqrtintensities[0], x) for x in candidate_ms2_sqrtintensities];
            ms2_corr = np.max(ms2_corr_list[1:]);

            label = 1; tag = 'target';
            if row['is_target'] <= 0: label = -1; tag = 'decoy';
            specId = '{}_0_{}_{}_{}'.format(tag, neighbor_edgeid, row['charge'], row['xcorr_rank']);

            charge_list = np.zeros(5, dtype=int);
            charge_list[row['charge'] - 1] = 1;

            lnEvalue = row['evalue_log']; lnEvalueAccpep = lnEvalue; lnEvalueCoop = lnEvalue;
            accpepEvalue_indices = np.where(neighbor_lnEvalues <= lnEvalue)[0];
            if len(accpepEvalue_indices) > 0: lnEvalueAccpep = logsumexp(neighbor_lnEvalues[accpepEvalue_indices]);

            ms1scan = row['ms1scannum'];
            coopEvalue_indices = np.where(np.abs(multi_scan - ms1scan) <= scan_gap*matchingParaDict["rtdiff_thres"] );
            assert len(coopEvalue_indices) > 0;
            if len(coopEvalue_indices) <= 1: lnEvalueCoop = logsumexp([lnEvalue, mean_lnEvalue]);
            else: lnEvalueCoop = logsumexp([lnEvalue, np.min(neighbor_lnEvalues[coopEvalue_indices]) ]);

            output.write('{}\t{}\t{}\t{}\t{}\t'
                         '{}\t{}\t{}\t{}\t{}\t'
                         '{}\t{}\t{}\t{}\t{}\t'
                         '{}\t{}\t{}\t{}\t{}\t'
                         '{}\t{}\t{}\t{}\t{}\t'
                         '{}\t{}\t{}\t{}\t{}\t'
                         '{}\t{}\t{}\t{}\t{}\t'
                         '{}\t{}\t{}\t{}\t{}\t'
                         '{}\t{}\t{}\n'.format(
                specId,                                             # 0. SpecId
                label,                                              # 1. Label
                neighbor_edgeid,                                    # 2. ScanNr
                neighbor_edgeid,                                    # 3. Edgeid
                row['pepid'],                                       # 4. Pepid
                row['ms1scannum'],                                  # 5. MS1Scan
                row['ms2scannum'],                                  # 6. MS2Scan
                row['scanwin'],                                     # 7. Scanwin
                row['peplen'],                                      # 8. Peplen
                row['charge'],                                      # 9. Charge
                charge_list[0],                                     # 10. Charge1
                charge_list[1],                                     # 11. Charge2
                charge_list[2],                                     # 12. Charge3
                charge_list[3],                                     # 13. Charge4
                charge_list[4],                                     # 14. Charge5
                row['DeltLcn'],                                     # 15. DeltLcn
                row['DeltCn'],                                      # 16. DeltCn
                np.log(row['NumSP']),                               # 17. lnNumSP
                row['xcorr'],                                       # 18. Xcorr
                row['xcorr_rank'],                                  # 19. XcorrRank
                row['tailor'],                                      # 20. Tailor
                row['pvalue_log'],                                  # 21. lnPvalue
                row['pvalue_log_mean'],                             # 22. lnPvalueMean
                lnEvalue,                                           # 23. lnEvalue
                lnEvalueAccpep,                                     # 24. lnEvalueAccpep
                lnEvalueCoop,                                       # 25. lnEvalueCoop
                row['rtdiff'],                                      # 26. Rtdiff
                candidate_logintensities[0][0],                     # 27. Intensity
                candidate_logintensities[0][1],                     # 28. IntensityM1
                candidate_logintensities[0][2],                     # 29. IntensityM2
                np.sum(candidate_intensity_evalues[0]),             # 30. lnIntensityRank1
                np.sum(candidate_intensity_pvalues[0]),             # 31. lnIntensityRank2
                intensityIsoCorr1,                                  # 32. IntensityIsoCorr1
                intensityIsoCorr2,                                  # 33. IntensityIsoCorr2
                ms1ms2_corr,                                        # 34. IntensityMS1MS2Corr
                ms2_corr,                                           # 35. IntensityMS2Corr
                ms2_pvalue,                                         # 36. MS2Pvalue1
                ms2_pvalue2,                                        # 37. MS2Pvalue2
                np.log(len(multi_charge)),                          # 38. CC
                np.log(min(1 + 2 * matchingParaDict["rtdiff_thres"], len(multi_scan))), # 39. TC
                ','.join([str(x) for x in neighbor_edgeids]),       # 40. NgbrEdgeids
                '{}.{}.{}'.format(row['FlankAA'][0], peptide, row['FlankAA'][1]),# 41. Peptide
                row['Proteins']                                     # 42. Proteins
            ));


    def deal_with_pepid(row_list):
        curr_ms1scan = -100; multiscan_row_list = [];
        for row in row_list:
            assert row['ms1scannum'] >= curr_ms1scan;
            if np.abs(curr_ms1scan - row['ms1scannum']) > matchingParaDict["rtdiff_thres"] * scan_gap:
                if len(multiscan_row_list) > 0: deal_with_multi(multiscan_row_list);
                multiscan_row_list = [];
            multiscan_row_list.append(row); curr_ms1scan = row['ms1scannum'];
        if len(multiscan_row_list) > 0: deal_with_multi(multiscan_row_list);


    curr_pepid = -1; row_list = [];
    for row in yield_bipartite_info(topmatch_bipartite_info_url):
        if curr_pepid != row['pepid']:
            assert row['pepid'] > curr_pepid;
            if len(row_list) > 0: deal_with_pepid(row_list);
            curr_pepid = row['pepid']; row_list = [];
        row_list.append(row);
    if len(row_list) > 0: deal_with_pepid(row_list);

    output.close();


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

    return evidences_url_sortbyscan;