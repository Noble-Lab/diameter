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

import utils.mzxml_utils as mzxml_utils;
import utils.spectrum_utils as spectrum_utils;
import utils.viz_utils as viz_utils;
import utils.debiase_utils as debiase_utils;
import utils.peptide_utils as peptide_utils;

import crux.read_xcorr_results as read_xcorr_results;



def get_bipartite_info(matchingParaDict, peptide_deeprt_map, top_match  ):
    inputFilename = matchingParaDict["inputFile"]; xcorrFilename = matchingParaDict["xcorrFile"];
    assert 'sorted' in xcorrFilename;

    result_dir = os.path.join(os.path.dirname(xcorrFilename), 'results_top{}'.format(top_match));
    if not os.path.exists(result_dir): os.makedirs(result_dir);

    topmatch_bipartite_info_url = os.path.join(result_dir, 'bipartite_info.txt');
    if os.path.isfile(topmatch_bipartite_info_url): return topmatch_bipartite_info_url;

    ms2scannum_scanwin_map = mzxml_utils.get_ms2scannum_scanwin_map_from_mzxml(inputFilename);
    ms2_to_ms1_scannum_map = mzxml_utils.get_ms2_to_ms1_scannum_map(inputFilename);

    pepid_len_map = read_xcorr_results.get_pepid_len_map(matchingParaDict["fastaFile"]);
    pepid_peptide_map = read_xcorr_results.get_pepid_peptide_map(matchingParaDict["fastaFile"]);
    pepid_label_map = read_xcorr_results.get_pepid_labels(matchingParaDict["fastaFile"]);

    obv_rt_arr = np.sort(list(ms2_to_ms1_scannum_map.keys()));
    deeprt_list = np.unique(list(peptide_deeprt_map.values())); pred_bins = 100; pred_rt_arr = np.linspace(np.min(deeprt_list), np.max(deeprt_list), pred_bins);
    obv_rt_quant_map = dict(zip(obv_rt_arr,  np.asarray(list(range(len(obv_rt_arr))), dtype=float) / len(obv_rt_arr)));

    output = open(topmatch_bipartite_info_url, 'w');
    curr_ms2scannum_charge = -1; xcorr_line_list = []; ms2scannum_charge_set = set();
    def deal_with_ms2scannum_charge(curr_ms2scannum_charge, xcorr_line_list):
        assert curr_ms2scannum_charge not in ms2scannum_charge_set;
        ms2scannum_charge_set.add(curr_ms2scannum_charge);

        whitelist_pepid_set = set();
        xcorr_line_list.sort(key=lambda x: -x[0]);

        xcorr_list = np.asarray([x[0] for x in xcorr_line_list]);
        # pvalue_list = np.sort([x[2] for x in xcorr_line_list])[::-1];
        label_list = np.asarray([x[3] for x in xcorr_line_list]);
        pepid_list = np.asarray([x[4] for x in xcorr_line_list]);

        target_indices = np.where(label_list > 0)[0]; decoy_indices = np.where(label_list <= 0)[0];

        for target_idx in target_indices[:top_match]: target_pepid = pepid_list[target_idx]; whitelist_pepid_set.add(target_pepid);
        for decoy_idx in decoy_indices[:top_match]: decoy_pepid = pepid_list[decoy_idx]; whitelist_pepid_set.add(decoy_pepid);

        for xcorr, evalue_log,  pvalue_log,  curr_label, pepid, line in xcorr_line_list:
            arr = line.split('\t');
            if pepid not in whitelist_pepid_set: continue;

            ms2_scannum = int(arr[0]); peptide = pepid_peptide_map[pepid]; peplen = pepid_len_map[pepid];
            ms1_scannum = ms2_to_ms1_scannum_map[ms2_scannum]; scanwin = ms2scannum_scanwin_map[ms2_scannum];

            # if peptide not in peptide_deeprt_map: logger.info('not find deeprt! {}'.format(peptide)); continue;
            if 'U' in peptide or 'O' in peptide: continue;
            obv_rt_quant = obv_rt_quant_map[ms2_scannum];
            pred_rt_quant = np.digitize(peptide_deeprt_map[peptide], bins=pred_rt_arr) * 1.0 / pred_bins;
            rtdiff = np.fabs(obv_rt_quant - pred_rt_quant);

            # pred_rt_quant2 = len(np.where(peptide_deeprt_map[peptide] >= deeprt_list)[0]) * 1.0 / len(deeprt_list);
            # rtdiff2 = np.fabs(obv_rt_quant - pred_rt_quant2);

            xcorr_rank = np.argmin(np.fabs(xcorr_list - xcorr));
            # pvalue_rank = np.argmin(np.fabs(pvalue_list - pvalue_log));
            pvalue_rank = 0;

            output.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(line, ms1_scannum, scanwin, peplen, rtdiff, rtdiff, xcorr_rank, pvalue_rank ));


    with open(xcorrFilename) as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip(); arr = line.split('\t');
            if cnt <= 0: output.write('{}\tms1scan\tscanwin\tpeplen\trtdiff\trtdiff2\txcorr_rank\tpvalue_rank\n'.format(line)); continue;

            ms2_scannum = int(arr[0]); charge = int(arr[2]); xcorr = float(arr[4]); pvalue_log = float(arr[5]); evalue_log = float(arr[6]);

            pepid = int(arr[1]); real_label = pepid_label_map[pepid];
            if real_label >= 0: curr_label = 1;
            else: curr_label = -1;
            arr[3] = str(curr_label);
            line = '\t'.join(arr);

            ms2scannum_charge = '{}_{}'.format(ms2_scannum, charge);
            if curr_ms2scannum_charge != ms2scannum_charge:
                if len(xcorr_line_list) > 0: deal_with_ms2scannum_charge(curr_ms2scannum_charge, xcorr_line_list);
                curr_ms2scannum_charge = ms2scannum_charge; xcorr_line_list = [];
            xcorr_line_list.append((xcorr, evalue_log,  pvalue_log,  curr_label, pepid, line));
        if len(xcorr_line_list) > 0: deal_with_ms2scannum_charge(curr_ms2scannum_charge, xcorr_line_list);

    output.close();

    return topmatch_bipartite_info_url;


def yield_bipartite_info(topmatch_bipartite_info_url):
    topmatch_bipartite_info_url_sortbypepid = topmatch_bipartite_info_url.replace('.txt', '_sorted.txt');
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

    with open(topmatch_bipartite_info_url_sortbypepid) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t');
        for row in reader:
            ms1scannum = int(row["ms1scan"]);
            ms2scannum = int(row["scan"]);
            scanwin = int(row["scanwin"]);
            peplen = int(row["peplen"]);
            rtdiff = float(row["rtdiff"]);
            rtdiff2 = float(row["rtdiff2"]);
            xcorr_rank = int(row["xcorr_rank"]);
            pvalue_rank = int(row["pvalue_rank"]);

            pepid = int(row["pepid"]);
            charge = int(row["charge"]);
            is_target = int(row["is_target"]);
            xcorr = float(row["xcorr"]);
            pvalue_log = float(row["pvalue_log"]);
            evalue_log = float(row["evalue_log"]);
            ExpMass = float(row["ExpMass"]);
            CalcMass = float(row["CalcMass"]);
            DeltCn = float(row["DeltCn"]);
            DeltLcn = float(row["DeltLcn"]);
            NumSP = int(row["NumSP"]);
            FlankAA = row["FlankAA"];
            Proteins = row["Proteins"];


            yield {'is_target':is_target, 'pepid':pepid, 'charge':charge, 'ms1scannum': ms1scannum, 'ms2scannum':ms2scannum,
                   'scanwin':scanwin, 'peplen':peplen, 'rtdiff':rtdiff, 'rtdiff2':rtdiff2, 'xcorr_rank':xcorr_rank, 'pvalue_rank':pvalue_rank,
                   'xcorr':xcorr, 'pvalue_log':pvalue_log, 'evalue_log':evalue_log, 'ExpMass':ExpMass, 'CalcMass':CalcMass,
                   'DeltCn':DeltCn, 'DeltLcn':DeltLcn, 'NumSP':NumSP, 'FlankAA':FlankAA, 'Proteins':Proteins,
            };


edgeid = 0;
aa_mass_dict = peptide_utils.getAnimoMassDict();

def construct_bipartite(matchingParaDict, peptide_deeprt_map, top_match, charge_states=[1,2,3,4,5]):
    inputFilename = matchingParaDict["inputFile"]; xcorrFilename = matchingParaDict["xcorrFile"]; fastaFilename = matchingParaDict["fastaFile"];

    result_dir = os.path.join(os.path.dirname(xcorrFilename), 'results_top{}'.format(top_match));
    if not os.path.exists(result_dir): os.makedirs(result_dir);

    # read_xcorr_results.get_pepid_normalized_predRT(fastaFilename, peptide_deeprt_map);
    xcorr_pepevalues_url = read_xcorr_results.getPeptideEvalue(inputFilename, xcorrFilename);


    # evidences_url = os.path.join(result_dir, 'pepid_evidence_scangap{}_ppm{}.txt'.format(matchingParaDict["rtdiff_thres"], matchingParaDict["ppm"] ));
    # if os.path.isfile(evidences_url): return evidences_url;
    #
    # scan_gap = mzxml_utils.get_scan_gap(inputFilename);
    # pepid_peptide_map = read_xcorr_results.get_pepid_peptide_map(fastaFilename);
    #
    # ms1_slope_intercept_map = mzxml_utils.get_ms1_slope_intercept(inputFilename);
    # ms1_mz_logintensity_map = mzxml_utils.get_ms1_mz_intensity(inputFilename);
    #
    # ms2_mzbin_map = mzxml_utils.get_ms2_mzbin(inputFilename, top=10);
    # ms2_mzbin_map2 = mzxml_utils.get_ms2_mzbin(inputFilename, top=50);
    # ms2_mzbin_sqrtintensity_map = mzxml_utils.get_ms2_mzbin_intensity(inputFilename);
    #
    #
    # ms2scannum_charge_slope_intercept_url = os.path.join(os.path.dirname(xcorrFilename), 'ms2scannum_charge_xcorr_slope_intercept.npy');
    # ms2scannum_charge_slope_intercept_map = np.load(ms2scannum_charge_slope_intercept_url, allow_pickle=True).item();
    #
    # topmatch_bipartite_info_url = get_bipartite_info(matchingParaDict, peptide_deeprt_map, top_match);
    # ms2scan_evalues_map = {}; pepid_evalues_map = {};
    # for row in yield_bipartite_info(topmatch_bipartite_info_url):
    #     ms2scannum = row['ms2scannum']; pepid = row['pepid']; evalue_log = row['evalue_log'];
    #
    #     if ms2scannum not in ms2scan_evalues_map: ms2scan_evalues_map[ms2scannum] = [];
    #     ms2scan_evalues_map[ms2scannum].append(evalue_log);
    #
    #     if pepid not in pepid_evalues_map: pepid_evalues_map[pepid] = [];
    #     pepid_evalues_map[pepid].append(evalue_log);
    #
    # for ms2scannum in ms2scan_evalues_map: ms2scan_evalues_map[ms2scannum] = np.sort(ms2scan_evalues_map[ms2scannum]);
    # for pepid in pepid_evalues_map: pepid_evalues_map[pepid] = np.sort(pepid_evalues_map[pepid]);
    #
    # ### in case for imputation
    # avg_ms1_peakbum = np.mean([len(ms1_mz_logintensity_map[ms1scan][1]) for ms1scan in ms1_mz_logintensity_map]);
    # avg_ms1_intercept = np.mean([ms1_slope_intercept_map[ms1scan][1] for ms1scan in ms1_slope_intercept_map]);
    # logger.info('avg_ms1_peakbum={}\tavg_ms1_intercept={}'.format(avg_ms1_peakbum, avg_ms1_intercept));
    #
    #
    # output = open(evidences_url, 'w');
    # output.write('SpecId\t'                                 # 0. SpecId
    #              'Label\t'                                  # 1. Label
    #              'ScanNr\t'                                 # 2. ScanNr
    #              'Edgeid\t'                                 # 3. Edgeid
    #              'Pepid\t'                                  # 4. Pepid
    #              'MS1Scan\t'                                # 5. MS1Scan
    #              'MS2Scan\t'                                # 6. MS2Scan
    #              'Scanwin\t'                                # 7. Scanwin
    #              'Peplen\t'                                 # 8. Peplen
    #              'Charge\t'                                 # 9. Charge
    #              'Charge1\t'                                # 10. Charge1
    #              'Charge2\t'                                # 11. Charge2
    #              'Charge3\t'                                # 12. Charge3
    #              'Charge4\t'                                # 13. Charge4
    #              'Charge5\t'                                # 14. Charge5
    #              'DeltLcn\t'                                # 15. DeltLcn
    #              'DeltCn\t'                                 # 16. DeltCn
    #              'lnNumSP\t'                                # 17. lnNumSP
    #              'Xcorr\t'                                  # 18. Xcorr
    #              'XcorrRank\t'                              # 19. XcorrRank
    #              'XcorrExp\t'                               # 20. XcorrExp
    #              'dXcorr\t'                                 # 21. dXcorr
    #              'lnPvalue\t'                               # 22. lnPvalue
    #              'lnPvalueRank\t'                           # 23. lnPvalueRank
    #              'lnEvalue\t'                               # 24. lnEvalue
    #              'lnEvalueRank1\t'                          # 25. lnEvalueRank1
    #              'lnEvalueRank2\t'                          # 26. lnEvalueRank2
    #              'Rtdiff1\t'                                # 27. Rtdiff1
    #              'Rtdiff2\t'                                # 28. Rtdiff2
    #              'Intensity\t'                              # 29. Intensity
    #              'IntensityM1\t'                            # 30. IntensityM1
    #              'IntensityM2\t'                            # 31. IntensityM2
    #              'lnIntensityRank1\t'                       # 32. lnIntensityRank1
    #              'lnIntensityRank2\t'                       # 33. lnIntensityRank2
    #              'IntensityIsoCorr1\t'                      # 34. IntensityIsoCorr1
    #              'IntensityIsoCorr2\t'                      # 35. IntensityIsoCorr2
    #              'IntensityMS1MS2Corr\t'                    # 36. IntensityMS1MS2Corr
    #              'IntensityMS2Corr\t'                       # 37. IntensityMS2Corr
    #              'MS2Pvalue1\t'                             # 38. MS2Pvalue1
    #              'MS2Pvalue2\t'                             # 39. MS2Pvalue2
    #              'CC\t'                                     # 40. CC
    #              'TC\t'                                     # 41. TC
    #              'TDCScore\t'                               # 42. TDCScore
    #              'NgbrEdgeids\t'                            # 43. NgbrEdgeids
    #              'Peptide\t'                                # 44. Peptide
    #              'Proteins\n'                               # 45. Proteins
    #              );
    #
    #
    # def deal_with_multi(multiscan_row_list):
    #     global edgeid;
    #
    #     multi_charge = np.unique([row['charge'] for row in multiscan_row_list]);
    #     multi_scan = np.unique([row['ms1scannum'] for row in multiscan_row_list]);
    #     neighbor_edgeids = list(range(edgeid, edgeid + len(multiscan_row_list)));
    #     edgeid += len(multiscan_row_list);
    #
    #     for neighbor_edgeid, row in zip(neighbor_edgeids, multiscan_row_list):
    #         cpepid = '{}_{}'.format(row['pepid'], row['charge']);
    #         ms2scan_charge = '{}_{}'.format(row['ms2scannum'], row['charge']);
    #
    #         peptide = pepid_peptide_map[row['pepid']];
    #         peptide1 = peptide_utils.getTransformPeptide(peptide);
    #         assert '[' not in peptide1;
    #
    #         calc_mz = spectrum_utils.calc_mz_from_neutralmass_charge(peptide_utils.neutral_mass(peptide1, aminoMass=aa_mass_dict), row['charge']);
    #         isotope_mzs = [calc_mz, calc_mz+1.0/row['charge'], calc_mz+2.0/row['charge'] ];
    #
    #         candidate_ms1scannum_list = [row['ms1scannum'], row['ms1scannum']+scan_gap, row['ms1scannum']-scan_gap, row['ms1scannum']+2*scan_gap, row['ms1scannum']-2*scan_gap];
    #         candidate_ms2scannum_list = [row['ms2scannum'], row['ms2scannum']+scan_gap, row['ms2scannum']-scan_gap, row['ms2scannum']+2*scan_gap, row['ms2scannum']-2*scan_gap];
    #
    #         candidate_logintensities = []; candidate_sqrtintensities = []; candidate_intensity_evalues = []; candidate_intensity_pvalues = [];
    #         for candidate_ms1scan in candidate_ms1scannum_list:
    #             if (candidate_ms1scan not in ms1_mz_logintensity_map) or (candidate_ms1scan not in ms1_slope_intercept_map):
    #                 candidate_logintensities.append((0,0,0));
    #                 candidate_sqrtintensities.append((0,0,0));
    #                 candidate_intensity_evalues.append(tuple([avg_ms1_intercept - np.log(40000)]*3));
    #                 candidate_intensity_pvalues.append(tuple([np.log(max(1, avg_ms1_peakbum)) - np.log(40000)]*3));
    #             else:
    #                 ms1_mz_arr, ms1_logintensity_arr = ms1_mz_logintensity_map[candidate_ms1scan];
    #                 slope, intercept = ms1_slope_intercept_map[candidate_ms1scan];
    #
    #                 log_intensities = []; sqrt_intensities = [];
    #                 for isotope_mz in isotope_mzs:
    #                     min_idx = np.argmin(np.fabs(ms1_mz_arr - isotope_mz)); curr_logintensity = 0; curr_sqrtintensity = 0;
    #                     ppm_closest = spectrum_utils.calc_ppm(ms1_mz_arr[min_idx], isotope_mz);
    #                     if ppm_closest <= matchingParaDict["ppm"]: curr_logintensity = ms1_logintensity_arr[min_idx]; curr_sqrtintensity = np.sqrt(np.exp(curr_logintensity));
    #                     log_intensities.append(curr_logintensity); sqrt_intensities.append(curr_sqrtintensity);
    #
    #                 candidate_logintensities.append(tuple(log_intensities)); candidate_sqrtintensities.append(tuple(sqrt_intensities));
    #
    #                 intensity_evalues = np.asarray([(slope * x + intercept) for x in log_intensities]) - np.log(40000);
    #                 intensity_pvalues = np.asarray([np.log(max(1, len(np.where(x <= ms1_logintensity_arr)[0]))) for x in log_intensities]) - np.log(40000);
    #                 candidate_intensity_evalues.append(tuple(intensity_evalues));
    #                 candidate_intensity_pvalues.append(tuple(intensity_pvalues));
    #
    #
    #         bion_map, yion_map = peptide_utils.fragmentIon(peptide1, row['charge'], aminoMass=aa_mass_dict);
    #         frag_mz_arr = [];
    #         for frag_charge in [1]:
    #             if frag_charge in bion_map: frag_mz_arr = np.concatenate((frag_mz_arr, bion_map[frag_charge]));
    #             if frag_charge in yion_map: frag_mz_arr = np.concatenate((frag_mz_arr, yion_map[frag_charge]));
    #         frag_mzbin_arr = np.unique(np.asarray([spectrum_utils.mz_to_binidx(mz) for mz in frag_mz_arr], dtype=int));
    #
    #         ms2_mzbin_arr = ms2_mzbin_map[row['ms2scannum']];
    #         ms2_coverage = 1.0 * len(ms2_mzbin_arr) / (np.max(ms2_mzbin_arr) - np.min(ms2_mzbin_arr) + 1);
    #         overlap_mzbin_arr = np.intersect1d(frag_mzbin_arr, ms2_mzbin_arr);
    #         ms2_pvalue = np.log( 1.0 - np.exp(binom.logcdf(len(overlap_mzbin_arr) - 1, len(frag_mzbin_arr), ms2_coverage)));
    #
    #         ms2_mzbin_arr2 = ms2_mzbin_map2[row['ms2scannum']];
    #         ms2_coverage2 = 1.0 * len(ms2_mzbin_arr2) / (np.max(ms2_mzbin_arr2) - np.min(ms2_mzbin_arr2) + 1);
    #         overlap_mzbin_arr2 = np.intersect1d(frag_mzbin_arr, ms2_mzbin_arr2);
    #         ms2_pvalue2 = np.log( 1.0 - np.exp(binom.logcdf(len(overlap_mzbin_arr2) - 1, len(frag_mzbin_arr), ms2_coverage2)));
    #
    #         candidate_ms2_sqrtintensities = [];
    #         for candidate_ms2scan in candidate_ms2scannum_list:
    #             frag_intensities = np.zeros_like(frag_mzbin_arr);
    #             if candidate_ms2scan in ms2_mzbin_sqrtintensity_map:
    #                 ms2_mzbin_arr, ms2_sqrtintensity_arr = ms2_mzbin_sqrtintensity_map[candidate_ms2scan]; ms2_mzbin_arr = np.asarray(ms2_mzbin_arr, dtype=int);
    #                 overlap_mzbins, frag_mzbin_indices, ms2_mzbin_indices =  np.intersect1d(frag_mzbin_arr, ms2_mzbin_arr, return_indices=True);
    #                 if len(overlap_mzbins) > 0: frag_intensities[frag_mzbin_indices] = ms2_sqrtintensity_arr[ms2_mzbin_indices];
    #             candidate_ms2_sqrtintensities.append(frag_intensities);
    #
    #         ms1_sqrtintensity_M0 = [x[0] for x in candidate_sqrtintensities];
    #         ms1_sqrtintensity_M1 = [x[1] for x in candidate_sqrtintensities];
    #         ms1_sqrtintensity_M2 = [x[2] for x in candidate_sqrtintensities];
    #         intensityIsoCorr1 = spectrum_utils.calc_normalized_innerproduct(ms1_sqrtintensity_M0, ms1_sqrtintensity_M1);
    #         intensityIsoCorr2 = spectrum_utils.calc_normalized_innerproduct(ms1_sqrtintensity_M0, ms1_sqrtintensity_M2);
    #
    #         ms2_sqrtintensity = [np.sum(x) for x in candidate_ms2_sqrtintensities];
    #         ms1ms2_corr = spectrum_utils.calc_normalized_innerproduct(ms1_sqrtintensity_M0, ms2_sqrtintensity);
    #
    #         ms2_corr_list = [spectrum_utils.calc_normalized_innerproduct(candidate_ms2_sqrtintensities[0], x) for x in candidate_ms2_sqrtintensities];
    #         ms2_corr = np.max(ms2_corr_list[1:]);
    #
    #
    #         label = 1; tag = 'target';
    #         if row['is_target'] <= 0: label = -1; tag = 'decoy';
    #         specId = '{}_0_{}_{}_{}'.format(tag, neighbor_edgeid, row['charge'], row['xcorr_rank']);
    #
    #         charge_list = np.zeros(5, dtype=int);
    #         charge_list[row['charge'] - 1] = 1;
    #
    #         xcorr_expected = row['xcorr'];
    #         if ms2scan_charge in ms2scannum_charge_slope_intercept_map:
    #             slope1, intercept1 = ms2scannum_charge_slope_intercept_map[ms2scan_charge];
    #             if slope1 < 0: xcorr_expected = np.exp((np.log(1 + row['xcorr_rank']) - intercept1) / slope1);
    #
    #         evalue_rank1 = len(np.where(row['evalue_log'] <= ms2scan_evalues_map[row['ms2scannum']])[0]);
    #         evalue_rank2 = len(np.where(row['evalue_log'] <= pepid_evalues_map[row['pepid']])[0]);
    #
    #         output.write('{}\t{}\t{}\t{}\t{}\t'
    #                      '{}\t{}\t{}\t{}\t{}\t'
    #                      '{}\t{}\t{}\t{}\t{}\t'
    #                      '{}\t{}\t{}\t{}\t{}\t'
    #                      '{}\t{}\t{}\t{}\t{}\t'
    #                      '{}\t{}\t{}\t{}\t{}\t'
    #                      '{}\t{}\t{}\t{}\t{}\t'
    #                      '{}\t{}\t{}\t{}\t{}\t'
    #                      '{}\t{}\t{}\t{}\t{}\t'
    #                      '{}\n'.format(
    #             specId,                                             # 0. SpecId
    #             label,                                              # 1. Label
    #             neighbor_edgeid,                                    # 2. ScanNr
    #             neighbor_edgeid,                                    # 3. Edgeid
    #             row['pepid'],                                       # 4. Pepid
    #             row['ms1scannum'],                                  # 5. MS1Scan
    #             row['ms2scannum'],                                  # 6. MS2Scan
    #             row['scanwin'],                                     # 7. Scanwin
    #             row['peplen'],                                      # 8. Peplen
    #             row['charge'],                                      # 9. Charge
    #             charge_list[0],                                     # 10. Charge1
    #             charge_list[1],                                     # 11. Charge2
    #             charge_list[2],                                     # 12. Charge3
    #             charge_list[3],                                     # 13. Charge4
    #             charge_list[4],                                     # 14. Charge5
    #             row['DeltLcn'],                                     # 15. DeltLcn
    #             row['DeltCn'],                                      # 16. DeltCn
    #             np.log(row['NumSP']),                               # 17. lnNumSP
    #             row['xcorr'],                                       # 18. Xcorr
    #             row['xcorr_rank'],                                  # 19. XcorrRank
    #             xcorr_expected,                                     # 20. XcorrExp
    #             row['xcorr']-xcorr_expected,                        # 21. dXcorr
    #             row['pvalue_log'],                                  # 22. lnPvalue
    #             row['pvalue_rank'],                                 # 23. lnPvalueRank
    #             row['evalue_log'],                                  # 24. lnEvalue
    #             evalue_rank1,                                       # 25. lnEvalueRank1
    #             evalue_rank2,                                       # 26. lnEvalueRank2
    #             row['rtdiff'],                                      # 27. Rtdiff1
    #             row['rtdiff2'],                                     # 28. Rtdiff2
    #             candidate_logintensities[0][0],                     # 29. Intensity
    #             candidate_logintensities[0][1],                     # 30. IntensityM1
    #             candidate_logintensities[0][2],                     # 31. IntensityM2
    #             np.sum(candidate_intensity_evalues[0]),             # 32. lnIntensityRank1
    #             np.sum(candidate_intensity_pvalues[0]),             # 33. lnIntensityRank2
    #             intensityIsoCorr1,                                  # 34. IntensityIsoCorr1
    #             intensityIsoCorr2,                                  # 35. IntensityIsoCorr2
    #             ms1ms2_corr,                                        # 36. IntensityMS1MS2Corr
    #             ms2_corr,                                           # 37. IntensityMS2Corr
    #             ms2_pvalue,                                         # 38. MS2Pvalue1
    #             ms2_pvalue2,                                        # 39. MS2Pvalue2
    #             np.log(len(multi_charge)),                          # 40. CC
    #             np.log(min(1 + 2 * matchingParaDict["rtdiff_thres"], len(multi_scan))), # 41. TC
    #             0,                                                  # 42 TDCScore
    #             ','.join([str(x) for x in neighbor_edgeids]),       # 43. NgbrEdgeids
    #             '{}.{}.{}'.format(row['FlankAA'][0], peptide, row['FlankAA'][1]),# 44. Peptide
    #             row['Proteins']                                     # 45. Proteins
    #         ));
    #
    #
    # def deal_with_pepid(row_list):
    #     curr_ms1scan = -100; multiscan_row_list = [];
    #     for row in row_list:
    #         assert row['ms1scannum'] >= curr_ms1scan;
    #         if np.abs(curr_ms1scan - row['ms1scannum']) > matchingParaDict["rtdiff_thres"] * scan_gap:
    #             if len(multiscan_row_list) > 0: deal_with_multi(multiscan_row_list);
    #             multiscan_row_list = [];
    #         multiscan_row_list.append(row); curr_ms1scan = row['ms1scannum'];
    #     if len(multiscan_row_list) > 0: deal_with_multi(multiscan_row_list);
    #
    #
    # curr_pepid = -1; row_list = [];
    # for row in yield_bipartite_info(topmatch_bipartite_info_url):
    #     if curr_pepid != row['pepid']:
    #         assert row['pepid'] > curr_pepid;
    #         if len(row_list) > 0: deal_with_pepid(row_list);
    #         curr_pepid = row['pepid']; row_list = [];
    #     row_list.append(row);
    # if len(row_list) > 0: deal_with_pepid(row_list);
    #
    # output.close();
    #
    #
    # evidences_url_sortbyscan = evidences_url.replace('.txt', '_sorted.txt');
    # if not os.path.isfile(evidences_url_sortbyscan):
    #     header_url = evidences_url.replace('.txt', '_tmp1.txt');
    #     body_url = evidences_url.replace('.txt', '_tmp2.txt');
    #
    #     command_line = "head -1 {} > {}".format(evidences_url, header_url);
    #     print(subprocess.check_output(command_line, shell=True));
    #
    #     command_line = "tail -n+2 {} | sort -k7n -k10n -k19nr -o {}".format(evidences_url, body_url);
    #     print(subprocess.check_output(command_line, shell=True));
    #
    #     command_line = "cat {} {} > {}".format(header_url, body_url, evidences_url_sortbyscan);
    #     print(subprocess.check_output(command_line, shell=True));
    #     os.remove(header_url); os.remove(body_url);
    #
    # return evidences_url_sortbyscan;