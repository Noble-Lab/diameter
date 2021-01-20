
import os, sys, random, csv, math, time
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import shutil
import subprocess
import numpy as np
from pyteomics import mzxml, mgf, mass
import matplotlib.pyplot as plt
from scipy.special import logsumexp

import run_crux;
import utils.mzxml_utils as mzxml_utils;
import utils.spectrum_utils as spectrum_utils;
import utils.peptide_utils as peptide_utils;
import utils.viz_utils as viz_utils

import algorithms.read_deeprt_results as read_deeprt_results;


def getStandardXcorrOutput(cruxParaDict, charge_states = [1, 2, 3, 4, 5], overwrite=False, useTailor=1):
    inputfilenames = [];
    for charge in charge_states:
        cruxParaDict['charge'] = str(charge);
        inputfile = os.path.join(run_crux.getCruxOutputName(cruxParaDict), "tide-search.txt"); # assert os.path.isfile(inputfile);
        inputfilenames.append(inputfile);

    cruxParaDict['charge'] = 'All';
    outputfoldername = run_crux.getCruxOutputName(cruxParaDict);
    if not os.path.exists(outputfoldername): os.makedirs(outputfoldername);

    outputfilename = os.path.join(outputfoldername, "xcorr.txt");
    sorted_outputfilename = outputfilename.replace('.txt', '_sorted.txt');
    if os.path.isfile(sorted_outputfilename) and not overwrite: return sorted_outputfilename;

    if not os.path.isfile(outputfilename):
        logger.info('inputfile={}'.format(cruxParaDict["inputFile"]));
        scannum_list, rt_list = mzxml_utils.get_scannum_rt_pair(cruxParaDict["inputFile"]);
        scan_rt_map = dict(zip(scannum_list, rt_list));

        for tidefilename in inputfilenames:
            logger.info('parsing: {}'.format(tidefilename));
            outputfilename_local = os.path.join(os.path.dirname(tidefilename), "xcorr.txt");
            if os.path.isfile(outputfilename_local): continue;
            logger.info('outputfilename_local={}'.format(outputfilename_local));

            output = open(outputfilename_local, 'w');
            with open(tidefilename) as csvfile:
                reader = csv.DictReader(csvfile, delimiter='\t');
                for row in reader:
                    rt = -1;
                    if int(row["scan"]) in scan_rt_map: rt = scan_rt_map[int(row["scan"])];

                    is_target = 1;
                    if "decoy_" in row["protein id"] or "rev_" in row["protein id"]: is_target = 0;

                    scan=row["scan"];
                    sequence=row["sequence"];
                    charge = row["charge"];

                    # mz = row["spectrum precursor m/z"];
                    # neutral_mass = row["spectrum neutral mass"];
                    # peptide_mass = row["peptide mass"];
                    delta_cn = row["delta_cn"];
                    delta_lcn = row["delta_lcn"];
                    distinct_match = row["distinct matches/spectrum"];
                    xcorr = row["xcorr score"];

                    tailor = 0;
                    if useTailor > 0: tailor = row["tailor score"];

                    flank_aa = row["flanking aa"];
                    proteinid = row["protein id"];

                    proteinid_arr = proteinid.split(',');
                    proteinid_arr_refined = [];
                    for protein in proteinid_arr:
                        if '(' in protein: proteinid_arr_refined.append(protein[:protein.find('(')]);
                    proteinid = ','.join(proteinid_arr_refined);

                    output.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(scan, sequence, charge, is_target, xcorr, tailor, delta_cn, delta_lcn, rt, distinct_match, flank_aa, proteinid ));
            output.close();

        output = open(outputfilename, 'w');
        output.write("scan\tsequence\tcharge\tis_target\txcorr\ttailor\tDeltCn\tDeltLcn\tRT\tNumSP\tFlankAA\tProteins\n");
        for tidefilename in inputfilenames:
            outputfilename_local = os.path.join(os.path.dirname(tidefilename), "xcorr.txt"); assert os.path.isfile(outputfilename_local);
            with open(outputfilename_local) as input:
                for line in input.readlines():
                    output.write(line);
        output.close();

    command_line = "sort {} -k1n -k3n -o {}".format(outputfilename, sorted_outputfilename);
    print(subprocess.check_output(command_line, shell=True));

    return sorted_outputfilename;


def getStandardXcorrOutput_var(cruxParaDict, charge_states = [1, 2, 3, 4, 5], overwrite=False, useTailor=1, denoisePeak=0):

    scanwin_ms2scannum_map = mzxml_utils.get_scanwin_ms2scannum_map_from_mzxml(cruxParaDict["inputFile"]);

    inputfilenames = [];
    for scanwin in scanwin_ms2scannum_map:
        for charge in charge_states:
            if denoisePeak == 0: xcorr_dir = os.path.join(cruxParaDict["outputFolder"], 'tmp', '{}-top{}-charge{}-scanwin{}'.format(cruxParaDict["label"], cruxParaDict["topMatch"], charge, scanwin));
            else: xcorr_dir = os.path.join(cruxParaDict["outputFolder"], 'tmp', '{}-denoised{}-top{}-charge{}-scanwin{}'.format(cruxParaDict["label"], denoisePeak, cruxParaDict["topMatch"], charge, scanwin));

            tidefile = os.path.join(xcorr_dir, "tide-search.txt"); assert os.path.isfile(tidefile);
            inputfilenames.append(tidefile);
    logger.info('inputfilenames={}'.format(len(inputfilenames)));

    cruxParaDict['charge'] = 'All';
    outputfoldername = run_crux.getCruxOutputName(cruxParaDict);
    if not os.path.exists(outputfoldername): os.makedirs(outputfoldername);

    outputfilename = os.path.join(outputfoldername, "xcorr.txt");
    sorted_outputfilename = outputfilename.replace('.txt', '_sorted.txt');
    if os.path.isfile(sorted_outputfilename) and not overwrite: return sorted_outputfilename;

    logger.info('inputfile={}'.format(cruxParaDict["inputFile"]));
    scannum_list, rt_list = mzxml_utils.get_scannum_rt_pair(cruxParaDict["inputFile"]);
    scan_rt_map = dict(zip(scannum_list, rt_list));

    output = open(outputfilename, 'w');
    output.write("scan\tsequence\tcharge\tis_target\txcorr\ttailor\tDeltCn\tDeltLcn\tRT\tNumSP\tFlankAA\tProteins\n");

    for tidefilename in inputfilenames:
        logger.info('parsing: {}'.format(tidefilename));

        with open(tidefilename) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                rt = -1;
                if int(row["scan"]) in scan_rt_map: rt = scan_rt_map[int(row["scan"])];

                is_target = 1;
                if "decoy_" in row["protein id"] or "rev_" in row["protein id"]: is_target = 0;

                scan=row["scan"];
                sequence=row["sequence"];
                charge = row["charge"];

                # mz = row["spectrum precursor m/z"];
                # neutral_mass = row["spectrum neutral mass"];
                # peptide_mass = row["peptide mass"];
                delta_cn = row["delta_cn"];
                delta_lcn = row["delta_lcn"];
                distinct_match = row["distinct matches/spectrum"];
                xcorr = row["xcorr score"];

                tailor = 0;
                if useTailor > 0: tailor = row["tailor score"];

                flank_aa = row["flanking aa"];
                proteinid = row["protein id"];

                proteinid_arr = proteinid.split(',');
                proteinid_arr_refined = [];
                for protein in proteinid_arr:
                    if '(' in protein: proteinid_arr_refined.append(protein[:protein.find('(')]);
                proteinid = ','.join(proteinid_arr_refined);

                output.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(scan, sequence, charge, is_target, xcorr, tailor, delta_cn, delta_lcn, rt, distinct_match, flank_aa, proteinid ));

    output.close();

    command_line = "sort {} -k1n -k3n -o {}".format(outputfilename, sorted_outputfilename);
    print(subprocess.check_output(command_line, shell=True));

    return sorted_outputfilename;


def fit_log_linear(scores, ignore_top = 20, ignore_bottom=0, fixed_slope=0):
    x_values = np.asarray(sorted(scores, reverse=True));
    y_values = np.log(np.asarray(range(len(x_values))) + 1 );

    if fixed_slope >= 0:
        from sklearn import linear_model;
        regr = linear_model.LinearRegression();
        regr.fit(x_values[ignore_top:(len(x_values)-ignore_bottom)].reshape(-1, 1), y_values[ignore_top:(len(x_values)-ignore_bottom)].reshape(-1, 1));
        slope = regr.coef_[0]; intercept = regr.intercept_;
    else:
        slope = fixed_slope;
        intercept = np.mean(y_values[ignore_top:(len(x_values)-ignore_bottom)] - x_values[ignore_top:(len(x_values)-ignore_bottom)] * slope);

    return slope, intercept, x_values, y_values;


def getStandardEvalueOutput(cruxParaDict):
    cruxParaDict['charge'] = 'All';
    outputfoldername = run_crux.getCruxOutputName(cruxParaDict);
    xcorrFilename = os.path.join(outputfoldername, "xcorr_sorted.txt"); logger.info('xcorrFilename={}'.format(xcorrFilename)); assert os.path.isfile(xcorrFilename);

    ### calculate the evalue on cpepid
    cpepid_xcorr_url = os.path.join(os.path.dirname(xcorrFilename), 'cpepid_xcorr.txt');
    sorted_cpepid_xcorr_url = cpepid_xcorr_url.replace('.txt', '_sorted.txt');
    if not os.path.isfile(sorted_cpepid_xcorr_url):
        if not os.path.isfile(cpepid_xcorr_url):
            output = open(cpepid_xcorr_url, 'w');
            peptide_pepid_map = get_peptide_pepid_map(cruxParaDict["fastaFile"]);
            with open(xcorrFilename) as csvfile:
                reader = csv.DictReader(csvfile, delimiter='\t');
                for row in reader:
                    peptide = row["sequence"];  charge = row["charge"];  xcorr = row["xcorr"];
                    pepid = peptide_pepid_map[peptide]; cpepid = '{}_{}'.format(pepid, charge);
                    output.write('{}\t{}\n'.format(cpepid, xcorr));
            output.close();

        command_line = "sort {} -k1 -o {}".format(cpepid_xcorr_url, sorted_cpepid_xcorr_url);
        print(subprocess.check_output(command_line, shell=True));
        os.remove(cpepid_xcorr_url);

    cpepid_xcorr_pvalues_url = os.path.join(os.path.dirname(xcorrFilename), 'cpepid_xcorr_pvalues.txt');
    if not os.path.isfile(cpepid_xcorr_pvalues_url):
        output = open(cpepid_xcorr_pvalues_url, 'w');

        with open(sorted_cpepid_xcorr_url) as input:
            curr_cpepid = ''; xcorr_list = [];
            for cnt, line in enumerate(input):
                arr = line.rstrip().split('\t'); cpepid = arr[0]; xcorr = float(arr[1]);
                if curr_cpepid != cpepid:
                    if len(xcorr_list) > 0:
                        if len(xcorr_list) > 1: output.write('{}\t{}\n'.format(curr_cpepid, ','.join([str(x) for x in np.sort(xcorr_list)[::-1] ]) ));
                    curr_cpepid = cpepid; xcorr_list = [];
                xcorr_list.append(xcorr);
            if len(xcorr_list) > 0 and len(xcorr_list) > 1: output.write( '{}\t{}\n'.format(curr_cpepid, ','.join([str(x) for x in np.sort(xcorr_list)[::-1]])));
        output.close();


    ms2scannum_charge_slope_intercept_url = os.path.join(os.path.dirname(xcorrFilename), 'ms2scannum_charge_xcorr_slope_intercept.npy');
    if os.path.isfile(ms2scannum_charge_slope_intercept_url): ms2scannum_charge_slope_intercept_map = np.load(ms2scannum_charge_slope_intercept_url, allow_pickle=True).item();
    else:
        ms2scannum_charge_slope_intercept_map = {};
        def deal_with_ms2scannum_charge(curr_ms2scannum_charge, xcorr_list, curr_candidatematches):
            assert curr_ms2scannum_charge not in ms2scannum_charge_slope_intercept_map;
            if len(xcorr_list) <= 500: return ;
            logger.info('ms2scannum_charge={}\txcorr_list={}\tcurr_candidatematches={}'.format(curr_ms2scannum_charge, len(xcorr_list), curr_candidatematches));

            ignore_top = 25; retain_cnt = min(1000, int((curr_candidatematches - ignore_top) * 0.2));
            slope, intercept, x_values, y_values = fit_log_linear(xcorr_list, ignore_top=ignore_top, ignore_bottom=max(len(xcorr_list) - retain_cnt - ignore_top, 200));
            ms2scannum_charge_slope_intercept_map[curr_ms2scannum_charge] = (slope[0], intercept[0]);
            logger.info('ms2scannum_charge={}\tslope={}\tintercept={}'.format(curr_ms2scannum_charge, slope, intercept));

        curr_ms2scannum_charge = -1; curr_candidatematches = -1; xcorr_list = [];
        with open(xcorrFilename) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                ms2scannum_charge = '{}_{}'.format(row["scan"], row["charge"]); xcorr = float(row["xcorr"]); candidatematches = int(row["NumSP"]);
                if curr_ms2scannum_charge != ms2scannum_charge:
                    if len(xcorr_list) > 0: deal_with_ms2scannum_charge(curr_ms2scannum_charge, xcorr_list, curr_candidatematches);
                    curr_ms2scannum_charge = ms2scannum_charge; xcorr_list = []; curr_candidatematches = candidatematches;
                xcorr_list.append(xcorr);
            if len(xcorr_list) > 0: deal_with_ms2scannum_charge(curr_ms2scannum_charge, xcorr_list, curr_candidatematches);

        np.save(ms2scannum_charge_slope_intercept_url, ms2scannum_charge_slope_intercept_map);
    logger.info('ms2scannum_charge_slope_intercept_map={}'.format(len(ms2scannum_charge_slope_intercept_map)));


    xcorr_evalues_url = os.path.join(os.path.dirname(xcorrFilename), 'xcorr_evalues_sorted.txt');
    if not os.path.isfile(xcorr_evalues_url):
        f = open(xcorr_evalues_url, 'w');
        f.write('scan\tpepid\tcharge\tis_target\txcorr\ttailor\tpvalue_log\tevalue_log\tDeltCn\tDeltLcn\tRT\tNumSP\tFlankAA\tProteins\n');

        peptide_pepid_map = get_peptide_pepid_map(cruxParaDict["fastaFile"]);
        cpepid_xcorr_pvalues_map = {};
        with open(cpepid_xcorr_pvalues_url) as input:
            for cnt, line in enumerate(input):
                arr = line.rstrip().split('\t'); cpepid = arr[0]; xcorr_list = np.asarray([float(x) for x in arr[1].split(',')]);
                if len(xcorr_list) <= 1: continue;
                cpepid_xcorr_pvalues_map[cpepid] = xcorr_list;
        logger.info('cpepid_xcorr_pvalues_map={}'.format(len(cpepid_xcorr_pvalues_map)));

        scanwin_ms2scannum_map = mzxml_utils.get_scanwin_ms2scannum_map_from_mzxml(cruxParaDict["inputFile"]);
        ms1_size = len(scanwin_ms2scannum_map[0]); logger.info('ms1_size={}'.format(ms1_size));

        with open(xcorrFilename) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                ms2scannum = int(row["scan"]);
                peptide = row["sequence"];
                charge = int(row["charge"]);
                is_target = int(row["is_target"]);
                xcorr = float(row["xcorr"]);
                tailor = float(row["tailor"]);

                ms2scannum_charge = '{}_{}'.format(ms2scannum, charge);

                pepid = peptide_pepid_map[peptide]; cpepid = '{}_{}'.format(pepid, charge);
                if cpepid not in cpepid_xcorr_pvalues_map: pvalue_log = -np.log(ms1_size);
                else:
                    xcorr_list = cpepid_xcorr_pvalues_map[cpepid];
                    pvalue_log = np.log(max(1, len(np.where(xcorr >= xcorr_list)[0]))) - np.log(ms1_size);

                evalue_log = 'NA';
                if ms2scannum_charge in ms2scannum_charge_slope_intercept_map:
                    slope1, intercept1 = ms2scannum_charge_slope_intercept_map[ms2scannum_charge];
                    evalue_log = xcorr * slope1 + intercept1;

                f.write('{}\t{}\t{}\t{}\t{:.6f}\t{:.6f}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    ms2scannum, pepid, charge, is_target, xcorr, tailor, pvalue_log, evalue_log,
                    row["DeltCn"], row["DeltLcn"], row["RT"],row["NumSP"], row["FlankAA"], row["Proteins"]
                ));
        f.close();


def getPeptideEvalue(inputFilename, xcorr_evalues_url):
    xcorr_pepevalues_url = os.path.join(os.path.dirname(xcorr_evalues_url), 'xcorr_pepevalues_sorted.txt');
    if os.path.isfile(xcorr_pepevalues_url): return xcorr_pepevalues_url;

    xcorr_evalues_url_sortbypepid = xcorr_evalues_url.replace('_sorted.txt', '_sortbypepid.txt');
    if not os.path.isfile(xcorr_evalues_url_sortbypepid):
        header_url = xcorr_evalues_url.replace('.txt', '_tmp1.txt');
        body_url = xcorr_evalues_url.replace('.txt', '_tmp2.txt');

        command_line = "head -1 {} > {}".format(xcorr_evalues_url, header_url);
        print(subprocess.check_output(command_line, shell=True));

        command_line = "tail -n+2 {} | sort -k2n -k7n -o {}".format(xcorr_evalues_url, body_url);
        print(subprocess.check_output(command_line, shell=True));

        command_line = "cat {} {} > {}".format(header_url, body_url, xcorr_evalues_url_sortbypepid);
        print(subprocess.check_output(command_line, shell=True));
        os.remove(header_url); os.remove(body_url);


    xcorr_pepevalues_url_sortbypepid = os.path.join(os.path.dirname(xcorr_evalues_url), 'xcorr_pepevalues_sortbypepid.txt');
    if not os.path.isfile(xcorr_pepevalues_url_sortbypepid):
        input_csvfile = open(xcorr_evalues_url_sortbypepid);
        reader = csv.DictReader(input_csvfile, delimiter='\t');
        header_list = [];
        for header in reader.fieldnames:
            if header not in ['ExpMz', 'ExpMass', 'CalcMass']: header_list.append(header);
        header_list = np.concatenate((header_list[:7], ['accpep_evalue_log', 'coop_evalue_log' ], header_list[7:]));
        logger.info('header_list={}'.format(header_list));

        output_csvfile = open(xcorr_pepevalues_url_sortbypepid, 'w');
        writer = csv.DictWriter(output_csvfile, fieldnames=header_list, delimiter='\t');
        writer.writeheader();

        scan_gap = mzxml_utils.get_scan_gap(inputFilename);
        ms2_to_ms1_scannum_map = mzxml_utils.get_ms2_to_ms1_scannum_map(inputFilename);

        def deal_with_pepid(curr_pepid, record_list):
            ms1scan_minevalue_map = {};
            for evalue_log, ms2scan, row in record_list:
                ms1scan = ms2_to_ms1_scannum_map[ms2scan];
                if ms1scan not in ms1scan_minevalue_map: ms1scan_minevalue_map[ms1scan] = evalue_log;
                else: ms1scan_minevalue_map[ms1scan] = min(ms1scan_minevalue_map[ms1scan], evalue_log);

            evalue_log_list = [record[0] for record in record_list]; mean_evalue_log = np.mean(evalue_log_list);
            for record_idx, record in enumerate(record_list):
                evalue_log, ms2scan, row = record;
                ms1scan = ms2_to_ms1_scannum_map[ms2scan];

                accpep_evalue_log = logsumexp(evalue_log_list[:(record_idx+1)]);
                coop_evalue_log = logsumexp([evalue_log, min(ms1scan_minevalue_map.get(ms1scan-scan_gap, mean_evalue_log), ms1scan_minevalue_map.get(ms1scan+scan_gap, mean_evalue_log)) ]);
                row['accpep_evalue_log'] = accpep_evalue_log; row['coop_evalue_log'] = coop_evalue_log;
                writer.writerow({key: value for key, value in row.items() if key in header_list});

        curr_pepid = -1; record_list = [];
        for row_idx, row in enumerate(reader):
            ms2scan = int(row["scan"]); pepid = int(row["pepid"]); evalue_log = float(row["evalue_log"]);

            if curr_pepid != pepid:
                if len(record_list) > 0: deal_with_pepid(curr_pepid, record_list);
                curr_pepid = pepid; record_list = [];
            record_list.append((evalue_log, ms2scan, row));
        if len(record_list) > 0: deal_with_pepid(curr_pepid, record_list);

        output_csvfile.close(); input_csvfile.close();


    if not os.path.isfile(xcorr_pepevalues_url):
        header_url = xcorr_pepevalues_url_sortbypepid.replace('.txt', '_tmp1.txt');
        body_url = xcorr_pepevalues_url_sortbypepid.replace('.txt', '_tmp2.txt');

        command_line = "head -1 {} > {}".format(xcorr_pepevalues_url_sortbypepid, header_url);
        print(subprocess.check_output(command_line, shell=True));

        command_line = "tail -n+2 {} | sort -k1n -k3n -o {}".format(xcorr_pepevalues_url_sortbypepid, body_url);
        print(subprocess.check_output(command_line, shell=True));

        command_line = "cat {} {} > {}".format(header_url, body_url, xcorr_pepevalues_url);
        print(subprocess.check_output(command_line, shell=True));
        os.remove(header_url); os.remove(body_url);

    return xcorr_pepevalues_url;


def read_xcorr_file(xcorr_results):
    with open(xcorr_results) as input:
        reader = csv.DictReader(input, delimiter='\t');
        for row in reader:
            yield {'is_decoy':('decoy_' in row['protein id']) or ('rev_' in row['protein id']), 'xcorr score':float(row['xcorr score']), 'sequence':"{}-{}".format(row['sequence'], row['charge'])}


def merge_umpire_allQs(cruxParaDict, ppm, Q_states = [1, 2, 3], overwrite=False):
    cruxParaDict['charge'] = 'None';
    cruxParaDict["topMatch"] = 1;
    output_filename = run_crux.getCruxOutputName(cruxParaDict, suffix='_umpire_ppm{}.txt'.format(ppm));
    logger.info('output_filename={}'.format(output_filename));
    if os.path.isfile(output_filename) and not overwrite:
        return output_filename;

    target_peptides = {}; decoy_peptides = {};
    for Q in Q_states:
        inputfile = os.path.join(run_crux.getCruxOutputName(cruxParaDict, suffix='_umpireQ{}_ppm{}'.format(Q, ppm)), "tide-search.txt");
        assert os.path.isfile(inputfile);

        for row in read_xcorr_file(inputfile):
            if row['is_decoy']: decoy_peptides[row['sequence']] = max(row['xcorr score'], decoy_peptides.get(row['sequence'], -100));
            else: target_peptides[row['sequence']] = max(row['xcorr score'], target_peptides.get(row['sequence'], -100));

    all_score = [];
    for key in target_peptides: all_score.append((key, target_peptides[key], 't'));
    for key in decoy_peptides: all_score.append((key, decoy_peptides[key], 'd'));
    all_score.sort(key=lambda x: -x[1]);

    with open(output_filename, 'w') as output:
        output.write("td\tpeptide\tcharge\tscore\n")

        for i in all_score:
            ss = i[0].split('-');
            assert len(ss) == 2;
            peptide = ss[0];
            charge = ss[1];
            output.write("%s\t%s\t%s\t%f\n"%(i[2], peptide, charge, i[1]));
    return output_filename;


def percolator_umpire(cruxParaDict, ppm, Q_states = [1, 2, 3], overwrite=False):
    cruxParaDict['charge'] = 'None';
    cruxParaDict["topMatch"] = 1;

    percolator_umpire_url = run_crux.getCruxOutputName(cruxParaDict, suffix='_ppm{}_umpire_percolator.txt'.format(ppm));
    if os.path.isfile(percolator_umpire_url) and not overwrite: return percolator_umpire_url;

    inputfile_list = [];
    for Q in Q_states:
        inputfile = os.path.join(run_crux.getCruxOutputName(cruxParaDict, suffix='_umpireQ{}_ppm{}'.format(Q, ppm)), "tide-search.txt");
        inputfile_list.append(inputfile); logger.info(inputfile); assert os.path.isfile(inputfile);

    input_csvfile = open(inputfile_list[0]);
    reader = csv.DictReader(input_csvfile, delimiter='\t');
    header = reader.fieldnames;
    input_csvfile.close();

    peptide_pepid_map = get_peptide_pepid_map(cruxParaDict["fastaFile"]);
    pepid_label_map = get_pepid_labels(cruxParaDict["fastaFile"]);

    output_csvfile = open(percolator_umpire_url, 'w');
    writer = csv.DictWriter(output_csvfile, fieldnames=header, delimiter='\t');
    writer.writeheader();

    scanNum=1;
    for inputfile in inputfile_list:
        with open(inputfile) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                # row['scan'] = str(scanNum); scanNum += 1;

                peptide = row['sequence']; pepid = peptide_pepid_map[peptide]; real_label = pepid_label_map[pepid];
                if real_label >= 0: td = 'target';
                else: td='decoy';
                row['target/decoy'] = td;

                writer.writerow(row);

    output_csvfile.close();
    return percolator_umpire_url;


def percolator_umpire_pin(cruxParaDict, ppm, Q_states = [1, 2, 3], overwrite=False):
    cruxParaDict['charge'] = 'None';
    cruxParaDict["topMatch"] = 1;

    percolator_umpire_url = run_crux.getCruxOutputName(cruxParaDict, suffix='_ppm{}_umpire_percolator.pin'.format(ppm));
    if os.path.isfile(percolator_umpire_url) and not overwrite: return percolator_umpire_url;

    inputfile_list = [];
    for Q in Q_states:
        inputfile = os.path.join(run_crux.getCruxOutputName(cruxParaDict, suffix='_umpireQ{}_ppm{}'.format(Q, ppm)), "tide-search.txt");
        inputfile_list.append(inputfile); logger.info(inputfile); assert os.path.isfile(inputfile);


    peptide_pepid_map = get_peptide_pepid_map(cruxParaDict["fastaFile"]);
    pepid_label_map = get_pepid_labels(cruxParaDict["fastaFile"]);

    output = open(percolator_umpire_url, 'w');
    output.write('SpecId\t'                 # 0. SpecId
                 'Label\t'                  # 1. Label
                 'ScanNr\t'                 # 2. ScanNr
                 'Peplen\t'                 # 3. Peplen
                 'Charge1\t'                # 4. Charge1
                 'Charge2\t'                # 5. Charge2
                 'Charge3\t'                # 6. Charge3
                 'Charge4\t'                # 7. Charge4
                 'Charge5\t'                # 8. Charge5
                 'DeltLcn\t'                # 9. DeltLcn
                 'DeltCn\t'                 # 10. DeltCn
                 'lnNumSP\t'                # 11. lnNumSP
                 'Xcorr\t'                  # 12. Xcorr
                 'XcorrRank\t'              # 13. XcorrRank
                 'Peptide\t'                # 14. Peptide
                 'Proteins\n'               # 15. Proteins
                 );


    for inputfile in inputfile_list:
        with open(inputfile) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                scanNum = row["scan"];
                peptide = row["sequence"];
                charge = int(row["charge"]);

                delta_cn = row["delta_cn"];
                delta_lcn = row["delta_lcn"];
                distinct_match = int(row["distinct matches/spectrum"]);
                xcorr = row["xcorr score"];
                xcorr_rank = row["xcorr rank"];

                flank_aa = row["flanking aa"];
                proteinid = row["protein id"];

                proteinid_arr = proteinid.split(',');
                proteinid_arr_refined = [];
                for protein in proteinid_arr:
                    protein1 = protein;
                    if '(' in protein: protein1 = protein[:protein.find('(')];

                    # if 'decoy_decoy_' in protein1: protein1 = protein1.replace('decoy_decoy_', 'decoy_d_' );
                    # elif 'decoy_' in protein1: protein1 = protein1.replace('decoy_', 'd_' );

                    proteinid_arr_refined.append(protein1);
                proteinid = ','.join(proteinid_arr_refined);

                pepid = peptide_pepid_map[peptide]; real_label = pepid_label_map[pepid];
                if real_label >= 0: label = 1; tag = 'target';
                else: label = -1; tag = 'decoy';

                peptide1 = peptide_utils.getTransformPeptide(peptide); assert '[' not in peptide1;
                peplen = len(peptide1);

                charge_list = np.zeros(5, dtype=int);
                charge_list[charge - 1] = 1;

                specId = '{}_0_{}_{}_{}'.format(tag, scanNum, charge, xcorr_rank);

                output.write('{}\t{}\t{}\t{}\t{}\t'
                             '{}\t{}\t{}\t{}\t{}\t'
                             '{}\t{}\t{}\t{}\t{}\t'
                             '{}\n'.format(
                    specId,                             # 0. SpecId
                    label,                              # 1. Label
                    scanNum,                            # 2. ScanNr
                    peplen,                             # 3. Peplen
                    charge_list[0],                     # 4. Charge1
                    charge_list[1],                     # 5. Charge2
                    charge_list[2],                     # 6. Charge3
                    charge_list[3],                     # 7. Charge4
                    charge_list[4],                     # 8. Charge5
                    delta_lcn,                          # 9. DeltLcn
                    delta_cn,                           # 10. DeltCn
                    np.log(distinct_match),             # 11. lnNumSP
                    xcorr,                              # 12. Xcorr
                    xcorr_rank,                         # 13. XcorrRank
                    '{}.{}.{}'.format(flank_aa[0], peptide, flank_aa[1]),# 14. Peptide
                    proteinid,                          # 15. Proteins
                ));


    output.close();


def get_protein_len_map(fasta_url):
    protein_proid_url = os.path.join(os.path.dirname(fasta_url), 'protein_proid_map.npy');
    proid_len_url = os.path.join(os.path.dirname(fasta_url), 'proid_len_map.npy');

    if os.path.isfile(protein_proid_url) and os.path.isfile(proid_len_url):
        protein_proid_map = np.load(protein_proid_url, allow_pickle=True).item();
        proid_len_map = np.load(proid_len_url, allow_pickle=True).item();
    else:
        protein_proid_map = {}; proid_len_map = {}; protein_cnt = 0;

        with open(fasta_url) as input:
            protein = ''; protein_len = 0;
            for cnt, line in enumerate(input):
                line = line.rstrip();
                if line.startswith('>'):
                    if protein_len > 0:
                        assert protein not in protein_proid_map;

                        protein_proid_map[protein] = protein_cnt; protein_cnt += 1;
                        protein_decoy = 'decoy_{}'.format(protein); protein_proid_map[protein_decoy] = protein_cnt; protein_cnt += 1;

                        proid_len_map[protein_proid_map[protein]] = protein_len; logger.info('protein={}\tproid={}\tlen={}'.format(protein, protein_proid_map[protein], proid_len_map[protein_proid_map[protein]]));
                        proid_len_map[protein_proid_map[protein_decoy]] = protein_len; logger.info('protein={}\tproid={}\tlen={}'.format(protein_decoy, protein_proid_map[protein_decoy], proid_len_map[protein_proid_map[protein_decoy]]));

                        protein_len = 0;

                    protein = line.split(' ')[0][1:];
                else: protein_len += len(line);

            if protein_len > 0:
                assert protein not in protein_proid_map;
                protein_proid_map[protein] = protein_cnt; protein_cnt += 1;
                protein_decoy = 'decoy_{}'.format(protein); protein_proid_map[protein_decoy] = protein_cnt; protein_cnt += 1;

                proid_len_map[protein_proid_map[protein]] = protein_len; logger.info('protein={}\tproid={}\tlen={}'.format(protein, protein_proid_map[protein], proid_len_map[protein_proid_map[protein]]));
                proid_len_map[protein_proid_map[protein_decoy]] = protein_len; logger.info('protein={}\tproid={}\tlen={}'.format(protein_decoy, protein_proid_map[protein_decoy], proid_len_map[protein_proid_map[protein_decoy]]));

        np.save(protein_proid_url, protein_proid_map);
        np.save(proid_len_url, proid_len_map);

    logger.info('protein_proid_map={}'.format(len(protein_proid_map)));
    logger.info('proid_len_map={}'.format(len(proid_len_map)));

    return protein_proid_map, proid_len_map;


def get_proid_interactions(fasta_url):
    protein_proid_url = os.path.join(os.path.dirname(fasta_url), 'protein_proid_map.npy'); assert os.path.isfile(protein_proid_url);
    protein_proid_map = np.load(protein_proid_url, allow_pickle=True).item();

    proteinid_proids_map = {};
    for protein in protein_proid_map:
        if 'decoy_' in protein: continue;
        if '|' not in protein: continue;

        proteinid = protein.split('|')[-1].split('_')[0]; organ = protein.split('|')[-1].split('_')[1];

        if proteinid not in proteinid_proids_map: proteinid_proids_map[proteinid] = [];
        proteinid_proids_map[proteinid].append((protein_proid_map[protein], organ));


    proid_interact_url = os.path.join(os.path.dirname(fasta_url), 'proid_interact_map.npy');
    if os.path.isfile(proid_interact_url):  proid_interact_maps = np.load(proid_interact_url, allow_pickle=True).item();
    else:
        biogrid_url = os.path.join(os.path.dirname(fasta_url), 'BIOGRID-ALL-3.5.175.tab2.txt'); assert os.path.isfile(biogrid_url);
        interact_set = set();
        with open(biogrid_url) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                intA = row["Official Symbol Interactor A"]; intB = row["Official Symbol Interactor B"];
                synAs = row["Synonyms Interactor A"]; synBs = row["Synonyms Interactor B"];

                intA_list = [];
                if intA in proteinid_proids_map: intA_list.append(intA);
                for synA in synAs.split('|'):
                    if synA in proteinid_proids_map: intA_list.append(synA);

                intB_list = [];
                if intB in proteinid_proids_map: intB_list.append(intB);
                for synB in synBs.split('|'):
                    if synB in proteinid_proids_map: intB_list.append(synB);

                if len(intA_list) <= 0 or len(intB_list) <= 0: continue;

                for intA in intA_list:
                    for intB in intB_list:

                        for proidA, organA in proteinid_proids_map[intA]:
                            for proidB, organB in proteinid_proids_map[intB]:
                                if organA != organB: continue;
                                interact_set.add('{}_{}'.format(min(proidA, proidB), max(proidA, proidB)));

        proid_interact_maps = {};
        for proid_tp_str in interact_set:
            arr = proid_tp_str.split('_'); proidA = int(arr[0]); proidB = int(arr[1]);
            if proidA not in proid_interact_maps: proid_interact_maps[proidA] = [];
            if proidB not in proid_interact_maps: proid_interact_maps[proidB] = [];

            proid_interact_maps[proidA].append(proidB);
            proid_interact_maps[proidB].append(proidA);

        np.save(proid_interact_url, proid_interact_maps);
    logger.info('proid_interact_maps={}'.format(len(proid_interact_maps)));
    # for proid in proid_interact_maps: logger.info('{}\t{}'.format(proid, proid_interact_maps[proid]));
    return proid_interact_maps;


def get_peptide_pepid_map(fasta_url, whitelist_pepids = None ):

    if whitelist_pepids:
        whitelist_len = len(whitelist_pepids);
        whitelist_sum = sum(whitelist_pepids);
        whitelist_peptide_pepid_url = os.path.join(os.path.dirname(fasta_url), 'peptide_pepid_map_whitelist{}_{}.npy'.format(whitelist_len, whitelist_sum));
        if os.path.isfile(whitelist_peptide_pepid_url):
            whitelist_peptide_pepid_map = np.load(whitelist_peptide_pepid_url, allow_pickle=True).item();
            logger.info('whitelist_peptide_pepid_map={}'.format(len(whitelist_peptide_pepid_map)));
            return whitelist_peptide_pepid_map;


    file_name, file_extension = os.path.splitext(fasta_url); index_dir = file_name + '-index';
    peptide_target_db_url = os.path.join(index_dir, 'tide-index.peptides.target.txt'); assert os.path.isfile(peptide_target_db_url);
    peptide_decoy_db_url = os.path.join(index_dir, 'tide-index.peptides.decoy.txt'); assert os.path.isfile(peptide_decoy_db_url);

    peptide_pepid_url = os.path.join(os.path.dirname(fasta_url), 'peptide_pepid_map.npy');
    if os.path.isfile(peptide_pepid_url): peptide_pepid_map = np.load(peptide_pepid_url, allow_pickle=True).item();
    else:
        peptide_pepid_map = {}; peptide_cnt = 0;
        for peptide_db_url in [peptide_target_db_url, peptide_decoy_db_url]:
            logger.info(peptide_db_url);
            with open(peptide_db_url) as fp:
                for cnt, line in enumerate(fp):
                    peptide=line.split('\t')[0];

                    if peptide not in peptide_pepid_map:
                        peptide_pepid_map[peptide] = peptide_cnt;
                        peptide_cnt += 1;

        np.save(peptide_pepid_url, peptide_pepid_map);
    logger.info('peptide_pepid_map={}'.format(len(peptide_pepid_map)));

    if whitelist_pepids:
        whitelist_peptide_pepid_map = {};
        for peptide in peptide_pepid_map:
            pepid =  peptide_pepid_map[peptide];
            if pepid in whitelist_pepids: whitelist_peptide_pepid_map[peptide] = pepid;
        np.save(whitelist_peptide_pepid_url, whitelist_peptide_pepid_map);
        logger.info('whitelist_peptide_pepid_map={}'.format(len(whitelist_peptide_pepid_map)));
        del peptide_pepid_map;
        return whitelist_peptide_pepid_map;
    else: return peptide_pepid_map;


def get_pepid_labels(fasta_url, whitelist_pepids = None ):

    if whitelist_pepids:
        whitelist_len = len(whitelist_pepids);
        whitelist_sum = sum(whitelist_pepids);
        whitelist_pepid_label_url = os.path.join(os.path.dirname(fasta_url), 'pepid_label_map_whitelist{}_{}.npy'.format(whitelist_len, whitelist_sum));
        if os.path.isfile(whitelist_pepid_label_url):
            whitelist_pepid_label_map = np.load(whitelist_pepid_label_url, allow_pickle=True).item();
            logger.info('whitelist_pepid_label_map={}'.format(len(whitelist_pepid_label_map)));
            return whitelist_pepid_label_map;


    file_name, file_extension = os.path.splitext(fasta_url);
    origin_file_name = file_name.replace('_PlusDecoy', ''); assert file_name != origin_file_name;

    pepid_label_url = os.path.join(os.path.dirname(fasta_url), 'pepid_label_map.npy');
    if os.path.isfile(pepid_label_url): pepid_label_map = np.load(pepid_label_url, allow_pickle=True).item();
    else:
        index_dir = file_name + '-index'; origin_index_dir = origin_file_name + '-origin';
        origin_target_url = os.path.join(origin_index_dir, 'tide-index.peptides.target.txt'); assert os.path.isfile(origin_target_url);
        curr_target_url = os.path.join(index_dir, 'tide-index.peptides.target.txt'); assert os.path.isfile(curr_target_url);
        curr_decoy_url = os.path.join(index_dir, 'tide-index.peptides.decoy.txt'); assert os.path.isfile(curr_decoy_url);

        real_target_cnt = 0; pseudo_target_cnt = 0; decoy_cnt = 0;
        pepid_label_map = {}; origin_target_set = set(); peptide_pepid_map = get_peptide_pepid_map(fasta_url);

        with open(origin_target_url) as fp:
            for cnt, line in enumerate(fp):
                peptide = line.split('\t')[0];
                origin_target_set.add(peptide);

        with open(curr_target_url) as fp:
            for cnt, line in enumerate(fp):
                peptide = line.split('\t')[0]; pepid = peptide_pepid_map[peptide];
                if peptide in origin_target_set:
                    real_target_cnt += 1;
                    pepid_label_map[pepid] = 1;
                else:
                    pseudo_target_cnt += 1;
                    pepid_label_map[pepid] = 0;

        with open(curr_decoy_url) as fp:
            for cnt, line in enumerate(fp):
                peptide = line.split('\t')[0]; pepid = peptide_pepid_map[peptide];
                decoy_cnt += 1;
                pepid_label_map[pepid] = -1;

        del peptide_pepid_map;
        np.save(pepid_label_url, pepid_label_map);
        logger.info('real_target_cnt={}\tpseudo_target_cnt={}\tdecoy_cnt={}'.format(real_target_cnt, pseudo_target_cnt, decoy_cnt));
    logger.info('pepid_label_map={}'.format(len(pepid_label_map)));

    if whitelist_pepids:
        whitelist_pepid_label_map = {};
        for pepid in whitelist_pepids:
            if pepid in pepid_label_map: whitelist_pepid_label_map[pepid] = pepid_label_map[pepid];
        np.save(whitelist_pepid_label_url, whitelist_pepid_label_map);
        logger.info('whitelist_pepid_label_map={}'.format(len(whitelist_pepid_label_map)));
        del pepid_label_map;
        return whitelist_pepid_label_map;
    else: return pepid_label_map;


def get_pepid_peptide_map(fasta_url, whitelist_pepids = None ):

    if whitelist_pepids:
        whitelist_len = len(whitelist_pepids);
        whitelist_sum = sum(whitelist_pepids);
        whitelist_pepid_peptide_url = os.path.join(os.path.dirname(fasta_url), 'pepid_peptide_map_whitelist{}_{}.npy'.format(whitelist_len, whitelist_sum));
        if os.path.isfile(whitelist_pepid_peptide_url):
            whitelist_pepid_peptide_map = np.load(whitelist_pepid_peptide_url, allow_pickle=True).item();
            logger.info('whitelist_pepid_peptide_map={}'.format(len(whitelist_pepid_peptide_map)));
            return whitelist_pepid_peptide_map;


    pepid_peptide_url = os.path.join(os.path.dirname(fasta_url), 'pepid_peptide_map.npy');
    if os.path.isfile(pepid_peptide_url): pepid_peptide_map = np.load(pepid_peptide_url, allow_pickle=True).item();
    else:
        pepid_peptide_map = {}; peptide_pepid_map = get_peptide_pepid_map(fasta_url);
        for peptide in peptide_pepid_map: pepid_peptide_map[peptide_pepid_map[peptide]] = peptide;
        del peptide_pepid_map;
        np.save(pepid_peptide_url, pepid_peptide_map);
    logger.info('pepid_peptide_map={}'.format(len(pepid_peptide_map)));

    if whitelist_pepids:
        whitelist_pepid_peptide_map = {};
        for pepid in whitelist_pepids:
            if pepid in pepid_peptide_map: whitelist_pepid_peptide_map[pepid] = pepid_peptide_map[pepid];
        np.save(whitelist_pepid_peptide_url, whitelist_pepid_peptide_map);
        logger.info('whitelist_pepid_peptide_map={}'.format(len(whitelist_pepid_peptide_map)));
        del pepid_peptide_map;
        return whitelist_pepid_peptide_map;
    else: return pepid_peptide_map;


def get_target_decoy_pepid_bimap(fasta_url, whitelist_pepids = None ):

    # if whitelist_pepids:
    #     whitelist_len = len(whitelist_pepids);
    #     whitelist_sum = sum(whitelist_pepids);
    #     whitelist_target_decoy_pepid_map_url = os.path.join(os.path.dirname(fasta_url), 'target_decoy_pepid_bimap_whitelist{}_{}.npy'.format(whitelist_len, whitelist_sum));
    #     if os.path.isfile(whitelist_target_decoy_pepid_map_url):
    #         whitelist_target_decoy_pepid_map = np.load(whitelist_target_decoy_pepid_map_url, allow_pickle=True).item();
    #         logger.info('whitelist_target_decoy_pepid_map={}'.format(len(whitelist_target_decoy_pepid_map)));
    #         return whitelist_target_decoy_pepid_map;


    target_decoy_pepid_map_url = os.path.join(os.path.dirname(fasta_url), 'target_decoy_pepid_bimap.npy');
    if whitelist_pepids:
        whitelist_len = len(whitelist_pepids);
        whitelist_sum = sum(whitelist_pepids);
        target_decoy_pepid_map_url = os.path.join(os.path.dirname(fasta_url), 'target_decoy_pepid_bimap_whitelist{}_{}.npy'.format( whitelist_len, whitelist_sum));


    file_name, file_extension = os.path.splitext(fasta_url); index_dir = file_name + '-index';
    if os.path.isfile(target_decoy_pepid_map_url): target_decoy_pepid_map = np.load(target_decoy_pepid_map_url, allow_pickle=True).item();
    else:
        peptide_pepid_map = get_peptide_pepid_map(fasta_url); target_decoy_pepid_map = {};

        peptide_target_decoy_url = os.path.join(index_dir, 'tide-index.peptides.target.decoy.txt');
        if os.path.isfile(peptide_target_decoy_url):
            with open(peptide_target_decoy_url) as fp:
                for cnt, line in enumerate(fp):
                    arr = line.rstrip().split('\t'); target_peptide = arr[0]; decoy_peptide = arr[1];
                    target_pepid = peptide_pepid_map[target_peptide];
                    decoy_pepid = peptide_pepid_map[decoy_peptide];

                    target_decoy_pepid_map[target_pepid] = decoy_pepid;
                    if decoy_pepid not in target_decoy_pepid_map: target_decoy_pepid_map[decoy_pepid] = target_pepid;

        else:
            peptide_target_url = os.path.join(index_dir, 'tide-index.peptides.target.txt'); assert os.path.isfile(peptide_target_url);
            peptide_decoy_url = os.path.join(index_dir, 'tide-index.peptides.decoy.txt'); assert os.path.isfile(peptide_decoy_url);

            def get_peptide_token_ascii(peptide):
                peptide_askii = np.sum([(i+1)*ord(c) for i, c in enumerate(peptide)]);
                return peptide_askii*10000+ord(peptide[0])*100+ord(peptide[-1]);

            def get_peptide_token_str(peptide):
                return '{}{}_{}'.format(peptide[0], peptide[-1], ''.join(sorted(peptide)) );


            token_pepids_map = {};
            with open(peptide_target_url) as fp:
                for cnt, line in enumerate(fp):
                    peptide = line.rstrip().split('\t')[0];
                    # peptide_token = get_peptide_token_ascii(peptide);
                    peptide_token = get_peptide_token_str(peptide);
                    if peptide_token not in token_pepids_map: token_pepids_map[peptide_token] = [];
                    token_pepids_map[peptide_token].append((1, peptide_pepid_map[peptide]));

            with open(peptide_decoy_url) as fp:
                for cnt, line in enumerate(fp):
                    peptide = line.rstrip().split('\t')[0];
                    # peptide_token = get_peptide_token_ascii(peptide);
                    peptide_token = get_peptide_token_str(peptide);
                    if peptide_token not in token_pepids_map: token_pepids_map[peptide_token] = [];
                    token_pepids_map[peptide_token].append((0, peptide_pepid_map[peptide]));

            for peptide_token in token_pepids_map:
                target_pepid_arr = [x[1] for x in token_pepids_map[peptide_token] if x[0] > 0 ];
                decoy_pepid_arr = [x[1] for x in token_pepids_map[peptide_token] if x[0] <= 0 ];
                # assert len(target_pepid_arr) >= len(decoy_pepid_arr);

                for idx in range(min(len(target_pepid_arr), len(decoy_pepid_arr))):
                    target_pepid = target_pepid_arr[idx]; decoy_pepid = decoy_pepid_arr[idx];

                    if whitelist_pepids:
                        if target_pepid in whitelist_pepids or decoy_pepid in whitelist_pepids:
                            target_decoy_pepid_map[target_pepid] = decoy_pepid;
                            if decoy_pepid not in target_decoy_pepid_map: target_decoy_pepid_map[decoy_pepid] = target_pepid;
                    else:
                        target_decoy_pepid_map[target_pepid] = decoy_pepid;
                        if decoy_pepid not in target_decoy_pepid_map: target_decoy_pepid_map[decoy_pepid] = target_pepid;

                # logger.info('token={}\ttarget_peptide_arr={}\tdecoy_peptide_arr={}'.format(target_token, len(target_peptide_arr), len(decoy_peptide_arr) ));

        np.save(target_decoy_pepid_map_url, target_decoy_pepid_map);
    logger.info('target_decoy_pepid_map={}'.format(len(target_decoy_pepid_map)));

    # if whitelist_pepids:
    #     whitelist_target_decoy_pepid_map = {};
    #     for target_pepid in target_decoy_pepid_map:
    #         decoy_pepid = target_decoy_pepid_map[target_pepid];
    #         if target_pepid in whitelist_pepids or decoy_pepid in whitelist_pepids: whitelist_target_decoy_pepid_map[target_pepid] = decoy_pepid;
    #
    #     np.save(whitelist_target_decoy_pepid_map_url, whitelist_target_decoy_pepid_map);
    #     logger.info('whitelist_target_decoy_pepid_map={}'.format(len(whitelist_target_decoy_pepid_map)));
    #     del target_decoy_pepid_map;
    #     return whitelist_target_decoy_pepid_map;
    # else: return target_decoy_pepid_map;
    return target_decoy_pepid_map;

def get_whitelist_pepids(xcorrFilename):
    pepid_url = os.path.join(os.path.dirname(xcorrFilename), 'pepid.npy');
    if os.path.isfile(pepid_url): pepid_set = set(np.asarray(np.load(pepid_url, allow_pickle=True), dtype=int));
    else:
        pepid_set = set();
        with open(xcorrFilename) as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t');
            for row in reader:
                pepid = int(row["pepid"]);
                pepid_set.add(pepid);
        np.save(pepid_url, np.asarray(list(pepid_set), dtype=int));
    logger.info('pepid_set={}'.format(len(pepid_set)));

    return pepid_set;


def get_pepid_maxcharge_map(fasta_url):
    pepid_maxcharge_url = os.path.join(os.path.dirname(fasta_url), 'pepid_maxcharge_map.npy');
    if os.path.isfile(pepid_maxcharge_url): pepid_maxcharge_map = np.load(pepid_maxcharge_url, allow_pickle=True).item();
    else:
        pepid_peptide_map = get_pepid_peptide_map(fasta_url); pepid_maxcharge_map = {};
        for pepid in pepid_peptide_map:
            peptide = pepid_peptide_map[pepid];
            maxcharge = peptide.count('H') + peptide.count('R') + peptide.count('K') + 1;
            pepid_maxcharge_map[pepid] = maxcharge;

        np.save(pepid_maxcharge_url, pepid_maxcharge_map);
    logger.info('pepid_maxcharge_map={}'.format(len(pepid_maxcharge_map)));
    return pepid_maxcharge_map;


def get_pepid_len_map(fasta_url):
    pepid_len_url = os.path.join(os.path.dirname(fasta_url), 'pepid_len_map.npy');
    if os.path.isfile(pepid_len_url): pepid_len_map = np.load(pepid_len_url, allow_pickle=True).item();
    else:
        pepid_len_map = {}; pepid_peptide_map = get_pepid_peptide_map(fasta_url);
        for pepid in pepid_peptide_map:
            peptide = pepid_peptide_map[pepid];
            while '[' in peptide:
                idx1 = peptide.find('['); idx2 = peptide.find(']');
                peptide = peptide[:idx1] + peptide[idx2 + 1:];
            pepid_len_map[pepid] = len(peptide);
        # np.save(pepid_len_url, pepid_len_map);
    logger.info('pepid_len_map={}'.format(len(pepid_len_map)));
    return pepid_len_map;


def get_pepid_chargecombo_map(fasta_url, mzxml_url, charge_states=[1,2,3,4,5]):
    pepid_chargecombo_url = os.path.join(os.path.dirname(fasta_url), 'pepid_chargecombo_map.npy');
    if os.path.isfile(pepid_chargecombo_url): pepid_chargecombo_map = np.load(pepid_chargecombo_url, allow_pickle=True).item();
    else:
        pepid_chargecombo_map = {};  min_mz, max_mz = mzxml_utils.get_ms1_mz_range(mzxml_url); aa_mass_dict = peptide_utils.getAnimoMassDict();

        peptide_pepid_map = get_peptide_pepid_map(fasta_url);
        for peptide in peptide_pepid_map:
            peptide1 = peptide_utils.getTransformPeptide(peptide); valid_charges = [];
            for pep_charge in charge_states:
                calc_mz = spectrum_utils.calc_mz_from_neutralmass_charge(peptide_utils.neutral_mass(peptide1, aminoMass=aa_mass_dict), pep_charge);
                if calc_mz >= min_mz and calc_mz <= max_mz: valid_charges.append(pep_charge);
            pepid_chargecombo_map[peptide_pepid_map[peptide]] = tuple(valid_charges);

        np.save(pepid_chargecombo_url, pepid_chargecombo_map);
    logger.info('pepid_chargecombo_map={}'.format(len(pepid_chargecombo_map)));
    return pepid_chargecombo_map;


def get_pepid_normalized_predRT(fasta_url, peptide_deeprt_map):
    pepid_predrt_map_url = os.path.join(os.path.dirname(fasta_url), 'pepid_norm_predRT_map.npy');
    if os.path.isfile(pepid_predrt_map_url): pepid_norm_predRT_map = np.load(pepid_predrt_map_url, allow_pickle=True).item();
    else:
        peptide_pepid_map = get_peptide_pepid_map(fasta_url);

        pred_bins = 200;
        deeprt_list = np.unique(list(peptide_deeprt_map.values()));
        pred_rt_arr = np.linspace(np.min(deeprt_list), np.max(deeprt_list), pred_bins);

        pepid_norm_predRT_map = {};
        for peptide in peptide_deeprt_map:
            if 'U' in peptide or 'O' in peptide: continue;
            pepid = peptide_pepid_map[peptide];
            pred_rt_quant = np.digitize(peptide_deeprt_map[peptide], bins=pred_rt_arr) * 1.0 / pred_bins;
            pepid_norm_predRT_map[pepid] = pred_rt_quant;

        np.save(pepid_predrt_map_url, pepid_norm_predRT_map);
    logger.info('pepid_norm_predRT_map={}'.format(len(pepid_norm_predRT_map)));
    return pepid_norm_predRT_map;


def get_cpepid_isotope_dist(fasta_url, mzxml_url, charge_states=[1,2,3,4,5]):
    cpepid_isotopes_url = os.path.join(os.path.dirname(fasta_url), 'cpepid_isotopes.npy');
    if os.path.isfile(cpepid_isotopes_url): cpepid_isotopes_map = np.load(cpepid_isotopes_url, allow_pickle=True).item();
    else:
        formular_charge_peptide_url = os.path.join(os.path.dirname(fasta_url), 'formular_charge_peptide.txt');
        pepid_isotope_info_url = os.path.join(os.path.dirname(fasta_url), 'pepid_isotope_info.txt');

        aa_mass_dict = peptide_utils.getAnimoMassDict(); aa_comp = peptide_utils.getTransformAnimoMassDict();

        # Prepare for the ComputeMS1 input
        if not os.path.isfile(formular_charge_peptide_url):
            min_mz, max_mz = mzxml_utils.get_ms1_mz_range(mzxml_url);
            peptide_pepid_map = get_peptide_pepid_map(fasta_url);


            f = open(formular_charge_peptide_url, 'w');
            for peptide in peptide_pepid_map:
                peptide1 = peptide_utils.getTransformPeptide(peptide);
                peptide2 = peptide_utils.getTransformPeptide2(peptide);

                peptide_comp = dict(mass.Composition(peptide2, aa_comp=aa_comp));
                formular = ''.join(['{}{}'.format(a, peptide_comp[a]) for a in peptide_comp]);

                # ## by default pyteomics doesn't account for cysteine alkylation
                # mass1 = peptide_utils.neutral_mass(peptide1, aminoMass=aa_mass_dict)
                # mass2 = mass.calculate_mass(peptide_comp);
                # if peptide != peptide1 and 'C' in peptide: logger.info('peptide={}\tformular={}\tmass1={}\tmass2={}'.format(peptide, formular, mass1, mass2))

                for pep_charge in charge_states:
                    calc_mz = spectrum_utils.calc_mz_from_neutralmass_charge(peptide_utils.neutral_mass(peptide1, aminoMass=aa_mass_dict), pep_charge);
                    if calc_mz >= min_mz and calc_mz <= max_mz: f.write("{}\t{}\t{}\t{}\n".format(formular, pep_charge, peptide, calc_mz));
            f.close();

        # Parse the ComputeMS1 output
        if not os.path.isfile(pepid_isotope_info_url):

            # Run ComputeMS1 for isotope distributions
            peptide_isotopes_url = os.path.join(os.path.dirname(fasta_url), 'peptide_isotopes.txt');
            if not os.path.isfile(peptide_isotopes_url):
                src_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'));

                isotope_dat_url = os.path.join(src_dir, 'Mercury8', 'ISOTOPE.DAT'); assert os.path.isfile(isotope_dat_url);
                command_arr = [os.path.join(src_dir, 'Mercury8', 'ComputeMS1'), formular_charge_peptide_url, isotope_dat_url, '>', peptide_isotopes_url];
                command_line = ' '.join(command_arr); logger.info(command_line);
                print(subprocess.check_output(command_line, shell=True));

            peptide_pepid_map = get_peptide_pepid_map(fasta_url);
            n = -1; isvals = False; peptide=''; charge=-1; mz_isotope_tuples = [];

            f = open(pepid_isotope_info_url, 'w');
            with open(peptide_isotopes_url) as fp:
                for cnt, line in enumerate(fp):
                    line = line.rstrip();
                    if 'successful' in line:
                        ### correct for cysteine alkylation
                        peptide1 = peptide_utils.getTransformPeptide(peptide);
                        calc_mz = spectrum_utils.calc_mz_from_neutralmass_charge(peptide_utils.neutral_mass(peptide1, aminoMass=aa_mass_dict), charge);
                        mono_mz = mz_isotope_tuples[0][0];
                        # if 'C' in peptide: logger.info('peptide={}\tcharge={}\tcalc_mz={}\tmono_mz={}'.format(peptide, charge, calc_mz, mono_mz ));

                        f.write("{}\t{}\t{}\t{}\n".format(peptide_pepid_map[peptide], peptide, charge, ' '.join(['{},{}'.format(tp[0]-mono_mz+calc_mz,tp[1]) for tp in mz_isotope_tuples]) ));
                        isvals = False; peptide=''; charge=-1; mz_isotope_tuples = [];
                        continue;

                    if 'Calculation' in line:
                        isvals = True;
                        continue;

                    if 'Sequence' in line: n += 1; peptide = line.split('\t')[1];
                    if 'Charge' in line: charge = int(line.split('\t')[1]);

                    if isvals:
                        arr = line.split('\t');
                        mz = float(arr[0]); intensity = float(arr[1]);
                        if intensity < 0.1: continue;
                        if len(mz_isotope_tuples) >= 5: continue;
                        mz_isotope_tuples.append((mz, intensity));
            f.close();

        cpepid_isotopes_map = {};
        with open(pepid_isotope_info_url) as fp:
            for cnt, line in enumerate(fp):
                arr = line.rstrip().split('\t');
                pepid = int(arr[0]); charge = int(arr[2]); cpepid = '{}_{}'.format(pepid, charge);
                mz_isotope_str_list = arr[3].split(' ')[:3]; mz_isotope_list = [];
                for mz_isotope_str in mz_isotope_str_list:
                    arr1 = mz_isotope_str.split(',');
                    mz_isotope_list.append(float(arr1[0])); mz_isotope_list.append(float(arr1[1]) / 100.0);
                cpepid_isotopes_map[cpepid] = tuple(mz_isotope_list);
        np.save(cpepid_isotopes_url, cpepid_isotopes_map);
    logger.info('cpepid_isotopes_map={}'.format(len(cpepid_isotopes_map)));

    return cpepid_isotopes_map;


def get_peptide_protein_bimap(cruxParaDict, charge_states = [1, 2, 3, 4, 5]):
    inputfilenames = [];
    for charge in charge_states:
        cruxParaDict['charge'] = str(charge);
        inputfile = os.path.join(run_crux.getCruxOutputName(cruxParaDict), "tide-search.txt");  assert os.path.isfile(inputfile);
        inputfilenames.append(inputfile);

    cruxParaDict['charge'] = 'All';
    outputfoldername = run_crux.getCruxOutputName(cruxParaDict);
    if not os.path.exists(outputfoldername): os.makedirs(outputfoldername);

    pepid_proids_url = os.path.join(outputfoldername, 'pepid_proids_map.npy');

    if os.path.isfile(pepid_proids_url): pepid_proids_map_uniq = np.load(pepid_proids_url, allow_pickle=True).item();
    else:
        pepid_proids_map = {}; protein_proid_map, _ = get_protein_len_map(cruxParaDict["fastaFile"]);
        peptide_pepid_map = get_peptide_pepid_map(cruxParaDict["fastaFile"]);

        for tidefilename in inputfilenames:
            logger.info('parsing: {}'.format(tidefilename));
            with open(tidefilename) as input:
                reader = csv.DictReader(input, delimiter='\t');
                for row in reader:
                    protein_list_str = row['protein id']; peptide = row['sequence'];

                    pepid = peptide_pepid_map[peptide];
                    if pepid not in pepid_proids_map: pepid_proids_map[pepid] = [];

                    for protein in protein_list_str.split(','):
                        if '(' in protein: protein = protein[:protein.index('(')];
                        assert protein in protein_proid_map;
                        pepid_proids_map[pepid].append(protein_proid_map[protein]);

        pepid_proids_map_uniq = {};
        for pepid in pepid_proids_map: pepid_proids_map_uniq[pepid] = list(np.unique(pepid_proids_map[pepid]));
        np.save(pepid_proids_url, pepid_proids_map_uniq);

    logger.info('pepid_proids_map={}'.format(len(pepid_proids_map_uniq)));
    return pepid_proids_map_uniq;


def get_target_pepid_map(fasta_url):
    file_name, file_extension = os.path.splitext(fasta_url); index_dir = file_name + '-index';

    target_pepid_map_url = os.path.join(os.path.dirname(fasta_url), 'target_pepid_map.npy');
    if os.path.isfile(target_pepid_map_url): target_pepid_map = np.load(target_pepid_map_url, allow_pickle=True).item();
    else:
        peptide_pepid_map = get_peptide_pepid_map(fasta_url); target_pepid_map = {};

        peptide_target_url = os.path.join(index_dir, 'tide-index.peptides.target.txt'); assert os.path.isfile(peptide_target_url);
        with open(peptide_target_url) as fp:
            for cnt, line in enumerate(fp):
                target_peptide = line.rstrip().split('\t')[0];
                target_pepid = peptide_pepid_map[target_peptide];
                target_pepid_map[target_peptide] = target_pepid;

        np.save(target_pepid_map_url, target_pepid_map);
    logger.info('target_pepid_map={}'.format(len(target_pepid_map)));
    return target_pepid_map;


