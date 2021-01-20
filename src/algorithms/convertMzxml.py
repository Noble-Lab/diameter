import os, sys, random, gc

sys.path.append('..')
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'));
# sys.path.append(parent_dir);


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np;
from pyteomics import mzxml,mgf

import utils.mzxml_utils as mzxml_utils;
import utils.spectrum_utils as spectrum_utils;



def convertMzxml2mgf_var(mzxmlFilename, charge_states = [1, 2, 3, 4, 5], overwrite=False):
    assert os.path.isfile(mzxmlFilename);
    file_name, file_extension = os.path.splitext(mzxmlFilename);
    base_name = os.path.basename(file_name);
    output_dir = os.path.join(os.path.dirname(mzxmlFilename), 'tmp');
    if not os.path.exists(output_dir): os.makedirs(output_dir);


    scanwin_ms2scannum_map = mzxml_utils.get_scanwin_ms2scannum_map_from_mzxml(mzxmlFilename);
    ms2scannum_scanwin_map = mzxml_utils.get_ms2scannum_scanwin_map_from_mzxml(mzxmlFilename);

    scanwin_output_map = {};
    for scanwin in scanwin_ms2scannum_map:
        mgfFilename = os.path.join(output_dir, '{}-scanwin{}-charge{}.mgf'.format(base_name, scanwin, 1));
        if os.path.isfile(mgfFilename) and not overwrite: continue;
        scanwin_output_map[scanwin] = open(mgfFilename, 'w');


    if len(scanwin_output_map) > 0:
        for row in mzxml.read(mzxmlFilename):
            level = int(row['msLevel']); scannum = int(row['num']); logger.info('level={}\tscannum={}'.format(level, scannum));
            mz_arr = np.asarray(row['m/z array']); intensity_arr = np.asarray(row['intensity array']);
            if scannum not in ms2scannum_scanwin_map: continue;
            if len(mz_arr) <= 1: continue;

            f = scanwin_output_map[ms2scannum_scanwin_map[scannum]];
            f.write('BEGIN IONS\n');
            f.write('TITLE={}\n'.format(row['id']));
            f.write('PEPMASS={}\n'.format(row["precursorMz"][0]["precursorMz"]));
            f.write('RTINSECONDS={}\n'.format(row["retentionTime"]));
            f.write('CHARGE={}+\n'.format(1));
            f.write('SCANS={}\n'.format(row['num']));
            for mz, intensity in zip(mz_arr, intensity_arr): f.write('{} {}\n'.format(mz, intensity));
            f.write('END IONS\n\n');

    for scanwin in scanwin_output_map: scanwin_output_map[scanwin].close();

    scanwin_charge_mgfFile_map = {};
    for scanwin in scanwin_ms2scannum_map:
        mgfFilename_ref = os.path.join(output_dir, '{}-scanwin{}-charge{}.mgf'.format(base_name, scanwin, 1)); assert os.path.isfile(mgfFilename_ref);
        for charge in charge_states:
            mgfFilename = os.path.join(output_dir, '{}-scanwin{}-charge{}.mgf'.format(base_name, scanwin, charge));
            scanwin_charge_mgfFile_map['{}_{}'.format(scanwin, charge)] = mgfFilename;

            if os.path.isfile(mgfFilename) and not overwrite: continue;
            f = open(mgfFilename, 'w');
            with open(mgfFilename_ref) as fp:
                for cnt, line in enumerate(fp):
                    if line.startswith('CHARGE='): f.write('CHARGE={}+\n'.format(charge));
                    else: f.write('{}\n'.format(line.strip()));
            f.close();

    return scanwin_charge_mgfFile_map;



def convertMzxml2mgf_denoised_var(mzxmlFilename, ppm, charge_states = [1, 2, 3, 4, 5], overwrite=False):
    assert os.path.isfile(mzxmlFilename);
    file_name, file_extension = os.path.splitext(mzxmlFilename);
    base_name = os.path.basename(file_name);
    output_dir = os.path.join(os.path.dirname(mzxmlFilename), 'tmp');
    if not os.path.exists(output_dir): os.makedirs(output_dir);


    scanwin_ms2scannum_map = mzxml_utils.get_scanwin_ms2scannum_map_from_mzxml(mzxmlFilename);
    ms2scannum_scanwin_map = mzxml_utils.get_ms2scannum_scanwin_map_from_mzxml(mzxmlFilename);

    scanwin_output_map = {};
    for scanwin in scanwin_ms2scannum_map:
        mgfFilename = os.path.join(output_dir, '{}-scanwin{}-charge{}-denoised-ppm{}.mgf'.format(base_name, scanwin, 1, ppm));
        if os.path.isfile(mgfFilename) and not overwrite: continue;
        scanwin_output_map[scanwin] = open(mgfFilename, 'w');


    if len(scanwin_output_map) > 0:
        scan_gap = mzxml_utils.get_scan_gap(mzxmlFilename);
        ms2_to_ms1_scannum_map = mzxml_utils.get_ms2_to_ms1_scannum_map(mzxmlFilename);
        scannum_mzintensity_map = mzxml_utils.get_unzipped_scannum_mzintensity_map_from_mzxml(mzxmlFilename);


        for row in mzxml.read(mzxmlFilename):
            level = int(row['msLevel']); scannum = int(row['num']); logger.info('level={}\tscannum={}'.format(level, scannum));
            mz_arr = np.asarray(row['m/z array']); intensity_arr = np.asarray(row['intensity array']);
            if scannum not in ms2scannum_scanwin_map: continue;
            if len(mz_arr) <= 1: continue;

            neighbor_scannum_list = [];
            if (scannum - scan_gap) in ms2_to_ms1_scannum_map: neighbor_scannum_list.append(scannum - scan_gap);
            if (scannum + scan_gap) in ms2_to_ms1_scannum_map: neighbor_scannum_list.append(scannum + scan_gap);

            valid_indices = np.zeros(len(mz_arr), dtype=bool);
            for neighbor_scannum in neighbor_scannum_list:
                if neighbor_scannum not in scannum_mzintensity_map: continue;
                neighbor_mz_arr, _ = scannum_mzintensity_map[neighbor_scannum];

                for mzIdx, mz in enumerate(mz_arr):
                    min_idx = np.argmin(np.fabs(neighbor_mz_arr - mz));
                    ppm_closest = spectrum_utils.calc_ppm(neighbor_mz_arr[min_idx], mz);
                    if ppm_closest <= ppm: valid_indices[mzIdx] = 1;
            if np.sum(valid_indices) <= 1: continue;


            f = scanwin_output_map[ms2scannum_scanwin_map[scannum]];
            f.write('BEGIN IONS\n');
            f.write('TITLE={}\n'.format(row['id']));
            f.write('PEPMASS={}\n'.format(row["precursorMz"][0]["precursorMz"]));
            f.write('RTINSECONDS={}\n'.format(row["retentionTime"]));
            f.write('CHARGE={}+\n'.format(1));
            f.write('SCANS={}\n'.format(row['num']));
            for mz, intensity in zip(mz_arr[valid_indices], intensity_arr[valid_indices]): f.write('{} {}\n'.format(mz, intensity));
            f.write('END IONS\n\n');


    for scanwin in scanwin_output_map: scanwin_output_map[scanwin].close();

    scanwin_charge_mgfFile_map = {};
    for scanwin in scanwin_ms2scannum_map:
        mgfFilename_ref = os.path.join(output_dir, '{}-scanwin{}-charge{}-denoised-ppm{}.mgf'.format(base_name, scanwin, 1, ppm)); assert os.path.isfile(mgfFilename_ref);
        for charge in charge_states:
            mgfFilename = os.path.join(output_dir, '{}-scanwin{}-charge{}-denoised-ppm{}.mgf'.format(base_name, scanwin, charge, ppm));
            scanwin_charge_mgfFile_map['{}_{}'.format(scanwin, charge)] = mgfFilename;

            if os.path.isfile(mgfFilename) and not overwrite: continue;
            f = open(mgfFilename, 'w');
            with open(mgfFilename_ref) as fp:
                for cnt, line in enumerate(fp):
                    if line.startswith('CHARGE='): f.write('CHARGE={}+\n'.format(charge));
                    else: f.write('{}\n'.format(line.strip()));
            f.close();

    return scanwin_charge_mgfFile_map;



def convertMzxml2mgf_denoised2_var(mzxmlFilename, ppm, charge_states = [1, 2, 3, 4, 5], overwrite=False):
    assert os.path.isfile(mzxmlFilename);
    file_name, file_extension = os.path.splitext(mzxmlFilename);
    base_name = os.path.basename(file_name);
    output_dir = os.path.join(os.path.dirname(mzxmlFilename), 'tmp');
    if not os.path.exists(output_dir): os.makedirs(output_dir);


    scanwin_ms2scannum_map = mzxml_utils.get_scanwin_ms2scannum_map_from_mzxml(mzxmlFilename);
    ms2scannum_scanwin_map = mzxml_utils.get_ms2scannum_scanwin_map_from_mzxml(mzxmlFilename);

    scanwin_output_map = {};
    for scanwin in scanwin_ms2scannum_map:
        mgfFilename = os.path.join(output_dir, '{}-scanwin{}-charge{}-denoised2-ppm{}.mgf'.format(base_name, scanwin, 1, ppm));
        if os.path.isfile(mgfFilename) and not overwrite: continue;
        scanwin_output_map[scanwin] = open(mgfFilename, 'w');


    if len(scanwin_output_map) > 0:
        scan_gap = mzxml_utils.get_scan_gap(mzxmlFilename);
        ms2_to_ms1_scannum_map = mzxml_utils.get_ms2_to_ms1_scannum_map(mzxmlFilename);
        scannum_mzintensity_map = mzxml_utils.get_unzipped_scannum_mzintensity_map_from_mzxml(mzxmlFilename);


        for row in mzxml.read(mzxmlFilename):
            level = int(row['msLevel']); scannum = int(row['num']); logger.info('level={}\tscannum={}'.format(level, scannum));
            mz_arr = np.asarray(row['m/z array']); intensity_arr = np.asarray(row['intensity array']);
            if scannum not in ms2scannum_scanwin_map: continue;
            if len(mz_arr) <= 1: continue;

            neighbor_scannum_list = [];
            if (scannum - scan_gap) in ms2_to_ms1_scannum_map: neighbor_scannum_list.append(scannum - scan_gap);
            if (scannum + scan_gap) in ms2_to_ms1_scannum_map: neighbor_scannum_list.append(scannum + scan_gap);

            valid_cnt_arr = np.zeros(len(mz_arr), dtype=int);
            for neighbor_scannum in neighbor_scannum_list:
                if neighbor_scannum not in scannum_mzintensity_map: continue;
                neighbor_mz_arr, _ = scannum_mzintensity_map[neighbor_scannum];

                for mzIdx, mz in enumerate(mz_arr):
                    min_idx = np.argmin(np.fabs(neighbor_mz_arr - mz));
                    ppm_closest = spectrum_utils.calc_ppm(neighbor_mz_arr[min_idx], mz);
                    if ppm_closest <= ppm: valid_cnt_arr[mzIdx] += 1;

            valid_indices = np.where(valid_cnt_arr >= len(neighbor_scannum_list))[0];
            if len(valid_indices) <= 1: continue;


            f = scanwin_output_map[ms2scannum_scanwin_map[scannum]];
            f.write('BEGIN IONS\n');
            f.write('TITLE={}\n'.format(row['id']));
            f.write('PEPMASS={}\n'.format(row["precursorMz"][0]["precursorMz"]));
            f.write('RTINSECONDS={}\n'.format(row["retentionTime"]));
            f.write('CHARGE={}+\n'.format(1));
            f.write('SCANS={}\n'.format(row['num']));
            for mz, intensity in zip(mz_arr[valid_indices], intensity_arr[valid_indices]): f.write('{} {}\n'.format(mz, intensity));
            f.write('END IONS\n\n');


    for scanwin in scanwin_output_map: scanwin_output_map[scanwin].close();

    scanwin_charge_mgfFile_map = {};
    for scanwin in scanwin_ms2scannum_map:
        mgfFilename_ref = os.path.join(output_dir, '{}-scanwin{}-charge{}-denoised2-ppm{}.mgf'.format(base_name, scanwin, 1, ppm)); assert os.path.isfile(mgfFilename_ref);
        for charge in charge_states:
            mgfFilename = os.path.join(output_dir, '{}-scanwin{}-charge{}-denoised2-ppm{}.mgf'.format(base_name, scanwin, charge, ppm));
            scanwin_charge_mgfFile_map['{}_{}'.format(scanwin, charge)] = mgfFilename;

            if os.path.isfile(mgfFilename) and not overwrite: continue;
            f = open(mgfFilename, 'w');
            with open(mgfFilename_ref) as fp:
                for cnt, line in enumerate(fp):
                    if line.startswith('CHARGE='): f.write('CHARGE={}+\n'.format(charge));
                    else: f.write('{}\n'.format(line.strip()));
            f.close();

    return scanwin_charge_mgfFile_map;



def convertMzxml2mgf(mzxmlFilename, charge_states = [1, 2, 3, 4, 5], overwrite=False):
    assert os.path.isfile(mzxmlFilename);
    mgfFilenameList = {};
    for charge in charge_states:
        file_name, file_extension = os.path.splitext(mzxmlFilename);
        mgfFilename = '{}-charge-{}.mgf'.format(file_name, charge);
        logger.info(mgfFilename);

        if os.path.isfile(mgfFilename) and not overwrite:
            mgfFilenameList[charge]=mgfFilename;
            continue;
        if os.path.isfile(mgfFilename):
            os.remove(mgfFilename);

        f = open(mgfFilename, 'w');
        for row in mzxml.read(mzxmlFilename):
            if int(row['msLevel']) == 2:
                if len(row['m/z array']) > 1:
                    f.write('BEGIN IONS\n');
                    f.write('TITLE={}\n'.format(row['id']));
                    f.write('PEPMASS={}\n'.format(row["precursorMz"][0]["precursorMz"]));
                    f.write('RTINSECONDS={}\n'.format(row["retentionTime"]));
                    f.write('CHARGE={}+\n'.format(charge));
                    f.write('SCANS={}\n'.format(row['num']));
                    for mz, intensity in zip(row['m/z array'], row['intensity array']):
                        f.write('{} {}\n'.format(mz, intensity));
                    f.write('END IONS\n\n');
                else:
                    logger.info('empty!')
        f.close();

        mgfFilenameList[charge] = mgfFilename
        gc.collect()
    return mgfFilenameList;



def convertMzxml2mgf_cutoff(mzxmlFilename, charge_states = [1, 2, 3, 4, 5], overwrite=False):
    assert os.path.isfile(mzxmlFilename);
    mgfFilenameList = {};

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


    for charge in charge_states:
        file_name, file_extension = os.path.splitext(mzxmlFilename);
        mgfFilename = '{}-charge-{}-cutoff.mgf'.format(file_name, charge);
        logger.info(mgfFilename);

        if os.path.isfile(mgfFilename) and not overwrite:
            mgfFilenameList[charge]=mgfFilename;
            continue;
        if os.path.isfile(mgfFilename):
            os.remove(mgfFilename);

        f = open(mgfFilename, 'w');
        for row in mzxml.read(mzxmlFilename):
            level = int(row['msLevel']); intensity_arr = np.asarray(row['intensity array']);
            if level != 2: continue;
            if len(intensity_arr) <= 500: continue;

            intensity_arr = np.log(intensity_arr);
            intensity_arr = np.asarray(sorted(intensity_arr, reverse=True));

            ignore_top = 25; retain_cnt = min(1000, int((len(intensity_arr) - ignore_top) * 0.2));
            slope, intercept, x_values, y_values = fit_log_linear(intensity_arr, ignore_top=ignore_top, ignore_bottom=max(len(intensity_arr) - retain_cnt - ignore_top, 200));

            min_logintensity = np.min(intensity_arr);
            for iter in range(3):
                curr_logintensity = (np.log(len(intensity_arr)) - intercept) / slope;
                if curr_logintensity > min_logintensity:
                    intensity_arr = intensity_arr[intensity_arr >= curr_logintensity];
                    min_logintensity = curr_logintensity;
                else: break;
            min_intensity = np.exp(min_logintensity);

            mz_arr = np.asarray(row['m/z array']); intensity_arr = np.asarray(row['intensity array']);
            valid_indices = np.where(intensity_arr >= min_intensity)[0];
            logger.info('mz_arr={}\t{}'.format(len(mz_arr), len(valid_indices)));

            f.write('BEGIN IONS\n');
            f.write('TITLE={}\n'.format(row['id']));
            f.write('PEPMASS={}\n'.format(row["precursorMz"][0]["precursorMz"]));
            f.write('RTINSECONDS={}\n'.format(row["retentionTime"]));
            f.write('CHARGE={}+\n'.format(charge));
            f.write('SCANS={}\n'.format(row['num']));
            for mz, intensity in zip(mz_arr[valid_indices], intensity_arr[valid_indices]):
                f.write('{} {}\n'.format(mz, intensity));
            f.write('END IONS\n\n');

        f.close();

        mgfFilenameList[charge] = mgfFilename
        gc.collect()
    return mgfFilenameList;


def convertMzxml2mgf_denoised2(mzxmlFilename, ppm, charge_states = [1, 2, 3, 4, 5]):
    assert os.path.isfile(mzxmlFilename);

    mgfFilenameList = {};
    for charge in charge_states:
        file_name, file_extension = os.path.splitext(mzxmlFilename);
        mgfFilename = '{}-charge-{}-denoised2-ppm{}.mgf'.format(file_name, charge, ppm);
        logger.info(mgfFilename);
        mgfFilenameList[charge] = mgfFilename;


    for charge in charge_states[:1]:
        mgfFilename = mgfFilenameList[charge];
        if os.path.isfile(mgfFilename): continue;

        scan_gap = mzxml_utils.get_scan_gap(mzxmlFilename);
        ms2_to_ms1_scannum_map = mzxml_utils.get_ms2_to_ms1_scannum_map(mzxmlFilename);
        scannum_mzintensity_map = mzxml_utils.get_unzipped_scannum_mzintensity_map_from_mzxml(mzxmlFilename);

        f = open(mgfFilename, 'w');
        for row in mzxml.read(mzxmlFilename):
            level = int(row['msLevel']); scannum = int(row['num']); mz_arr = np.asarray(row['m/z array']); intensity_arr = np.asarray(row['intensity array']);
            if level != 2: continue;
            if len(mz_arr) <= 1: continue;

            neighbor_scannum_list = [];
            if (scannum - scan_gap) in ms2_to_ms1_scannum_map: neighbor_scannum_list.append(scannum - scan_gap);
            if (scannum + scan_gap) in ms2_to_ms1_scannum_map: neighbor_scannum_list.append(scannum + scan_gap);
            # logger.info('scannum={}\tneighbor_scannums={}'.format(scannum, neighbor_scannum_list));

            valid_cnt_arr = np.zeros(len(mz_arr), dtype=int);
            for neighbor_scannum in neighbor_scannum_list:
                if neighbor_scannum not in scannum_mzintensity_map: continue;
                neighbor_mz_arr, _ = scannum_mzintensity_map[neighbor_scannum];

                for mzIdx, mz in enumerate(mz_arr):
                    min_idx = np.argmin(np.fabs(neighbor_mz_arr - mz));
                    ppm_closest = spectrum_utils.calc_ppm(neighbor_mz_arr[min_idx], mz);
                    if ppm_closest <= ppm: valid_cnt_arr[mzIdx] += 1;

            valid_indices = np.where(valid_cnt_arr >= len(neighbor_scannum_list))[0];
            if len(valid_indices) <= 1: continue;

            if int(row['msLevel']) == 2:
                if len(row['m/z array']) > 1:
                    f.write('BEGIN IONS\n');
                    f.write('TITLE={}\n'.format(row['id']));
                    f.write('PEPMASS={}\n'.format(row["precursorMz"][0]["precursorMz"]));
                    f.write('RTINSECONDS={}\n'.format(row["retentionTime"]));
                    f.write('CHARGE={}+\n'.format(charge));
                    f.write('SCANS={}\n'.format(row['num']));
                    for mz, intensity in zip(mz_arr[valid_indices], intensity_arr[valid_indices]):
                        f.write('{} {}\n'.format(mz, intensity));
                    f.write('END IONS\n\n');
                else:
                    logger.info('empty!')
        f.close();
        gc.collect()


    mgfFilename_ref = mgfFilenameList[charge_states[0]];
    for charge in charge_states[1:]:
        mgfFilename = mgfFilenameList[charge];
        if os.path.isfile(mgfFilename): continue;

        f = open(mgfFilename, 'w');
        with open(mgfFilename_ref) as fp:
            for cnt, line in enumerate(fp):
                if line.startswith('CHARGE='): f.write('CHARGE={}+\n'.format(charge));
                else: f.write('{}\n'.format(line.strip()));
        f.close();


    return mgfFilenameList;


def convertMzxml2mgf_denoised(mzxmlFilename, ppm, charge_states = [1, 2, 3, 4, 5]):
    assert os.path.isfile(mzxmlFilename);

    mgfFilenameList = {};
    for charge in charge_states:
        file_name, file_extension = os.path.splitext(mzxmlFilename);
        mgfFilename = '{}-charge-{}-denoised-ppm{}.mgf'.format(file_name, charge, ppm);
        logger.info(mgfFilename);
        mgfFilenameList[charge] = mgfFilename;


    for charge in charge_states[:1]:
        mgfFilename = mgfFilenameList[charge];
        if os.path.isfile(mgfFilename): continue;

        scan_gap = mzxml_utils.get_scan_gap(mzxmlFilename);
        ms2_to_ms1_scannum_map = mzxml_utils.get_ms2_to_ms1_scannum_map(mzxmlFilename);
        scannum_mzintensity_map = mzxml_utils.get_unzipped_scannum_mzintensity_map_from_mzxml(mzxmlFilename);

        f = open(mgfFilename, 'w');
        for row in mzxml.read(mzxmlFilename):
            level = int(row['msLevel']); scannum = int(row['num']); mz_arr = np.asarray(row['m/z array']); intensity_arr = np.asarray(row['intensity array']);
            if level != 2: continue;
            if len(mz_arr) <= 1: continue;

            neighbor_scannum_list = [];
            if (scannum - scan_gap) in ms2_to_ms1_scannum_map: neighbor_scannum_list.append(scannum - scan_gap);
            if (scannum + scan_gap) in ms2_to_ms1_scannum_map: neighbor_scannum_list.append(scannum + scan_gap);
            # logger.info('scannum={}\tneighbor_scannums={}'.format(scannum, neighbor_scannum_list));

            valid_indices = np.zeros(len(mz_arr), dtype=bool);
            for neighbor_scannum in neighbor_scannum_list:
                if neighbor_scannum not in scannum_mzintensity_map: continue;
                neighbor_mz_arr, _ = scannum_mzintensity_map[neighbor_scannum];

                for mzIdx, mz in enumerate(mz_arr):
                    min_idx = np.argmin(np.fabs(neighbor_mz_arr - mz));
                    ppm_closest = spectrum_utils.calc_ppm(neighbor_mz_arr[min_idx], mz);
                    if ppm_closest <= ppm: valid_indices[mzIdx] = 1;
            if np.sum(valid_indices) <= 1: continue;

            if int(row['msLevel']) == 2:
                if len(row['m/z array']) > 1:
                    f.write('BEGIN IONS\n');
                    f.write('TITLE={}\n'.format(row['id']));
                    f.write('PEPMASS={}\n'.format(row["precursorMz"][0]["precursorMz"]));
                    f.write('RTINSECONDS={}\n'.format(row["retentionTime"]));
                    f.write('CHARGE={}+\n'.format(charge));
                    f.write('SCANS={}\n'.format(row['num']));
                    for mz, intensity in zip(mz_arr[valid_indices], intensity_arr[valid_indices]):
                        f.write('{} {}\n'.format(mz, intensity));
                    f.write('END IONS\n\n');
                else:
                    logger.info('empty!')
        f.close();
        gc.collect()


    mgfFilename_ref = mgfFilenameList[charge_states[0]];
    for charge in charge_states[1:]:
        mgfFilename = mgfFilenameList[charge];
        if os.path.isfile(mgfFilename): continue;

        f = open(mgfFilename, 'w');
        with open(mgfFilename_ref) as fp:
            for cnt, line in enumerate(fp):
                if line.startswith('CHARGE='): f.write('CHARGE={}+\n'.format(charge));
                else: f.write('{}\n'.format(line.strip()));
        f.close();


    return mgfFilenameList;


