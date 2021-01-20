import os, sys, re;
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import subprocess
import shutil
from multiprocessing import Pool
import gc
import copy

import numpy as np;
import utils.mzxml_utils as mzxml_utils;


def getCruxOutputName(cruxParaDict, suffix=""):
    return os.path.join(cruxParaDict["outputFolder"], '{}-top{}-charge{}-prewindow{}-mzbinwidth{}-mzbinoffset{}{}'.format(cruxParaDict["label"], cruxParaDict["topMatch"], cruxParaDict["charge"], cruxParaDict["precursorWindow"], cruxParaDict["mzBinWidth"], cruxParaDict["mzBinOffset"], suffix));


def getCruxParaDict(
    pwd,
    label,
    cruxExecFile,
    fastaFile,
    precursorWindow,
    mzBinWidth,
    mzBinOffset,
    cleavages,
    topMatch,
    charge,
    modsSpec,
    enzyme,
    inputFile,
    outputFolder
):
    cruxParaDict = {};
    cruxParaDict["pwd"] = pwd;
    cruxParaDict["label"] = label;
    cruxParaDict["cruxExecFile"] = cruxExecFile;
    cruxParaDict["fastaFile"] = fastaFile;
    cruxParaDict["precursorWindow"] = precursorWindow;
    cruxParaDict["mzBinWidth"] = mzBinWidth;
    cruxParaDict["mzBinOffset"] = mzBinOffset;
    cruxParaDict["cleavages"] = cleavages;
    cruxParaDict["topMatch"] = topMatch;
    cruxParaDict["charge"] = charge;
    cruxParaDict["modsSpec"] = modsSpec;
    cruxParaDict["enzyme"] = enzyme;
    cruxParaDict["inputFile"] = inputFile;
    cruxParaDict["outputFolder"] = outputFolder;

    return cruxParaDict;


def run_crux(cruxParaDict, mgf_file_list, charge_states = [1, 2, 3, 4, 5], overwrite=False, useTailor=1):

    for charge in charge_states:
        cruxParaDict_copy = copy.deepcopy(cruxParaDict);
        cruxParaDict_copy['charge'] = str(charge);
        cruxParaDict_copy['outputFolder'] = getCruxOutputName(cruxParaDict_copy);

        if overwrite:
            shutil.rmtree(cruxParaDict_copy['outputFolder']);
        elif os.path.isfile(os.path.join(cruxParaDict_copy['outputFolder'], "tide-search.txt")):
            logger.info('{} exists, do nothing...'.format(os.path.join(cruxParaDict_copy['outputFolder'], "tide-search.txt")));
            continue;

        cruxParaDict_copy["inputFile"] = mgf_file_list[charge];

        run_crux_sub(cruxParaDict_copy, useTailor=useTailor);
        gc.collect();


def run_crux_var(cruxParaDict, scanwin_charge_mgfFile_map, target_scanwin, overwrite=False, useTailor=1, denoisePeak=0):
    ms1_scanwins_arr = mzxml_utils.get_ms1_scanwins(cruxParaDict["inputFile"]);
    for scanwin_charge in scanwin_charge_mgfFile_map:
        scanwin, charge = [int(x) for x in scanwin_charge.split('_')];
        precursorWindow = (ms1_scanwins_arr[scanwin, 1] - ms1_scanwins_arr[scanwin, 0]) / 2;
        if scanwin != target_scanwin and target_scanwin >= 0: continue;

        logger.info('charge={}\tscanwin={}\t{}\tprecursorWindow={}'.format(charge, scanwin, ms1_scanwins_arr[scanwin], precursorWindow));

        cruxParaDict_copy = copy.deepcopy(cruxParaDict);
        cruxParaDict_copy["precursorWindow"] = str(precursorWindow);
        cruxParaDict_copy['charge'] = str(charge);

        if denoisePeak == 0: cruxParaDict_copy['outputFolder'] = os.path.join(cruxParaDict_copy["outputFolder"], 'tmp', '{}-top{}-charge{}-scanwin{}'.format(cruxParaDict_copy["label"], cruxParaDict_copy["topMatch"], cruxParaDict_copy["charge"], scanwin));
        else: cruxParaDict_copy['outputFolder'] = os.path.join(cruxParaDict_copy["outputFolder"], 'tmp', '{}-denoised{}-top{}-charge{}-scanwin{}'.format(cruxParaDict_copy["label"], denoisePeak, cruxParaDict_copy["topMatch"], cruxParaDict_copy["charge"], scanwin));

        cruxParaDict_copy["inputFile"] = scanwin_charge_mgfFile_map[scanwin_charge];
        logger.info(cruxParaDict_copy['outputFolder']);

        if overwrite:
            shutil.rmtree(cruxParaDict_copy['outputFolder']);
        elif os.path.isfile(os.path.join(cruxParaDict_copy['outputFolder'], "tide-search.txt")):
            logger.info('{} exists, do nothing...'.format(os.path.join(cruxParaDict_copy['outputFolder'], "tide-search.txt")));
            continue;

        run_crux_sub(cruxParaDict_copy, useTailor=useTailor);
        gc.collect();


def run_crux_umpire(cruxParaDict, ppm, Q_states = [1, 2, 3], overwrite=False):
    for Q in Q_states:
        cruxParaDict_copy = copy.deepcopy(cruxParaDict);
        cruxParaDict_copy['charge'] = 'None';
        cruxParaDict_copy["topMatch"] = 1;
        cruxParaDict_copy['outputFolder'] = getCruxOutputName(cruxParaDict_copy, suffix='_umpireQ{}_ppm{}'.format(Q, ppm));

        if overwrite:
            shutil.rmtree(cruxParaDict_copy['outputFolder']);
        elif os.path.isfile(os.path.join(cruxParaDict_copy['outputFolder'], "tide-search.txt")):
            logger.info('{} exists, do nothing...'.format(os.path.join(cruxParaDict_copy['outputFolder'], "tide-search.txt")));
            continue;


        file_name, file_extension = os.path.splitext(cruxParaDict["inputFile"]);
        cruxParaDict_copy["inputFile"] = '{}_Q{}.mgf'.format(file_name, Q);
        cruxParaDict_copy["precursorWindow"] = str(ppm);

        run_crux_umpire_sub(cruxParaDict_copy);
        gc.collect();


def run_crux_sub(cruxParaDict, useTailor=1):
    if useTailor > 0: exec_file = 'run_crux1.sh';
    else: exec_file = 'run_crux.sh';

    command_line = "sh {}/{} {} {} {} {} {} {} {} {} {} {} {} {}".format(
        os.path.dirname(os.path.realpath(__file__)),
        exec_file,
        cruxParaDict["cruxExecFile"],
        cruxParaDict["fastaFile"],
        cruxParaDict["precursorWindow"],
        cruxParaDict["mzBinWidth"],
        cruxParaDict["mzBinOffset"],
        cruxParaDict["cleavages"],
        cruxParaDict["topMatch"],
        cruxParaDict["charge"],
        cruxParaDict["modsSpec"],
        cruxParaDict["enzyme"],
        cruxParaDict["inputFile"],
        cruxParaDict["outputFolder"]);
    print(subprocess.check_output(command_line, shell=True));


def run_gen_peptides(cruxParaDict):
    exec_file = 'run_gen_peptides.sh';
    command_line = "sh {}/{} {} {} {} {} {}".format(
        os.path.dirname(os.path.realpath(__file__)),
        exec_file,
        cruxParaDict["cruxExecFile"],
        cruxParaDict["fastaFile"],
        cruxParaDict["cleavages"],
        cruxParaDict["enzyme"],
        cruxParaDict["modsSpec"],
    );
    print(subprocess.check_output(command_line, shell=True));


def run_crux_umpire_sub(cruxParaDict):
    command_line = "sh {}/run_crux_umpire.sh {} {} {} {} {} {} {} {} {} {} {} {}".format(
        os.path.dirname(os.path.realpath(__file__)),
        cruxParaDict["cruxExecFile"],
        cruxParaDict["fastaFile"],
        cruxParaDict["precursorWindow"],
        cruxParaDict["mzBinWidth"],
        cruxParaDict["mzBinOffset"],
        cruxParaDict["cleavages"],
        cruxParaDict["topMatch"],
        cruxParaDict["charge"],
        cruxParaDict["modsSpec"],
        cruxParaDict["enzyme"],
        cruxParaDict["inputFile"],
        cruxParaDict["outputFolder"]);

    print(subprocess.check_output(command_line, shell=True));
