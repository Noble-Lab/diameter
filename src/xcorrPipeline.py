import os, sys, re;
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'algorithms'));
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crux'));
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'utils'));

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import argparse

from crux import run_crux, read_xcorr_results;
from algorithms import convertMzxml, read_deeprt_results;
from utils import mzxml_utils;

def main(args):
    pwd = os.path.abspath('.');

    ## check input arguments
    assert args.label;
    # assert os.path.isfile(args.inputFile);
    assert os.path.isfile(args.fastaFile);

    if not args.outputFolder: args.outputFolder = os.path.join(pwd, 'results');
    if not os.path.exists(args.outputFolder): os.makedirs(args.outputFolder);

    file_name, file_extension = os.path.splitext(args.inputFile);
    assert file_extension.lower() == '.mzxml';

    if args.overwrite.lower() in ["f", "false", "0"]: overwrite = False;
    elif args.overwrite.lower() in ["t", "true", "1"]: overwrite = True;
    else: assert False;

    charge_states = [int(x) for x in args.charges.split(',')]; # charge_states = [1,2,3,4,5];

    cruxParaDict = run_crux.getCruxParaDict(pwd, args.label,
                             args.cruxExecFile,
                             args.fastaFile,
                             args.precursorWindow,
                             args.mzBinWidth,
                             args.mzBinOffset,
                             args.cleavages,
                             args.topMatch,
                             charge_states,
                             args.modsSpec,
                             args.enzyme,
                             args.inputFile,
                             args.outputFolder);


    # run_crux.run_gen_peptides(cruxParaDict);


    # if int(args.denoisePeak) == 0: mgf_file_list = convertMzxml.convertMzxml2mgf(args.inputFile, charge_states=charge_states);
    # elif int(args.denoisePeak) == 1: mgf_file_list = convertMzxml.convertMzxml2mgf_denoised(args.inputFile, ppm=int(args.ppm), charge_states=charge_states);
    # elif int(args.denoisePeak) == 2: mgf_file_list = convertMzxml.convertMzxml2mgf_denoised2(args.inputFile, ppm=int(args.ppm), charge_states=charge_states);
    # else: assert False;
    #
    # run_crux.run_crux(cruxParaDict, mgf_file_list, charge_states=charge_states, useTailor=int(args.useTailor));
    # read_xcorr_results.getStandardXcorrOutput(cruxParaDict, charge_states=charge_states, useTailor=int(args.useTailor));
    # read_xcorr_results.getStandardEvalueOutput(cruxParaDict);


    run_crux.run_crux_umpire(cruxParaDict, ppm=int(args.ppm), overwrite=overwrite);
    read_xcorr_results.merge_umpire_allQs(cruxParaDict, ppm=int(args.ppm), overwrite=overwrite);
    read_xcorr_results.percolator_umpire_pin(cruxParaDict, ppm=int(args.ppm), overwrite=overwrite);


    # read_deeprt_results.generate_deeprt_input(args.fastaFile, split=500000);
    # read_deeprt_results.read_peptide_deeprt(args.fastaFile, split=500000);
    # read_deeprt_results.generate_prosit_input(args.fastaFile);



    # read_xcorr_results.get_pepid_labels(args.fastaFile);
    # read_xcorr_results.get_target_decoy_pepid_bimap(args.fastaFile);


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--label', type=str, help='label');
    parser.add_argument('--inputFile', type=str,  help='inputFile');
    parser.add_argument('--fastaFile', type=str, help='fastaFile');
    parser.add_argument('--cruxExecFile', type=str, help='cruxExecFile');
    parser.add_argument('--outputFolder', type=str, help='outputFolder');
    parser.add_argument('--cleavages', type=int, help='cleavages', default=2);
    parser.add_argument('--topMatch', type=int, help='topMatch', default=1);
    parser.add_argument('--modsSpec', type=str, help='modsSpec', default="C+57.02146");
    parser.add_argument('--enzyme', type=str, help='enzyme', default="trypsin");
    parser.add_argument('--precursorWindow', type=str, help='precursorWindow');
    parser.add_argument('--mzBinWidth', type=str, help='mzBinWidth', default=0.02);
    parser.add_argument('--mzBinOffset', type=str, help='mzBinOffset', default=0);
    parser.add_argument('--ppm', type=int, help='ppm', default=10);
    parser.add_argument('--charges', type=str, help='charges', default="1,2,3,4,5");
    parser.add_argument('--overwrite', type=str, help='overwrite', default="False");

    parser.add_argument('--denoisePeak', type=int, help='denoisePeak', default=0);
    parser.add_argument('--useTailor', type=int, help='useTailor', default=0);


    args = parser.parse_args();
    main(args);
pass
