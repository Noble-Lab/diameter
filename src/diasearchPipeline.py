import os, sys, argparse;
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'algorithms'));
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crux'));
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'utils'));

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np;
from algorithms import run_matching, eval_topmatch, plot_manuscript  # run_bipartite,

def main(args):
    pwd = os.path.abspath('.');

    logger.info('inputFile={}'.format(args.inputFile)); assert os.path.isfile(args.inputFile);
    logger.info('fastaFile={}'.format(args.fastaFile)); assert os.path.isfile(args.fastaFile);

    if not args.outputFolder: args.outputFolder = os.path.join(pwd, 'results');
    if not os.path.exists(args.outputFolder): os.makedirs(args.outputFolder);

    file_name, file_extension = os.path.splitext(args.inputFile);
    assert file_extension.lower() == '.mzxml';

    top_match_list = [int(x) for x in args.topMatches.split(',')]; logger.info('top_match_list={}'.format(top_match_list));

    metric_type_list = []; bias_type_list = [];
    for x in args.metric_debiase_list.split(','):
        arr = x.split('_');
        metric_type_list.append(int(arr[0]));
        if len(arr) <= 1: bias_type_list.append(0);
        else: bias_type_list.append(int(arr[1]));
    logger.info('metric_type_list={}\tbias_type_list={}'.format(metric_type_list, bias_type_list));

    neighbor_labels = [];
    if args.neighbor_labels is not None:
        if len(args.neighbor_labels) > 0:
            neighbor_labels = args.neighbor_labels.split(',');
            assert neighbor_labels[0] in args.inputFile;
    logger.info('neighbor_labels={}'.format(neighbor_labels));


    for top_match in top_match_list:
        for file in os.listdir(args.outputFolder):

            if 'top1000-chargeAll' in file:
                xcorrFile = os.path.join(args.outputFolder, file, 'xcorr_evalues_sorted.txt'); logger.info(xcorrFile);

                matchingParaDict = run_matching.getMatchingParamDict(float(args.alphaPreP), float(args.alphaMS2P), float(args.alphaInt), float(args.alphaRT), float(args.alphaTCC), float(args.betaTCC), float(args.lamdaTCC),
                                                                    int(args.ppm), float(args.rtdiff_thres), int(args.evalue_mode),
                                                                    metric_type_list, bias_type_list,
                                                                    args.inputFile, xcorrFile, args.cruxExecFile, args.fastaFile);

                result_dir = os.path.join(os.path.dirname(xcorrFile), 'results_top{}'.format(top_match));
                evidences_url = os.path.join(result_dir, 'pepid_evidence_scangap{}_ppm{}.txt'.format(matchingParaDict["rtdiff_thres"], matchingParaDict["ppm"]));

                # if not os.path.isfile(evidences_url): run_bipartite.construct_bipartite(matchingParaDict, top_match);
                #
                if bias_type_list[0] == 0: debiase_types = []; qn_type = 1;
                elif bias_type_list[0] == 1: debiase_types = [1]; qn_type = 1;
                elif bias_type_list[0] == 2: debiase_types = [1, 2, 3, 4, 1]; qn_type = 1;
                elif bias_type_list[0] == 3: debiase_types = [4, 3, 2, 1]; qn_type = 1;
                elif bias_type_list[0] == 4: debiase_types = [1, 2, 3, 1]; qn_type = 1;
                elif bias_type_list[0] == 5: debiase_types = [3, 2, 1]; qn_type = 1;
                elif bias_type_list[0] == 6: debiase_types = [1, 2, 1]; qn_type = 1;
                elif bias_type_list[0] == 7: debiase_types = [2, 1]; qn_type = 1;
                elif bias_type_list[0] == 8: debiase_types = [1]; qn_type = 2;
                else: assert False;

                # calibrated_evidences_url = run_matching.get_debiased_evidence(evidences_url, debiase_types, qn_type=qn_type); logger.info('calibrated_evidences_url={}'.format(calibrated_evidences_url));
                # # run_matching.plot_debiased_evidence(calibrated_evidences_url, matchingParaDict);
                # # run_matching.get_minmax_norm_shift(calibrated_evidences_url, matchingParaDict);
                #
                # filter_type = int(args.evalue_mode);
                # # run_matching.evidence_tdc_by_fix_weights(calibrated_evidences_url, matchingParaDict, neighbor_labels=neighbor_labels, filter_type=filter_type);
                # run_matching.evidence_tdc_marginal2(calibrated_evidences_url, matchingParaDict);


                # run_matching.plot_qvalue_via_pseudolabel_diaumpire(matchingParaDict);
                # run_matching.plot_qvalue_via_pseudolabel_topDiameter(evidences_url, matchingParaDict);

                # if int(args.evalue_mode) == 1: tag='results_prosit_PlusDecoy';
                # elif int(args.evalue_mode) == 2: tag='results_prosit_PlusDecoy2';
                # elif int(args.evalue_mode) == 3: tag='results_prosit_PlusDecoy3';
                # elif int(args.evalue_mode) == 4: tag='results_walnut_PlusDecoy';
                # elif int(args.evalue_mode) == 5: tag='results_walnut_PlusDecoy2';
                # elif int(args.evalue_mode) == 6: tag='results_walnut_PlusDecoy3';
                # else: assert False;
                # run_matching.postprecess_walnut_prosit_features(matchingParaDict, tag=tag);
                # run_matching.plot_qvalue_via_pseudolabel_encyclopedia(matchingParaDict, tag=tag);


                # eval_topmatch.plot_perf_hyperparam();
                # eval_topmatch.plot_comp_instrument();
                # eval_topmatch.plot_orbitrap();

                # plot_manuscript.plot_tdc_manuscript();
                # eval_topmatch.plot_percolator_nonMs1Int_chromatogram();


                # eval_topmatch.plot_prosit_nonconf_versus_diameter();
                # eval_topmatch.plot_marginal();
                eval_topmatch.plot_marginal2();

                # eval_topmatch.plot_comp_qvalues(psm_level=False);
                # eval_topmatch.plot_orbitrap_sibling_comp();



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--inputFile', type=str, help='inputFile');
    parser.add_argument('--fastaFile', type=str, help='fastaFile');
    parser.add_argument('--cruxExecFile', type=str, help='cruxExecFile');
    parser.add_argument('--outputFolder', type=str, help='outputFolder');
    parser.add_argument('--percolatorWeightFile', type=str, default='NA', help='percolatorWeightFile');

    parser.add_argument('--selmethod', type=str, help='selmethod', default="BP");
    parser.add_argument('--ppm', type=int, help='ppm', default=30);

    parser.add_argument('--alphaPreP', type=float, help='weight for precursor-specific ', default=1.0);
    parser.add_argument('--alphaMS2P', type=float, help='weight for peak-match p-value', default=1.0);
    parser.add_argument('--alphaInt', type=float, help='weight for intensity', default=1.0);
    parser.add_argument('--alphaRT', type=float, help='weight for RTdiff', default=1.0);
    parser.add_argument('--alphaTCC', type=float, help='weight for time and charge cooperativity', default=1.0);

    parser.add_argument('--betaTCC', type=float, help='betaTC', default=1.0);
    parser.add_argument('--lamdaTCC', type=float, help='lamdaTC', default=1.0);

    parser.add_argument('--topMatches', type=str, help='topMatches', default='1');
    parser.add_argument('--metric_debiase_list', type=str, help='metric_debiase_list');

    parser.add_argument('--rtdiff_thres', type=float, help='rtdiff_thres', default=2.0);
    parser.add_argument('--overwrite', type=str, help='overwrite', default="False");

    parser.add_argument('--evalue_mode', type=int, help='evalue_mode', default=1);
    parser.add_argument('--neighbor_labels', type=str, help='neighbor_labels', default=None);


    args = parser.parse_args()
    main(args)
