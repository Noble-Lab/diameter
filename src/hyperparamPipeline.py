import os, sys, argparse;
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'algorithms'));
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crux'));
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'utils'));

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np;
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
# plt.switch_backend('agg')
import itertools
import seaborn as sns

from algorithms import run_matching, eval_topmatch;
from crux import read_xcorr_results;




def comp_across_dataset(args):
    ### param processing
    alphaPreP_list = [float(x) for x in args.alphaPrePlist.split('|')]; logger.info('alphaPreP_list={}'.format(alphaPreP_list));
    alphaMS2P_list = [float(x) for x in args.alphaMS2Plist.split('|')]; logger.info('alphaMS2P_list={}'.format(alphaMS2P_list));
    alphaInt_list = [float(x) for x in args.alphaIntlist.split('|')]; logger.info('alphaInt_list={}'.format(alphaInt_list));
    alphaRT_list = [float(x) for x in args.alphaRTlist.split('|')]; logger.info('alphaRT_list={}'.format(alphaRT_list));
    alphaTCC_list = [float(x) for x in args.alphaTCClist.split('|')]; logger.info('alphaTCC_list={}'.format(alphaTCC_list));

    betaTCC_list = [float(x) for x in args.betaTCClist.split('|')]; logger.info('betaTCC_list={}'.format(betaTCC_list));
    lamdaTCC_list = [float(x) for x in args.lamdaTCClist.split('|')]; logger.info('lamdaTCC_list={}'.format(lamdaTCC_list));


    metric_type_list = []; bias_type_list = [];
    for x in args.metric_debiase_list.split(','):
        arr = x.split('_');
        metric_type_list.append(int(arr[0]));
        if len(arr) <= 1: bias_type_list.append(0);
        else: bias_type_list.append(int(arr[1]));
    logger.info('metric_type_list={}\tbias_type_list={}'.format(metric_type_list, bias_type_list));

    param_str_list = [];
    param_list_combo = [alphaPreP_list, alphaMS2P_list, alphaInt_list, alphaRT_list, alphaTCC_list,  betaTCC_list, lamdaTCC_list];
    for alphaPreP, alphaMS2P, alphaInt, alphaRT, alphaTCC, betaTCC, lamdaTCC in itertools.product( *param_list_combo):
        matchingParaDict = run_matching.getMatchingParamDict(alphaPreP, alphaMS2P, alphaInt, alphaRT, alphaTCC, betaTCC, lamdaTCC, args.ppm, args.rtdiff_thres, args.evalue_mode, metric_type_list=metric_type_list, bias_type_list=bias_type_list);

        if 'mod' in args.selmethod: param_str = run_matching.getMatchingParamStr(matchingParaDict, modular=True);
        else: param_str = run_matching.getMatchingParamStr(matchingParaDict, modular=False);
        param_str = '{}_{}'.format(args.selmethod, param_str);
        param_str_list.append(param_str);
    logger.info('param_str_list={}'.format(len(param_str_list)));

    dataset_metric_rank_map = {};
    dataset_list = args.datasetlist.split('|'); logger.info('dataset_list={}'.format(dataset_list));
    label_list = args.labellist.split('|'); logger.info('label_list={}'.format(label_list));
    assert len(dataset_list) == len(label_list);

    for dataset in dataset_list:
        results_dir = os.path.join(args.outputFolder, dataset, 'results' ); search_name = '';
        for file in os.listdir(results_dir):
            if 'umpire' not in file: search_name = file; break;
        assert len(search_name) > 0;
        results_dir = os.path.join(results_dir, search_name, 'results_top{}'.format(args.topMatches), 'results'); logger.info('results_dir={}'.format(results_dir)); assert os.path.exists(results_dir);

        metric_list = [];
        for param_str in param_str_list:
            eval_url = os.path.join(results_dir, 'rst_{}_eval.txt'.format(param_str));
            if os.path.isfile(eval_url):
                with open(eval_url, 'r') as myfile:
                    eval_data = myfile.readlines();
                    if len(eval_data) <= 0: continue;
                    target_precursor = [int(x) for x in eval_data[0].rstrip().split('\t')[1].split(',')][0];
                    target_peptide = [int(x) for x in eval_data[1].rstrip().split('\t')[1].split(',')][0];
                    metric_list.append((target_precursor, target_peptide));
        logger.info('metric_list={}'.format(len(metric_list))); assert len(param_str_list) == len(metric_list);

        target_precursor_list = [x[0] for x in metric_list]; target_peptide_list = [x[1] for x in metric_list];
        target_precursor_ranks = np.zeros_like(target_precursor_list); target_precursor_ranks[np.argsort(target_precursor_list)] = list(range(len(metric_list)));
        target_peptide_ranks = np.zeros_like(target_peptide_list); target_peptide_ranks[np.argsort(target_peptide_list)] = list(range(len(metric_list)));
        rank_list = zip(target_precursor_ranks, target_peptide_ranks);

        dataset_metric_rank_map[dataset] = (metric_list, rank_list);

    ### view each pair
    for levelIdx, levelName in enumerate(['tgt_prec', 'tgt_pep']):
        for valueIdx, valueType in enumerate(['val', 'rank']):

            figure = plt.figure(figsize=(20, 16));
            for dataset_idx1, dataset1 in enumerate(dataset_list):
                metric_list1, rank_list1 = dataset_metric_rank_map[dataset1];

                for dataset_idx2, dataset2 in enumerate(dataset_list):
                    metric_list2, rank_list2 = dataset_metric_rank_map[dataset2];

                    if dataset_idx2 <= dataset_idx1: continue;

                    fig_idx = dataset_idx1*(len(dataset_list)-1)+dataset_idx2;
                    ax = figure.add_subplot(len(dataset_list)-1, len(dataset_list)-1, fig_idx);

                    if valueIdx == 0: ax.plot([x[levelIdx] for x in metric_list1], [x[levelIdx] for x in metric_list2], 's', marker='.', alpha=0.1);
                    else: ax.plot([x[levelIdx] for x in rank_list1], [x[levelIdx] for x in rank_list2], 's', marker='.', alpha=0.1);

                    ax.set_xlabel(label_list[dataset_idx1], fontsize=14);
                    ax.set_ylabel(label_list[dataset_idx2], fontsize=14);

            plt.savefig(
                os.path.join(args.outputFolder, 'param_{}_{}_{}_top{}_met{}_aPreP{},{}_aMS2P{},{}_aInt{},{}_aRT{},{}_aTCC{},{}.png'.format(
                    levelName, valueType, args.labellist, args.topMatches, args.metric_debiase_list,
                    np.min(alphaPreP_list), np.max(alphaPreP_list),
                    np.min(alphaMS2P_list), np.max(alphaMS2P_list),
                    np.min(alphaInt_list), np.max(alphaInt_list),
                    np.min(alphaRT_list), np.max(alphaRT_list),
                    np.min(alphaTCC_list), np.max(alphaTCC_list),
                )), dpi=100);
            plt.close(figure);



def main(args):
    comp_across_dataset(args);



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--outputFolder', type=str, help='outputFolder');
    parser.add_argument('--datasetlist', type=str, help='datasetlist');
    parser.add_argument('--labellist', type=str, help='labellist');

    parser.add_argument('--alphaPrePlist', type=str, help='alphaPrePlist');
    parser.add_argument('--alphaMS2Plist', type=str, help='alphaMS2Plist');
    parser.add_argument('--alphaIntlist', type=str, help='alphaIntlist');
    parser.add_argument('--alphaRTlist', type=str, help='alphaRTlist');
    parser.add_argument('--alphaTCClist', type=str, help='alphaTCClist');

    parser.add_argument('--betaTCClist', type=str, help='betaTCClist');
    parser.add_argument('--lamdaTCClist', type=str, help='lamdaTCClist');

    parser.add_argument('--selmethod', type=str, help='selmethod');
    parser.add_argument('--ppm', type=int, help='ppm', default=30);
    parser.add_argument('--rtdiff_thres', type=float, help='rtdiff_thres', default=2.0);
    parser.add_argument('--evalue_mode', type=int, help='evalue_mode', default=1);

    parser.add_argument('--topMatches', type=str, help='topMatches');
    parser.add_argument('--metric_debiase_list', type=str, help='metric_debiase_list');

    args = parser.parse_args()
    main(args)
pass
