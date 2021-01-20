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


def comp_across_keywords(args):
    ### param processing

    alphaPreP_list = [float(x) for x in args.alphaPrePlist.split('|')]; logger.info('alphaPreP_list={}'.format(alphaPreP_list));
    alphaMS2P_list = [float(x) for x in args.alphaMS2Plist.split('|')]; logger.info('alphaMS2P_list={}'.format(alphaMS2P_list));
    alphaInt_list = [float(x) for x in args.alphaIntlist.split('|')]; logger.info('alphaInt_list={}'.format(alphaInt_list));
    alphaRT_list = [float(x) for x in args.alphaRTlist.split('|')]; logger.info('alphaRT_list={}'.format(alphaRT_list));
    alphaTCC_list = [float(x) for x in args.alphaTCClist.split('|')]; logger.info('alphaTCC_list={}'.format(alphaTCC_list));

    betaTCC_list = [float(x) for x in args.betaTCClist.split('|')]; logger.info('betaTCC_list={}'.format(betaTCC_list));
    lamdaTCC_list = [float(x) for x in args.lamdaTCClist.split('|')]; logger.info('lamdaTCC_list={}'.format(lamdaTCC_list));

    selmethod_list = [x for x in args.selmethodlist.split('|')]; logger.info('selmethod_list={}'.format(selmethod_list));
    topMatches_list = [int(x) for x in args.topMatcheslist.split('|')]; logger.info('topMatches_list={}'.format(topMatches_list));

    keyword_list = [x for x in args.keywordlist.split('|')]; logger.info('keyword_list={}'.format(keyword_list));
    label_list = [x for x in args.labellist.split('|')]; logger.info('label_list={}'.format(label_list));
    assert len(keyword_list) == len(label_list);

    metriclist_list = [];  biaslist_list = [];
    for metric_debiase_list in args.metricdebiaselist.split('|'):
        metric_type_list = []; bias_type_list = [];
        for x in metric_debiase_list.split(','):
            arr = x.split('_');
            metric_type_list.append(int(arr[0]));
            if len(arr) <= 1: bias_type_list.append(0);
            else: bias_type_list.append(int(arr[1]));
        logger.info('metric_type_list={}\tbias_type_list={}'.format(metric_type_list, bias_type_list));
        metriclist_list.append(metric_type_list); biaslist_list.append(bias_type_list);


    ### load baseline
    baseline_target_precursor = eval_topmatch.comp_target_precursor_by_fdr(args.baseline_url);
    baseline_target_peptide = eval_topmatch.comp_target_peptide_by_fdr(args.baseline_url);
    baseline_arr = [baseline_target_precursor, baseline_target_peptide];


    param_metric_map = {};
    param_list_combo = [selmethod_list, metriclist_list, biaslist_list, alphaPreP_list, alphaMS2P_list, alphaInt_list, alphaRT_list, alphaTCC_list,  betaTCC_list, lamdaTCC_list];
    for topMatches in topMatches_list:
        logger.info('topMatches={}'.format(topMatches));

        for selmethod, metriclist, biaslist, alphaPreP, alphaMS2P, alphaInt, alphaRT, alphaTCC, betaTCC, lamdaTCC in itertools.product(*param_list_combo):
            matchingParaDict = run_matching.getMatchingParamDict(alphaPreP, alphaMS2P, alphaInt, alphaRT, alphaTCC, betaTCC, lamdaTCC,
                                                                 args.ppm, args.rtdiff_thres, args.evalue_mode, metric_type_list=metriclist, bias_type_list=biaslist);

            if 'mod' in selmethod: param_str = run_matching.getMatchingParamStr(matchingParaDict, modular=True);
            else: param_str = run_matching.getMatchingParamStr(matchingParaDict, modular=False);
            param_str = '{}_{}'.format(selmethod, param_str);

            eval_url = os.path.join(args.outputFolder, 'results_top{}'.format(topMatches), 'results', 'rst_{}_eval.txt'.format(param_str)); # logger.info(eval_url);
            if os.path.isfile(eval_url):
                with open(eval_url, 'r') as myfile:
                    eval_data = myfile.readlines();
                    if len(eval_data) <= 0: continue;
                    target_precursor = float(eval_data[0].rstrip().split('\t')[1]);
                    target_peptide = float(eval_data[1].rstrip().split('\t')[1]);

                    param_str1 = 'top{}_{}'.format(topMatches, param_str);
                    param_metric_map[param_str1] = (target_precursor, target_peptide, param_str1);
    logger.info('param_metric_map={}'.format(len(param_metric_map)));


    keyword_metric_map = {};
    for keyword in keyword_list:
        keyword_metric_map[keyword] = [];
        for param in param_metric_map:
            keyword1 = '{}_'.format(keyword);
            if keyword1 in param: keyword_metric_map[keyword].append(param_metric_map[param]);


    level_arr = []; param_arr = []; value_arr = [];
    for levelIdx, levelName in enumerate(['target precursor', 'target peptide']):
        for keyword, label in zip(keyword_list, label_list):
            values = [x[levelIdx] for x in keyword_metric_map[keyword]];
            param_strs = [x[-1] for x in keyword_metric_map[keyword]];

            level_arr = np.concatenate((level_arr, [levelName]*len(values) ));
            param_arr = np.concatenate((param_arr, [label]*len(values) ));
            value_arr = np.concatenate((value_arr, values ));

            top10_indices = np.argsort(values)[::-1]; logger.info('\n---------------------------------------------------------\nlevelName={}\tkeyword={}\tlabel={}'.format(levelName, keyword, label))
            for top10_idx in top10_indices[:20]: logger.info( 'top10_idx={}\tvalue={}\tparam_str={}'.format(top10_idx+1, values[top10_idx], param_strs[top10_idx]));

        level_arr = np.append(level_arr, levelName); param_arr = np.append(param_arr, 'baseline'); value_arr = np.append(value_arr, baseline_arr[levelIdx]);

    values_pd = pd.DataFrame({'level': level_arr, 'param': param_arr, 'value': value_arr, });
    figure = plt.figure(figsize=(30, 15));
    ax = sns.violinplot(x="level", y="value", hue="param", data=values_pd, palette="Set1", split=False);
    ax.set_xlabel('');
    ax.set_ylabel('# of detection (<1% FDR)', fontsize=30);
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.legend(fontsize=25);
    # plt.show();

    plt.savefig(os.path.join(args.outputFolder, 'comp_{}_top{}_met{}_aPreP{},{}_aMS2P{},{}_aInt{},{}_aRT{},{}_aTCC{},{}.png'.format(
        args.keywordlist, args.topMatcheslist, args.metricdebiaselist,
        np.min(alphaPreP_list), np.max(alphaPreP_list),
        np.min(alphaMS2P_list), np.max(alphaMS2P_list),
        np.min(alphaInt_list), np.max(alphaInt_list),
        np.min(alphaRT_list), np.max(alphaRT_list),
        np.min(alphaTCC_list), np.max(alphaTCC_list),
        )),dpi=100);
    plt.close(figure);


def main(args):
    comp_across_keywords(args);



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    parser.add_argument('--baseline_url', type=str, help='baseline_url');
    parser.add_argument('--outputFolder', type=str, help='outputFolder');
    parser.add_argument('--ppm', type=int, help='ppm', default=30);
    parser.add_argument('--rtdiff_thres', type=float, help='rtdiff_thres', default=2.0);

    parser.add_argument('--alphaPrePlist', type=str, help='alphaPrePlist');
    parser.add_argument('--alphaMS2Plist', type=str, help='alphaMS2Plist');
    parser.add_argument('--alphaIntlist', type=str, help='alphaIntlist');
    parser.add_argument('--alphaRTlist', type=str, help='alphaRTlist');
    parser.add_argument('--alphaTCClist', type=str, help='alphaTCClist');

    parser.add_argument('--betaTCClist', type=str, help='betaTCClist');
    parser.add_argument('--lamdaTCClist', type=str, help='lamdaTCClist');

    parser.add_argument('--selmethodlist', type=str, help='selmethodlist');
    parser.add_argument('--keywordlist', type=str, help='keywordlist');
    parser.add_argument('--labellist', type=str, help='labellist');

    parser.add_argument('--metricdebiaselist', type=str, help='metricdebiaselist');
    parser.add_argument('--topMatcheslist', type=str, help='topMatcheslist');

    parser.add_argument('--evalue_mode', type=int, help='evalue_mode', default=1);


    args = parser.parse_args()
    main(args)
pass
