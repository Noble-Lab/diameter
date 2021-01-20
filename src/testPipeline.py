import os, sys, argparse;
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'algorithms'));
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'crux'));
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'utils'));

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np;
import networkx as nx;
import matplotlib
import matplotlib.pyplot as plt


from algorithms import eval_topmatch;

def load_paramperform_data(paramperform_url, index=0):
    param_score_map = {};
    with open(paramperform_url) as fp:
        for cnt, line in enumerate(fp):
            arr = line.rstrip().split('\t');
            param = arr[0]; score = int(arr[1].split(',')[index]);
            if param not in param_score_map: param_score_map[param] = score;
    return param_score_map;

def comp_agressive_grassfiltering(filename1, tag1, filename2, tag2, title=''):
    src_param_score_map = load_paramperform_data(filename1, index=0);
    tgt_param_score_map = load_paramperform_data(filename2, index=0);

    src_score_list = []; tgt_score_list = [];
    for src_param in src_param_score_map:
        if src_param not in tgt_param_score_map: continue;
        src_score_list.append(src_param_score_map[src_param]);
        tgt_score_list.append(tgt_param_score_map[src_param]);

    figure1 = plt.figure();
    ax = figure1.add_subplot(1, 1, 1);
    ax.autoscale();

    min_score = min(np.min(src_score_list), np.min(tgt_score_list))-10; max_score = max(np.max(src_score_list), np.max(tgt_score_list))+10;
    ax.plot(src_score_list, tgt_score_list, 's', color='green',  marker='.');
    ax.plot([min_score, max_score], [min_score, max_score], color='black');
    ax.set_xlabel(tag1, fontsize=12);
    ax.set_ylabel(tag2, fontsize=12);
    ax.set_title(title);
    plt.show();


def comp_cv_train_test_consistency(cv_paramperform_url, index=0):
    figure = plt.figure(); partition_all = 5;

    param_trainscore_map = {}; param_testscore_map = {};
    for partition_idx in range(partition_all):
        src_score_list = []; tgt_score_list = [];

        with open(cv_paramperform_url) as fp:
            for cnt, line in enumerate(fp):
                arr = line.rstrip().split('\t');
                param = arr[0]; partition = int(arr[1]); train_score = int(arr[2].split(',')[index]); test_score = int(arr[3].split(',')[index]);
                if param not in param_trainscore_map: param_trainscore_map[param] = 0;
                param_trainscore_map[param] += train_score;

                if param not in param_testscore_map: param_testscore_map[param] = 0;
                param_testscore_map[param] += test_score;

        with open(cv_paramperform_url) as fp:
            for cnt, line in enumerate(fp):
                arr = line.rstrip().split('\t');
                param = arr[0]; partition = int(arr[1]); test_score = int(arr[3].split(',')[index]);
                if partition != partition_idx: continue;
                src_score_list.append(param_trainscore_map[param]);
                tgt_score_list.append(test_score);
                # tgt_score_list.append(param_testscore_map[param]);

        ax = figure.add_subplot(2, 3, partition_idx + 1);
        ax.autoscale();

        min_score = min(np.min(src_score_list), np.min(tgt_score_list)); max_score = max(np.max(src_score_list), np.max(tgt_score_list));
        ax.plot(src_score_list, tgt_score_list, 's', color='green', marker='.');
        # ax.plot([min_score, max_score], [min_score, max_score], color='black');
        ax.set_xlabel('Train', fontsize=12);
        ax.set_ylabel('Test', fontsize=12);
        ax.set_title('Fold={}'.format(partition_idx+1));

    plt.show();

def comp_paramperform_debaisedInt(paramperform_url_list, tag_list):
    assert len(paramperform_url_list) == len(tag_list);
    key_list = []; values_list = [];
    for paramperform_url, tag in zip(paramperform_url_list, tag_list):
        src_param_score_map = load_paramperform_data(paramperform_url, index=0);
        logger.info('tag={}\tsrc_param_score_map={}'.format(tag, len(src_param_score_map)));
        key_list.append(tag); values_list.append(src_param_score_map.values());

    figure = plt.figure();
    ax1 = figure.add_subplot(1,1,1);
    ax1.autoscale();
    ax1.violinplot(values_list, showmedians=True);
    ax1.yaxis.grid(True);
    ax1.set_xticks([y + 1 for y in range(len(key_list))]);
    ax1.tick_params(axis='both', which='major', labelsize=12);
    ax1.set_ylabel('Area under the target-decoy curve', fontsize=12);
    plt.xticks(rotation=15)
    plt.setp(ax1, xticks=[y + 1 for y in range(len(key_list))], xticklabels=key_list)
    plt.show();

def comp_single_param_top12(paramperform_url_top1_list):
    paramperform_url_top2_list = paramperform_url_top1_list.replace('top1', 'top2'); assert os.path.isfile(paramperform_url_top2_list);
    metric_list = ['aInt', 'aRT', 'aCC', 'aTC'];
    metric_param_map = {}; baseline_param = '';

    for target_metric in metric_list:
        metric_param_map[target_metric] = [];
        src_param_score_map = load_paramperform_data(paramperform_url_top1_list, index=0);
        for param in src_param_score_map.keys():
            metric_value_list = [(x, float(param[param.find(x) + len(x):param.find('_', param.find(x))])) for x in metric_list];
            target_metric_value = 0; other_metric_value = 0;
            for metric, value in metric_value_list:
                if metric == target_metric: target_metric_value += value;
                else: other_metric_value += value;

            if target_metric_value > 0 and other_metric_value <= 0: metric_param_map[target_metric].append(param);
            elif target_metric_value <= 0 and other_metric_value <= 0 and len(baseline_param) <= 0: baseline_param = param;
    logger.info('baseline_param={}'.format(baseline_param));

    figure = plt.figure();
    for metric_idx, metric in enumerate(metric_list):
        logger.info('metric={}'.format(metric));
        for param in metric_param_map[metric]: logger.info(param);

        src_param_score_map = load_paramperform_data(paramperform_url_top1_list, index=0);
        tgt_param_score_map = load_paramperform_data(paramperform_url_top2_list, index=0);

        src_score_list = [src_param_score_map[target_param] for target_param in metric_param_map[metric]];
        tgt_score_list = [tgt_param_score_map[target_param] for target_param in metric_param_map[metric]];

        ax = figure.add_subplot(2, 2, metric_idx+1);
        ax.autoscale();

        ax.plot(np.ones(len(src_score_list))*1, src_score_list, 's', color='green', marker='.', label='top1');
        ax.plot(np.ones(len(tgt_score_list))*2, tgt_score_list, 's', color='red', marker='.', label='top2');
        ax.plot([1, 2], [src_param_score_map[baseline_param], tgt_param_score_map[baseline_param]], 's', color='black', marker='x', label='xcorr');

        ax.set_xlabel('TopK', fontsize=12);
        ax.set_ylabel('Area under the target-decoy curve', fontsize=12);
        ax.set_title(metric);


    plt.show();



def main(args):

    calibration = 0; top = 1;


    # filename = 'paramperform_param_charged_peptide_calibration_evaluelinear_modular_ppm30_aInt0.0,0.5,1.0_aRT0.0,3.0,6.0_aCC0.0,0.5,1.0,1.5,2.0_aTC0.0,0.5,1.0,1.5,2.0_bCC0.0_bTC0.0_lCC0.0_lTC0.0_cal{}_int100.0_rtd1.0.txt'.format(calibration);
    filename = 'paramperform_Norm0_param_charged_peptide_calibration_evaluelinear_modular_ppm30_aInt0.5,1.0,1.5,2.0,2.5,3.0_aRT0.0,3.0,6.0_aCC0.0,0.5,1.0,1.5,2.0,2.5,3.0_aTC0.0,0.5,1.0,1.5,2.0,2.5,3.0_bCC0.0_bTC0.0_lCC0.0_lTC0.0_cal0_int1_rtd2.0.txt';

    logger.info(filename.replace('.','_'));

    dir_list = [
        '/media/yanglu/TOSHIBA/data/dia_search/MSQC1/20150601_011_nodil_2/results/20150601_011_nodil_2-top{}-chargeAll-prewindow13-mzbinwidth0.02-mzbinoffset0'.format(top),
        '/media/yanglu/TOSHIBA/data/dia_search/NCI60/guot_L130610_003_SW-NCI60_2/results/guot_L130610_003_SW-NCI60_2-top{}-chargeAll-prewindow13-mzbinwidth0.02-mzbinoffset0'.format(top),
        '/media/yanglu/TOSHIBA/data/dia_search/Navarro/HYE124_TTOF5600_32fix_lgillet_L150206_001/results/HYE124_TTOF5600_32fix_lgillet_L150206_001-top{}-chargeAll-prewindow13-mzbinwidth0.02-mzbinoffset0'.format(top),
        '/media/yanglu/TOSHIBA/data/dia_search/diaumpire_data/Ecoli/18484_REP3_1ug_Ecoli_NewStock2_SWATH_1/results/18484_REP3_1ug_Ecoli_NewStock2_SWATH_1-top{}-chargeAll-prewindow13-mzbinwidth0.02-mzbinoffset0'.format(top),
        # '/media/yanglu/TOSHIBA/data/dia_search/diaumpire_data/Human/18300_REP2_500ng_HumanLysate_SWATH_1/results/18300_REP2_500ng_HumanLysate_SWATH_1-top{}-chargeAll-prewindow13-mzbinwidth0.02-mzbinoffset0'.format(top)
    ];
    tag_list = [
        'MSQC1/20150601_011_nodil_2',
        'NCI60/guot_L130610_003_SW-NCI60_2',
        'Navarro/HYE124_TTOF5600_32fix_lgillet_L150206_001',
        'Ecoli/18484_REP3_1ug_Ecoli_NewStock2_SWATH_1',
        # 'Human/18300_REP2_500ng_HumanLysate_SWATH_1'
    ];
    umpire_list = [
        '/media/yanglu/TOSHIBA/data/dia_search/MSQC1/20150601_011_nodil_2/results/20150601_011_nodil_2-top1-chargeNone-prewindow13-mzbinwidth0.02-mzbinoffset0_umpire_ppm30.txt',
        '/media/yanglu/TOSHIBA/data/dia_search/NCI60/guot_L130610_003_SW-NCI60_2/results/guot_L130610_003_SW-NCI60_2-top1-chargeNone-prewindow13-mzbinwidth0.02-mzbinoffset0_umpire_ppm30.txt',
        '/media/yanglu/TOSHIBA/data/dia_search/Navarro/HYE124_TTOF5600_32fix_lgillet_L150206_001/results/HYE124_TTOF5600_32fix_lgillet_L150206_001-top1-chargeNone-prewindow13-mzbinwidth0.02-mzbinoffset0_umpire_ppm30.txt',
        '/media/yanglu/TOSHIBA/data/dia_search/diaumpire_data/Ecoli/18484_REP3_1ug_Ecoli_NewStock2_SWATH_1/results/18484_REP3_1ug_Ecoli_NewStock2_SWATH_1-top1-chargeNone-prewindow13-mzbinwidth0.02-mzbinoffset0_umpire_ppm30.txt',
        # '/media/yanglu/TOSHIBA/data/dia_search/diaumpire_data/Human/18300_REP2_500ng_HumanLysate_SWATH_1/results/18300_REP2_500ng_HumanLysate_SWATH_1-top1-chargeNone-prewindow13-mzbinwidth0.02-mzbinoffset0_umpire_ppm30.txt'
    ];
    mzxml_list = [
        '/media/yanglu/TOSHIBA/data/dia_search/MSQC1/20150601_011_nodil_2/20150601_011_nodil_2.mzXML',
        '/media/yanglu/TOSHIBA/data/dia_search/NCI60/guot_L130610_003_SW-NCI60_2/guot_L130610_003_SW-NCI60_2.mzXML',
        '/media/yanglu/TOSHIBA/data/dia_search/Navarro/HYE124_TTOF5600_32fix_lgillet_L150206_001/HYE124_TTOF5600_32fix_lgillet_L150206_001.mzXML',
        '/media/yanglu/TOSHIBA/data/dia_search/diaumpire_data/Ecoli/18484_REP3_1ug_Ecoli_NewStock2_SWATH_1/18484_REP3_1ug_Ecoli_NewStock2_SWATH_1.mzXML',
        # '/media/yanglu/TOSHIBA/data/dia_search/diaumpire_data/Human/18300_REP2_500ng_HumanLysate_SWATH_1/18300_REP2_500ng_HumanLysate_SWATH_1.mzXML'
    ];

    assert len(dir_list) == len(tag_list); assert len(dir_list) == len(umpire_list); assert len(dir_list) == len(mzxml_list);



    def load_baseline(umpire_url, index=0):
        baseline_psm_target_list, baseline_peptide_target_list, baseline_decoy_thres_list = eval_topmatch.comp_thres_target_decoy(umpire_url);
        return baseline_psm_target_list[index];

    def load_data1(edge_list, peptide_id_map, edgeid_evalue_map):
        metric_list = [];
        id_peptide_map = {};
        for peptide in peptide_id_map.keys(): id_peptide_map[peptide_id_map[peptide]] = peptide;

        for edge in edge_list:
            pepid = int(edge[0]); charge = int(edge[1]); scannum = int(edge[2]);
            edgeID = '{}_{}_{}'.format(id_peptide_map[pepid], charge, scannum);

            intensity = float(edge[4]) ;
            rtdiff1 = float(edge[11]); rtdiff2 = float(edge[9]);
            xcorr = np.max(edgeid_evalue_map.get(edgeID, (0, 0))[0], 0);

            multicharge_max = int(edge[8]) + 1.0;
            multicharge = float(edge[6])*1.0 / multicharge_max;
            multiscan = float(edge[7]);

            metric_list.append((xcorr,intensity,multicharge,multiscan,rtdiff1,rtdiff2));
        return metric_list;


    # navarro_url = os.path.join(dir_list[2], filename); title = tag_list[2]+' (each dot is a hyper-param setting)';
    # comp_agressive_grassfiltering(navarro_url, 'regular <5% grass filtering', navarro_url.replace('results', 'results1'), 'aggresive <20% grass filtering', title)

    # comp_cv_train_test_consistency(os.path.join(dir_list[2], filename.replace('paramperform_param', 'paramperform_cv_param')));

    paramperform_url = os.path.join(dir_list[2],  filename);
    # tag_list = ['Int2_Norm{}'.format(x) for x in range(20)]; label_list = tag_list;
    tag_list = ['Int0_Norm0','Int0_Norm1','Int0_Norm2','Int1_Norm0','Int1_Norm1','Int1_Norm2','Int2_Norm0','Int2_Norm1','Int2_Norm2'];


    paramperform_url_list = [paramperform_url.replace('Norm0', x) for x in tag_list];
    comp_paramperform_debaisedInt(paramperform_url_list, tag_list);

    # comp_single_param_top12(paramperform_url.replace('debaisInt0tic0', 'debaisInt0tic0'));




    #
    #
    # for src_idx in range(len(dir_list)):
    #     logger.info('src_idx={}\t{}'.format(src_idx, dir_list[src_idx]));
    #     src_edge_list = np.load(os.path.join(dir_list[src_idx], 'all_valid_bipartite_edges_ppm30.npy'));
    #     src_peptide_id_map = np.load(os.path.join(dir_list[src_idx], 'all_peptide_id_ppm30.npy')).item();
    #     src_edgeid_evalue_map = np.load(os.path.join(dir_list[src_idx], 'edgeid_evalue.npy')).item();
    #     src_metric_list = load_data1(src_edge_list, src_peptide_id_map, src_edgeid_evalue_map);
    #
    #     src_param_score_map = load_paramperform_data(os.path.join(dir_list[src_idx], filename), index=0);
    #
    #     for tgt_idx in range(src_idx+1, len(dir_list)):
    #         logger.info('tgt_idx={}\t{}'.format(tgt_idx, dir_list[tgt_idx]));
    #         tgt_edge_list = np.load(os.path.join(dir_list[tgt_idx], 'all_valid_bipartite_edges_ppm30.npy'));
    #         tgt_peptide_id_map = np.load(os.path.join(dir_list[tgt_idx], 'all_peptide_id_ppm30.npy')).item();
    #         tgt_edgeid_evalue_map = np.load(os.path.join(dir_list[tgt_idx], 'edgeid_evalue.npy')).item();
    #         tgt_metric_list = load_data1(tgt_edge_list, tgt_peptide_id_map, tgt_edgeid_evalue_map);
    #
    #
    #         tgt_param_score_map = load_paramperform_data(os.path.join(dir_list[tgt_idx], filename), index=0);
    #         src_score_list = []; tgt_score_list = [];
    #         for src_param in src_param_score_map:
    #             if src_param not in tgt_param_score_map: continue;
    #             src_score_list.append(src_param_score_map[src_param]);
    #             tgt_score_list.append(tgt_param_score_map[src_param]);
    #         test_train_score_list = zip(src_score_list, np.asarray(tgt_score_list) );
    #         test_train_score_list = sorted(test_train_score_list, key=lambda x: x[1]);
    #
    #         figure1 = plt.figure();
    #         ax = figure1.add_subplot(1,1,1);
    #         ax.autoscale();
    #         ax.plot(list(range(len(test_train_score_list))), [x[1] for x in test_train_score_list], 's', color='green', marker='.', label='Train');
    #         ax.plot(list(range(len(test_train_score_list))), [x[0] for x in test_train_score_list], 's', color='red', marker='.', label='Test');
    #         ax.set_xlabel('Sorted hyper-param', fontsize=12);
    #         ax.set_ylabel('Area under Target-decoy curve', fontsize=12);
    #         ax.set_title('{} vs. {}'.format(tag_list[src_idx], tag_list[tgt_idx]));
    #         ax.legend(loc='lower right', fontsize=8);
    #
    #
    #         label_metric_map = {};
    #         label_metric_map['xcorr'] = ([x[0] for x in src_metric_list], [x[0] for x in tgt_metric_list]);
    #         label_metric_map['intensity'] = ([x[1] for x in src_metric_list], [x[1] for x in tgt_metric_list]);
    #         label_metric_map['multicharge'] = ([x[2] for x in src_metric_list], [x[2] for x in tgt_metric_list]);
    #         label_metric_map['multiscan'] = ([x[3] for x in src_metric_list], [x[3] for x in tgt_metric_list]);
    #         label_metric_map['abs_zscorert_diff'] = ([x[4] for x in src_metric_list], [x[4] for x in tgt_metric_list]);
    #         label_metric_map['abs_quant_diff'] = ([x[5] for x in src_metric_list], [x[5] for x in tgt_metric_list]);
    #
    #         figure = plt.figure();
    #         figure.suptitle('{} vs. {}'.format(tag_list[src_idx], tag_list[tgt_idx]), fontsize=16);
    #         for metric_idx, metric_name in enumerate(['xcorr', 'intensity', 'multicharge', 'multiscan', 'abs_zscorert_diff', 'abs_quant_diff']):
    #             ax = figure.add_subplot(2, 3, metric_idx + 1);
    #             ax.autoscale();
    #
    #             score_max = -np.inf; score_min = np.inf;
    #             score_max = max(score_max, np.max(label_metric_map[metric_name][0])); score_min = min(score_min, np.min(label_metric_map[metric_name][0]));
    #             score_max = max(score_max, np.max(label_metric_map[metric_name][1])); score_min = min(score_min, np.min(label_metric_map[metric_name][1]));
    #
    #             n, bins, patches = ax.hist(label_metric_map[metric_name][0], bins=30, log=False, range=(score_min, score_max), alpha=0.4, label=tag_list[src_idx]);
    #             n, bins, patches = ax.hist(label_metric_map[metric_name][1], bins=30, log=False, range=(score_min, score_max), alpha=0.4, label=tag_list[tgt_idx]);
    #
    #             ax.legend(loc='upper right', fontsize=8);
    #             ax.set_xlabel(metric_name, fontsize=15);
    #             ax.set_ylabel('# of edges', fontsize=15);
    #         plt.show();



    # figure = plt.figure(); count = 1;
    # for src_idx in range(len(dir_list)):
    #     logger.info('src_idx={}\t{}'.format(src_idx, tag_list[src_idx]));
    #     src_paramperform_url = os.path.join(dir_list[src_idx], filename);
    #     src_param_score_map = load_paramperform_data(src_paramperform_url, index=0);
    #     src_baseline = load_baseline(umpire_list[src_idx], index=0);
    #
    #
    #     targetall_param_score_map = {};
    #     for tgt_idx in range(len(dir_list)):
    #         if tgt_idx == src_idx: continue;
    #         logger.info('tgt_idx={}\t{}'.format(tgt_idx, tag_list[tgt_idx]));
    #
    #         tgt_paramperform_url = os.path.join(dir_list[tgt_idx], filename);
    #         tgt_param_score_map = load_paramperform_data(tgt_paramperform_url, index=0);
    #
    #         for tgt_param in tgt_param_score_map.keys():
    #             targetall_param_score_map[tgt_param] = targetall_param_score_map.get(tgt_param, 0)+tgt_param_score_map[tgt_param];
    #
    #     src_score_list = []; tgt_score_list = [];
    #     for src_param in src_param_score_map:
    #         if src_param not in tgt_param_score_map: continue;
    #         src_score_list.append(src_param_score_map[src_param]);
    #         tgt_score_list.append(targetall_param_score_map[src_param]);
    #
    #     test_train_score_list = zip(src_score_list, np.asarray(tgt_score_list)/len(tag_list));
    #     test_train_score_list = sorted(test_train_score_list, key=lambda x: x[1]);
    #     logger.info(test_train_score_list);
    #
    #     ax = figure.add_subplot(2, 2, src_idx+1);
    #     ax.autoscale();
    #     ax.plot(list(range(len(test_train_score_list))), [x[1] for x in test_train_score_list], 's', color='green', marker='.', label='Train');
    #     ax.plot(list(range(len(test_train_score_list))), [x[0] for x in test_train_score_list], 's', color='red', marker='.', label='Test');
    #
    #     ax.plot([0,len(test_train_score_list)], [src_baseline, src_baseline], color='black');
    #
    #     ax.set_xlabel('Sorted hyper-param', fontsize=12);
    #     ax.set_ylabel('Area under Target-decoy curve', fontsize=12);
    #     ax.set_title('Test:{}'.format(tag_list[src_idx]), fontsize=12);
    #     ax.legend(loc='lower right', fontsize=8);
    #
    # plt.show()












if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description');
    args = parser.parse_args();
    main(args);

