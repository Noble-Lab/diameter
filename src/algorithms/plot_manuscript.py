import os, sys, random, csv, math, re
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np;
import networkx as nx;
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import plotly;
import plotly.express as px;
import plotly.graph_objects as go;
from plotly.subplots import make_subplots;
import seaborn as sns


from scipy import interpolate
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from sklearn import linear_model, preprocessing
import itertools
import pandas as pd;

import utils.viz_utils as viz_utils;
import utils.mzxml_utils as mzxml_utils;
import utils.peptide_utils as peptide_utils;
import utils.spectrum_utils as spectrum_utils;
import utils.diaumpire_utils as diaumpire_utils;


def plot_tdc_manuscript():
    result_dir = '/media/yanglu/OS/Users/isaac/Documents/2017_wenruo_dia-matching/doc/paper_jpr/data/qvalues';

    datasets1_list = ['qvalue_e01306_diameter.txt', 'qvalue_e01306_diaumpire.txt', 'qvalue_e01306_pecan.txt', 'qvalue_e01306_prosit.txt'];
    datasets2_list = ['qvalue_e01307-e01308_diameter.txt', 'qvalue_e01307-e01308_diaumpire.txt', 'qvalue_e01307-e01308_pecan.txt', 'qvalue_e01307-e01308_prosit.txt'];
    datasets3_list = ['qvalue_e01309-e01312_diameter.txt', 'qvalue_e01309-e01312_diaumpire.txt', 'qvalue_e01309-e01312_pecan.txt', 'qvalue_e01309-e01312_prosit.txt'];
    datasets4_list = ['qvalue_HYE124_TTOF5600_32fix_001_diameter.txt', 'qvalue_HYE124_TTOF5600_32fix_001_diaumpire.txt', 'qvalue_HYE124_TTOF5600_32fix_001_pecan.txt', 'qvalue_HYE124_TTOF5600_32fix_001_prosit.txt'];
    datasets5_list = ['qvalue_HYE124_TTOF5600_64var_007_diameter.txt', 'qvalue_HYE124_TTOF5600_64var_007_diaumpire.txt', 'qvalue_HYE124_TTOF5600_64var_007_pecan.txt', 'qvalue_HYE124_TTOF5600_64var_007_prosit.txt'];
    datasets6_list = ['qvalue_HYE124_TTOF6600_32fix_002_diameter.txt', 'qvalue_HYE124_TTOF6600_32fix_002_diaumpire.txt', 'qvalue_HYE124_TTOF6600_32fix_002_pecan.txt', 'qvalue_HYE124_TTOF6600_32fix_002_prosit.txt'];
    datasets7_list = ['qvalue_HYE124_TTOF6600_64var_008_diameter.txt', 'qvalue_HYE124_TTOF6600_64var_008_diaumpire.txt', 'qvalue_HYE124_TTOF6600_64var_008_pecan.txt', 'qvalue_HYE124_TTOF6600_64var_008_prosit.txt'];
    datasets8_list = ['qvalue_yanliu_L130104_007_diameter.txt', 'qvalue_yanliu_L130104_007_diaumpire.txt', 'qvalue_yanliu_L130104_007_pecan.txt', 'qvalue_yanliu_L130104_007_prosit.txt'];

    # dataset_name_list = ['OxMetYeast_20mz', 'OxMetYeast_10mz', 'OxMetYeast_5mz', 'LFQbench_TTOF5600_32fix', 'LFQbench_TTOF5600_64var', 'LFQbench_TTOF6600_32fix', 'LFQbench_TTOF6600_64var', 'Plasma_TTOF5600_32fix'];
    # datasets_all_list = [datasets1_list, datasets2_list, datasets3_list, datasets4_list, datasets5_list, datasets6_list, datasets7_list, datasets8_list];
    # label_list = ['DIAmeter', 'DIA-Umpire', 'PECAN', 'Prosit+EncyclopeDIA'];
    # color_list = ['blue', 'red', 'orange', 'green'];


    datasets9_list = ['qvalue_e01306_diameter.txt', 'qvalue_e01306_diaumpire.txt', 'qvalue_e01306_pecan.txt', 'qvalue_e01306_prosit.txt', 'qvalue_new_e01306_top20_bestfdr0.01.txt', 'qvalue_new_e01306_top20_bestfdr0.05.txt'];
    datasets10_list = ['qvalue_e01307-e01308_diameter.txt', 'qvalue_e01307-e01308_diaumpire.txt', 'qvalue_e01307-e01308_pecan.txt', 'qvalue_e01307-e01308_prosit.txt', 'qvalue_new_e01307-e01308_top5_bestfdr0.01.txt', 'qvalue_new_e01307-e01308_top5_bestfdr0.05.txt', 'qvalue_new_e01307-e01308_top20_bestfdr0.01.txt', 'qvalue_new_e01307-e01308_top20_bestfdr0.05.txt'];
    dataset_name_list = ['OxMetYeast_20mz_test', 'OxMetYeast_10mz_test'];
    datasets_all_list = [datasets9_list, datasets10_list];
    label_list = ['DIAmeter(k=5, best 1%FDR)', 'DIA-Umpire', 'PECAN', 'Prosit+EncyclopeDIA',  'DIAmeter(k=20, best 1%FDR)', 'DIAmeter(k=20, best 1%+5%FDR)'];
    color_list = ['blue', 'red', 'orange', 'green', 'cyan', 'magenta'];



    def parse_result(result_url):
        qvalue_list = [];
        with open(result_url) as fp:
            for cnt, line in enumerate(fp):
                line = line.rstrip();
                qvalue = float(line.split('\t')[0]);
                qvalue_list.append(qvalue);

            for idx in range(len(qvalue_list) - 1, 0, -1): qvalue_list[idx-1] = min(qvalue_list[idx-1], qvalue_list[idx] );
            assert all(qvalue_list[i] <= qvalue_list[i + 1] for i in range(len(qvalue_list) - 1));

            tenp_idx=-1; onep_idx = -1;
            for idx in range(len(qvalue_list) - 1, 0, -1):
                if qvalue_list[idx] > 0.1: tenp_idx = idx;
                if qvalue_list[idx] > 0.01:  onep_idx = idx;

            logger.info('tenp_idx={}\t{}\t{}'.format(tenp_idx, qvalue_list[tenp_idx-1], qvalue_list[tenp_idx] ));
            logger.info('onep_idx={}\t{}\t{}'.format(onep_idx, qvalue_list[onep_idx-1], qvalue_list[onep_idx] ));
            return np.asarray(qvalue_list[:tenp_idx]);


    for datasets_name, datasets_list in zip(dataset_name_list, datasets_all_list):
        logger.info('datasets_name={}'.format(datasets_name));
        figure = plt.figure(figsize=(16, 9));
        ax1 = figure.add_subplot(1, 1, 1);
        ax1.autoscale();

        max_detect=-1;
        for label_name, dataset_name, color_name in zip(label_list, datasets_list, color_list):
            result_url = os.path.join(result_dir, dataset_name); assert os.path.isfile(result_url);
            qvalue_list = parse_result(result_url); logger.info('label={}\tqvalue_list={}'.format(label_name, len(qvalue_list) ));

            if len(qvalue_list) <= 1: ax1.plot([0, 0.1], [0, 0], linestyle='-', color=color_name, linewidth=1.5, label=label_name);
            else: ax1.plot(qvalue_list, list(range(1, len(qvalue_list) + 1)), linestyle='-', color=color_name, linewidth=1.5, label=label_name);

            max_detect = max(max_detect, len(qvalue_list)+1);
        max_detect *= 1.05;

        ax1.plot([0.01, 0.01], [0, max_detect], linestyle='--', linewidth=3, color='black');
        ax1.set_xlim(0, 0.1);
        ax1.set_ylim(0, max_detect); # bottom=0

        ax1.set_xlabel('q-value', fontsize=25);
        ax1.set_ylabel('Number of accepted peptides', fontsize=25);
        ax1.xaxis.set_tick_params(labelsize=18);
        ax1.yaxis.set_tick_params(labelsize=18);
        ax1.legend(fontsize=28);

        ax1.grid(linestyle=':');
        # plt.tight_layout()
        # plt.show();

        pp = PdfPages(os.path.join(result_dir,  'fig_tdc_{}.pdf'.format(datasets_name)));
        pp.savefig(figure, bbox_inches='tight');
        pp.close();
        plt.close(figure);
        del figure;






