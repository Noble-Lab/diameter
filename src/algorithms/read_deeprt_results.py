import os, sys, random, gc, csv, subprocess
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pyteomics import mzxml,mgf
import numpy as np;

import matplotlib
import matplotlib.pyplot as plt


def generate_prosit_input(fastaFilename, partition=5):

    file_name, file_extension = os.path.splitext(fastaFilename);
    peptide_target_db_url = os.path.join(file_name + '-index', 'tide-index.peptides.target.txt'); assert os.path.isfile(peptide_target_db_url);
    peptide_decoy_db_url = os.path.join(file_name + '-index', 'tide-index.peptides.decoy.txt'); assert os.path.isfile(peptide_decoy_db_url);

    # charge_states = [1, 2, 3, 4, 5]; adj_NCEs = 33*0.9/np.asarray([1.0, 0.9, 0.85, 0.8, 0.75], dtype=float);
    charge_states = [2, 3]; adj_NCEs = 33*0.9/np.asarray([0.9, 0.85], dtype=float);


    for charge, nce_val in zip(charge_states, adj_NCEs):
        f_arr = [];
        for partition_idx in range(partition):
            output_prosit_url = os.path.join(file_name + '-index', 'prosit.peptides.charge{}.part{}_{}.csv'.format(charge, partition_idx, partition));
            f = open(output_prosit_url, 'w');
            f.write('modified_sequence,collision_energy,precursor_charge\n');
            f_arr.append(f);

        for peptide_db_url in [peptide_target_db_url, peptide_decoy_db_url]:
            logger.info(peptide_db_url);

            with open(peptide_db_url) as fp:
                for cnt, line in enumerate(fp):
                    peptide = line.split('\t')[0];

                    peptide = peptide.replace('M[15.99]', 'M(ox)');
                    peptide = peptide.replace('S[79.97]', 'S(ph)');
                    peptide = peptide.replace('T[79.97]', 'T(ph)');
                    peptide = peptide.replace('Y[79.97]', 'Y(ph)');

                    if 'U' in peptide or 'O' in peptide: continue;
                    if len(peptide) < 7 or len(peptide) > 30: continue;


                    f_arr[cnt % partition].write('{},{},{}\n'.format(peptide, int(nce_val), charge));
        for partition_idx in range(partition): f_arr[partition_idx].close();




def generate_deeprt_input(fastaFilename, split=-1):
    file_name, file_extension = os.path.splitext(fastaFilename); index_dir = file_name + '-index';
    peptide_target_db_url = os.path.join(index_dir, 'tide-index.peptides.target.txt'); assert os.path.isfile(peptide_target_db_url);
    peptide_decoy_db_url = os.path.join(index_dir, 'tide-index.peptides.decoy.txt'); assert os.path.isfile(peptide_decoy_db_url);

    for peptide_db_url in [peptide_target_db_url, peptide_decoy_db_url]:
        logger.info(peptide_db_url);
        output_deeprt_url = os.path.join(peptide_db_url.replace('tide-index', 'deeprt'));

        if not os.path.isfile(output_deeprt_url):
            f = open(output_deeprt_url, 'w');
            f.write('sequence\tRT\n');

            with open(peptide_db_url) as fp:
                for cnt, line in enumerate(fp):
                    peptide=line.split('\t')[0];
                    if 'U' in peptide: continue;

                    peptide = peptide.replace('M[15.99]', '1');
                    peptide = peptide.replace('S[79.97]', '2');
                    peptide = peptide.replace('T[79.97]', '3');
                    peptide = peptide.replace('Y[79.97]', '4');

                    # if peptide != line.split('\t')[0]: logger.info('{}\t{}'.format(peptide, line.split('\t')[0]));
                    assert '[' not in peptide;

                    f.write('{}\t0\n'.format(peptide));
            f.close();

        if split <= 0: continue;

        curr_fileID = -1; foutput = None;
        with open(output_deeprt_url) as fp:
            for cnt, line in enumerate(fp):
                tmp_fileID = cnt / split;
                if cnt <= 0: continue;
                if tmp_fileID > curr_fileID:
                    if foutput: foutput.close();
                    curr_fileID = tmp_fileID;
                    foutput = open(output_deeprt_url.replace('.txt', '_part{}.txt'.format(curr_fileID)), "w");
                    foutput.write('sequence\tRT\n');

                foutput.write(line);

        if foutput: foutput.close();




def read_peptide_deeprt(fastaFilename, split=-1):
    file_name, file_extension = os.path.splitext(fastaFilename); index_dir = file_name + '-index';
    peptide_deeprt_map_url = os.path.join(file_name + '-index', 'peptide_deeprt_map.npy');
    if os.path.isfile(peptide_deeprt_map_url): peptide_deeprt_map_all = np.load(peptide_deeprt_map_url, allow_pickle=True).item();
    else:
        peptide_deeprt_map_all = {};
        for db_type in ['target', 'decoy']:
            logger.info('db_type={}'.format(db_type));

            peptide_deeprt_map_url_tmp = peptide_deeprt_map_url.replace('.npy', '_{}.npy'.format(db_type));
            if os.path.isfile(peptide_deeprt_map_url_tmp): peptide_deeprt_map = np.load(peptide_deeprt_map_url_tmp, allow_pickle=True).item();
            else:
                peptide_db_url_list = []; deeprt_pred_url_list = []; peptide_deeprt_map = {};
                if split <= 0:
                    peptide_db_url = os.path.join(index_dir, 'deeprt.peptides.{}.txt'.format(db_type)); assert os.path.isfile(peptide_db_url);
                    deeprt_pred_url = os.path.join('{}.pred', peptide_db_url); assert os.path.isfile(deeprt_pred_url);
                    peptide_db_url_list.append(peptide_db_url); deeprt_pred_url_list.append(deeprt_pred_url);
                else:
                    curr_fileID = 0;
                    while True:
                        peptide_db_url = os.path.join(index_dir, 'deeprt.peptides.{}_part{}.txt'.format(db_type, curr_fileID));
                        if not os.path.isfile(peptide_db_url): break;

                        deeprt_pred_url = os.path.join('{}.pred', peptide_db_url); assert os.path.isfile(deeprt_pred_url);
                        peptide_db_url_list.append(peptide_db_url); deeprt_pred_url_list.append(deeprt_pred_url);
                        curr_fileID += 1;

                logger.info('peptide_db_url_list={}\t'.format(len(peptide_db_url_list)));

                for peptide_db_url, deeprt_pred_url in zip(peptide_db_url_list, deeprt_pred_url_list):
                    peptide_list = []; rt_list = [];
                    with open(peptide_db_url) as fp1:
                        for cnt, line in enumerate(fp1):
                            if cnt <= 0: continue;
                            peptide = line.rstrip().split('\t')[0];

                            peptide = peptide.replace('1', 'M[15.99]');
                            peptide = peptide.replace('2', 'S[79.97]');
                            peptide = peptide.replace('3', 'T[79.97]');
                            peptide = peptide.replace('4', 'Y[79.97]');

                            peptide_list.append(peptide);

                    with open(deeprt_pred_url) as fp2:
                        for cnt, line in enumerate(fp2):
                            if cnt <= 0: continue;
                            rt = float(line.rstrip().split('\t')[1]);
                            rt_list.append(rt);

                    assert len(peptide_list) == len(rt_list);
                    assert len(peptide_list) == len(np.unique(peptide_list));
                    peptide_deeprt_map.update(dict(zip(peptide_list, rt_list)));
                    logger.info('peptide_deeprt_map={}'.format(len(peptide_deeprt_map)));

                np.save(peptide_deeprt_map_url_tmp, peptide_deeprt_map)

            peptide_deeprt_map_all.update(peptide_deeprt_map);
        np.save(peptide_deeprt_map_url, peptide_deeprt_map_all)

    logger.info('peptide_deeprt_map_all={}'.format(len(peptide_deeprt_map_all)));
    return peptide_deeprt_map_all;


