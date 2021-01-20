
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages

from . import spectrum_utils;

COLORS = ['red', 'green', 'blue', 'orange', 'black', 'cyan', 'magenta', '0.7', 'yellow', 'gray'];
MARKS = ['o', 'x', 'd', 'v', '^', 's', 'p', 'P', '*', '.'];

LINESTYLES = ['-', '--', '-.', ':', (0, (1, 10)), (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1)) ];

def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=["black", "white"], threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_spectrum(mz_list, intensity_list, title=''):
    figure = plt.figure(figsize=(20, 10));
    ax = figure.add_subplot(1, 1, 1);
    ax.autoscale();

    peak_lines = [];
    for mz, intensity in zip(mz_list, intensity_list): peak_lines.append([(mz, 0), (mz, intensity)])
    ax.add_collection(mc.LineCollection(peak_lines, colors=[(0.8, 0.8, 0.8, 0.8)] * len(peak_lines)));

    ax.plot([np.min(mz_list) - 20, np.max(mz_list) + 20], [0.0, 0.0], linewidth=2, color='black');
    ax.autoscale();
    ax.set_xlabel('Mz', fontsize=18);
    ax.set_ylabel('Intensity', fontsize=18);
    if len(title) > 0: ax.set_title(title, fontsize=18);

    return figure;


def plot_binned_spectrum_with_theoretical(obv_mzbin_arr, obv_intensity_arr, ion_mzbin_arr, ion_label_arr, title='', save_url=''):
    figure = plt.figure(figsize=(16, 9))
    ax = figure.add_subplot(1, 1, 1);
    ax.autoscale();

    obv_intensity_arr /= (1.05 * np.max(obv_intensity_arr));
    obv_intensity_arr *= 100;

    miss_peak_lines = []; hit_peak_lines = [];
    hit_obv_indices = np.asarray([(x in ion_mzbin_arr) for x in obv_mzbin_arr], dtype=bool); miss_obv_indices = ~hit_obv_indices;
    if np.any(hit_obv_indices):
        for mzbin, intensity in zip(obv_mzbin_arr[hit_obv_indices], obv_intensity_arr[hit_obv_indices]): hit_peak_lines.append([(mzbin*0.02, 0), (mzbin*0.02, intensity)]);
    if np.any(miss_obv_indices):
        for mzbin, intensity in zip(obv_mzbin_arr[miss_obv_indices], obv_intensity_arr[miss_obv_indices]): miss_peak_lines.append([(mzbin*0.02, 0), (mzbin*0.02, intensity)]);


    hit_ion_labels = [];
    for ion_mzbin, ion_label in zip(ion_mzbin_arr, ion_label_arr):
        if ion_mzbin not in obv_mzbin_arr: continue;
        hit_ion_labels.append(ion_label);

        obv_mzbin_idx = np.where(obv_mzbin_arr==ion_mzbin)[0][0];
        position = (ion_mzbin*0.02, obv_intensity_arr[obv_mzbin_idx]);
        ax.annotate(ion_label, xy=position, fontsize=18, rotation=90, rotation_mode='anchor');

    ax.add_collection(mc.LineCollection(miss_peak_lines, colors=[(0.4, 0.4, 0.4, 0.3)] * len(miss_peak_lines), linewidths=(np.ones(len(miss_peak_lines)) * 1.0).tolist() ));
    ax.add_collection(mc.LineCollection(hit_peak_lines, colors=['green'] * len(hit_peak_lines), linewidths=(np.ones(len(hit_peak_lines)) * 3.0).tolist() ));

    x_min = (min(np.min(obv_mzbin_arr), np.min(ion_mzbin_arr)) - 50)*0.02;
    x_max = (max(np.max(obv_mzbin_arr), np.max(ion_mzbin_arr)) + 50)*0.02;
    logger.info('x_min={}\tx_max={}'.format(x_min, x_max));

    ax.plot([x_min, x_max], [0.0, 0.0], linewidth=2, color='black');
    ax.autoscale();
    ax.set_xlabel('m/z', fontsize=25);
    ax.set_ylabel('Intensity', fontsize=25);
    ax.xaxis.set_tick_params(labelsize=22);
    ax.yaxis.set_tick_params(labelsize=22);

    yticks = mtick.FormatStrFormatter('%.0f%%');
    ax.yaxis.set_major_formatter(yticks);
    ax.grid(linestyle=':');

    ax.set_xlim(x_min, min(x_max, 2000));
    ax.set_ylim(0, 100);

    if len(title) > 0: ax.set_title(title, fontsize=18);

    if len(save_url) > 0:
        pp = PdfPages(save_url);
        pp.savefig(figure, bbox_inches='tight');
        pp.close();
        plt.close(figure);
        del figure;

    return hit_ion_labels;


def plot_ion_elution_chromatogram(ms1_chrom_list, candidate_ms1scan_list, ion_chrom_map, candidate_ms2scan_list, hit_ion_labels, title='', save_url=''):
    filtered_ion_list = [];
    for ion_label in ion_chrom_map:
        if np.sum(ion_chrom_map[ion_label]) <= 0: continue;
        if ion_label not in hit_ion_labels: continue;
        filtered_ion_list.append(ion_label);
    logger.info('ion_chrom_map={}\tfiltered_ion_list={}'.format(len(ion_chrom_map), len(filtered_ion_list) ));
    if len(filtered_ion_list) <= 1: return ;

    # import networkx as nx;
    # corr_graph = nx.Graph();
    # for ionidx1 in range(len(filtered_ion_list)):
    #     ion_label1 = filtered_ion_list[ionidx1];
    #     for ionidx2 in range(ionidx1+1, len(filtered_ion_list)):
    #         ion_label2 = filtered_ion_list[ionidx2];
    #
    #         corr = spectrum_utils.calc_normalized_innerproduct(np.sqrt(ion_chrom_map[ion_label1]), np.sqrt(ion_chrom_map[ion_label2]));
    #         if corr > 0.9: corr_graph.add_edge(ion_label1, ion_label2);
    #
    # corr_cluster_list = [list(c) for c in nx.connected_components(corr_graph) if len(c) > 1];
    # corr_size_list = [len(c) for c in corr_cluster_list];
    #
    # max_idx = np.argmax(corr_size_list); max_size = corr_size_list[max_idx]; max_corr_cluster = corr_cluster_list[max_idx];
    # logger.info('max_corr_cluster={}\t={}'.format(len(max_corr_cluster), max_corr_cluster));

    max_corr_cluster = filtered_ion_list;

    figure = plt.figure(figsize=(16, 8))
    ax = figure.add_subplot(1, 1, 1);
    ax.autoscale();

    max_ms2_signal = 1.05 * np.max([np.max(ion_chrom_map[ion_label]) for ionidx, ion_label in enumerate(max_corr_cluster)]);
    max_ms1_signal = max(1.05*np.max(ms1_chrom_list), 0.3 * max_ms2_signal );

    min_scan = min(np.min(candidate_ms1scan_list), np.min(candidate_ms2scan_list));
    max_scan = max(np.max(candidate_ms1scan_list), np.max(candidate_ms2scan_list));
    logger.info('max_ms1_signal={}\tmax_ms2_signal={}\tmin_scan={}\tmax_scan={}'.format(max_ms1_signal, max_ms2_signal, min_scan, max_scan));

    ms1_chrom_list_mz0 = -np.asarray([x[0] for x in ms1_chrom_list], dtype=float) / max_ms1_signal * 100;
    ms1_chrom_list_mz1 = -np.asarray([x[1] for x in ms1_chrom_list], dtype=float) / max_ms1_signal * 100;
    ms1_chrom_list_mz2 = -np.asarray([x[2] for x in ms1_chrom_list], dtype=float) / max_ms1_signal * 100;

    ax.plot(candidate_ms1scan_list, ms1_chrom_list_mz0, color='1.0', linestyle='-', linewidth=1);
    ax.plot(candidate_ms1scan_list, ms1_chrom_list_mz1, color='0.6', linestyle='--', linewidth=1);
    ax.plot(candidate_ms1scan_list, ms1_chrom_list_mz2, color='0.2', linestyle=':', linewidth=1);

    for ionidx, ion_label in enumerate(max_corr_cluster):
        ax.plot(candidate_ms2scan_list, ion_chrom_map[ion_label] / max_ms2_signal * 100,  linestyle='-', linewidth=1, label=ion_label);

    ax.plot([min_scan, max_scan], [0, 0], linestyle='-', linewidth=1, color='black');
    ax.plot([np.median(candidate_ms2scan_list), np.median(candidate_ms2scan_list)], [-100.0, 100.0], linestyle='--', linewidth=2, color='black');

    # ax.annotate('MS2 chromatograms', xy=(min_scan, max_ms2_signal - (max_ms2_signal-max_ms1_signal)/20 ), fontsize=22);
    # ax.annotate('MS1 chromatogram', xy=(min_scan, max_ms1_signal+10), fontsize=22);

    ax.autoscale();

    ax.set_xlabel('Retention time', fontsize=25);
    ax.set_ylabel('Intensity', fontsize=25);
    ax.xaxis.set_tick_params(labelsize=22);
    ax.yaxis.set_tick_params(labelsize=22);
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,y: '{:.1f}%'.format(abs(x)) ));
    # yticks = mtick.FormatStrFormatter('%.0f%%');
    # ax.yaxis.set_major_formatter(yticks);

    ax.set_xlim(min_scan, max_scan);
    ax.set_ylim(-100, 100);
    # ax.set_ylim(max_ms1_signal, max_ms2_signal);



    ax.legend(fontsize=15, bbox_to_anchor=(1, 0.5), loc='center left');
    ax.grid(linestyle=':');

    ax.set_title(title, fontsize=18);

    if len(save_url) > 0:
        pp = PdfPages(save_url);
        pp.savefig(figure, bbox_inches='tight');
        pp.close();
        plt.close(figure);
        del figure;





