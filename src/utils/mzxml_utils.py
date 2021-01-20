import os, sys, re;
sys.path.append('..')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from enum import Enum
import numpy as np
from scipy import spatial
from pyteomics import mgf, mzxml
from scipy.stats import pearsonr, spearmanr;

import spectrum_utils, debiase_utils;


def get_cache_dir(mzxml_url):
    cache_dir = os.path.join(os.path.dirname(mzxml_url), 'cache');
    if not os.path.exists(cache_dir): os.makedirs(cache_dir);
    return cache_dir;


def get_scan_gap(mzxml_url):
    scan_gap1 = len(get_ms1_scanwins(mzxml_url))+1;
    logger.info('scan_gap1: ' + str(scan_gap1));
    return scan_gap1;


def get_ms1_scanwins(mzxml_url):

    cached_ms1_scanwin_url = '{}/{}.npy'.format(get_cache_dir(mzxml_url), 'ms1_scanwin_range');
    if os.path.exists(cached_ms1_scanwin_url):
        logger.info('cache exists in '+cached_ms1_scanwin_url);
        ms1_scanwins_arr = np.load(cached_ms1_scanwin_url, allow_pickle=True);
    else:
        tmp_url = '{}/tmp_{}.txt'.format(get_cache_dir(mzxml_url), str(np.random.randint(0,1e10)));
        os.system('grep precursorMz ' + mzxml_url + ' > ' + tmp_url);

        ms1_scanwins_arr = []; mz_arr = []; winWidth_arr = [];

        # with open(tmp_url, 'r') as tempReader:
        #     for line in tempReader:
        #         mzStr = line.split('>')[1].split('<')[0];
        #         mz = float(mzStr);
        #         window_search = re.search('windowWideness="(.*)">', line, re.IGNORECASE);
        #         winWidth = float(window_search.group(1));
        #         tp = (mz - 0.5 * winWidth, mz + 0.5 * winWidth);
        #
        #         if len(ms1_scanwins_arr) <= 0:
        #             ms1_scanwins_arr.append(tp);
        #         elif ms1_scanwins_arr[-1][0] < tp[0]:
        #             ms1_scanwins_arr.append(tp);
        #         else:
        #             break;

        with open(tmp_url, 'r') as tempReader:
            for line in tempReader:
                mzStr = line.split('>')[1].split('<')[0];
                mz = float(mzStr);

                winWidth = -1;
                if 'windowWideness' in line:
                    window_search = re.search('windowWideness="(.*)">', line, re.IGNORECASE);
                    winWidth = float(window_search.group(1));

                if len(mz_arr) <= 0:
                    mz_arr.append(mz); winWidth_arr.append(winWidth);
                elif mz_arr[-1] < mz:
                    mz_arr.append(mz); winWidth_arr.append(winWidth);
                else:
                    break;

        for mz, winWidth in zip(mz_arr, winWidth_arr):
            if winWidth < 0: winWidth = float(mz_arr[2]-mz_arr[1]);
            ms1_scanwins_arr.append((mz - 0.5 * winWidth, mz + 0.5 * winWidth));

        logger.info('mz_arr={}\twinWidth_arr={}\tms1_scanwins_arr={}'.format(mz_arr, winWidth_arr, ms1_scanwins_arr));

        os.remove(tmp_url);
        np.save(cached_ms1_scanwin_url, ms1_scanwins_arr);
        logger.info('Now save cache to ' + cached_ms1_scanwin_url);
    logger.info('ms1_scanwins_arr loaded! size: ' + str(len(ms1_scanwins_arr)));
    return ms1_scanwins_arr;


def get_ms1_mz_range(mzxml_url):
    ms1_scanwins = get_ms1_scanwins(mzxml_url);
    min_mz = ms1_scanwins[0][0];
    max_mz = ms1_scanwins[-1][1];
    return min_mz, max_mz;


def get_ms1_scanwinindices_by_mz(ms1_scanwins_arr, mz):

    winidx_list = [];
    for winIdx in range(len(ms1_scanwins_arr)):
        if (mz >= ms1_scanwins_arr[winIdx][0]) and (mz <= ms1_scanwins_arr[winIdx][1]):
            winidx_list.append(winIdx);

    # if len(winidx_list) <= 0:
    #     if mz < ms1_scanwins_arr[0][0]: winidx_list.append(0);
    #     elif mz > ms1_scanwins_arr[-1][1]: winidx_list.append(len(ms1_scanwins_arr) - 1);
    return winidx_list;


def get_ms1_scannums_from_mzxml(mzxml_url):
    ms1_scannum_arr = [];

    cached_ms1_scannums_url = '{}/{}.npy'.format(get_cache_dir(mzxml_url), 'ms1_scannums');

    if os.path.exists(cached_ms1_scannums_url):
        logger.info('cache exists in '+cached_ms1_scannums_url);
        ms1_scannum_arr = np.load(cached_ms1_scannums_url, allow_pickle=True);
    else:
        with mzxml.read(mzxml_url) as reader:
            for spectrum in reader:
                scanNum = int(spectrum['num']);
                msLevel = int(spectrum['msLevel']);

                if msLevel == 1:
                    ms1_scannum_arr.append(scanNum);

        np.save(cached_ms1_scannums_url, ms1_scannum_arr);
        logger.info('Now save cache to ' + cached_ms1_scannums_url);
    logger.info('ms1_scannum_arr loaded! size: ' + str(len(ms1_scannum_arr)));
    return np.asarray(ms1_scannum_arr);

def get_scanwin_ms2scannum_map_from_mzxml(mzxml_url):
    scanwin_ms2scannum_map = {};

    cached_ms2_scanwinidx_scannum_url = '{}/{}.npy'.format(get_cache_dir(mzxml_url), 'ms2_scanwinidx_scannum');

    if os.path.exists(cached_ms2_scanwinidx_scannum_url):
        logger.info('cache exists in '+cached_ms2_scanwinidx_scannum_url);
        scanwin_ms2scannum_map = np.load(cached_ms2_scanwinidx_scannum_url, allow_pickle=True).item();
    else:
        # scan_gap = get_scan_gap(mzxml_url);
        ms1_scanwins_arr = get_ms1_scanwins(mzxml_url);
        for winIdx in range(len(ms1_scanwins_arr)):
            scanwin_ms2scannum_map[winIdx] = [];

        with mzxml.read(mzxml_url) as reader:
            for spectrum in reader:
                scanNum = int(spectrum['num']); msLevel = int(spectrum['msLevel']);
                if msLevel == 2:
                    precursorMz = spectrum['precursorMz'][0]['precursorMz'];
                    winidx_list = get_ms1_scanwinindices_by_mz(ms1_scanwins_arr, precursorMz); assert len(winidx_list) == 1;
                    scanwin_ms2scannum_map[winidx_list[0]].append(scanNum);

        np.save(cached_ms2_scanwinidx_scannum_url, scanwin_ms2scannum_map);
        logger.info('Now save cache to ' + cached_ms2_scanwinidx_scannum_url);
    logger.info('scanwin_ms2scannum_map loaded! size: ' + str(len(scanwin_ms2scannum_map)));
    return scanwin_ms2scannum_map;


def get_ms2scannum_scanwin_map_from_mzxml(mzxml_url):
    ms2scannum_scanwin_map = {};
    scanwin_ms2scannum_map = get_scanwin_ms2scannum_map_from_mzxml(mzxml_url);
    for scanwinidx, ms2_scannum_arr in scanwin_ms2scannum_map.items():
        ms2scannum_scanwin_map.update(dict(zip(ms2_scannum_arr, np.ones(len(ms2_scannum_arr), dtype=int) * scanwinidx)));
    logger.info('ms2scannum_scanwin_map loaded! size: ' + str(len(ms2scannum_scanwin_map)));
    return ms2scannum_scanwin_map;


def get_scannum_rt_pair(mzxml_url):
    cached_scannum_rt_pair_url = '{}/{}.npy'.format(get_cache_dir(mzxml_url), 'scannum_rt_pair');
    scannum_list = []; rt_list = [];

    if os.path.exists(cached_scannum_rt_pair_url):
        logger.info('cache exists in '+cached_scannum_rt_pair_url);
        scannum_list, rt_list = np.load(cached_scannum_rt_pair_url, allow_pickle=True);
    else:
        with mzxml.read(mzxml_url) as reader:
            for spectrum in reader:
                scanNum = int(spectrum['num']); rt = float(spectrum['retentionTime']);
                scannum_list.append(scanNum); rt_list.append(rt);
        np.save(cached_scannum_rt_pair_url, (scannum_list, rt_list));
        logger.info('Now save cache to ' + cached_scannum_rt_pair_url);
    logger.info('scannum_rt_pair loaded! size: ' + str(len(scannum_list)));
    return np.asarray(scannum_list, dtype=int), np.asarray(rt_list);


def get_scannum_mzintensity_map_from_mzxml(mzxml_url):
    scannum_mzintensity_map = {};
    cached_scannum_mzintensity_map_url = '{}/{}.npy'.format(get_cache_dir(mzxml_url), 'scannum_mzintensity');

    if os.path.exists(cached_scannum_mzintensity_map_url):
        logger.info('cache exists in '+cached_scannum_mzintensity_map_url);
        scannum_mzintensity_map = np.load(cached_scannum_mzintensity_map_url, allow_pickle=True).item();
    else:
        with mzxml.read(mzxml_url) as reader:
            for spectrum in reader:
                scanNum = int(spectrum['num']);
                msLevel = int(spectrum['msLevel']);

                logger.info('scan=' + str(scanNum) + '\tm/z array and intensity array: ');
                mz_arr = spectrum['m/z array'];
                intensity_arr = spectrum['intensity array'];

                # if msLevel > 1:
                #     intensity_arr = spectrum_utils.get_transformed_intensity(mz_arr, intensity_arr, spectrum_utils.IntTransformType.Sqrt);
                #     mz_arr, intensity_arr = spectrum_utils.get_grassfiltered_mz_intensity(mz_arr, intensity_arr);
                #     intensity_arr = spectrum_utils.get_transformed_intensity(mz_arr, intensity_arr, spectrum_utils.IntTransformType.SegmentedBasePeakNorm);
                #
                if len(intensity_arr) <= 0: continue;
                encoded_source = spectrum_utils.encode_mzintensity_arr(mz_arr, intensity_arr);
                scannum_mzintensity_map[scanNum] = encoded_source;

        np.save(cached_scannum_mzintensity_map_url, scannum_mzintensity_map);
        logger.info('Now save cache to ' + cached_scannum_mzintensity_map_url);
    logger.info('scannum_mzintensity_map loaded! size: ' + str(len(scannum_mzintensity_map)));
    return scannum_mzintensity_map;


def get_unzipped_scannum_mzintensity_map_from_mzxml(mzxml_url):
    unzipped_scannum_mzintensity_map = {};
    cached_scannum_mzbin_intensity_map_url = '{}/{}.npy'.format(get_cache_dir(mzxml_url), 'scannum_mzintensity_unzipped');

    if os.path.exists(cached_scannum_mzbin_intensity_map_url):
        logger.info('cache exists in '+cached_scannum_mzbin_intensity_map_url);
        unzipped_scannum_mzintensity_map = np.load(cached_scannum_mzbin_intensity_map_url, allow_pickle=True).item();
    else:
        with mzxml.read(mzxml_url) as reader:
            for spectrum in reader:
                scanNum = int(spectrum['num']);

                logger.info('scan=' + str(scanNum) + '\tm/z array and intensity array: ');
                mz_arr = spectrum['m/z array'];
                intensity_arr = spectrum['intensity array'];
                if len(intensity_arr) <= 0: continue;
                unzipped_scannum_mzintensity_map[scanNum] = (np.asarray(mz_arr), np.asarray(intensity_arr));

        # np.save(cached_scannum_mzbin_intensity_map_url, unzipped_scannum_mzintensity_map);
        logger.info('Now save cache to ' + cached_scannum_mzbin_intensity_map_url);
    logger.info('unzipped_scannum_mzintensity_map loaded! size: ' + str(len(unzipped_scannum_mzintensity_map)));
    return unzipped_scannum_mzintensity_map;


def get_ms2_to_ms1_scannum_map(mzxml_url):
    scan_gap = get_scan_gap(mzxml_url);
    ms1_scannum_arr = get_ms1_scannums_from_mzxml(mzxml_url);
    ms2_scannum_arr_map = get_scanwin_ms2scannum_map_from_mzxml(mzxml_url);

    ms2_to_ms1_scannum_map = {};
    for scanwinidx, ms2_scannum_arr in ms2_scannum_arr_map.items():
        # logger.info('ms2_scannum_arr={}'.format(np.asarray(ms2_scannum_arr)));
        while len(ms1_scannum_arr) < len(ms2_scannum_arr):
            ms1_scannum_arr = np.append(ms1_scannum_arr, ms1_scannum_arr[-1] + scan_gap);
        while len(ms2_scannum_arr) < len(ms1_scannum_arr):
            ms2_scannum_arr = np.append(ms2_scannum_arr, ms2_scannum_arr[-1] + scan_gap);
        assert len(ms1_scannum_arr) == len(ms2_scannum_arr);
        ms2_to_ms1_scannum_map.update(dict(zip(ms2_scannum_arr, ms1_scannum_arr)));
    return ms2_to_ms1_scannum_map;


def get_ms1_slope_intercept(mzxml_url, del_lognoise=False):
    ms1_slope_intercept_url = '{}/ms1_slope_intercept_dellognoise{}.npy'.format(get_cache_dir(mzxml_url), int(del_lognoise));

    if os.path.isfile(ms1_slope_intercept_url): ms1_slope_intercept_map = np.load(ms1_slope_intercept_url, allow_pickle=True).item();
    else:
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

        ms1_slope_intercept_map = {};
        ms1_mz_intensity_map = get_ms1_mz_intensity(mzxml_url, intensity_type=2);
        ms1_lognoise_map = get_ms1_lognoise(mzxml_url);

        for ms1scan in ms1_mz_intensity_map:
            _, intensity_arr = ms1_mz_intensity_map[ms1scan];
            if len(intensity_arr) < 500: continue;
            if del_lognoise: intensity_arr -= ms1_lognoise_map[ms1scan];
            intensity_arr = intensity_arr[intensity_arr>0];

            ignore_top = 20; retain_cnt = min(500, int((len(intensity_arr) - ignore_top) * 0.2));
            slope, intercept, x_values, y_values = fit_log_linear(intensity_arr, ignore_top=ignore_top, ignore_bottom=max(len(intensity_arr)-retain_cnt-ignore_top, 0));
            ms1_slope_intercept_map[ms1scan] = (slope[0], intercept[0]);

        np.save(ms1_slope_intercept_url, ms1_slope_intercept_map);

    logger.info('ms1_slope_intercept_map={}'.format(len(ms1_slope_intercept_map)));
    return ms1_slope_intercept_map;


def get_ms2_mzbin(mzxml_url, top=-1):
    if top <=0: cached_ms2_mzbin_url = '{}/ms2_mzbin.npy'.format(get_cache_dir(mzxml_url));
    else: cached_ms2_mzbin_url = '{}/ms2_mzbin_top{}.npy'.format(get_cache_dir(mzxml_url), top);

    if os.path.isfile(cached_ms2_mzbin_url): ms2_mzbin_map = np.load(cached_ms2_mzbin_url, allow_pickle=True).item();
    else:
        ms2_mzbin_map = {};
        with mzxml.read(mzxml_url) as reader:
            for spectrum in reader:
                scannum = int(spectrum['num']); msLevel = int(spectrum['msLevel']);
                if msLevel <= 1: continue;

                mz_arr = np.asarray(spectrum['m/z array']);
                if len(mz_arr) <= 0: continue;

                mzbin_arr = np.asarray([spectrum_utils.mz_to_binidx(mz) for mz in mz_arr], dtype=int);
                intensity_arr = np.asarray(spectrum['intensity array']); intensity_arr = np.sqrt(intensity_arr);

                uniq_mzbin_arr = np.unique(mzbin_arr);
                if len(uniq_mzbin_arr) != len(mz_arr):
                    indices = np.argsort(intensity_arr);
                    mzbin_intensity_map = dict(zip(mzbin_arr[indices], intensity_arr[indices]));

                    mzbin_arr = uniq_mzbin_arr;
                    intensity_arr = np.asarray([mzbin_intensity_map[mzbin] for mzbin in uniq_mzbin_arr]);

                intensity_arr /= np.max(intensity_arr);
                valid_indices = np.where(intensity_arr >= 0.05)[0];
                mzbin_arr = mzbin_arr[valid_indices]; intensity_arr = intensity_arr[valid_indices];
                logger.info('scannum={}\tmzbin_arr={}'.format(scannum, len(mzbin_arr) ));

                # segment-norm
                seg_num = 10;
                window_mzbin = np.linspace(np.min(mzbin_arr), np.max(mzbin_arr), seg_num + 1);

                mzbin_arr_new = np.asarray([], dtype=int);
                for segIdx in range(seg_num):
                    seg_indices = np.where((mzbin_arr >= window_mzbin[segIdx]) & (mzbin_arr <= window_mzbin[segIdx + 1]))[0];
                    if len(seg_indices) <= 0: continue;

                    mzbin_arr_sub = mzbin_arr[seg_indices]; intensity_arr_sub = intensity_arr[seg_indices];
                    max_indices = np.argsort(intensity_arr_sub)[::-1];
                    if top > 0: max_indices = max_indices[:min(top, len(max_indices))];
                    mzbin_arr_new = np.concatenate((mzbin_arr_new, mzbin_arr_sub[max_indices] ));
                mzbin_arr = mzbin_arr_new;

                ms2_mzbin_map[scannum] = mzbin_arr;
        np.save(cached_ms2_mzbin_url, ms2_mzbin_map);
    logger.info('ms2_mzbin_map={}'.format(len(ms2_mzbin_map)));
    return ms2_mzbin_map;


def get_ms2_mzbin_intensity(mzxml_url):
    cached_ms2_mzbin_intensity_url = '{}/ms2_mzbin_intensity.npy'.format(get_cache_dir(mzxml_url));
    if os.path.isfile(cached_ms2_mzbin_intensity_url): ms2_mzbin_intensity_map = np.load(cached_ms2_mzbin_intensity_url, allow_pickle=True).item();
    else:
        ms2_mzbin_intensity_map = {};
        with mzxml.read(mzxml_url) as reader:
            for spectrum in reader:
                scannum = int(spectrum['num']); msLevel = int(spectrum['msLevel']);
                if msLevel <= 1: continue;

                mz_arr = np.asarray(spectrum['m/z array']);
                if len(mz_arr) <= 0: continue;

                mzbin_arr = np.asarray([spectrum_utils.mz_to_binidx(mz) for mz in mz_arr], dtype=int);
                intensity_arr = np.sqrt(spectrum['intensity array']);

                uniq_mzbin_arr = np.unique(mzbin_arr);
                if len(uniq_mzbin_arr) != len(mz_arr):
                    indices = np.argsort(intensity_arr);
                    mzbin_intensity_map = dict(zip(mzbin_arr[indices], intensity_arr[indices]));

                    mzbin_arr = uniq_mzbin_arr;
                    intensity_arr = np.asarray([mzbin_intensity_map[mzbin] for mzbin in uniq_mzbin_arr]);
                ms2_mzbin_intensity_map[scannum] = (mzbin_arr, intensity_arr);

        np.save(cached_ms2_mzbin_intensity_url, ms2_mzbin_intensity_map);
    logger.info('ms2_mzbin_intensity_map={}'.format(len(ms2_mzbin_intensity_map)));
    return ms2_mzbin_intensity_map;



def get_ms1_mz_intensity(mzxml_url, intensity_type=2):
    assert intensity_type in [0, 1, 2];
    if intensity_type == 0: cached_ms1_mz_intensity_url = '{}/ms1_mz_intensity.npy'.format(get_cache_dir(mzxml_url) );
    else: cached_ms1_mz_intensity_url = '{}/ms1_mz_intensity_type{}.npy'.format(get_cache_dir(mzxml_url), intensity_type );

    if os.path.isfile(cached_ms1_mz_intensity_url): ms1_mz_intensity_map = np.load(cached_ms1_mz_intensity_url, allow_pickle=True).item();
    else:
        ms1_scanwins_arr = get_ms1_scanwins(mzxml_url); min_mz = ms1_scanwins_arr[0][0]; max_mz = ms1_scanwins_arr[-1][1];
        logger.info('min_mz={}\tmax_mz={}'.format(min_mz, max_mz));

        ms1_mz_intensity_map = {};
        with mzxml.read(mzxml_url) as reader:
            for spectrum in reader:
                scannum = int(spectrum['num']); msLevel = int(spectrum['msLevel']);
                if msLevel > 1: continue;

                mz_arr = np.asarray(spectrum['m/z array']); intensity_arr = np.asarray(spectrum['intensity array']);
                valid_indices = np.where((mz_arr >= min_mz) & (mz_arr <= max_mz))[0];

                mz_arr = mz_arr[valid_indices]; intensity_arr = intensity_arr[valid_indices];
                if intensity_type == 1: intensity_arr = np.sqrt(intensity_arr);
                elif intensity_type == 2: intensity_arr = np.log(intensity_arr + 1);

                ms1_mz_intensity_map[scannum] = (mz_arr, intensity_arr);
        np.save(cached_ms1_mz_intensity_url, ms1_mz_intensity_map);
    logger.info('ms1_mz_intensity_map={}'.format(len(ms1_mz_intensity_map)));
    return ms1_mz_intensity_map;



def get_ms1_mzbin_freq(mzxml_url):
    cached_ms1_mzbin_freq_url = '{}/ms1_mzbin_freq.npy'.format(get_cache_dir(mzxml_url));
    if os.path.isfile(cached_ms1_mzbin_freq_url): ms1_mzbin_freq_map = np.load(cached_ms1_mzbin_freq_url, allow_pickle=True).item();
    else:
        ms1_mz_intensity_map = get_ms1_mz_intensity(mzxml_url); mzbin_ms1scans_map = {}; ms1_mzbin_freq_map = {}; total_ms1scans = len(ms1_mz_intensity_map);
        for ms1scan in ms1_mz_intensity_map:
            mz_arr, intensity_arr = ms1_mz_intensity_map[ms1scan];
            mzbin_arr = np.asarray([spectrum_utils.mz_to_binidx(mz) for mz in mz_arr], dtype=int);

            for mzbin in np.unique(mzbin_arr):
                if mzbin not in mzbin_ms1scans_map: mzbin_ms1scans_map[mzbin] = [];
                mzbin_ms1scans_map[mzbin].append(ms1scan);

        for mzbin in mzbin_ms1scans_map:
            ms1scans = len(mzbin_ms1scans_map[mzbin]);
            ms1_mzbin_freq_map[mzbin] = 1.0*ms1scans/total_ms1scans;
            # logger.info('mzbin={}\ttotal_ms1scans={}\tmzbin_freq={}'.format(mzbin, total_ms1scans, ms1_mzbin_freq_map[mzbin]));

        np.save(cached_ms1_mzbin_freq_url, ms1_mzbin_freq_map);
    logger.info('ms1_mzbin_freq_map={}'.format(len(ms1_mzbin_freq_map)));
    return ms1_mzbin_freq_map;


def get_ms1_mz_coverage(mzxml_url, ppm):
    cached_ms1_mz_coverage_url = '{}/ms1_mz_coverage_ppm{}.npy'.format(get_cache_dir(mzxml_url), ppm);
    if os.path.isfile(cached_ms1_mz_coverage_url): ms1_mz_coverage_map = np.load(cached_ms1_mz_coverage_url, allow_pickle=True).item();
    else:
        ms1_mz_intensity_map = get_ms1_mz_intensity(mzxml_url); ms1_mz_coverage_map = {};
        for ms1scan in ms1_mz_intensity_map:
            mz_arr, intensity_arr = ms1_mz_intensity_map[ms1scan]; mz_tuple_arr = [];
            for mz, intensity in zip(mz_arr, intensity_arr):
                mz_low, mz_high = spectrum_utils.calc_mztol_range_by_ppm(mz, ppm);
                mz_tuple_arr.append((mz_low, mz_high));
            mz_tuple_arr.sort(key=lambda x: x[0]);

            prev_high = -1; mz_sum = 0.0; mz_span = mz_tuple_arr[-1][1] - mz_tuple_arr[0][0];
            for mz_low, mz_high in mz_tuple_arr:
                if mz_low >= prev_high: mz_sum += (mz_high - mz_low);
                else:
                    assert mz_high >= prev_high;
                    mz_sum += (mz_high - prev_high);
                prev_high = mz_high;
            coverage = mz_sum / mz_span;
            ms1_mz_coverage_map[ms1scan] = coverage;
            # logger.info('ms1scan={}\tmz_sum={}\tmz_span={}\tcoverage={}'.format(ms1scan, mz_sum, mz_span, coverage));
        np.save(cached_ms1_mz_coverage_url, ms1_mz_coverage_map);

    values = list(ms1_mz_coverage_map.values());
    logger.info('ms1_mz_coverage_map={}\tmin={}\tmax={}'.format(len(ms1_mz_coverage_map), np.min(values), np.max(values) ));
    return ms1_mz_coverage_map;


def get_ms1_expected_intensity(mzxml_url):
    cached_ms1_expected_intensity_url = '{}/ms1_expected_intensity.npy'.format(get_cache_dir(mzxml_url));
    if os.path.isfile(cached_ms1_expected_intensity_url): ms1_expected_intensity_map = np.load(cached_ms1_expected_intensity_url, allow_pickle=True).item();
    else:
        ms1_slope_intercept_map = get_ms1_slope_intercept(mzxml_url, del_lognoise=False);
        ms1_mz_intensity_map = get_ms1_mz_intensity(mzxml_url, intensity_type=2);
        ms1_lognoise_map = get_ms1_lognoise(mzxml_url);

        ms1_expected_intensity_map = {};
        ms1scan_arr = np.sort(list(ms1_mz_intensity_map.keys()));
        for ms1scan in ms1scan_arr:
            slope, intercept = ms1_slope_intercept_map[ms1scan];
            # mz_arr, intensity_arr = ms1_mz_intensity_map[ms1scan];

            max_logrank = intercept; max_logInt = -intercept / slope;
            # max_rank = int(np.floor(np.exp(max_logrank)));
            # exp_intensity1 = np.mean((np.log(np.asarray(list(range(1, max_rank+1))))-intercept) / slope);

            total_partition = 1e4;
            exp_intensity2 = np.mean((np.linspace(0, max_logrank, total_partition)-intercept) / slope)

            ms1_expected_intensity_map[ms1scan] = exp_intensity2;
            logger.info('ms1scan={}\tlognoise={}\texp_intensity={}'.format(ms1scan, ms1_lognoise_map[ms1scan], exp_intensity2));

        np.save(cached_ms1_expected_intensity_url, ms1_expected_intensity_map);
    logger.info('ms1_expected_intensity_map={}'.format(len(ms1_expected_intensity_map)));
    return ms1_expected_intensity_map;




def get_ms1_scanwin_tic(mzxml_url):
    cached_ms1_scanwin_tic_url = '{}/{}.npy'.format(get_cache_dir(mzxml_url), 'ms1_scanwin_tic');
    if os.path.isfile(cached_ms1_scanwin_tic_url): ms1_scanwin_tic_map = np.load(cached_ms1_scanwin_tic_url, allow_pickle=True).item();
    else:
        ms1_scanwins_arr = get_ms1_scanwins(mzxml_url);
        ms1_scanwin_tic_map = {};

        with mzxml.read(mzxml_url) as reader:
            for spectrum in reader:
                scannum = int(spectrum['num']); msLevel = int(spectrum['msLevel']);
                if msLevel > 1: continue;

                mz_arr = np.asarray(spectrum['m/z array']); intensity_arr = np.asarray(spectrum['intensity array']);
                for scanwin_idx, scanwin in enumerate(ms1_scanwins_arr):
                    valid_intensity_arr = intensity_arr[np.where((mz_arr<=scanwin[1])&(mz_arr>=scanwin[0]))];
                    if len(valid_intensity_arr) <= 0: continue;

                    ms1_scanwin = '{}_{}'.format(scannum, scanwin_idx);
                    ms1_scanwin_tic_map[ms1_scanwin] = (np.mean(valid_intensity_arr), np.sum(valid_intensity_arr));

        np.save(cached_ms1_scanwin_tic_url, ms1_scanwin_tic_map);
    logger.info('ms1_scanwin_tic_map={}'.format(len(ms1_scanwin_tic_map)));
    return ms1_scanwin_tic_map;


def get_ms1_tic(mzxml_url):
    cached_ms1_tic_url = '{}/{}.npy'.format(get_cache_dir(mzxml_url), 'ms1_tic');
    if os.path.isfile(cached_ms1_tic_url): ms1_tic_map = np.load(cached_ms1_tic_url, allow_pickle=True).item();
    else:
        ms1_tic_map = {};
        with mzxml.read(mzxml_url) as reader:
            for spectrum in reader:
                scannum = int(spectrum['num']); msLevel = int(spectrum['msLevel']);
                if msLevel > 1: continue;

                intensity_arr = np.asarray(spectrum['intensity array']);
                ms1_tic_map[scannum] = (np.mean(intensity_arr), np.sum(intensity_arr));
        np.save(cached_ms1_tic_url, ms1_tic_map);
    logger.info('ms1_tic_map={}'.format(len(ms1_tic_map)));
    return ms1_tic_map;


def get_ms1_lognoise(mzxml_url):
    cached_ms1_lognoise_map_url = '{}/{}.npy'.format(get_cache_dir(mzxml_url), 'ms1_lognoise');
    if os.path.exists(cached_ms1_lognoise_map_url): ms1_lognoise_map = np.load(cached_ms1_lognoise_map_url, allow_pickle=True).item();
    else:
        leftover_ms1scan_list = []; lognoise_list = []; ms1_lognoise_map = {};
        with mzxml.read(mzxml_url) as reader:
            for spectrum in reader:
                scannum = int(spectrum['num']); msLevel = int(spectrum['msLevel']);
                if msLevel > 1: continue;

                ms1_intensity_arr = np.asarray(spectrum['intensity array']);
                if len(ms1_intensity_arr) <= 100:
                    leftover_ms1scan_list.append(scannum);
                    continue;

                estimated_noise = spectrum_utils.estimate_noiselevel(ms1_intensity_arr, percent=10);
                if estimated_noise > 0:
                    estimated_lognoise = np.log(estimated_noise + 1);
                    lognoise_list.append(estimated_lognoise);
                    ms1_lognoise_map[scannum] = estimated_lognoise;
                else: leftover_ms1scan_list.append(scannum);

        logger.info('leftover_ms1scan_list={}'.format(leftover_ms1scan_list));
        for leftover_ms1scan in leftover_ms1scan_list: ms1_lognoise_map[leftover_ms1scan] = np.mean(lognoise_list);
        np.save(cached_ms1_lognoise_map_url, ms1_lognoise_map);

    logger.info('ms1_lognoise_map={}'.format(len(ms1_lognoise_map)));
    return ms1_lognoise_map;


