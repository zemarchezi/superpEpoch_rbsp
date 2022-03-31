#%%

import numpy as np
import re
from urllib3 import PoolManager
from loguru import logger
from scipy import interpolate as interp
import pandas as pd
from scipy.signal import butter, lfilter
from waveletFunctions import wavelet as wavelt
from waveletFunctions import wave_signif



def find(condition):
    """Returns the indices where ravel(condition) is true."""
    res, = np.nonzero(np.ravel(condition))
    return res

def normD(a):
    norm = 0
    for i in range(3):
        norm += a[i]*a[i]
    return np.sqrt(norm)

def crossD(a,b):
    cross = [0]*3
    cross[0] = a[1]*b[2]+a[2]*b[1]
    cross[1] = a[2]*b[0]+a[0]*b[2]
    cross[2] = a[0]*b[1]+a[1]*b[2]
    return cross


def replace_at_index1(tup, ix, val):
     lst = list(tup)
     for i in range(0,len(ix)):
         lst[ix[i]] = val[i]
     return tuple(lst)

# Butterword filter coefficients
def butter_bandpass(lowcut, highcut, fs, order):
     nyq = 0.5 * fs
     low = lowcut / nyq
     high = highcut / nyq
     b, a = butter(order, [low, high], btype='band')
     return b, a

# Band-pass butterword filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
     y = lfilter(b, a, data)
     return y

# fill the gaps with nans
def fill_nan(A):
     '''
     interpolate to fill nan values
     '''
     if np.isnan(A[0]):
         A[0] = 0.5
     inds = np.arange(A.shape[0])
     good = np.where(np.isfinite(A))
     f = interp.interp1d(inds[good], A[good], bounds_error=False)
     B = np.where(np.isfinite(A), A, f(inds))
     return B

def calcExEFW(efield, bfield):
    ey = efield[:,1]
    ez = efield[:,2]
    bx = bfield[:,0]
    by = bfield[:,1]
    bz = bfield[:,2]

    ex = -((ey*by) + (ez*bz))/(bx)

    return ex

# def cutFlux_lshell(enSignal,lValue, EnChanel, lArray, timeArray):
#     l = float(lValue)
#     cutF = list()
#     cut_date = list()
#     for i, ll in enumerate(lArray):
#         if ll > l-0.01 and ll < l+0.01:
#             cutF.append(enSignal[i, EnChanel])
#             cut_date.append(timeArray[i])

#     fy = interp.interp1d((np.arange(0, len(cutF))), cutF, bounds_error=False)
#     xnewy = np.linspace(0, len(cutF), len(timeArray))
#     eyi = fy(xnewy)

#     return timeArray, eyi

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)

    Version history
    ---------------
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    listMinimun = list(x[ind])
    try:
        minIndex = listMinimun.index(min(listMinimun))
    except:
        minIndex = int(len(x)/2)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.axvline(ind[minIndex])
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()

def totalfilesSize(evList, paramLoadSat, config_file_sat, cutDay0, cutDay1):
    evDate = evList[0]
    numof_days = ((cutDay1 + 1) - (cutDay0 + 1)) * len(evList)
    satellite = paramLoadSat['satellite']
    prb = paramLoadSat['probe']
    level = paramLoadSat['level']
    instrument = paramLoadSat['instrument']
    datatype = paramLoadSat['datatype']
    rel = paramLoadSat['rel']

    remote_path = config_file_sat[satellite]['remote_data_dir']
    subpathformat = config_file_sat[satellite]['remote_subpath'][str(prb)][instrument][datatype]['subpath']
    subpathformat = eval(f"f'{subpathformat}'")
    sat_filename = config_file_sat[satellite]['remote_subpath'][str(prb)][instrument][datatype]['filename']
    sat_filename = eval(f"f'{sat_filename}'")

    pool = PoolManager()
    responseSubpath = pool.request("GET", evDate.strftime(remote_path + subpathformat), preload_content=False)

    site_content = responseSubpath.data
    responseSubpath.close()
    logger.info(f'Looking for the file: {evDate.strftime(sat_filename)}')
    search = re.search(evDate.strftime(sat_filename).replace('.cdf', '.*cdf'), str(site_content))
    if search:
        test_filename = search.group().split('">')[0]
        logger.info("Gettinf the file size...")
        response = pool.request("GET", evDate.strftime(remote_path + subpathformat) + test_filename,
                                preload_content=False)

        sizeInMbytes = float(response.headers.get("Content-Length"))/1e6
        logger.info(f"The size of one file is {sizeInMbytes} MB")
        total_size = (sizeInMbytes * numof_days) / 1e3
        response.close()
        if_Download = False
        return total_size, if_Download
    else:
        logger.error(f"The file, {evDate.strftime(sat_filename)} could not be found")
        sizeInMbytes = 0
        total_size = (sizeInMbytes * numof_days) / 1e3
        if_Download = False
        return total_size, if_Download

def set_columnNames(parameters):
    columnNames = {}
    for i in parameters:
        if i == 'IMF':
            columnNames[i] = 'IMF [nT]'
        elif i == 'F':
            columnNames[i] = 'B [nT]'
        elif i == 'flow_speed':
            columnNames[i] = 'Vsw [km/s]'
        elif i == 'proton_density':
            columnNames[i] = 'Np [cm$^{-3}$]'
        elif i in ['BX_GSE', 'BY_GSE', 'BZ_GSE', 'BX_GSM', 'BY_GSM', 'Bz_GSM']:
            new = i.split('_')[0]
            columnNames[i] = f"{new[0]}{new[1].lower()} [nT]"
        elif i in ['Vx', 'Vy', 'Vz']:
            columnNames[i] = f"{i} [nT]"
        elif i in ['AE_INDEX', 'AL_INDEX', 'AU_INDEX']:
            new = i.split('_')[0]
            columnNames[i] = f"{new} [nT]"
        elif i in ['SYM_D', 'SYM_H', 'ASY_D', 'ASY_H']:
            columnNames[i] = f"{i} [nT]"
        else:
            columnNames[i] = i

    return columnNames

def calcMeanMedian(times, parameters, dados, dates, columnNames, powerParam=None):
    medianDf = pd.DataFrame(index=times)
    meanDf = pd.DataFrame(index=times)
    quartile25Df = pd.DataFrame(index=times)
    quartile75Df = pd.DataFrame(index=times)
    skewDf = pd.DataFrame(index=times)
    for pp in parameters:
        tempDict = {}
        for dd in dates:
            if powerParam:
                if len(dados[dd][powerParam]['data'][pp]) != len(times):
                    tempDict[dd] = dados[dd][powerParam]['data'][pp][0:len(times)]
                else:
                    print(pp)
                    print(powerParam)
                    tempDict[dd] = dados[dd][powerParam]['data'][pp]
            else:
                if len(dados[dd]['data'][pp]) != len(times):
                    tempDict[dd] = dados[dd]['data'][pp][0:len(times)]
                else:
                    tempDict[dd] = dados[dd]['data'][pp]

        tempDf = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in tempDict.items()]))
        tempDf.index = times
        tempDf[tempDf > 5000] = np.nan

        tempDf = tempDf.interpolate()

        # tempDf = tempDf.rolling(12).mean()

        medianDf.loc[:, pp] = tempDf.median(axis=1)
        meanDf.loc[:, pp] = tempDf.mean(axis=1)
        quartile25Df.loc[:, pp] = tempDf.quantile(q=0.25, axis=1)
        quartile75Df.loc[:, pp] = tempDf.quantile(q=0.75, axis=1)
        if not powerParam:
            medianDf.rename(columns=columnNames, inplace=True)
            meanDf.rename(columns=columnNames, inplace=True)
            quartile25Df.rename(columns=columnNames, inplace=True)
            quartile75Df.rename(columns=columnNames, inplace=True)

    return medianDf, meanDf, quartile25Df, quartile75Df


def calcMeanMedianFlux(dataJson, eventType,cutDay0, cutDay1):
    
    probe_keys = [i.split('_')[0] for i in list(dataJson.keys()) if eventType in i]

    daysKey = list(dataJson[f"{eventType}"].keys())
    lss = list(dataJson[f"{eventType}"][daysKey[0]]['data'].keys())
    medianns = pd.DataFrame()
    means = pd.DataFrame()
    for ll in lss:
        print(ll)
        dictL = {}
        for pk in probe_keys:
            for i in dataJson[f"{eventType}"].keys():
                dadoL5 = dataJson[f"{eventType}"][i]['data'][ll]
                tt = pd.to_datetime(dataJson[f"{eventType}"][i]['time'])
                timeDeltaDays = (tt[-1] - tt[0]).days
                dictL[f"{pk}_{i}"] = dadoL5

        dfL = pd.DataFrame.from_dict(dictL, orient='index')
        # dfL = dfL.interpolate('linear')
        # dfL = pd.DataFrame(dictL, index=times)

        median = dfL.median(axis=0)
        mean = dfL.mean(axis=0)
        medianns[f'{ll}'] = median
        means[f'{ll}'] = mean

        times = np.linspace(-4 * 24, 4 * 24, len(medianns))

    return medianns, means, times

def calcMEdianQuartilesFlux(dataJson, eventType,cutDay0, cutDay1):
    
    probe_keys = [i.split('_')[0] for i in list(dataJson.keys()) if eventType in i]

    daysKey = list(dataJson[f"{eventType}"].keys())
    lss = list(dataJson[f"{eventType}"][daysKey[0]]['data'].keys())
    medianns = pd.DataFrame()
    means = pd.DataFrame()
    uppQuart = pd.DataFrame()
    lowQuart = pd.DataFrame()
    for ll in lss:
        print(ll)
        dictL = {}
        for pk in probe_keys:
            for i in dataJson[f"{eventType}"].keys():
                dadoL5 = dataJson[f"{eventType}"][i]['data'][ll]
                tt = pd.to_datetime(dataJson[f"{eventType}"][i]['time'])
                timeDeltaDays = (tt[-1] - tt[0]).days
                dictL[f"{pk}_{i}"] = dadoL5

        dfL = pd.DataFrame.from_dict(dictL, orient='index')
        # dfL = dfL.interpolate('linear')
        # dfL = pd.DataFrame(dictL, index=times)

        median = dfL.median(axis=0)
        mean = dfL.mean(axis=0)
        uppQ = dfL.quantile(q=0.75, axis=0)
        lowQ = dfL.quantile(q=0.25, axis=0)
        medianns[f'{ll}'] = median
        means[f'{ll}'] = mean
        uppQuart[f'{ll}'] = uppQ
        lowQuart[f'{ll}'] = lowQ

        times = np.linspace(-cutDay0 * 24, cutDay1 * 24, len(medianns))

    return medianns, means, lowQuart, uppQuart,times

def countingEventsBins(dataFlux, eventType,cutDay0, cutDay1):
    lss = ['1.0', '1.25', '1.5', '1.75', '2.0', '2.25', 
			'2.5', '2.75', '3.0', '3.25', '3.5', '3.75', 
			'4.0', '4.25', '4.5', '4.75', '5.0', '5.25', 
			'5.5', '5.75', '6.0', '6.25', '6.5', '6.75', 
			'7.0']
    days = list(dataFlux[eventType[0]].keys())
    numbBin = len(dataFlux[eventType[0]][days[0]]['data'][lss[4]])
    numL = len(lss)
    totalEvents = len(dataFlux[eventType[0]].keys()) + len(dataFlux[eventType[1]].keys()) + len(dataFlux[eventType[2]].keys())
    totalBins = numbBin * totalEvents
    countingsL = {}

    for ll in lss:
        print(ll)
        tempC = []
        for es in eventType:
            for i in dataFlux[es].keys():
                dadoL5 = dataFlux[es][i]['data'][ll]
                # tempL = []
                # for dd in dadoL5:
                # 	if not np.isnan(dd):
                # 		tempC.append(float(ll))
                tempC.append(np.count_nonzero(~np.isnan(np.asarray(dadoL5))))
            totalL = np.sum(tempC)
        # # countingPercent = tempDf.count().values
        countingsL[float(ll)] = (totalL/totalBins) * 100
        # dfL = dfL.interpolate('linear')
    dfL = pd.DataFrame({'Time bins [$\%$]':countingsL.values(), 'L-star bin':countingsL.keys()})

    return dfL

def countingEventsBins2(dataFlux, eventType,cutDay0, cutDay1):

    lss = ['1.0', '1.25', '1.5', '1.75', '2.0', '2.25', 
			'2.5', '2.75', '3.0', '3.25', '3.5', '3.75', 
			'4.0', '4.25', '4.5', '4.75', '5.0', '5.25', 
			'5.5', '5.75', '6.0', '6.25', '6.5', '6.75', 
			'7.0']
    days = list(dataFlux[eventType[0]].keys())
    numbBin = len(dataFlux[eventType[0]][days[0]]['data'][lss[4]])
    numL = len(lss)
    totalEvents = len(dataFlux[eventType[0]].keys()) + len(dataFlux[eventType[1]].keys())
    totalBins = numbBin * totalEvents
    countingsL = {}

    for ll in lss:
        print(ll)
        tempC = []
        for es in eventType:
            for i in dataFlux[es].keys():
                dadoL5 = dataFlux[es][i]['data'][ll]
                # tempL = []
                # for dd in dadoL5:
                # 	if not np.isnan(dd):
                # 		tempC.append(float(ll))
                tempC.append(dadoL5)
            tempDf = pd.DataFrame(tempC)
        countingPercent = tempDf.count().values
        countingsL[float(ll)] = countingPercent
        # dfL = dfL.interpolate('linear')
    dfL = pd.DataFrame.from_dict(countingsL)

    return dfL

def countingEventsBins3(dataFlux, eventType,cutDay0, cutDay1):

    lss = ['1.0', '1.25', '1.5', '1.75', '2.0', '2.25', 
			'2.5', '2.75', '3.0', '3.25', '3.5', '3.75', 
			'4.0', '4.25', '4.5', '4.75', '5.0', '5.25', 
			'5.5', '5.75', '6.0', '6.25', '6.5', '6.75', 
			'7.0']
    days = list(dataFlux[eventType[0]].keys())
    lenghs= []
    dfBin = {}
    counts = 0
    for ee in eventType:
        days2 = list(dataFlux[ee].keys())
        for dd in days2:
            tempDf = pd.DataFrame(dataFlux[ee][dd]['data'])
            tempDf_bin = tempDf.notnull().astype("int")
            dfBin[f'{len(tempDf_bin)}_{counts}'] = tempDf_bin
            lenghs.append(len(tempDf_bin))
            counts += 1

    maxLen = np.max(lenghs)
    keysdf = list(dfBin.keys())
    probeDf = pd.DataFrame(np.zeros((maxLen, dfBin[keysdf[0]].shape[1])), columns=dfBin[keysdf[0]].columns)
    sorted_dict = dict(sorted(dfBin.items()))
    for i in sorted_dict.keys():
        probeDf = probeDf + sorted_dict[i]
        probeDf.fillna(0, inplace=True)
    probeDf = (probeDf / len(sorted_dict)) * 100
    probeDf['5.5'] = probeDf['5.5']*1.2 
    return probeDf, sorted_dict

# def comp_wavelet(signal,t0, dt, dj, s0, J, mother, scales_int):
#     N = len(signal)
#     t = np.arange(0, N) * dt + t0
#     p = np.polyfit(t - t0, signal, 1)
#     dat_notrend = signal - np.polyval(p, t - t0)
#     std = dat_notrend.std()  # Standard deviation
#     var = std ** 2  # Variance
#     dat_norm = dat_notrend / std  # Normalized dataset
#
#
#     alpha, _, _ = wavelet.ar1(signal)  # Lag-1 autocorrelation for red noise
#
#     wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J,
#                                                           mother)
#
#     iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std
#
#
#     power = (np.abs(wave)) ** 2
#     fft_power = np.abs(fft) ** 2
#     period = 1 / freqs
#
#     power /= scales[:, None]
#
#     signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
#                                              significance_level=0.95,
#                                              wavelet=mother)
#     sig95 = np.ones([1, N]) * signif[:, None]
#     sig95 = power / sig95
#     # sig95 = 0
#
#     glbl_power = power.mean(axis=1)
#     dof = N - scales  # Correction for padding at edges
#     glbl_signif, tmp = wavelet.significance(var, dt, scales, 1, alpha,
#                                             significance_level=0.95, dof=dof,
#                                             wavelet=mother)
#
#     sel = find((period >= scales_int[0]) & (period < scales_int[1]))
#     Cdelta = mother.cdelta
#     # scale_avg = (scales * np.ones((N, 1))).transpose()
#     scale_avg = power
#     # scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
#     scale_avg = scale_avg[sel, :].sum(axis=0)
#     # scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
#     scale_avg_signif, tmp = wavelet.significance(var, dt, scales, 2, alpha,
#                                                  significance_level=0.95,
#                                                  dof=[scales[sel[0]],
#                                                       scales[sel[-1]]],
#                                                  wavelet=mother)
#
#     out_dict = {'time': t,
#                 'dat_norm': dat_norm,
#                 'iwave': iwave,
#                 'period':period,
#                 'power':power,
#                 'coi': coi,
#                 'sig95': sig95,
#                 'glbl_signif': glbl_signif,
#                 'fft_theor': fft_theor,
#                 'var': var,
#                 'fft_power': fft_power,
#                 'fftfreqs': fftfreqs,
#                 'glbl_power': glbl_power,
#                 'scale_avg_signif': scale_avg_signif,
#                 'scale_avg': scale_avg
#     }
#
#     return out_dict

def wavelet_calc(data, dt, powers):
    ######################################################################
    #  Wavelet analysis - Based on the C. Torrence Matlab scripts and the
    #   Evgeniya Predybaylo python version
    ######################################################################

    sst = data
#    rec_text = hrra2 - 0.7

    variance = np.std(sst, ddof=1) ** 2
    # sst = (sst - np.mean(sst)) / np.std(sst, ddof=1)
    n = len(sst)
    dt = dt

    pad = 1  # pad the time series with zeroes (recommended)
    dj = 0.1  # this will do 4 sub-octaves per octave
    s0 = 4 * dt

    j1 = powers / dj  # this says do 7 powers-of-two with dj sub-octaves each
    lag1 = 0.2  # lag-1 autocorrelation for red noise background
    mother = 'MORLET'

    # Wavelet transform:
    wave, period, scale, coi = wavelt(sst, dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave)) ** 2  # compute wavelet power spectrum

    # Significance levels: (variance=1 for the normalized SST)
    signif = wave_signif((1.5*sst), dt=dt, sigtest=0, scale=scale, lag1=lag1,
                         mother=mother)
    sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # expand signif --> (J+1)x(N) array
    sig95 = power / sig95  # where ratio > 1, power is significant

    # Global wavelet spectrum & significance levels:
    global_ws = variance * (np.sum(power, axis=1) / n)  # time-average over all times
    dof = n - scale  # the -scale corrects for padding at edges
    global_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=1,
                                lag1=lag1, dof=dof, mother=mother)

    # Scale-average between El Nino periods of 2--8 years
    avg = np.logical_and(scale >= 2, scale < 8)
    Cdelta = 0.776  # this is for the MORLET wavelet
    scale_avg = scale[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # expand scale --> (J+1)x(N) array
    scale_avg = power / scale_avg  # [Eqn(24)]
    scale_avg = variance * dj * dt / Cdelta * sum(scale_avg[avg, :])  # [Eqn(24)]
    scaleavg_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=2,
                                  lag1=lag1, dof=([2, 7.9]), mother=mother)

    return (power, sst, period, coi, sig95)



def get_IntegratePowerSD(signal, dt, lshell, perMin, perMax, lShellMin):

    lshell = np.asarray(lshell)
    signal = np.asarray(signal)
    sds = find((lshell <= lShellMin))
    signal[sds] = 0
    lshell[sds] = np.nan

    power, sst, period, coi, sig95 = wavelet_calc(signal, dt, 10)

    sel = find((period >= perMin) & (period < perMax))
    power[:, sds] = 0
    scale_Power = power[sel, :]
    sum_Power = scale_Power.sum(axis=0)

    return sum_Power

# %%
