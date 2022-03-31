from __future__ import print_function
import sys
# sys.path.insert(0, '')
from pathlib import Path
import os
import os.path
import pytz
from pysatdata.utils.library_functions import *
from pysatdata.loaders.load import *
from pysatdata.utils.interpolate_flux_rbsp import *
from pysatdata.utils.flux2PhSD.getData2PhSD import *    
from loguru import logger
import numpy as np
import datetime
from libfunctions import *
from ftplib import FTP
import spacepy.LANLstar as sl
#%%
# def separateZeroEpoch(instDate, cutDay0, cutDay1,
#                       EnChanel, lCutValue, LorLstar=False):

def find(condition):
    """Returns the indices where ravel(condition) is true."""
    res, = np.nonzero(np.ravel(condition))
    return res
def splitDataframe(a):
     return [a.iloc[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(a['L'].values))]


def separateFlux_L(enSignal,lArray, EnChanel, timeArray):
    temps = []
    for i in lArray:

        ssd = cutFluxss(lval=i,enSignal=enSignal,
                        EnChanel=EnChanel,
                        lArray=lArray,
                        timeArray=np.array(timeArray))

        temps.append(ssd)
    finDf = pd.concat(temps, axis=1)
    resampledDf = finDf.resample('6H').mean()


def cutFlux_lshell(enSignal,lValue, EnChanel, lArray, timeArray, rollingMean, interpolate):
    l = float(lValue)
    cutF = list()
    cut_date = list()
    cutFlux = pd.DataFrame([np.nan]*len(timeArray), index=timeArray)
    for i, ll in enumerate(lArray):
        if ll > l-0.01 and ll < l+0.01:
            if len(enSignal.shape)<2:
                cutFlux.loc[timeArray[i],0] = enSignal[i]
            else:
                cutFlux.loc[timeArray[i],0] = enSignal[i, EnChanel]

    if interpolate:
        cutFlux = cutFlux.interpolate()
    if rollingMean:
        cutFlux = cutFlux.rolling('12h').mean()
        cutFlux = cutFlux.resample('1h').mean()

    return cutFlux

def cutPower_lshell(signal,lValue, lArray, timeArray, rollingMean, interpolate):
    l = float(lValue)
    cutF = list()
    cut_date = list()
    cutPower = pd.DataFrame([np.nan]*len(timeArray), index=timeArray)
    for i, ll in enumerate(lArray):
        if ll > l-0.01 and ll < l+0.01:
            cutPower.loc[timeArray[i],0] = signal[i]

    if interpolate:
        cutFlux = cutFlux.interpolate()
    if rollingMean:
        cutFlux = cutFlux.rolling('12h').mean()
        cutFlux = cutFlux.resample('1h').mean()

    return cutPower

def separateCutPeriodRBSP(instDate, cutDay0, cutDay1,
                          EnChanel, lCutValue,
                          config_file_sat,
                          paramLoadSat,LorLstar=False):

    instDate_str = instDate.strftime('%Y-%m-%d %H:%M:%S')
    inidate = instDate - datetime.timedelta(days = cutDay0+1)
    enddate = instDate + datetime.timedelta(days = cutDay1+1)
    trange = [f"{inidate.year}-{inidate.month}-{inidate.day}", 
              f"{enddate.year}-{enddate.month}-{enddate.day}"]

    pytplot.del_data()
    varss_rept = load_sat(trange=trange, satellite=paramLoadSat['satellite'],
                          probe=[paramLoadSat['probe']], level=paramLoadSat['level'], 
                          rel='rel03', instrument=paramLoadSat['instrument'],
                          datatype=paramLoadSat['datatype'], 
                          config_file=config_file_sat, downloadonly=False, 
                          usePandas=False, usePyTplot=True)
    quants_fedu_rept = pytplot.data_quants['FEDU']
    flux_rept = quants_fedu_rept.values
    flux_rept[flux_rept == -9999999848243207295109594873856.000] = np.nan
    flux_rept[flux_rept == -1e31] = np.nan
    flux_rept_spec = np.nanmean(flux_rept, axis=1)
    if LorLstar:
        l_rept = pytplot.data_quants['L_star'].values
    else:
        l_rept = pytplot.data_quants['L'].values
    l_rept[l_rept == -9999999848243207295109594873856.000] = np.nan
    l_rept[l_rept == -1e31] = np.nan

    time_rept = quants_fedu_rept.coords['time'].values
    time_dt_rept = [datetime.datetime.fromtimestamp(i, pytz.timezone("UTC")) for i in time_rept]
    print(len(time_dt_rept))
    #
    # flux_at_level = flux_rept_spec[:,EnChanel]

    ldict = {'data':{}}

    cutAtL = separateFlux_L(enSignal=flux_rept_spec,
                            lArray=lCutValue, 
                            EnChanel=EnChanel, 
                            timeArray=time_dt_rept)
    # for ll in lCutValue:
    #     lStr = f"L{ll:.1f}"

    #     cutAtL = cutFlux_lshell(enSignal=flux_rept_spec, lValue=ll, EnChanel=EnChanel, lArray=l_rept,
    #                             timeArray=time_dt_rept)
    #     ldict['data'][lStr] = list(cutAtL[0].values)

    # ldict['time'] = [i.strftime('%Y-%m-%d %H:%M:%S') for i in cutAtL.index]

    return instDate_str, cutAtL


def cutFluxss(dictData, EnChanel, minuteRes, hourXRes):
    secondRes = 60*minuteRes
    logger.warning(f"Resampling {secondRes}")
    # hourXRes = 3
    # dictData = dats
    # hourXRes = 3
    # dictData = dats
    if len(dictData['a']['enData'].shape)<2:
        energyA = dictData['a']['enData']
        energyB = dictData['b']['enData']
    else:
        energyA = dictData['a']['enData'][:,EnChanel]
        energyB = dictData['b']['enData'][:,EnChanel]

    dfA = pd.DataFrame({'L':dictData['a']['lStar'], 
                            'en':energyA}, 
                            index=dictData['a']['time'])


    dfB = pd.DataFrame({'L':dictData['b']['lStar'], 
                        'en':energyB}, 
                        index=dictData['b']['time'])
    daRes = dfA.resample(f'{secondRes}S').agg(pd.Series.mean, skipna=False)
    dbRes = dfB.resample(f'{secondRes}S').agg(pd.Series.mean, skipna=False)

    timeDelA = daRes.index[-1] - daRes.index[0]
    timeDelB = daRes.index[-1] - daRes.index[0]
    
    print(timeDelA)
    print(timeDelB)

    totalSeconds = np.nanmean([timeDelA.days+1, timeDelB.days+1])*24*60*60
    logger.warning(f"Total seconds {totalSeconds}")
    step = int((hourXRes*60*60)//secondRes)

    xGridSize = int(np.round(len(daRes)/step))

    lss = daRes['L'].values
    ens = daRes['en'].values
    timea = daRes.index

    lssb = dbRes['L'].values
    ensb = dbRes['en'].values
    timeb = dbRes.index

    x = np.arange(0, xGridSize, 1)
    y = np.arange(1, 7.25, 0.25)

    xx, yy = np.meshgrid(x,y)
    mma = np.zeros((len(y), len(x)))
    tta = []
    k=0
    l=(k+step)
    for j,k in enumerate(np.arange(0,len(lss)-1,step)):
        l=(k+step)
        tempDa = lss[k:l]
        tempDb = lssb[k:l]
        tempTa = timea[k]
        tempEna = ens[k:l]
        tempEnb = ensb[k:l]
        tempTb = timea[k]
        tta.append(tempTa+(tempTb-tempTa)/2)
        for i in range(len(y)):
            tempLa = []
            tempLb = []
            for nE, ts in enumerate(tempDa):
                if ts > y[i]-0.12 and ts<= y[i]+0.12: 
                    tempLa.append(tempEna[nE])
            for nE, ts in enumerate(tempDb):
                if ts > y[i]-0.12 and ts<= y[i]+0.12: 
                    tempLb.append(tempEnb[nE])
            
            if len(tempLa) == 0:
                tempLa = [np.nan]
            if len(tempLb) == 0:
                tempLb = [np.nan]
            meanA = np.nanmean(tempLa)                
            meanB = np.nanmean(tempLb)    
            
            mma[i,j] = np.nanmean([meanA, meanB])
    
    # colunsNam = [f'{f"L{ll:.2f}"}' for ll in y]

    return pd.DataFrame(np.transpose(mma), index=tta, columns=y)
#%%
def separateCutPeriodRBSP_AB2(instDate, cutDay0, cutDay1,
                          EnChanel, lCutValue,
                          config_file_sat,
                          paramLoadSat,LorLstar=False):

    instDate_str = instDate.strftime('%Y-%m-%d %H:%M:%S')
    inidate = instDate - datetime.timedelta(days = cutDay0+1)
    enddate = instDate + datetime.timedelta(days = cutDay1+1)
    trange = [f"{inidate.year}-{inidate.month}-{inidate.day}", 
              f"{enddate.year}-{enddate.month}-{enddate.day}"]
    datsPhSD = {}
    dats = {}
    for probe in ['a', 'b']:
        key_dic_d = f"data_{probe}"
        key_dic_t = f"time_{probe}"
        key_dic_tPhsd = f"timePhsd_{probe}"
        pytplot.del_data()
        varss_rept = load_sat(trange=trange, satellite=paramLoadSat['satellite'],
                            probe=[probe], level=paramLoadSat['level'], 
                            rel='rel03', instrument=paramLoadSat['instrument']['rept'],
                            datatype=paramLoadSat['datatype']['rept'], 
                            config_file=config_file_sat, downloadonly=False, 
                            usePandas=False, usePyTplot=True)
        quants_fedu_rept = pytplot.data_quants['FEDU']
        flux_rept = quants_fedu_rept.values
        flux_rept[flux_rept == -9999999848243207295109594873856.000] = np.nan
        flux_rept[flux_rept == -1e31] = np.nan
        flux_rept_spec = np.nanmean(flux_rept, axis=1)
        if LorLstar:
            l_rept = pytplot.data_quants['L_star'].values
        else:
            l_rept = pytplot.data_quants['L'].values
        l_rept[l_rept == -9999999848243207295109594873856.000] = np.nan
        l_rept[l_rept == -1e31] = np.nan

        time_rept = quants_fedu_rept.coords['time'].values
        time_dt_rept = [datetime.datetime.fromtimestamp(i, pytz.timezone("UTC")) for i in time_rept]
        print(len(time_dt_rept))
        #
        dats[probe] = {'time' : time_dt_rept,
                   'enData': flux_rept_spec,
                   'lStar': l_rept}
        # flux_at_level = flux_rept_spec[:,EnChanel]
        alphaRange = [70,90]
        Kd = None
        # Kd=.1 #desired value of K; change
        MUd=2000 #desired MU in MeV/G; change 

        probeList=['a', 'b']
        ## Phase Space Density Calculation using pySatData module
        epochPhsd, lstarPhsd, phsd = getData2PhSD(trange=trange, dictParams=paramLoadSat, 
                                                  probeList=probe, alphaRange=alphaRange, 
                                                  Kd=Kd, MUd=MUd, config_file_sat=config_file_sat)

        datsPhSD[probe] = {'time' : epochPhsd,
                   'enData': phsd,
                   'lStar': lstarPhsd}
        # ldict[key_dic_d] = {}
        # for ll in lCutValue:
        #     lStr = f"L{ll:.1f}"
        #     lStrPhsd = f"L{ll:.1f}_Phsd"

        #     cutAtL = cutFlux_lshell(enSignal=flux_rept_spec, lValue=ll, EnChanel=EnChanel, lArray=l_rept,
        #                             timeArray=time_dt_rept, rollingMean=False, interpolate=False)
        #     cutPhsdAtL = cutFlux_lshell(enSignal=phsd, lValue=ll, EnChanel=EnChanel, lArray=lstarPhsd,
        #                             timeArray=epochPhsd, rollingMean=False, interpolate=False)
        #     ldict[key_dic_d][lStr] = list(cutAtL[0].values)
        #     ldict[key_dic_d][lStrPhsd] = list(cutPhsdAtL[0].values)
    ddas = cutFluxss(dictData=dats, EnChanel=EnChanel, minuteRes=1/6, hourXRes=3)
    ddasPhsd = cutFluxss(dictData=datsPhSD, EnChanel=EnChanel, minuteRes=5, hourXRes=3)

    # ldict[key_dic_t] = [i.strftime('%Y-%m-%d %H:%M:%S') for i in cutAtL.index]
    # ldict[key_dic_tPhsd] = [i.strftime('%Y-%m-%d %H:%M:%S') for i in cutPhsdAtL.index]

    return instDate_str, ddas, ddasPhsd



def separateCutPeriodRBSP_AB(instDate, cutDay0, cutDay1,
                          EnChanel, lCutValue,
                          config_file_sat,
                          paramLoadSat,LorLstar=False):

    instDate_str = instDate.strftime('%Y-%m-%d %H:%M:%S')
    inidate = instDate - datetime.timedelta(days = cutDay0+1)
    enddate = instDate + datetime.timedelta(days = cutDay1+1)
    trange = [f"{inidate.year}-{inidate.month}-{inidate.day}", 
              f"{enddate.year}-{enddate.month}-{enddate.day}"]
    ldict = {}
    for probe in ['a', 'b']:
        key_dic_d = f"data_{probe}"
        key_dic_t = f"time_{probe}"
        key_dic_tPhsd = f"timePhsd_{probe}"
        pytplot.del_data()
        varss_rept = load_sat(trange=trange, satellite=paramLoadSat['satellite'],
                            probe=[probe], level=paramLoadSat['level'], 
                            rel='rel03', instrument=paramLoadSat['instrument']['rept'],
                            datatype=paramLoadSat['datatype']['rept'], 
                            config_file=config_file_sat, downloadonly=False, 
                            usePandas=False, usePyTplot=True)
        quants_fedu_rept = pytplot.data_quants['FEDU']
        flux_rept = quants_fedu_rept.values
        flux_rept[flux_rept == -9999999848243207295109594873856.000] = np.nan
        flux_rept[flux_rept == -1e31] = np.nan
        flux_rept_spec = np.nanmean(flux_rept, axis=1)
        if LorLstar:
            l_rept = pytplot.data_quants['L_star'].values
        else:
            l_rept = pytplot.data_quants['L'].values
        l_rept[l_rept == -9999999848243207295109594873856.000] = np.nan
        l_rept[l_rept == -1e31] = np.nan

        time_rept = quants_fedu_rept.coords['time'].values
        time_dt_rept = [datetime.datetime.fromtimestamp(i, pytz.timezone("UTC")) for i in time_rept]
        print(len(time_dt_rept))
        #
        # flux_at_level = flux_rept_spec[:,EnChanel]
        alphaRange = [80,90]
        Kd = None
        # Kd=.1 #desired value of K; change
        MUd=1500 #desired MU in MeV/G; change 

        probeList=['a', 'b']
        ## Phase Space Density Calculation using pySatData module
        epochPhsd, lstarPhsd, phsd = getData2PhSD(trange=trange, dictParams=paramLoadSat, 
                                                  probeList=probe, alphaRange=alphaRange, 
                                                  Kd=Kd, MUd=MUd, config_file_sat=config_file_sat)

        ldict[key_dic_d] = {}
        for ll in lCutValue:
            lStr = f"L{ll:.1f}"
            lStrPhsd = f"L{ll:.1f}_Phsd"

            cutAtL = cutFlux_lshell(enSignal=flux_rept_spec, lValue=ll, EnChanel=EnChanel, lArray=l_rept,
                                    timeArray=time_dt_rept, rollingMean=False, interpolate=False)
            cutPhsdAtL = cutFlux_lshell(enSignal=phsd, lValue=ll, EnChanel=EnChanel, lArray=lstarPhsd,
                                    timeArray=epochPhsd, rollingMean=False, interpolate=False)
            ldict[key_dic_d][lStr] = list(cutAtL[0].values)
            ldict[key_dic_d][lStrPhsd] = list(cutPhsdAtL[0].values)

        ldict[key_dic_t] = [i.strftime('%Y-%m-%d %H:%M:%S') for i in cutAtL.index]
        ldict[key_dic_tPhsd] = [i.strftime('%Y-%m-%d %H:%M:%S') for i in cutPhsdAtL.index]

    return instDate_str, ldict

#%%
def separateCutPeriodOMNI(instDate, cutDay0, cutDay1,
                          config_file_sat,
                          paramLoadSat):

    instDate_str = instDate.strftime('%Y-%m-%d %H:%M:%S')
    inidate = instDate - datetime.timedelta(days = cutDay0)
    strInidate = inidate.strftime('%Y-%m-%d %H:%M:%S.%f')
    enddate = instDate + datetime.timedelta(days = cutDay1)
    strEnddate = enddate.strftime('%Y-%m-%d %H:%M:%S.%f')
    trange = [f"{inidate.year}-{inidate.month}-{inidate.day}", f"{enddate.year}-{enddate.month}-{enddate.day}"]

    varss_omni = load_sat(trange=trange, satellite=paramLoadSat['satellite'],
                         probe=[paramLoadSat['probe']], rel='rel03',
                         instrument=paramLoadSat['instrument'],datatype=paramLoadSat['datatype'],
                         config_file=config_file_sat, downloadonly=False,
                         usePandas=True, usePyTplot=False)
    varss_omni = varss_omni.interpolate()
    varss_omni = varss_omni.rolling('3h').mean()
    varss_omni = varss_omni.resample('15T').mean()
    mask = (varss_omni.index > pd.Timestamp(strInidate,tz="UTC")) & (varss_omni.index < pd.Timestamp(strEnddate,tz="UTC"))
    cutFlux = varss_omni[mask]


    omniDict = cutFlux.to_dict('list')

    outDict = {'time': [i.strftime('%Y-%m-%d %H:%M:%S') for i in cutFlux.index],
               'data': omniDict}


    return instDate_str, outDict


def separateCutPeriodEFW_EMF2(instDate, cutDay0, cutDay1,
                          EnChanel, config_file_sat,
                          paramLoadEm, paramLoadEfw,LorLstar=False):

    instDate_str = instDate.strftime('%Y-%m-%d %H:%M:%S')
    inidate = instDate - datetime.timedelta(days = cutDay0)
    strInidate = inidate.strftime('%Y-%m-%d %H:%M:%S.%f')
    enddate = instDate + datetime.timedelta(days = cutDay1)
    strEnddate = enddate.strftime('%Y-%m-%d %H:%M:%S.%f')
    trange = [f"{inidate.year}-{inidate.month}-{inidate.day}", f"{enddate.year}-{enddate.month}-{enddate.day}"]

    datsFields = {}
    for probe in ['a', 'b']:
        varss_emfisis = load_sat(trange=trange, satellite=paramLoadEm['satellite'],
                                probe=[probe], rel='rel03', level=paramLoadEm['level'],
                                instrument=paramLoadEm['instrument'], datatype=paramLoadEm['datatype'],
                                cadence=paramLoadEm['cadence'], coord=paramLoadEm['coord'],
                                config_file=config_file_sat, varnames=paramLoadEm['varnames'], downloadonly=False,
                                usePandas=True, usePyTplot=False)
        varss_emfisis[varss_emfisis <= -99999.898437] = np.nan
        varss_emfisis[varss_emfisis == -1e31] = np.nan
        pytplot.del_data()
        varss_rept = load_sat(trange=trange, satellite='rbsp',
                            probe=[probe], level='3', 
                            rel='rel03', instrument='rept',
                            datatype='sectors', varnames=['L_star'],
                            config_file=config_file_sat, downloadonly=False, 
                            usePandas=False, usePyTplot=True)
        # quants_fedu_rept = pytplot.data_quants['FEDU']
        l_star = pytplot.data_quants['L_star'].values
        pytplot.del_data()
        varss_efw = load_sat(trange=trange, satellite=paramLoadEfw['satellite'],
                            probe=[probe], level=paramLoadEfw['level'], rel='rel03',
                            instrument=paramLoadEfw['instrument'], datatype=paramLoadEfw['datatype'],
                            config_file=config_file_sat, varnames=paramLoadEfw['varnames'], downloadonly=False,
                            usePandas=False, usePyTplot=True)

        quants_efw = pytplot.data_quants['efield_mgse']
        fieldEfw = quants_efw.values
        fieldEfw[fieldEfw <= -999999984824320729510959487.000] = np.nan
        fieldEfw[fieldEfw == -1e31] = np.nan
        l_efw = pytplot.data_quants['lshell'].values
        fl = interp.interp1d((np.arange(0, len(l_efw))), l_efw, bounds_error=False)
        lnewy = np.linspace(0, len(l_efw), len(varss_emfisis.index))
        l_efw = fl(lnewy)

        fls = interp.interp1d((np.arange(0, len(l_star))), l_star, bounds_error=False)
        lsnewy = np.linspace(0, len(l_star), len(varss_emfisis.index))
        l_star = fls(lsnewy)



        eField = list()
        for i in range(3):
            ey = fieldEfw[:, i]
            fy = interp.interp1d((np.arange(0, len(ey))), ey, bounds_error=False)
            xnewy = np.linspace(0, len(ey), len(varss_emfisis.index))
            eField.append(fy(xnewy))
        eyi = np.asarray(np.transpose(eField))



        mag = varss_emfisis[['Mag_x', 'Mag_y', 'Mag_z']].values
        coords = varss_emfisis[['coordinates_x', 'coordinates_y', 'coordinates_z']].values
        eyi[:,0] = calcExEFW(eyi, mag)

        outerDf = rotate_field_fac(coords[:,0], coords[:,1], coords[:,2],
                                mag[:,0], mag[:,1], mag[:,2],
                                eyi[:,0], eyi[:,1], eyi[:,2])
        outerDf.index = varss_emfisis.index


        outerDf.interpolate()
        outerDf.fillna(0, inplace=True)

        if LorLstar:
            outerDf['L'] = l_star
        else:
            outerDf['L'] = l_efw

        mask = (outerDf.index > pd.Timestamp(strInidate, tz="UTC")) & (
                outerDf.index < pd.Timestamp(strEnddate, tz="UTC"))
        cutFields = outerDf[mask]

        logger.info("Power Spectrum Density")

        datsFields[probe] = cutFields

    dataDict = {'data' : {}}
    dataProbe = {}
    for cc in cutFields.columns:
        dataDict['data'][f'{cc}'] = {}
        for ppr in ['a', 'b']:
            tempdf = datsFields[ppr]
            if cc not in ['L', 'x', 'y', 'z']:
                signal = butter_bandpass_filter(tempdf.loc[:,cc], 1. / 1000, 1. / 100, 1., order=3)
                integratedPower = get_IntegratePowerSD(signal, 1., tempdf['L'].values, 100, 1000, 3.5)
                
                dataDict['data'][f'{cc}'][f'{ppr}'] = {'time' : tempdf.index,
                'enData': integratedPower,
                'lStar': tempdf['L'].values}
                # else:
                #     cutFields.loc[:, cc] = cutFields.loc[:,cc]
        # dataDict['time'] = [i.strftime('%Y-%m-%d %H:%M:%S') for i in cutAtL.index]
    sdData = {}
    sdDataInterp = {}
    for iic in list(dataDict['data'].keys()):
        if iic not in ['L', 'x', 'y', 'z']:
            ddas = cutFluxss(dictData=dataDict['data'][iic], EnChanel=EnChanel, minuteRes=1/6, hourXRes=3)
            ddasInterp = ddas.interpolate()
            sdData[iic] = {'time': [i.strftime('%Y-%m-%d %H:%M:%S') for i in ddas.index],
                            'data': ddas.to_dict('list')}
            sdDataInterp[iic] = {'time': [i.strftime('%Y-%m-%d %H:%M:%S') for i in ddasInterp.index],
                            'data': ddasInterp.to_dict('list')}
    # for iic in list(dataDict['data'].keys()):

    #     ddas = cutFluxss(dictData=dataDict['data'][iic], EnChanel=EnChanel, minuteRes=1/6, hourXRes=3)
    #     sdData[iic] = {'time': [i.strftime('%Y-%m-%d %H:%M:%S') for i in ddas.index],
    #                     'data': ddas.to_dict('list')}
    # # omniDict = cutFields.to_dict('list')
    #
    # outDict = {'time': [i.strftime('%Y-%m-%d %H:%M:%S') for i in cutFields.index],
    #            'data': omniDict}


    return instDate_str, sdData, sdDataInterp




def separateCutPeriodEFW_EMF(instDate, cutDay0, cutDay1,
                          EnChanel, lCutValue,
                          config_file_sat,
                          paramLoadEm, paramLoadEfw,LorLstar=False):

    instDate_str = instDate.strftime('%Y-%m-%d %H:%M:%S')
    inidate = instDate - datetime.timedelta(days = cutDay0)
    strInidate = inidate.strftime('%Y-%m-%d %H:%M:%S.%f')
    enddate = instDate + datetime.timedelta(days = cutDay1)
    strEnddate = enddate.strftime('%Y-%m-%d %H:%M:%S.%f')
    trange = [f"{inidate.year}-{inidate.month}-{inidate.day}", f"{enddate.year}-{enddate.month}-{enddate.day}"]


    varss_emfisis = load_sat(trange=trange, satellite=paramLoadEm['satellite'],
                             probe=[paramLoadEm['probe']], rel='rel03', level=paramLoadEm['level'],
                             instrument=paramLoadEm['instrument'], datatype=paramLoadEm['datatype'],
                             cadence=paramLoadEm['cadence'], coord=paramLoadEm['coord'],
                             config_file=config_file_sat, varnames=paramLoadEm['varnames'], downloadonly=False,
                             usePandas=True, usePyTplot=False)
    varss_emfisis[varss_emfisis <= -99999.898437] = np.nan
    varss_emfisis[varss_emfisis == -1e31] = np.nan


    pytplot.del_data()
    varss_efw = load_sat(trange=trange, satellite=paramLoadEfw['satellite'],
                         probe=[paramLoadEfw['probe']], level=paramLoadEfw['level'], rel='rel03',
                         instrument=paramLoadEfw['instrument'], datatype=paramLoadEfw['datatype'],
                         config_file=config_file_sat, varnames=paramLoadEfw['varnames'], downloadonly=False,
                         usePandas=False, usePyTplot=True)

    quants_efw = pytplot.data_quants['efield_mgse']
    fieldEfw = quants_efw.values
    fieldEfw[fieldEfw <= -999999984824320729510959487.000] = np.nan
    fieldEfw[fieldEfw == -1e31] = np.nan
    l_efw = pytplot.data_quants['lshell'].values
    fl = interp.interp1d((np.arange(0, len(l_efw))), l_efw, bounds_error=False)
    lnewy = np.linspace(0, len(l_efw), len(varss_emfisis.index))
    l_efw = fl(lnewy)

    eField = list()
    for i in range(3):
        ey = fieldEfw[:, i]
        fy = interp.interp1d((np.arange(0, len(ey))), ey, bounds_error=False)
        xnewy = np.linspace(0, len(ey), len(varss_emfisis.index))
        eField.append(fy(xnewy))
    eyi = np.asarray(np.transpose(eField))



    mag = varss_emfisis[['Mag_x', 'Mag_y', 'Mag_z']].values
    coords = varss_emfisis[['coordinates_x', 'coordinates_y', 'coordinates_z']].values
    eyi[:,0] = calcExEFW(eyi, mag)

    outerDf = rotate_field_fac(coords[:,0], coords[:,1], coords[:,2],
                               mag[:,0], mag[:,1], mag[:,2],
                               eyi[:,0], eyi[:,1], eyi[:,2])
    outerDf.index = varss_emfisis.index


    outerDf.interpolate()
    outerDf.fillna(0, inplace=True)

    outerDf['L'] = l_efw

    mask = (outerDf.index > pd.Timestamp(strInidate, tz="UTC")) & (
            outerDf.index < pd.Timestamp(strEnddate, tz="UTC"))
    cutFields = outerDf[mask]

    logger.info("Power Spectrum Density")
    dataDict = {'data' : {}}
    for cc in cutFields.columns:

        if cc not in ['L', 'x', 'y', 'z']:
            signal = butter_bandpass_filter(cutFields.loc[:,cc], 1. / 1000, 1. / 100, 1., order=3)
            integratedPower = get_IntegratePowerSD(signal, 1., cutFields['L'].values, 100, 1000, 3.5)
            dataDict['data'][f'{cc}'] = {}
            logger.info(f"Cutting Values at Lshell for {cc}")
            for lsl in lCutValue:
                lStr = f"L{lsl:.1f}"
                logger.info(f"Extracting cut at L = {lStr}")

                cutAtL = cutPower_lshell(signal=integratedPower, lValue=lsl,
                                        lArray=cutFields['L'].values,timeArray=cutFields.index, rollingMean=False, interpolate=False)
                dataDict['data'][f'{cc}'][lStr] = list(cutAtL[0].values)

        # else:
        #     cutFields.loc[:, cc] = cutFields.loc[:,cc]

    dataDict['time'] = [i.strftime('%Y-%m-%d %H:%M:%S') for i in cutAtL.index]


    
    # omniDict = cutFields.to_dict('list')
    #
    # outDict = {'time': [i.strftime('%Y-%m-%d %H:%M:%S') for i in cutFields.index],
    #            'data': omniDict}


    return instDate_str, dataDict

##################################################################################
def down_kp(timeini, timeend, configKp, downloadData, filename = 'none'):

    potsdam = downloadData # if 1, download the kp data from potsdam, if 0 you must import manually from spidr
    timeini = timeini
    timeend = timeend

    # the filename is used only when the data is from SPIDR repository
    ###

    # directory of the data
    # local data path
    local_path = str(Path.home().joinpath(configKp['local_data_dir'], 'Kp'))
    dataDownlDir = local_path


    str_temp = 'Kp_ap_{:}.txt'.format(int(timeini.year))
    print(str_temp)
    if os.path.isfile(dataDownlDir + str_temp):
        print(f"{str_temp} already at: {dataDownlDir}")
        downloadData = 0
    else:
        print(f"Downloading {str_temp}")
        downloadData = 1

    if downloadData == 1:
        # Download data

        # define the directory and host in the ftp
        host = 'ftp.gfz-potsdam.de'
        working_directory = '/pub/home/obs/Kp_ap_Ap_SN_F107/'
        #### download of the data #################
        mx = MarxeDownloader(host)
        # Connecting
        mx.connect()
        # Set downloaded data directory
        mx.set_output_directory(dataDownlDir)
        # Set FTP directory to download
        mx.set_directory(working_directory)
        # Download single data
        mx.download_one_data(str_temp)

    year, mm, dd, hh, kp = np.loadtxt(dataDownlDir + str_temp,
                                      usecols=(0, 1, 2, 3, 7), unpack=True)

    time = list()
    for ii in range(len(year)):
        t_string = f"{int(year[ii])}-{int(mm[ii])}-{int(dd[ii])}:{int(hh[ii])}"
        time.append(datetime.datetime.strptime(t_string, '%Y-%m-%d:%H'))

    time = pd.to_datetime(time)

    df_kp = pd.DataFrame(kp, index=time, columns=['Kp'])

    return df_kp

def separateCutPeriodKp(instDate, cutDay0, cutDay1, configKp):

    instDate_str = instDate.strftime('%Y-%m-%d %H:%M:%S')
    inidate = instDate - datetime.timedelta(days = cutDay0)
    strInidate = inidate.strftime('%Y-%m-%d %H:%M:%S.%f')
    enddate = instDate + datetime.timedelta(days = cutDay1)
    strEnddate = enddate.strftime('%Y-%m-%d %H:%M:%S.%f')
    trange = [f"{inidate.year}-{inidate.month}-{inidate.day}", f"{enddate.year}-{enddate.month}-{enddate.day}"]

    kpDf = down_kp(datetime.datetime.strptime(trange[0], '%Y-%m-%d'), datetime.datetime.strptime(trange[1], '%Y-%m-%d'),
                 configKp=configKp, downloadData=1)

    mask = (kpDf.index > pd.Timestamp(strInidate)) & (kpDf.index < pd.Timestamp(strEnddate))
    cutKP = kpDf[mask]


    omniDict = cutKP.to_dict('list')

    outDict = {'time': [i.strftime('%Y-%m-%d %H:%M:%S') for i in cutKP.index],
               'data': omniDict}


    return instDate_str, outDict


#########################################################################################

class MarxeDownloader():
    def __init__(self, hostname, user='', passwd=''):
        # Parametros
        self.host = hostname
        self.user = user
        self.passwd = passwd
        self.directory = None
        self.ftp = None
        self.output = None

    def set_user_and_password(self, user, passwd):
        self.user = user
        self.passwd = passwd

    def set_output_directory(self, output):
        self.output = str(output)

    def connect(self):
        try:
            self.ftp = FTP(str(self.host), user=str(self.user), passwd=str(self.passwd))
            self.ftp.login()
            print('Connected to: ' + str(self.host))
        except (Exception) as e:
            print(e)

    def set_directory(self, directory):
        try:
            self.ftp.cwd(directory)
            print('..')
        except (Exception) as e:
            print('Failed to set directory.\n' + str(e))

    def download_one_data(self, filename):
        try:
            self.ftp.retrbinary(str('RETR ' + filename), open(self.output + filename, 'wb').write)
            print("Downloaded: " + str(filename))
        except (Exception) as e:
            print(e)

    def download_many_data(self, filename_list):
        try:
            for filename in filename_list:
                self.ftp.retrbinary(str('RETR ' + filename), open(self.output + filename, 'wb').write)
                print('Downloaded: ' + str(filename))
        except (Exception) as e:
            raise e

    def close(self):
        self.ftp.close()


################################################################
def calcLNAN(date, dataOm, dataKp):
    nn = len(dataOm[date]['data']['SYM_H'])
    Bmodels = ['OPDYN','OPQUIET','T01QUIET','T01STORM','T89','T96','T05']
    # for i in range(varss_aceSwe.shape[0]):
    tt = dataKp[date]['data']['Kp']
    fls = interp.interp1d((np.arange(0, len(tt))), tt, bounds_error=False)
    lsnewy = np.linspace(0, len(tt), nn)
    kp_intep = fls(lsnewy)
    outs = []
    for i in range(nn):
        ddatae = pd.to_datetime(dataOm[date]['time'][i])
        yys = ddatae.year
        doys = ddatae.strftime('%j')
        housr = ((ddatae.hour * 60) + ddatae.minute) / 60 
        dat = {
                    'Kp'     : np.array([kp_intep[i]]),
                    'Dst'    : np.array([dataOm[date]['data']['SYM_H'][i]]),
                    'dens'   : np.array([dataOm[date]['data']['proton_density'][i]]),
                    'velo'   : np.array([dataOm[date]['data']['flow_speed'][i]]),
                    'Pdyn'   : np.array([dataOm[date]['data']['Pressure'][i]]),
                    'ByIMF'  : np.array([dataOm[date]['data']['BY_GSM'][i]]),
                    'BzIMF'  : np.array([dataOm[date]['data']['BZ_GSM'][i]]),
                    'G1'     : np.array([1.02966]),
                    'G2'     : np.array([0.54933]),
                    'G3'     : np.array([0.81399]),
                    'W1'     : np.array([0.12244]),
                    'W2'     : np.array([0.2514 ]),
                    'W3'     : np.array([0.0892 ]),
                    'W4'     : np.array([0.0478 ]),
                    'W5'     : np.array([0.2258 ]),
                    'W6'     : np.array([1.0461 ]),
                    'Year'   : np.array([yys]),
                    'DOY'    : np.array([doys]),
                    'Hr'     : np.array([housr]),
                    'PA'     : np.array([87.3875])}
        
        outs.append(sl.LANLmax(dat, Bmodels))
    
    lcdsDict = {'lcdsT05':[float(x['T05'][0]) for x in outs],    
                'lcdsT96':[float(x['T96'][0]) for x in outs],
                'lcdsT96':[float(x['T96'][0]) for x in outs],
                'lcdsT89':[float(x['T89'][0]) for x in outs],
                'lcdsT01STORM':[float(x['T01STORM'][0]) for x in outs],
                'lcdsT01QUIET':[float(x['T01QUIET'][0]) for x in outs]}
    
    outDict = {'time': dataOm[date]['time'],
               'data': lcdsDict}
            
    return outDict