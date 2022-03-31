#%%
import sys
# sys.path.insert(0, '')
import json
import numpy as np
import pandas as pd
import gc
from pysatdata.loaders.load import *
from pysatdata.utils.interpolate_flux_rbsp import *
from pysatdata.utils.plotFunc.plot_funtions import *
from utils.downloadSeparateData import (separateCutPeriodRBSP_AB2, 
                                  separateCutPeriodEFW_EMF2, 
                                  separateCutPeriodKp,
                                  separateCutPeriodOMNI, calcLNAN)
from loguru import logger

#%%
class get_RBSP_data():
    """

    """
    def __init__(self, config_file_sat :str, eventKind :str,
                 listEvents :pd.core.frame.DataFrame,
                 intervalDaysDownload :list, lvalues :np.ndarray, downloadData :bool) -> None:
        
        self.config_file_sat = config_file_sat
        self.eventKind = eventKind
        self.listEvents = listEvents
        self.listDate = pd.to_datetime(self.listEvents.index)
        self.cutDay0 = intervalDaysDownload[0]
        self.cutDay1 = intervalDaysDownload[1]
        self.down = downloadData
        self.lvalues = lvalues


    def getFluxData(self, EnChanel :int, LorLstar :bool, 
                    outputDir :str, paramLoadSat :dict):
        ssdJson = {}
        ssdJsonInterp = {}
        if self.down:
            logger.warning(f"All the cut data will be stored in a json file at ./dataJson")
            for n, ds in enumerate(self.listDate):
                logger.info(f"Event.... {n}/{len(self.listDate)}")
                try:
                    outputFlux = separateCutPeriodRBSP_AB2(instDate=ds, cutDay0=self.cutDay0, 
                                                           cutDay1=self.cutDay1, EnChanel=EnChanel, 
                                                           lCutValue=self.lvalues, config_file_sat=self.config_file_sat,
                                                           paramLoadSat=paramLoadSat, LorLstar=LorLstar)
                    
                    instDate_str = outputFlux[0]
                    dictCutL = outputFlux[1]
                    dictCutLPhsd = outputFlux[2]
                    interpCutL = dictCutL.interpolate()
                    interpCutLPhsd = dictCutLPhsd.interpolate()
                    # ssd[instDate_str] = {'normal':dictCutL,
                    #                         'phsd': dictCutLPhsd}

                    dictJsonNormal = {'time': [i.strftime('%Y-%m-%d %H:%M:%S') for i in dictCutL.index],
                                        'data': dictCutL.to_dict('list')}
                    dictJsonNormalInterp = {'time': [i.strftime('%Y-%m-%d %H:%M:%S') for i in interpCutL.index],
                                        'data': interpCutL.to_dict('list')}
                    dictJsonPhsd = {'time': [i.strftime('%Y-%m-%d %H:%M:%S') for i in dictCutLPhsd.index],
                                        'data': dictCutLPhsd.to_dict('list')}
                    dictJsonPhsdInterp = {'time': [i.strftime('%Y-%m-%d %H:%M:%S') for i in interpCutLPhsd.index],
                                        'data': interpCutLPhsd.to_dict('list')}

                    ssdJson[instDate_str] = {'normal': dictJsonNormal,
                                                'phsd': dictJsonPhsd
                                            }
                    ssdJsonInterp[instDate_str] = {'normal': dictJsonNormalInterp,
                                                'phsd': dictJsonPhsdInterp
                                            }
                except (Exception) as e:
                    logger.error(e)
                pytplot.del_data()
                gc.collect()
            logger.warning("Done!")

            with open(f'{outputDir}{self.eventKind}_electronFlux_PhsdCut2_AB_Lst2_asfreq.json', 'w') as js:
                json.dump(ssdJson, js)
            with open(f'{outputDir}{self.eventKind}_electronFlux_PhsdCut2_AB_LstInterp2_asfreq.json', 'w') as js:
                json.dump(ssdJsonInterp, js)
        else:
            logger.warning("See you next time...")

    
    def getMagneticElectricData(self, outputDir :str, LorLstar :bool, flux_enh_red :str,
                                paramLoadEfw :dict, paramLoadEm :dict):
        
        # total_size, ifDownload = totalfilesSize(listDate, paramLoadSat, config_file_sat, 2, 4)
        # logger.warning(f"The total Batch size is {total_size} GB")
        # logger.warning(f"Do you want to continue?... Y/N")
        ssd = {}
        ssdInterp = {}
        if self.down:
            logger.warning(f"All the cut data will be stored in a json file at ./dataJson")
            for n, ds in enumerate(self.listDate):
                logger.info(f"Event.... {n}/{len(self.listDate)}")
                try:
                    outputFlux = separateCutPeriodEFW_EMF2(instDate=ds, cutDay0=self.cutDay0, 
                                                            cutDay1=self.cutDay1, EnChanel=1,
                                                            config_file_sat=self.config_file_sat,
                                                            paramLoadEm=paramLoadEm, 
                                                            paramLoadEfw=paramLoadEfw, LorLstar=LorLstar)
                    instDate_str = outputFlux[0]
                    sdData = outputFlux[1]
                    sdDataInterp = outputFlux[2]
                    ssd[instDate_str] = sdData
                    ssdInterp[instDate_str] = sdDataInterp
                except (Exception) as e:
                    logger.error(e)
                pytplot.del_data()
                gc.collect()
            logger.warning("Done!")
            with open(f'{outputDir}{self.eventKind}_{flux_enh_red}_electronFluxCut_Efw2.json',
                    'w') as j:
                json.dump(ssd, j)
            with open(f'{outputDir}{self.eventKind}_{flux_enh_red}_electronFluxCut_Efw2_Interp.json',
                    'w') as ji:
                json.dump(ssdInterp, ji)
        else:
            logger.warning("See you next time...")

    def getKpIndexData(self, outputDir :str, flux_enh_red :str,
                        configKp :dict):

        # total_size, ifDownload = totalfilesSize(listDate, paramLoadSat, config_file_sat, 2, 4)
        # logger.warning(f"The total Batch size is {total_size} GB")
        # logger.warning(f"Do you want to continue?... Y/N")
        ssd = {}
        if self.down:
            logger.warning(f"All the cut data will be stored in a json file at ./dataJson")
            for n, ds in enumerate(self.listDate):
                logger.info(f"Event.... {n}/{len(self.listDate)}")
                try:
                    instDate_str, dictCut = separateCutPeriodKp(instDate=ds, 
                                                                cutDay0=self.cutDay0, 
                                                                cutDay1=self.cutDay1, 
                                                                configKp=configKp)

                    ssd[instDate_str] = dictCut
                except (Exception) as e:
                    logger.error(e)
                pytplot.del_data()
                gc.collect()
            logger.warning("Done!")
            with open(f'{outputDir}{self.eventKind}_{flux_enh_red}_CutKp.json', 'w') as j:
                json.dump(ssd, j)
        else:
            logger.warning("See you next time...")


    def getOmniData(self, outputDir :str, flux_enh_red :str,
                    paramLoadSat :dict):
        
        ssd = {}
        if self.down:
            logger.warning(f"All the cut data will be stored in a json file at ./dataJson")
            for n, ds in enumerate(self.listDate):
                logger.info(f"Event.... {n}/{len(self.listDate)}")
                try:
                    instDate_str, dictCut = separateCutPeriodOMNI(instDate=ds, cutDay0=self.cutDay0, 
                                                                  cutDay1=self.cutDay1,
                                                                    config_file_sat=self.config_file_sat,
                                                                    paramLoadSat=paramLoadSat)

                    ssd[instDate_str] = dictCut
                except (Exception) as e:
                    logger.error(e)
                pytplot.del_data()
                gc.collect()
            logger.warning("Done!")
            with open(f'{outputDir}{self.eventKind}_{flux_enh_red}_electronFluxCut_Omni.json', 'w') as j:
                json.dump(ssd, j)
        else:
            logger.warning("See you next time...")


    def getLcdsData(self, outputDir :str, flux_enh_red :str,
                    dataOMNI :dict, data_Kp :dict):
        """
        You must first download OMNI and KP data using using  "getOmniData" and "getKpIndexData".
        This will download and create the json files needed to load the dataOMNI and data_Kp 
        dictionaries.
        """

        listDate = list(dataOMNI.keys())
        ssd = {}
        if self.down:
            logger.warning(f"All the cut data will be stored in a json file at ./dataJson")
            for n, ds in enumerate(listDate):
                logger.info(f"Event.... {n}/{len(listDate)}")
                try:
                    dictCut = calcLNAN(ds, dataOMNI, data_Kp)

                    ssd[ds] = dictCut
                except (Exception) as e:
                    logger.error(e)
                pytplot.del_data()
                gc.collect()
            logger.warning("Done!")
            with open(f'{outputDir}{self.eventKind}_{flux_enh_red}_electronFluxCut_LCDS.json', 'w') as j:
                json.dump(ssd, j)
        else:
            logger.warning("See you next time...")

    