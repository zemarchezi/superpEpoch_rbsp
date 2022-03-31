#%%
import json
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from matplotlib import ticker
from utils.libfunctions import *

def replace_at_index1(tup, ix, val):
    lst = list(tup)
    for i in range(0, len(ix)):
        lst[ix[i]] = val[i]
    return tuple(lst)

class plotSEA():
    def __init__(self, eventKind :str, eventType :list, 
                 fluxType :str, geomagModel :str) -> None:

        self.eventKind = eventKind
        self.eventType = eventType
        self.fluxType = fluxType
        self.geomagModel = geomagModel
        self.eventsDicts = {'CME': 'Interplanetary Coronal Mass Ejection',
			   'HSS': 'High-Speed Streams'}
               
    def plotFluxOmni(self, jsonDirectory, cutdays, 
                     xlimFlux, ylimFlux, savePLot, 
                     outputDir, plotParameters):
        self.plotParameters = plotParameters
        
        matplotlib.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
        matplotlib.rc('text', usetex=True)
        matplotlib.rc('font', size=26, family='serif')
        figprops = dict(nrows=len(self.plotParameters['param'])+1, ncols=len(self.eventType),
                        constrained_layout=True, figsize=(21, 26),dpi=120)
        fig, axes = plt.subplots(**figprops)
        fig.suptitle(f'{self.eventsDicts[self.eventKind]}',va='center', fontsize=35)
        fig.supxlabel('Epoch [hours]', va='center', fontsize=35)

        for ee in range(len(self.eventType)):
            if self.fluxType == 'Flux':
                with open(f'{jsonDirectory}{self.eventKind}_FluxEnhRed2_AB_Lst_asfreq.json', 'r') as f:
                    self.dataFlux = json.load(f)
            else:
                with open(f'{jsonDirectory}{self.eventKind}_PhsdEnhRed2_AB_Lst_asfreq.json', 'r') as f:
                    dataFlux = json.load(f)

            with open(f'{jsonDirectory}{self.eventKind}_{self.eventType[ee]}_electronFluxCut_Omni.json',
                    'r') as f:
                data = json.load(f)

            with open(f'{jsonDirectory}{self.eventKind}_{self.eventType[ee]}_CutKp.json',
                    'r') as f:
                dataKP = json.load(f)

            with open(f'{jsonDirectory}{self.eventKind}_{self.eventType[ee]}_electronFluxCut_LCDS.json',
                    'r') as f:
                dataLcds = json.load(f)

            dates = list(data.keys())
            datesKP = list(dataKP.keys())
            datesLcds = list(dataLcds.keys())
            parameters = list(data[dates[0]]['data'].keys())
            parametersKP = list(dataKP[datesKP[0]]['data'].keys())
            parametersLcds = list(dataLcds[datesLcds[0]]['data'].keys())
            tt = pd.to_datetime(data[dates[0]]['time'])
            ttKP = pd.to_datetime(dataKP[datesKP[0]]['time'])
            ttLcds = pd.to_datetime(dataLcds[datesLcds[0]]['time'])
            times = np.linspace(-cutdays[0] * 24, cutdays[1] * 24, len(tt))
            timesKP = np.linspace(-cutdays[0] * 24, cutdays[1] * 24, len(ttKP))
            timesLcds = np.linspace(-cutdays[0] * 24, cutdays[1] * 24, len(ttLcds))

            columnNames = set_columnNames(parameters)

            mediannsFlux, meansFlux, timesFlux = calcMeanMedianFlux(dataFlux, self.eventType[ee], cutdays[0], cutdays[1])

            # mediannsFlux = mediannsFlux.interpolate('linear')

            lls = np.arange(3.5,5.5,0.25)
            # print(len(mediannsFlux[0]))
            # print(len(lls))
            # print(len(timesFlux))
            # plot1 = axes[0, ee].pcolormesh(timesFlux, mediannsFlux.columns,(mediannsFlux.T),
            # 					 norm=colors.LogNorm(vmin=1e-10, vmax=5e-5), cmap='jet')
            if self.fluxType == 'Flux':
                plot1 = axes[0, ee].pcolormesh(timesFlux, mediannsFlux.columns,(mediannsFlux.T),
                                    norm=colors.LogNorm(vmin=1e2, vmax=5e5), cmap='jet', shading='auto')
                contours = axes[0, ee].contour(timesFlux, mediannsFlux.columns,mediannsFlux.T, locator=ticker.LogLocator(),colors='k')
                fmt = ticker.LogFormatterMathtext()
                fmt.create_dummy_axis()
                axes[0, ee].clabel(contours,contours.levels, inline=True, fontsize=25, fmt=fmt)
                axes[0, ee].set_title(f'{self.eventType[ee].capitalize()} \n Median 2.10 MeV electrons')
            else:
                plot1 = axes[0, ee].pcolormesh(timesFlux, mediannsFlux.columns,(mediannsFlux.T),
                                norm=colors.LogNorm(vmin=1e-10, vmax=5e-5), cmap='jet',shading='auto')
                contours = axes[0, ee].contour(timesFlux, mediannsFlux.columns,mediannsFlux.T, locator=ticker.LogLocator(),colors='k')
                fmt = ticker.LogFormatterMathtext()
                fmt.create_dummy_axis()
                axes[0, ee].clabel(contours,contours.levels, inline=True, fontsize=25, fmt=fmt)
                axes[0, ee].set_title(f'{self.eventType[ee].capitalize()}'+ '\n Median $\mu$ = 1500 MeV/G,  $K$ = 0.02 $G^{1/2}R_E$ PhSD')
            
            # axes[0, ee].set_title(f'{eventType[ee].capitalize()}'+ '\n Median $\mu$ = 1500 MeV/G,  $K$ = 0.02 $G^{1/2}R_E$ PhSD')
            axes[0, ee].set_ylabel('L-Star')
            axes[0, ee].axvline(x=0, color='k', linewidth=2)
            if ee == 1:
                axLoc = make_axes_locatable(axes[0, ee]).get_position()
                cbar_coord = replace_at_index1(axLoc, [0,1, 2,3], [1.01,0.795, 0.015,0.146])
                cbar_ax = fig.add_axes(cbar_coord)
                cbar = fig.colorbar(plot1, cax=cbar_ax)
                if self.fluxType == 'Flux':
                    cbar.set_label('$[cm^{-2}s^{-1}sr^{-1}MeV^{-1}]$')
                else:
                    cbar.set_label('$[(c/MeV - cm)^3]$')
                axes[0, ee].set_ylabel('')
                axes[0, ee].set_yticklabels([])
                axes[0, ee].text(-0.035, 1.08, '(b)', transform=axes[0, ee].transAxes, size=30)
            else:
                axes[0, ee].text(-0.035, 1.08, '(a)', transform=axes[0, ee].transAxes, size=30)

            axes[0, ee].tick_params(direction='in', length=14, width=2, colors='k',
                                    grid_color='k', grid_alpha=0.1, which='major')
            axes[0, ee].tick_params(direction='in', length=7, width=0.8, colors='k',
                                    grid_color='k', grid_alpha=0.1, which='minor')


            medianDf, meanDf, quartile25Df, quartile75Df = calcMeanMedian(times, parameters,
                                                                        data, dates, columnNames)

            medianDfKP, meanDfKP, quartile25DfKP, quartile75DfKP = calcMeanMedian(timesKP, parametersKP,
                                                                            dataKP, datesKP, {'Kp':'Kp'})
            medianDfLcds, meanDfLcds, quartile25DfLcds, quartile75DfLcds = calcMeanMedian(timesLcds, parametersLcds,
                                                                            dataLcds, datesLcds, {f'lcds{self.geomagModel}':'LCDS [L*]'})

            # axes[0, ee].plot(medianDfLcds['LCDS [L*]']+16.25, color="orangered",
            # 						   label="Median", linewidth=2)
            # axes[0, ee].plot(meanDfLcds['LCDS [L*]']+16.25, color="gray",
            # 					label="Mean", linewidth=2)
            # axes[0, ee].plot(quartile25DfLcds['LCDS [L*]']+16.25, '--', color="#0b91ff",
            # 					label="Upper Quartile", linewidth=2)
            # axes[0, ee].plot(quartile75DfLcds['LCDS [L*]']+16.25, '--', color="#0b91ff",
            # 					label="Lower Quartile", linewidth=2)
            axes[0, ee].set_xlim(-xlimFlux[0] * 24, xlimFlux[1] * 24)
            axes[0, ee].set_ylim(ylimFlux[0], ylimFlux[1])
            for par in range(1,len(self.plotParameters['param'])+1):
                cc = par-1

                if self.plotParameters['param'][cc] == 'RMP [Re]':
                    np75 =quartile75Df['Np [cm$^{-3}$]']
                    np25 =quartile25Df['Np [cm$^{-3}$]']
                    meanNp =meanDf['Np [cm$^{-3}$]']
                    medianNp =medianDf['Np [cm$^{-3}$]']
                    medianV = medianDf['Vsw [km/s]']
                    meanV = meanDf['Vsw [km/s]']
                    V75 =quartile75Df['Vsw [km/s]']
                    V25 =quartile25Df['Vsw [km/s]']
                    medianRMP = 107.4 / ((medianNp * medianV ** 2) ** (1 / 6))
                    meanRMP = 107.4 / ((meanNp * meanV ** 2) ** (1 / 6))
                    quar75RMP = 107.4 / ((np75 * V75 ** 2) ** (1 / 6))
                    quart25RMP = 107.4 / ((np25 * V25 ** 2) ** (1 / 6))

                    axes[par, ee].plot(medianRMP, color="orangered",
                                    label="Median", linewidth=2)
                    axes[par, ee].plot(meanRMP, color="gray",
                                    label="Mean", linewidth=2)
                    axes[par, ee].plot(quart25RMP, '--', color="#0b91ff",
                                    label="Upper Quartile", linewidth=2)
                    axes[par, ee].plot(quar75RMP, '--', color="#0b91ff",
                                    label="Lower Quartile", linewidth=2)
                if self.plotParameters['param'][cc] == 'Kp':
                    axes[par, ee].plot(medianDfKP['Kp'], color="orangered",
                                    label="Median", linewidth=2)
                    axes[par, ee].plot(meanDfKP['Kp'], color="gray",
                                    label="Mean", linewidth=2)
                    axes[par, ee].plot(quartile25DfKP['Kp'], '--', color="#0b91ff",
                                    label="Upper Quartile", linewidth=2)
                    axes[par, ee].plot(quartile75DfKP['Kp'], '--', color="#0b91ff",
                                    label="Lower Quartile", linewidth=2)
                if self.plotParameters['param'][cc] == 'LCDS [L*]':
                    axes[par, ee].plot(medianDfLcds['LCDS [L*]'], color="orangered",
                                    label="Median", linewidth=2)
                    axes[par, ee].plot(meanDfLcds['LCDS [L*]'], color="gray",
                                    label="Mean", linewidth=2)
                    axes[par, ee].plot(quartile25DfLcds['LCDS [L*]'], '--', color="#0b91ff",
                                    label="Upper Quartile", linewidth=2)
                    axes[par, ee].plot(quartile75DfLcds['LCDS [L*]'], '--', color="#0b91ff",
                                    label="Lower Quartile", linewidth=2)
                if self.plotParameters['param'][cc] not in ['RMP [Re]', 'Kp', 'LCDS [L*]']:
                    axes[par,ee].plot((medianDf[self.plotParameters['param'][cc]]), color="orangered",
                                    label="Median", linewidth=2)
                    axes[par, ee].plot((meanDf[self.plotParameters['param'][cc]]), color="gray",
                                    label="Mean", linewidth=2)
                    axes[par,ee].plot((quartile25Df[self.plotParameters['param'][cc]]), '--', color="#0b91ff",
                                    label="Upper Quartile", linewidth=2)
                    axes[par,ee].plot((quartile75Df[self.plotParameters['param'][cc]]), '--', color="#0b91ff",
                                    label="Lower Quartile", linewidth=2)
                # axes.fill_between(times,(min_ser), (max_ser), color='#0b89ff', alpha=0.05)
                if ee == 1:
                    axes[par,ee].set_yticklabels([])
                axes[par,ee].set_xlim(-xlimFlux[0] * 24, xlimFlux[1] * 24)
                
                axes[par,ee].set_ylim(self.plotParameters['ylim'][cc][0], self.plotParameters['ylim'][cc][1])
                axes[par,ee].tick_params(direction='in', length=14, width=2, colors='k',
                            grid_color='k', grid_alpha=0.1, which='major')
                axes[par,ee].tick_params(direction='in', length=7, width=0.8, colors='k',
                            grid_color='k', grid_alpha=0.1, which='minor')
                if ee != 1:
                    axes[par,ee].set_ylabel(self.plotParameters['param'][cc].replace('_', ' '))
                if self.plotParameters['param'][cc] == 'Bz [nT]':
                    axes[par,ee].axhline(y=0, color='k', linewidth=2)
                axes[par,ee].axvline(x=0, color='k', linewidth=2)
                # # axes.set_yscale('log')
                axes[par,ee].text(-0.035, 1.08, self.plotParameters['letter'][ee][cc], transform=axes[par,ee].transAxes, size=30)
                if self.plotParameters['letter'][ee][cc] == '(d)':
                    axes[par,ee].legend(loc='lower right', prop={'size': 22})
                #
                #|
                axes[par,ee].grid()
        if savePLot:
            plt.savefig(f"{outputDir}{self.eventKind}_IndexData_AB_{self.fluxType}_Lst_LCDS{self.geomagModel}.png", bbox_inches='tight')
            plt.savefig(f"{outputDir}{self.eventKind}_IndexData_AB_{self.fluxType}_Lst_LCDS{self.geomagModel}.pdf", bbox_inches='tight')

    
    def plotULFPower(self, plotComponent, legendy, 
                     titles, plotParameters, jsonDirectory, 
                     cutdays, xlimP, savePLot, 
                     outputDir):
        # plotComponent = 'bp'
        # legendy = 'nT^2'
        # titles = "B$_{\parallel}$"
        # plotComponent = 'ea'
        # legendy = '(mV/m)^2'
        # titles = "E$_{\phi}$"
        # plotComponent = 'br'
        # legendy = 'nT^2'
        # titles = "B$_{r}$"
        # letters = [['(a)', '(c)', '(e)', '(g)', '(i)'], ['(b)', '(d)', '(f)', '(h)', '(j)']]
        # plotParameters = {'param': ['5.5', '5.0', '4.5', '4.0', '3.5'],
        #                 'ylim': [[1e0,5e3], [1e0,5e3], [1e0, 5e3], [1e0,5e3],[1e0,5e3]], #
        #                 'letter': letters}
        pathJson = jsonDirectory
        matplotlib.rc('font', **{'family': 'serif', 'serif': ['Palatino']})
        matplotlib.rc('text', usetex=True)
        matplotlib.rc('font', size=26, family='serif')
        figprops = dict(nrows=len(plotParameters['param']), ncols=len(self.eventType),
                        constrained_layout=True, figsize=(21, 26),dpi=120)
        fig, axes = plt.subplots(**figprops)
        fig.suptitle(f'{titles} -- {self.eventsDicts[self.eventKind]}',va='center', fontsize=35)
        fig.supxlabel('Epoch [hours]', va='center', fontsize=35)
        for ee in range(len(self.eventType)):
            with open(f'{pathJson}{self.eventKind}_{self.eventType[ee]}_electronFluxCut_Efw2_Interp.json', 'r') as f:
                data = json.load(f)

            dates = list(data.keys())
            lenTimes = [len(pd.to_datetime(data[i][plotComponent]['time'])) for i in dates]
            tt = pd.to_datetime(data[dates[0]][plotComponent]['time'])
            times = np.linspace(-cutdays[0] * 24, cutdays[1] * 24, max(lenTimes))

            # lshells = list(data[dates[0]]['data'][plotComponent].keys())

            medianDf, meanDf, quartile25Df, quartile75Df = calcMeanMedian(times, plotParameters['param'],
                                                                        data, dates,
                                                                        plotParameters['param'], plotComponent)

            for par in range(0, len(plotParameters['param'])):
                cc = par

                axes[par, ee].plot((medianDf[plotParameters['param'][cc]]), color="orangered", label="Median", linewidth=2)
                axes[par, ee].plot((meanDf[plotParameters['param'][cc]]), color="gray", label="Mean", linewidth=2)
                axes[par, ee].plot((quartile25Df[plotParameters['param'][cc]]), '--', color="#0b91ff", label="Upper Quartile",
                                linewidth=2)
                axes[par, ee].plot((quartile75Df[plotParameters['param'][cc]]), '--', color="#0b91ff", label="Lower Quartile",
                                linewidth=2)
                # axes.fill_between(times,(min_ser), (max_ser), color='#0b89ff', alpha=0.05)
                axes[par, ee].set_yscale('log', base=10)
                if ee == 1:
                    axes[par, ee].set_yticklabels([])
                axes[par, ee].set_xlim(-xlimP[0] * 24, xlimP[1] * 24)
                axes[par, ee].set_ylim(plotParameters['ylim'][cc][0], plotParameters['ylim'][cc][1])
                axes[par, ee].tick_params(direction='in', length=14, width=2, colors='k',
                                        grid_color='k', grid_alpha=0.1, which='major')
                axes[par, ee].tick_params(direction='in', length=7, width=0.8, colors='k',
                                        grid_color='k', grid_alpha=0.1, which='minor')
                if ee != 1:
                    axes[par, ee].set_ylabel(f"Power / ${legendy}$")
                axes[par, ee].axvline(x=0, color='k', linewidth=2)
                # axes.set_yscale('log')
                axes[par, ee].text(-0.035, 1.08, plotParameters['letter'][ee][cc], transform=axes[par, ee].transAxes, size=30)
                titL = f"L* = {plotParameters['param'][cc].split('L')[-1]}"
                axes[par, ee].text(0.035, 1.03, titL, transform=axes[par, ee].transAxes, size=30)
                if plotParameters['letter'][ee][cc] in ['(a)', '(b)']:
                    title = f"{self.eventType[ee].capitalize():}"
                    axes[par, ee].set_title(title, loc='center')
                if plotParameters['letter'][ee][cc] == '(b)':
                    axes[par, ee].legend(loc='upper right', prop={'size': 22})

                axes[par, ee].grid()
        if savePLot:
            plt.savefig(f"{outputDir}{self.eventKind}_Power{plotComponent}LS2.png", bbox_inches='tight')
            plt.savefig(f"{outputDir}{self.eventKind}_Power{plotComponent}LS2.pdf", bbox_inches='tight')
