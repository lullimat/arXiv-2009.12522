# Figure 5, panels (e), (f), (g) and (h)

import sys
sys.path.append("../../")

device_str, lang, _dpi = sys.argv[1], sys.argv[2], int(sys.argv[3])

###########################################################

from pathlib import Path
reproduced_results = Path("reproduced-results")

from sympy import exp as sp_exp
from sympy import symbols as sp_symbols
from sympy import Rational as sp_Rational
from sympy import lambdify as sp_lambdify
from IPython.display import display
import numpy as np
from collections import defaultdict

n = sp_symbols('n')
psis = [sp_exp(-1/n), 1 - sp_exp(-n)]
psi_codes = {psis[0]: 'exp((NType)(-1./ln))', 
             psis[1]: '1. - exp(-(NType)ln)',}

Gs = {psis[0]: [-3.6], 
      psis[1]: [-1.75]}

Ls = [255, 351]

E6_P2F6_sym = sp_symbols("\\bf{E}^{(6)}_{P2\,F6}")
E8_P2F8_sym = sp_symbols("\\bf{E}^{(8)}_{P2\,F8}")
E6_P4F6_sym = sp_symbols("\\bf{E}^{(6)}_{P4\,F6}")
E8_P4F6_sym = sp_symbols("\\bf{E}^{(8)}_{P4\,F6}")
E10_P2F10_sym = sp_symbols("\\bf{E}^{(10)}_{P2\,F10}")
E10_P4F6_sym = sp_symbols("\\bf{E}^{(10)}_{P4\,F6}")
E12_P2F12_sym = sp_symbols("\\bf{E}^{(12)}_{P2\,F12}")
E12_P4F6_sym = sp_symbols("\\bf{E}^{(12)}_{P4\,F6}")


stencil_string = {E6_P2F6_sym: 'E6_P2F6', 
                  E6_P4F6_sym: 'E6_P4F6', 
                  E8_P2F8_sym: 'E8_P2F8', 
                  E8_P4F6_sym: 'E8_P4F6',
                  E10_P2F10_sym: 'E10_P2F10', 
                  E10_P4F6_sym: 'E10_P4F6',
                  E12_P2F12_sym: 'E12_P2F12', 
                  E12_P4F6_sym: 'E12_P4F6'}

stencil_sym_list = [E6_P2F6_sym, E6_P4F6_sym, 
                    E8_P2F8_sym, E8_P4F6_sym,
                    E10_P2F10_sym, E10_P4F6_sym,
                    E12_P2F12_sym, E12_P4F6_sym]

def LaplaceFileName(stencil_sym, psi):
    psi_str = str(psi).replace("/", "_").replace("-", "_")
    psi_str = psi_str.replace(" ", "_")
    psi_str = psi_str.replace("(", "").replace(")","")
    lang_str = str(lang) + "_" + device_str

    return (lang_str + stencil_string[stencil_sym] + "_" + 
            psi_str + "_laplace")

def DropletsFileName(stencil_sym, psi):
    psi_str = str(psi).replace("/", "_").replace("-", "_")
    psi_str = psi_str.replace(" ", "_")
    psi_str = psi_str.replace("(", "").replace(")","")
    lang_str = str(lang) + "_" + device_str

    return (lang_str + stencil_string[stencil_sym] + "_" + 
            psi_str + "_droplets")

def StencilPsiKey(stencil_sym, psi):
    return str(stencil_sym) + "_" + str(psi)

droplets_files = \
    {StencilPsiKey(E6_P2F6_sym, psis[0]): reproduced_results / DropletsFileName(E6_P2F6_sym, psis[0]),  
     StencilPsiKey(E6_P4F6_sym, psis[0]): reproduced_results / DropletsFileName(E6_P4F6_sym, psis[0]), 
     StencilPsiKey(E8_P2F8_sym, psis[0]): reproduced_results / DropletsFileName(E8_P2F8_sym, psis[0]),  
     StencilPsiKey(E8_P4F6_sym, psis[0]): reproduced_results / DropletsFileName(E8_P4F6_sym, psis[0]),
     StencilPsiKey(E10_P2F10_sym, psis[0]): reproduced_results / DropletsFileName(E10_P2F10_sym, psis[0]),  
     StencilPsiKey(E10_P4F6_sym, psis[0]): reproduced_results / DropletsFileName(E10_P4F6_sym, psis[0]),
     StencilPsiKey(E12_P2F12_sym, psis[0]): reproduced_results / DropletsFileName(E12_P2F12_sym, psis[0]),  
     StencilPsiKey(E12_P4F6_sym, psis[0]): reproduced_results / DropletsFileName(E12_P4F6_sym, psis[0]),
     StencilPsiKey(E6_P2F6_sym, psis[1]): reproduced_results / DropletsFileName(E6_P2F6_sym, psis[1]),  
     StencilPsiKey(E6_P4F6_sym, psis[1]): reproduced_results / DropletsFileName(E6_P4F6_sym, psis[1]), 
     StencilPsiKey(E8_P2F8_sym, psis[1]): reproduced_results / DropletsFileName(E8_P2F8_sym, psis[1]),  
     StencilPsiKey(E8_P4F6_sym, psis[1]): reproduced_results / DropletsFileName(E8_P4F6_sym, psis[1]),
     StencilPsiKey(E10_P2F10_sym, psis[1]): reproduced_results / DropletsFileName(E10_P2F10_sym, psis[1]),  
     StencilPsiKey(E10_P4F6_sym, psis[1]): reproduced_results / DropletsFileName(E10_P4F6_sym, psis[1]),
     StencilPsiKey(E12_P2F12_sym, psis[1]): reproduced_results / DropletsFileName(E12_P2F12_sym, psis[1]),  
     StencilPsiKey(E12_P4F6_sym, psis[1]): reproduced_results / DropletsFileName(E12_P4F6_sym, psis[1])}


E_sym_YN = {E8_P2F8_sym: 'No', E8_P4F6_sym: 'Yes',
            E10_P2F10_sym: 'No', E10_P4F6_sym: 'Yes',
            E12_P2F12_sym: 'No', E12_P4F6_sym: 'Yes'}

from idpy.Utils.ManageData import ManageData            

UFields = {}
E_sym_dict = {E6_P2F6_sym: 'E6P2', E6_P4F6_sym: 'E6P4', 
              E8_P2F8_sym: 'E8P2', E8_P4F6_sym: 'E8P4',
              E10_P2F10_sym: 'E10P2', E10_P4F6_sym: 'E10P4',
              E12_P2F12_sym: 'E12P2', E12_P4F6_sym: 'E12P4'}

for L in [255, 351]:
    for _psi in psis:
        for G in Gs[_psi]:
            for _stencil in stencil_sym_list:
                _data_swap = ManageData(dump_file = droplets_files[StencilPsiKey(_stencil, _psi)])
                _is_file_there = _data_swap.Read()
                if not _is_file_there:
                    raise Exception("Could not find file!", 
                                    droplets_files[StencilPsiKey(_stencil, _psi)])        

                _data_key = str(G) + "_" + str(L)
                UFields['G' + str(G) + 'LX' + str(L) + E_sym_dict[_stencil]] = \
                    _data_swap.PullData(_data_key)
    
################################################
########### END DATA HANDLING ##################
################################################

# https://stackoverflow.com/questions/14737681/fill-the-right-column-of-a-matplotlib-legend-first

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, mark_inset)

import matplotlib.ticker as ticker
from sklearn.neighbors import KernelDensity

def GetCumulative(u_bins = None, logprob = None):
    du = u_bins[1] - u_bins[0]
    cumulative = np.zeros(logprob.shape[0])
    c_swap = 0
    for elem_i in range(len(logprob)):
        c_swap += np.exp(logprob[elem_i])*du
        cumulative[elem_i] = c_swap

    print(len(u_bins), len(logprob), c_swap*du)
    return cumulative

##################################################
#################### FIGURE 3 ####################
##################################################

dashed = {}
dashed[-2.6] = '-'
dashed[-3.1] = '--'
dashed[-3.6] = '-'

dashed[-1.4] = '-'
dashed[-1.6] = '-'
dashed[-1.75] = '-'
    
u_bins_iso_E6, u_bins_niso_E6 = {}, {}
logprob_iso_E6, logprob_niso_E6 = {}, {}

u_bins_iso_E8, u_bins_niso_E8 = {}, {}
logprob_iso_E8, logprob_niso_E8 = {}, {}

u_bins_iso_E10, u_bins_niso_E10 = {}, {}
logprob_iso_E10, logprob_niso_E10 = {}, {}

u_bins_iso_E12, u_bins_niso_E12 = {}, {}
logprob_iso_E12, logprob_niso_E12 = {}, {}

MDH = ManageData(dump_file = 'IsoPlotsHistos')
if not MDH.Read():
    G = -3.6
    LX = 255
    LXA = LX**2
    c2 = 1/3.

    for G in [-1.75, -3.6]:
        print("G: ", G)
        sim_2iso_u = UFields['G' + str(G) + 'LX' + str(LX) + 'E6P4']
        sim_2niso_u = UFields['G' + str(G) + 'LX' + str(LX) + 'E6P2']

        sim_2iso_U = np.array([sim_2iso_u[i] for i in range(LXA)])
        sim_2iso_V = np.array([sim_2iso_u[i + LXA] for i in range(LXA)])
        sim_2iso_U = sim_2iso_U.reshape((LX, LX))
        sim_2iso_V = sim_2iso_V.reshape((LX, LX))

        sim_2niso_U = np.array([sim_2niso_u[i] for i in range(LXA)])
        sim_2niso_V = np.array([sim_2niso_u[i + LXA] for i in range(LXA)])
        sim_2niso_U = sim_2niso_U.reshape((LX, LX))
        sim_2niso_V = sim_2niso_V.reshape((LX, LX))

        norm_2niso = np.sqrt((sim_2niso_U**2 + sim_2niso_V**2)/float(c2))
        norm_2iso = np.sqrt((sim_2iso_U**2 + sim_2iso_V**2)/float(c2))

        norm_2iso_histo = norm_2iso.reshape([LXA])
        norm_2niso_histo = norm_2niso.reshape([LXA])

        u_bins_iso_E6[G] = np.linspace(np.log(np.amin(norm_2iso_histo)),
                                    np.log(np.amax(norm_2iso_histo)), 2**12)

        u_bins_niso_E6[G] = np.linspace(np.log(np.amin(norm_2niso_histo)),
                                     np.log(np.amax(norm_2niso_histo)), 2**12)

        kde_iso = KernelDensity(bandwidth=0.25, kernel='gaussian')
        kde_iso.fit(np.log(norm_2iso_histo)[:, None])
        logprob_iso_E6[G] = kde_iso.score_samples(u_bins_iso_E6[G][:, None])

        kde_niso = KernelDensity(bandwidth=0.25, kernel='gaussian')
        kde_niso.fit(np.log(norm_2niso_histo)[:, None])
        logprob_niso_E6[G] = kde_niso.score_samples(u_bins_niso_E6[G][:, None])

    MDH.PushData(data = u_bins_iso_E6, key = 'u_bins_iso' + 'E6')
    MDH.PushData(data = u_bins_niso_E6, key = 'u_bins_niso' + 'E6')
    MDH.PushData(data = logprob_iso_E6, key = 'logprob_iso' + 'E6')
    MDH.PushData(data = logprob_niso_E6, key = 'logprob_niso' + 'E6')

    for G in [-1.75, -3.6]:
        print("G: ", G)
        sim_2iso_u = UFields['G' + str(G) + 'LX' + str(LX) + 'E8P4']
        sim_2niso_u = UFields['G' + str(G) + 'LX' + str(LX) + 'E8P2']

        sim_2iso_U = np.array([sim_2iso_u[i] for i in range(LXA)])
        sim_2iso_V = np.array([sim_2iso_u[i + LXA] for i in range(LXA)])
        sim_2iso_U = sim_2iso_U.reshape((LX, LX))
        sim_2iso_V = sim_2iso_V.reshape((LX, LX))

        sim_2niso_U = np.array([sim_2niso_u[i] for i in range(LXA)])
        sim_2niso_V = np.array([sim_2niso_u[i + LXA] for i in range(LXA)])
        sim_2niso_U = sim_2niso_U.reshape((LX, LX))
        sim_2niso_V = sim_2niso_V.reshape((LX, LX))

        norm_2niso = np.sqrt((sim_2niso_U**2 + sim_2niso_V**2)/float(c2))
        norm_2iso = np.sqrt((sim_2iso_U**2 + sim_2iso_V**2)/float(c2))

        norm_2iso_histo = norm_2iso.reshape([LXA])
        norm_2niso_histo = norm_2niso.reshape([LXA])

        u_bins_iso_E8[G] = np.linspace(np.log(np.amin(norm_2iso_histo)),
                                    np.log(np.amax(norm_2iso_histo)), 2**12)

        u_bins_niso_E8[G] = np.linspace(np.log(np.amin(norm_2niso_histo)),
                                     np.log(np.amax(norm_2niso_histo)), 2**12)

        kde_iso = KernelDensity(bandwidth=0.25, kernel='gaussian')
        kde_iso.fit(np.log(norm_2iso_histo)[:, None])
        logprob_iso_E8[G] = kde_iso.score_samples(u_bins_iso_E8[G][:, None])

        kde_niso = KernelDensity(bandwidth=0.25, kernel='gaussian')
        kde_niso.fit(np.log(norm_2niso_histo)[:, None])
        logprob_niso_E8[G] = kde_niso.score_samples(u_bins_niso_E8[G][:, None])

    MDH.PushData(data = u_bins_iso_E8, key = 'u_bins_iso' + 'E8')
    MDH.PushData(data = u_bins_niso_E8, key = 'u_bins_niso' + 'E8')
    MDH.PushData(data = logprob_iso_E8, key = 'logprob_iso' + 'E8')
    MDH.PushData(data = logprob_niso_E8, key = 'logprob_niso' + 'E8')

    for G in [-1.75, -3.6]:
        print("G: ", G)
        sim_2iso_u = UFields['G' + str(G) + 'LX' + str(LX) + 'E10P4']
        sim_2niso_u = UFields['G' + str(G) + 'LX' + str(LX) + 'E10P2']

        sim_2iso_U = np.array([sim_2iso_u[i] for i in range(LXA)])
        sim_2iso_V = np.array([sim_2iso_u[i + LXA] for i in range(LXA)])
        #sim_2iso_U = sim_2iso_U.reshape((LX, LX))
        #sim_2iso_V = sim_2iso_V.reshape((LX, LX))

        sim_2niso_U = np.array([sim_2niso_u[i] for i in range(LXA)])
        sim_2niso_V = np.array([sim_2niso_u[i + LXA] for i in range(LXA)])
        #sim_2niso_U = sim_2niso_U.reshape((LX, LX))
        #sim_2niso_V = sim_2niso_V.reshape((LX, LX))

        norm_2niso_histo = np.sqrt((sim_2niso_U**2 + sim_2niso_V**2)/float(c2))
        norm_2iso_histo = np.sqrt((sim_2iso_U**2 + sim_2iso_V**2)/float(c2))

        #norm_2iso_histo = norm_2iso.reshape([LXA])
        #norm_2niso_histo = norm_2niso.reshape([LXA])

        print(norm_2iso_histo.shape, norm_2niso_histo.shape)        

        u_bins_iso_E10[G] = np.linspace(np.log(np.amin(norm_2iso_histo)),
                                    np.log(np.amax(norm_2iso_histo)), 2**12)

        u_bins_niso_E10[G] = np.linspace(np.log(np.amin(norm_2niso_histo)),
                                     np.log(np.amax(norm_2niso_histo)), 2**12)

        kde_iso = KernelDensity(bandwidth=0.25, kernel='gaussian')
        kde_iso.fit(np.log(norm_2iso_histo)[:, None])
        logprob_iso_E10[G] = kde_iso.score_samples(u_bins_iso_E10[G][:, None])

        kde_niso = KernelDensity(bandwidth=0.25, kernel='gaussian')
        kde_niso.fit(np.log(norm_2niso_histo)[:, None])
        logprob_niso_E10[G] = kde_niso.score_samples(u_bins_niso_E10[G][:, None])

    MDH.PushData(data = u_bins_iso_E10, key = 'u_bins_iso' + 'E10')
    MDH.PushData(data = u_bins_niso_E10, key = 'u_bins_niso' + 'E10')
    MDH.PushData(data = logprob_iso_E10, key = 'logprob_iso' + 'E10')
    MDH.PushData(data = logprob_niso_E10, key = 'logprob_niso' + 'E10')    

    for G in [-1.75, -3.6]:
        print("G: ", G)
        sim_2iso_u = UFields['G' + str(G) + 'LX' + str(LX) + 'E12P4']
        sim_2niso_u = UFields['G' + str(G) + 'LX' + str(LX) + 'E12P2']

        sim_2iso_U = np.array([sim_2iso_u[i] for i in range(LXA)])
        sim_2iso_V = np.array([sim_2iso_u[i + LXA] for i in range(LXA)])
        #sim_2iso_U = sim_2iso_U.reshape((LX, LX))
        #sim_2iso_V = sim_2iso_V.reshape((LX, LX))

        sim_2niso_U = np.array([sim_2niso_u[i] for i in range(LXA)])
        sim_2niso_V = np.array([sim_2niso_u[i + LXA] for i in range(LXA)])
        #sim_2niso_U = sim_2niso_U.reshape((LX, LX))
        #sim_2niso_V = sim_2niso_V.reshape((LX, LX))

        norm_2niso_histo = np.sqrt((sim_2niso_U**2 + sim_2niso_V**2)/float(c2))
        norm_2iso_histo = np.sqrt((sim_2iso_U**2 + sim_2iso_V**2)/float(c2))

        #norm_2iso_histo = norm_2iso.reshape([LXA])
        #norm_2niso_histo = norm_2niso.reshape([LXA])

        print(norm_2iso_histo.shape, norm_2niso_histo.shape)        

        u_bins_iso_E12[G] = np.linspace(np.log(np.amin(norm_2iso_histo)),
                                    np.log(np.amax(norm_2iso_histo)), 2**12)

        u_bins_niso_E12[G] = np.linspace(np.log(np.amin(norm_2niso_histo)),
                                     np.log(np.amax(norm_2niso_histo)), 2**12)

        kde_iso = KernelDensity(bandwidth=0.25, kernel='gaussian')
        kde_iso.fit(np.log(norm_2iso_histo)[:, None])
        logprob_iso_E12[G] = kde_iso.score_samples(u_bins_iso_E12[G][:, None])

        kde_niso = KernelDensity(bandwidth=0.25, kernel='gaussian')
        kde_niso.fit(np.log(norm_2niso_histo)[:, None])
        logprob_niso_E12[G] = kde_niso.score_samples(u_bins_niso_E12[G][:, None])

    MDH.PushData(data = u_bins_iso_E12, key = 'u_bins_iso' + 'E12')
    MDH.PushData(data = u_bins_niso_E12, key = 'u_bins_niso' + 'E12')
    MDH.PushData(data = logprob_iso_E12, key = 'logprob_iso' + 'E12')
    MDH.PushData(data = logprob_niso_E12, key = 'logprob_niso' + 'E12')    

    MDH.Dump()
else:
    u_bins_iso_E6 = MDH.PullData('u_bins_iso' + 'E6')
    u_bins_niso_E6 = MDH.PullData('u_bins_niso' + 'E6')
    logprob_iso_E6 = MDH.PullData('logprob_iso' + 'E6')
    logprob_niso_E6 = MDH.PullData('logprob_niso' + 'E6')

    u_bins_iso_E8 = MDH.PullData('u_bins_iso' + 'E8')
    u_bins_niso_E8 = MDH.PullData('u_bins_niso' + 'E8')
    logprob_iso_E8 = MDH.PullData('logprob_iso' + 'E8')
    logprob_niso_E8 = MDH.PullData('logprob_niso' + 'E8')

    u_bins_iso_E10 = MDH.PullData('u_bins_iso' + 'E10')
    u_bins_niso_E10 = MDH.PullData('u_bins_niso' + 'E10')
    logprob_iso_E10 = MDH.PullData('logprob_iso' + 'E10')
    logprob_niso_E10 = MDH.PullData('logprob_niso' + 'E10')

    u_bins_iso_E12 = MDH.PullData('u_bins_iso' + 'E12')
    u_bins_niso_E12 = MDH.PullData('u_bins_niso' + 'E12')
    logprob_iso_E12 = MDH.PullData('logprob_iso' + 'E12')
    logprob_niso_E12 = MDH.PullData('logprob_niso' + 'E12')    

from matplotlib import rc, rcParams
from idpy.Utils.Plots import SetAxPanelLabelCoords, SetMatplotlibLatexParamas, CreateFiguresPanels, SetAxTicksFont
from idpy.Utils.Plots import SetAxTicksFont

SetMatplotlibLatexParamas([rc], [rcParams])

if False:
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    rcParams['text.latex.preamble']=[r"\usepackage{amsmath}\usepackage{xcolor}"]
    rcParams["xtick.top"] = True

x_range = [7e-8,2e-2]
x_range = [2e-8,2e-2]
y_range = [0, 0.62]
f_s = 12
pos_c_label = (0.8e-4, 0.55)
pos_c_label0 = (1.5e-6, 0.59)
pos_label = (4.5e-3, 0.55)
## G_label_pos = (0.05, 1.215)
G_label_pos = (0.05, 0.9125)
lw = 2.9
lw_c = 2

#################### SIZES ####################
fig = plt.figure(figsize=(6.5*0.95, 2.8*0.95))
###############################################

#################### PANEL (e) ####################
ax1 = plt.subplot2grid((1,2), (0,0), colspan=1, rowspan=1)
G = -1.75

cumulative_iso = GetCumulative(u_bins = u_bins_iso_E6[G], logprob = logprob_iso_E6[G])
cumulative_niso = GetCumulative(u_bins = u_bins_niso_E6[G], logprob = logprob_niso_E6[G])

ax1.plot(np.exp(u_bins_iso_E6[G]), np.exp(logprob_iso_E6[G]), dashed[G], color = 'red',
         linewidth = lw)
ax1.plot(np.exp(u_bins_niso_E6[G]), np.exp(logprob_niso_E6[G]), dashed[G], color = 'blue')

##### INSET CUMULATIVE #####
ax1_bis = inset_axes(ax1, width='40%', height='40%', loc = 'center left',
                     bbox_to_anchor=(0.04,0.135,1,1),
                     bbox_transform=ax1.transAxes)

ax1_bis.plot(np.exp(u_bins_iso_E6[G]), 1 - cumulative_iso, dashed[G], color = 'red', linewidth=0.7*lw_c)
ax1_bis.plot(np.exp(u_bins_niso_E6[G]), 1 - cumulative_niso, dashed[G], color = 'blue', linewidth=0.7)

ax1_bis.set_xscale('log')
ax1_bis.set_xlim([0.8e-5, x_range[1]])
ax1_bis.set_xticks([1e-5, 1e-3])

ax1_bis.yaxis.set_label_coords(0.15,0.5)
ax1_bis.set_ylabel('$1 - F$', fontsize=8)

## Setting tivks size
if False:
    for tick in ax1_bis.yaxis.get_major_ticks():
        tick.label.set_fontsize(5)
    for tick in ax1_bis.xaxis.get_major_ticks():
        tick.label.set_fontsize(5)
else:
    SetAxTicksFont(ax1_bis, 5)                

###############################

#ax1.axvline(x = 2.5e-4, ls = '--', color = 'black', linewidth = 0.5)

## Try using colors and speciy P and F  \;\;\, (a)

_e_label_pos00 = (0., 1.05)
ax1.text(_e_label_pos00[0], _e_label_pos00[1],
         r'$\bf{E}^{(6)}_{P4,F6}$', color = 'red',
         transform = ax1.transAxes, fontsize = f_s)

_e_label_pos10 = (0.25, 1.05)
ax1.text(_e_label_pos10[0], _e_label_pos10[1],
         r'$\bf{E}^{(6)}_{P2,F6}$', color = 'blue',
         transform = ax1.transAxes, fontsize = f_s)

ax1.text(5e-4, 0.65, '$\\psi = 1 - \\exp(-n)$', fontsize = f_s)


ax1.text(G_label_pos[0], G_label_pos[1], '$G c_s^2 = -1.75$',
         transform = ax1.transAxes, fontsize = f_s)

ax1.text(pos_label[0], pos_label[1], '$(e)$', fontsize = f_s)
ax1.set_xscale('log')
ax1.set_xlabel('$u/c_s$', fontsize = f_s)
ax1.set_ylabel('$p(\\log(u/c_s))$', fontsize = f_s)
ax1.set_xlim(x_range)

ax1.set_ylim(y_range)
ax1.set_yticks(np.arange(0, 0.8, 0.2))

# Remove all labels and set ticks (major and minor) inside
#ax1.tick_params(which = "both", axis="x",direction="in")
ax1.minorticks_off()

plt.setp(ax1.get_yticklabels()[0], visible=False)

#################### PANEL (f) ####################
ax1_1 = plt.subplot2grid((1,2), (0,1), colspan=1, rowspan=1, sharey = ax1)

G = -1.75

cumulative_iso = GetCumulative(u_bins = u_bins_iso_E8[G], logprob = logprob_iso_E8[G])
cumulative_niso = GetCumulative(u_bins = u_bins_niso_E8[G], logprob = logprob_niso_E8[G])

ax1_1.plot(np.exp(u_bins_iso_E8[G]), np.exp(logprob_iso_E8[G]), dashed[G], color = 'red',
           linewidth = lw)
ax1_1.plot(np.exp(u_bins_niso_E8[G]), np.exp(logprob_niso_E8[G]), dashed[G], color = 'blue')

##### INSET CUMULATIVE #####
ax1_1_bis = inset_axes(ax1_1, width='40%', height='40%', loc = 'center left',
                       bbox_to_anchor=(0.04,0.135,1,1),
                       bbox_transform=ax1_1.transAxes)

ax1_1_bis.plot(np.exp(u_bins_iso_E8[G]), 1 - cumulative_iso, dashed[G], color = 'red', linewidth=0.7*lw_c)
ax1_1_bis.plot(np.exp(u_bins_niso_E8[G]), 1 - cumulative_niso, dashed[G], color = 'blue', linewidth=0.7)

ax1_1_bis.set_xscale('log')
ax1_1_bis.set_xlim([0.8e-5, x_range[1]])
ax1_1_bis.set_xticks([1e-5, 1e-3])

ax1_1_bis.yaxis.set_label_coords(0.15,0.5)
ax1_1_bis.set_ylabel('$1 - F$', fontsize=8)

## Setting tivks size
if False:
    for tick in ax1_1_bis.yaxis.get_major_ticks():
        tick.label.set_fontsize(5)
    for tick in ax1_1_bis.xaxis.get_major_ticks():
        tick.label.set_fontsize(5)
else:
    SetAxTicksFont(ax1_1_bis, 5)                

###############################

#ax1_1.axvline(x = 3.2e-4, ls = '--', color = 'black', linewidth = 0.5)

_e_label_pos01 = (0.55, 1.05)
ax1_1.text(_e_label_pos01[0], _e_label_pos01[1],
           r'$\bf{E}^{(8)}_{P4,F6}$', color = 'red',
           transform = ax1_1.transAxes, fontsize = f_s)

_e_label_pos11 = (0.8, 1.05)
ax1_1.text(_e_label_pos11[0], _e_label_pos11[1],
           r'$\bf{E}^{(8)}_{P2,F8}$', color = 'blue',
           transform = ax1_1.transAxes, fontsize = f_s)

ax1_1.text(G_label_pos[0], G_label_pos[1], '$G c_s^2 = -1.75$',
            transform = ax1_1.transAxes, fontsize = f_s)

ax1_1.text(pos_label[0], pos_label[1], '$(f)$', fontsize = f_s)
ax1_1.set_xlim(x_range)
#ax1_1.set_xticks([1e-5, 1e-4, 1e-3])
ax1_1.set_ylim(y_range)
ax1_1.set_xscale('log')
ax1_1.set_xlabel('$u/c_s$', fontsize = f_s)
ax1_1.minorticks_off()

#ax1_1.set_yticks([])

# Remove all labels and set ticks (major and minor) inside
ax1_1.get_yaxis().set_visible(False)

# Remove 0 label
#plt.setp(ax1_1.get_yticklabels()[0], visible=False)

fig.subplots_adjust(hspace=0)
fig.subplots_adjust(wspace=0)
#################### SAVING ####################

from pathlib import Path
reproduced_figures = Path("reproduced-figures")
if not reproduced_figures.is_dir():
    reproduced_figures.mkdir()

plt.savefig(reproduced_figures / 'figure_7_ef.png', 
            bbox_inches = 'tight', dpi = _dpi)
plt.close()
