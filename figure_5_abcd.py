# Figure 5, panels (a), (b), (c) and (d)
###########################################################

import sys
sys.path.append("../../")

device_str, lang, _dpi = sys.argv[1], sys.argv[2], int(sys.argv[3])

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

Gs = {psis[0]: [-2.6, -3.1, -3.6], 
      psis[1]: [-1.4, -1.6]}

Ls = [127, 159, 191, 223, 255, 287, 319, 351]

E6_P2F6_sym = sp_symbols("\\boldsymbol{E}^{(6)}_{P2\,F6}")
E8_P2F8_sym = sp_symbols("\\boldsymbol{E}^{(8)}_{P2\,F8}")
E6_P4F6_sym = sp_symbols("\\boldsymbol{E}^{(6)}_{P4\,F6}")
E8_P4F6_sym = sp_symbols("\\boldsymbol{E}^{(8)}_{P4\,F6}")

stencil_string = {E6_P2F6_sym: 'E6_P2F6', 
                  E6_P4F6_sym: 'E6_P4F6', 
                  E8_P2F8_sym: 'E8_P2F8', 
                  E8_P4F6_sym: 'E8_P4F6'}

stencil_sym_list = [E6_P2F6_sym, E6_P4F6_sym, 
                    E8_P2F8_sym, E8_P4F6_sym]

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
     StencilPsiKey(E6_P2F6_sym, psis[1]): reproduced_results / DropletsFileName(E6_P2F6_sym, psis[1]),  
     StencilPsiKey(E6_P4F6_sym, psis[1]): reproduced_results / DropletsFileName(E6_P4F6_sym, psis[1]), 
     StencilPsiKey(E8_P2F8_sym, psis[1]): reproduced_results / DropletsFileName(E8_P2F8_sym, psis[1]),  
     StencilPsiKey(E8_P4F6_sym, psis[1]): reproduced_results / DropletsFileName(E8_P4F6_sym, psis[1])}


E_sym_YN = {E8_P2F8_sym: 'No', E8_P4F6_sym: 'Yes'}

from idpy.Utils.ManageData import ManageData            

UFields = {}
E_sym_dict = {E6_P2F6_sym: 'E6P2', E6_P4F6_sym: 'E6P4', 
              E8_P2F8_sym: 'E8P2', E8_P4F6_sym: 'E8P4'}

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

from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

import matplotlib.ticker as ticker
from sklearn.neighbors import KernelDensity

def GetCumulative(u_bins = None, logprob = None):
    du = u_bins[1] - u_bins[0]
    cumulative = np.zeros(logprob.shape[0])
    c_swap = 0
    for elem_i in range(len(logprob)):
        c_swap += np.exp(logprob[elem_i])
        cumulative[elem_i] = c_swap

    print(len(u_bins), len(logprob), c_swap*du)
    return np.copy(cumulative*du)

##################################################
#################### FIGURE 3 ####################
##################################################

dashed = {}
dashed[-2.6] = '-'
dashed[-3.1] = '--'
dashed[-3.6] = '-'

dashed[-1.4] = '-'
dashed[-1.6] = '-'
    
u_bins_iso_E6, u_bins_niso_E6 = {}, {}
logprob_iso_E6, logprob_niso_E6 = {}, {}

u_bins_iso_E8, u_bins_niso_E8 = {}, {}
logprob_iso_E8, logprob_niso_E8 = {}, {}

MDH = ManageData(dump_file = 'IsoPlotsHistos')
if not MDH.Read():
    G = -3.6
    LX = 255
    LXA = LX**2
    c2 = 1/3.
    grains = 2**13

    for G in [-1.4, -1.6, -2.6, -3.1, -3.6]:
        print("G: ", G)
        sim_2iso_u = UFields['G' + str(G) + 'LX' + str(LX) + 'E6P4']
        sim_2niso_u = UFields['G' + str(G) + 'LX' + str(LX) + 'E6P2']

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

        u_bins_iso_E6[G] = np.linspace(np.log(np.amin(norm_2iso_histo)),
                                    np.log(np.amax(norm_2iso_histo)), grains)

        u_bins_niso_E6[G] = np.linspace(np.log(np.amin(norm_2niso_histo)),
                                     np.log(np.amax(norm_2niso_histo)), grains)

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

    for G in [-1.4, -1.6, -2.6, -3.1, -3.6]:
        print("G: ", G)
        sim_2iso_u = UFields['G' + str(G) + 'LX' + str(LX) + 'E8P4']
        sim_2niso_u = UFields['G' + str(G) + 'LX' + str(LX) + 'E8P2']

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

        u_bins_iso_E8[G] = np.linspace(np.log(np.amin(norm_2iso_histo)),
                                    np.log(np.amax(norm_2iso_histo)), grains)

        u_bins_niso_E8[G] = np.linspace(np.log(np.amin(norm_2niso_histo)),
                                     np.log(np.amax(norm_2niso_histo)), grains)

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

from matplotlib import rc, rcParams
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}\usepackage{xcolor}"]
rcParams["xtick.top"] = True

x_range = [5e-7,2e-2]
#y_range = [0, 1.05]
y_range = [0, 0.62]
f_s = 12
pos_c_label = (1.5e-4, 0.535)
pos_c_label0 = (1e-6, 0.535)
pos_label = (0.7e-2, 0.55)
lw = 2.9
lw_c = 2

#################### SIZES ####################
fig = plt.figure(figsize=(6.5*0.95, 5.6*0.95))
###############################################

#################### PANEL (a) ####################
ax1 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)
G = -3.6

cumulative_iso = GetCumulative(u_bins = u_bins_iso_E6[G], logprob = logprob_iso_E6[G])
cumulative_niso = GetCumulative(u_bins = u_bins_niso_E6[G], logprob = logprob_niso_E6[G])

ax1.plot(np.exp(u_bins_iso_E6[G]), np.exp(logprob_iso_E6[G]), dashed[G], color = 'red',
         linewidth = lw)
ax1.plot(np.exp(u_bins_niso_E6[G]), np.exp(logprob_niso_E6[G]), dashed[G], color = 'blue')

##### INSET CUMULATIVE #####
ax1_bis = inset_axes(ax1, width='40%', height='40%', loc = 'center left',
                     bbox_to_anchor=(0.04,0,1,1),
                     bbox_transform=ax1.transAxes)

ax1_bis.plot(np.exp(u_bins_iso_E6[G]), 1 - cumulative_iso, dashed[G], color = 'red', linewidth=0.7*lw_c)
ax1_bis.plot(np.exp(u_bins_niso_E6[G]), 1 - cumulative_niso, dashed[G], color = 'blue', linewidth=0.7)

ax1_bis.set_xscale('log')
ax1_bis.set_xlim(x_range)
ax1_bis.set_xticks([1e-5, 1e-3])

ax1_bis.yaxis.set_label_coords(0.15,0.5)
ax1_bis.set_ylabel('$1 - F$', fontsize=8)

## Setting tivks size
for tick in ax1_bis.yaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax1_bis.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)

###############################

## Try using colors and speciy P and F  \;\;\, (a)
ax1.text(5e-7, 0.63, r'$\boldsymbol{E}^{(6)}_{P4,F6}$', color = 'red', fontsize = f_s)
ax1.text(1e-5, 0.63, r'$\boldsymbol{E}^{(6)}_{P2,F6}$', color = 'blue', fontsize = f_s)
ax1.text(2e-3, 0.65, '$\\psi = \\exp(-1/n)$', fontsize = f_s)
ax1.text(pos_c_label0[0], pos_c_label0[1], '$G c_s^2 = -3.6$', fontsize = f_s)
ax1.text(pos_label[0], pos_label[1], '$(a)$', fontsize = f_s)
ax1.set_xscale('log')
ax1.set_xlim(x_range)
ax1.set_ylim(y_range)

# Remove all labels and set ticks (major and minor) inside
ax1.tick_params(which = "both", axis="x",direction="in")
ax1.set_yticks(np.arange(0, 0.8, 0.2))
plt.setp(ax1.get_xticklabels()[:], visible=False)

plt.setp(ax1.get_yticklabels()[0], visible=False)

#################### PANEL (c) ####################
ax1_1 = plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1, sharey = ax1)

G = -3.6

cumulative_iso = GetCumulative(u_bins = u_bins_iso_E8[G], logprob = logprob_iso_E8[G])
cumulative_niso = GetCumulative(u_bins = u_bins_niso_E8[G], logprob = logprob_niso_E8[G])

ax1_1.plot(np.exp(u_bins_iso_E8[G]), np.exp(logprob_iso_E8[G]), dashed[G], color = 'red',
           linewidth = lw)
ax1_1.plot(np.exp(u_bins_niso_E8[G]), np.exp(logprob_niso_E8[G]), dashed[G], color = 'blue')

##### INSET CUMULATIVE #####
ax1_1_bis = inset_axes(ax1_1, width='40%', height='40%', loc = 'center left',
                       bbox_to_anchor=(0.04,0,1,1),
                       bbox_transform=ax1_1.transAxes)

ax1_1_bis.plot(np.exp(u_bins_iso_E8[G]), 1 - cumulative_iso, dashed[G], color = 'red', linewidth=0.7*lw_c)
ax1_1_bis.plot(np.exp(u_bins_niso_E8[G]), 1 - cumulative_niso, dashed[G], color = 'blue', linewidth=0.7)

ax1_1_bis.set_xscale('log')
ax1_1_bis.set_xlim(x_range)
ax1_1_bis.set_xticks([1e-5, 1e-3])

ax1_1_bis.yaxis.set_label_coords(0.15,0.5)
ax1_1_bis.set_ylabel('$1 - F$', fontsize=8)

## Setting tivks size
for tick in ax1_1_bis.yaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax1_1_bis.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)

###############################

##ax1_1.axvline(x = 3.2e-4, ls = '--', color = 'black', linewidth = 0.5)

ax1_1.text(1e-4, 0.63, r'$\boldsymbol{E}^{(8)}_{P4,F6}$', color = 'red', fontsize = f_s)
ax1_1.text(2e-3, 0.63, r'$\boldsymbol{E}^{(8)}_{P2,F8}$', color = 'blue', fontsize = f_s)
ax1_1.text(pos_c_label0[0], pos_c_label0[1], '$G c_s^2 = -3.6$', fontsize = f_s)
ax1_1.text(pos_label[0], pos_label[1], '$(c)$', fontsize = f_s)
ax1_1.set_xlim(x_range)
ax1_1.set_ylim(y_range)

ax1_1.set_xscale('log')

# Remove all labels and set ticks (major and minor) inside
ax1_1.get_yaxis().set_visible(False)
ax1_1.tick_params(which = "both", axis="x",direction="in")
plt.setp(ax1_1.get_xticklabels()[:], visible=False)

# Remove 0 label
#plt.setp(ax1_1.get_yticklabels()[0], visible=False)

#################### PANEL (b) ####################

ax3 = plt.subplot2grid((2,2), (1,0), colspan=1, rowspan=1, sharex=ax1)
G = -2.6

cumulative_iso = GetCumulative(u_bins = u_bins_iso_E6[G], logprob = logprob_iso_E6[G])
cumulative_niso = GetCumulative(u_bins = u_bins_niso_E6[G], logprob = logprob_niso_E6[G])

ax3.plot(np.exp(u_bins_iso_E6[G]), np.exp(logprob_iso_E6[G]), dashed[G], color = 'red',
         linewidth = lw)
ax3.plot(np.exp(u_bins_niso_E6[G]), np.exp(logprob_niso_E6[G]), dashed[G], color = 'blue')

##### INSET CUMULATIVE #####
ax3_bis = inset_axes(ax3, width='40%', height='40%', loc = 'right',
                     bbox_to_anchor=(0.,0,1,1),
                     bbox_transform=ax3.transAxes)

ax3_bis.plot(np.exp(u_bins_iso_E6[G]), 1 - cumulative_iso, dashed[G], color = 'red', linewidth=0.7*lw_c)
ax3_bis.plot(np.exp(u_bins_niso_E6[G]), 1 - cumulative_niso, dashed[G], color = 'blue', linewidth=0.7)

ax3_bis.set_xscale('log')
ax3_bis.set_xlim([x_range[0], 5e-4])
ax3_bis.set_xticks([1e-5, 1e-4])

ax3_bis.yaxis.set_label_coords(-0.05,0.5)
ax3_bis.set_ylabel('$1 - F$', fontsize=8)

## Setting tivks size
for tick in ax3_bis.yaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax3_bis.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)

###############################


ax3.set_ylabel('$p(\\log(u/c_s))$', fontsize = f_s, labelpad=+10)
ax3.set_yticks(np.arange(0, 0.8, 0.2))
ax3.yaxis.set_label_coords(-0.15,1)
ax3.set_xlabel('$u/c_s$', fontsize = f_s)
ax3.text(pos_c_label[0], pos_c_label[1], '$G c_s^2 = -2.6$', fontsize = f_s)
ax3.text(pos_label[0], pos_label[1], '$(b)$', fontsize = f_s)

ax3.set_xscale('log')
ax3.set_xlim(x_range)

ax3.set_ylim(y_range)

# Set ticks (major and minor) inside
ax3.tick_params(which = "both", axis="x",direction="in", pad = +9)

#################### PANEL (d) ####################
ax3_1 = plt.subplot2grid((2,2), (1,1), colspan=1, rowspan=1, sharey = ax3)

G = -2.6

cumulative_iso = GetCumulative(u_bins = u_bins_iso_E8[G], logprob = logprob_iso_E8[G])
cumulative_niso = GetCumulative(u_bins = u_bins_niso_E8[G], logprob = logprob_niso_E8[G])

ax3_1.plot(np.exp(u_bins_iso_E8[G]), np.exp(logprob_iso_E8[G]), dashed[G], color = 'red', linewidth = lw)
ax3_1.plot(np.exp(u_bins_niso_E8[G]), np.exp(logprob_niso_E8[G]), dashed[G], color = 'blue')

##### INSET CUMULATIVE #####
ax3_1_bis = inset_axes(ax3_1, width='40%', height='40%', loc = 'right',
                       bbox_to_anchor=(0.,0,1,1),
                       bbox_transform=ax3_1.transAxes)

ax3_1_bis.plot(np.exp(u_bins_iso_E8[G]), 1 - cumulative_iso, dashed[G], color = 'red', linewidth=0.7*lw_c)
ax3_1_bis.plot(np.exp(u_bins_niso_E8[G]), 1 - cumulative_niso, dashed[G], color = 'blue', linewidth=0.7)

ax3_1_bis.set_xscale('log')
ax3_1_bis.set_xlim([x_range[0], 5e-4])
ax3_1_bis.set_xticks([1e-5, 1e-4])

ax3_1_bis.yaxis.set_label_coords(-0.05,0.5)
ax3_1_bis.set_ylabel('$1 - F$', fontsize=8)

## Setting tivks size
for tick in ax3_1_bis.yaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax3_1_bis.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)

###############################


ax3_1.set_xlabel('$u/c_s$', fontsize = f_s)
ax3_1.text(pos_c_label[0], pos_c_label[1], '$G c_s^2 = -2.6$', fontsize = f_s)
ax3_1.text(pos_label[0], pos_label[1], '$(d)$', fontsize = f_s)

ax3_1.set_xlim(x_range)
ax3_1.set_ylim(y_range)
ax3_1.set_xscale('log')

# Set ticks (major and minor) inside
ax3_1.tick_params(which = "both", axis="x",direction="in", pad = +9)
ax3_1.get_yaxis().set_visible(False)

fig.subplots_adjust(hspace=0)
fig.subplots_adjust(wspace=0)
#################### SAVING ####################

from pathlib import Path
reproduced_figures = Path("reproduced-figures")
if not reproduced_figures.is_dir():
    reproduced_figures.mkdir()
    
plt.savefig(reproduced_figures / 'figure_5_abcd.png',
            bbox_inches = 'tight', dpi = _dpi)

plt.close()
