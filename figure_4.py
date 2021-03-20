import sys
sys.path.append("../../")

device_str, lang, _dpi = sys.argv[1], sys.argv[2], int(sys.argv[3])

from sympy import exp as sp_exp
from sympy import symbols as sp_symbols
from sympy import Rational as sp_Rational
from collections import defaultdict
import numpy as np

from idpy.Utils.ManageData import ManageData
from idpy.LBM.LBM import XIStencils
from idpy.LBM.SCFStencils import SCFStencils, BasisVectors
from idpy.LBM.SCThermo import ShanChanEquilibriumCache

from pathlib import Path
reproduced_results = Path("reproduced-results")

##########################################################################

n = sp_symbols('n')
psis = [sp_exp(-1/n), 1 - sp_exp(-n)]
psi_codes = {psis[0]: 'exp((NType)(-1./ln))', 
             psis[1]: '1. - exp(-(NType)ln)',}

Gs = {psis[0]: [-2.6, -3.1, -3.6], 
      psis[1]: [-1.4, -1.6, -1.75]}

Ls = [127, 159, 191, 223, 255, 287, 319, 351]

E6_P2F6_sym = sp_symbols("\\boldsymbol{E}^{(6)}_{P2\,F6}")
E6_P4F6_sym = sp_symbols("\\boldsymbol{E}^{(6)}_{P4\,F6}")
E8_P2F8_sym = sp_symbols("\\boldsymbol{E}^{(8)}_{P2\,F8}")
E8_P4F6_sym = sp_symbols("\\boldsymbol{E}^{(8)}_{P4\,F6}")
E10_P2F10_sym = sp_symbols("\\boldsymbol{E}^{(10)}_{P2\,F10}")
E10_P4F6_sym = sp_symbols("\\boldsymbol{E}^{(10)}_{P4\,F6}")
E12_P2F12_sym = sp_symbols("\\boldsymbol{E}^{(12)}_{P2\,F12}")
E12_P4F6_sym = sp_symbols("\\boldsymbol{E}^{(12)}_{P4\,F6}")

'''
Getting usual weights
'''

S5_E6_P2F6 = SCFStencils(E = BasisVectors(x_max = 2), 
                         len_2s = [1, 2, 4])
S5_E6_P2F6_W = S5_E6_P2F6.FindWeights()

S5_E8_P2F8 = SCFStencils(E = BasisVectors(x_max = 2), 
                         len_2s = [1, 2, 4, 5, 8])
S5_E8_P2F8_W = S5_E8_P2F8.FindWeights()

S5_E10_P2F10 = SCFStencils(E = BasisVectors(x_max = 3), 
                           len_2s = [1, 2, 4, 5, 8, 9, 10])
S5_E10_P2F10_W = S5_E10_P2F10.FindWeights()

S5_E12_P2F12 = SCFStencils(E = BasisVectors(x_max = 4), 
                           len_2s = [1, 2, 4, 5, 8, 9, 10, 13, 16, 17])
S5_E12_P2F12_W = S5_E12_P2F12.FindWeights()

'''
File Names
'''
stencil_string = {E6_P2F6_sym: 'E6_P2F6', 
                  E6_P4F6_sym: 'E6_P4F6',
                  E8_P2F8_sym: 'E8_P2F8', 
                  E8_P4F6_sym: 'E8_P4F6',
                  E10_P2F10_sym: 'E10_P2F10', 
                  E10_P4F6_sym: 'E10_P4F6',
                  E12_P2F12_sym: 'E12_P2F12', 
                  E12_P4F6_sym: 'E12_P4F6'}

stencil_dict = {E6_P2F6_sym: S5_E6_P2F6, 
                E8_P2F8_sym: S5_E8_P2F8, 
                E10_P2F10_sym: S5_E10_P2F10, 
                E12_P2F12_sym: S5_E12_P2F12}

stencil_sym_list = [E6_P2F6_sym, E6_P4F6_sym,
                    E8_P2F8_sym, E8_P4F6_sym,
                    E10_P2F10_sym, E10_P4F6_sym,
                    E12_P2F12_sym, E12_P4F6_sym]


def FlatFileName(stencil_sym, psi):
    psi_str = str(psi).replace("/", "_").replace("-", "_")
    psi_str = psi_str.replace(" ", "_")
    psi_str = psi_str.replace("(", "").replace(")","")
    lang_str = str(lang) + "_" + device_str

    return (lang_str + stencil_string[stencil_sym] + "_" + 
            psi_str + "_flat_profile")

def LaplaceFileName(stencil_sym, psi):
    psi_str = str(psi).replace("/", "_").replace("-", "_")
    psi_str = psi_str.replace(" ", "_")
    psi_str = psi_str.replace("(", "").replace(")","")
    lang_str = str(lang) + "_" + device_str

    return (lang_str + stencil_string[stencil_sym] + "_" + 
            psi_str + "_laplace")

def StencilPsiKey(stencil_sym, psi):
    return str(stencil_sym) + "_" + str(psi)

flat_files, laplace_files = {}, {}
for key in stencil_string:
    for _psi in psis:
        flat_files[StencilPsiKey(key, _psi)] = \
            reproduced_results / FlatFileName(key, _psi)
        laplace_files[StencilPsiKey(key, _psi)] = \
            reproduced_results / LaplaceFileName(key, _psi)


rho_fields = {}
E_sym_dict = {E6_P2F6_sym: 'E6P2', E6_P4F6_sym: 'E6P4',
              E8_P2F8_sym: 'E8P2', E8_P4F6_sym: 'E8P4',
              E10_P2F10_sym: 'E10P2', E10_P4F6_sym: 'E10P4',
              E12_P2F12_sym: 'E12P2', E12_P4F6_sym: 'E12P4'}

for _psi in psis:

    for _stencil in [E6_P2F6_sym, E6_P4F6_sym,
                     E8_P2F8_sym, E8_P4F6_sym,
                     E10_P2F10_sym, E10_P4F6_sym,
                     E12_P2F12_sym, E12_P4F6_sym]:
        _data_swap = ManageData(dump_file = flat_files[StencilPsiKey(_stencil, _psi)])
        _is_file_there = _data_swap.Read()
        if not _is_file_there:
            raise Exception("Could not find file!", 
                            flat_files[StencilPsiKey(_stencil, _psi)])
        
        for G in Gs[_psi]:
            rho_fields['G' + str(G) + E_sym_dict[_stencil]] = _data_swap.PullData(G)

gibbs_rad = defaultdict( # G
    lambda: defaultdict( #  'B2F6'
        lambda: defaultdict(  # 'P4Iso=' + YN
            lambda: defaultdict(dict) # 'droplet'
        )
    )
)

delta_p = defaultdict( # G
    lambda: defaultdict( #  'B2F6'
        lambda: defaultdict(  # 'P4Iso=' + YN
            lambda: defaultdict(dict) # 'droplet'
        )
    )
)

E_sym_YN = {E6_P2F6_sym: 'No', E6_P4F6_sym: 'Yes',
            E8_P2F8_sym: 'No', E8_P4F6_sym: 'Yes',
            E10_P2F10_sym: 'No', E10_P4F6_sym: 'Yes',
            E12_P2F12_sym: 'No', E12_P4F6_sym: 'Yes'}

if False:
    for _psi in psis:
        for _stencil in [E6_P2F6_sym, E6_P4F6_sym,
                         E8_P2F8_sym, E8_P4F6_sym,
                         E10_P2F10_sym, E10_P4F6_sym,
                         E12_P2F12_sym, E12_P4F6_sym]:
            _data_swap = ManageData(dump_file = laplace_files[StencilPsiKey(_stencil, _psi)])
            _is_file_there = _data_swap.Read()
            if not _is_file_there:
                raise Exception("Could not find file!", 
                                laplace_files[StencilPsiKey(_stencil, _psi)])        

            for G in Gs[_psi]:
                _swap_gibbs_rad, _swap_delta_p = [], []
                for L in Ls:
                    _data_key = str(G) + "_" + str(L)
                    _swap_gibbs_rad.append(_data_swap.PullData(_data_key)['R_Gibbs'])
                    _swap_delta_p.append(_data_swap.PullData(_data_key)['delta_p'])

                gibbs_rad['G=' + str(G)][stencil_string[_stencil]]['P4Iso=' + E_sym_YN[_stencil]]['droplet'] = \
                    np.array(_swap_gibbs_rad)
                delta_p['G=' + str(G)][stencil_string[_stencil]]['P4Iso=' + E_sym_YN[_stencil]]['droplet'] = \
                    np.array(_swap_delta_p)
            
sigma_f = defaultdict( # G
    lambda: defaultdict( #  'B2F6'
        lambda: defaultdict(  # 'P4Iso=' + YN
            lambda: defaultdict(dict) # 'droplet'
        )
    )
)

for _psi in psis:
    for _stencil in [E6_P2F6_sym, E8_P2F8_sym, E10_P2F10_sym, E12_P2F12_sym]:
        for G in Gs[_psi]:
            _sc_eq_cache = ShanChanEquilibriumCache(stencil = stencil_dict[_stencil], 
                                                    psi_f = _psi, G = G, 
                                                    c2 = XIStencils['D2Q9']['c2'])
            
            sigma_f['G=' + str(G)][stencil_string[_stencil]]['P4Iso=' + E_sym_YN[_stencil]]['droplet'] = \
                _sc_eq_cache.GetFromCache()['sigma_f']
            print("Surface tension (G = ", str(G), ": ", 
                  _sc_eq_cache.GetFromCache()['sigma_f'], "), psi = ", _psi)

            
##################################################
############# END OF DATA PREPARATION ############
##################################################


# https://stackoverflow.com/questions/14737681/fill-the-right-column-of-a-matplotlib-legend-first


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

import matplotlib.ticker as ticker

##################################################
#################### FIGURE 1 ####################
##################################################

from matplotlib import rc, rcParams
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
## To align latex text and symbols!!!
## https://stackoverflow.com/questions/40424249/vertical-alignment-of-matplotlib-legend-labels-with-latex-math
rcParams['text.latex.preview'] = True
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

x_lim = 0.042
rm1_axis = np.linspace(0, x_lim, 2**7)

_panel_label_pos = (0.02, 0.89)
_panel_label_pos = (0.89, 1.05)

a = 0.9
 
b_height = 0.8
legend_size = 10

dashed = {}
dashed[-2.6] = '-'
dashed[-3.1] = '--'
dashed[-3.6] = '-.'

dashed[-1.4] = '-'
dashed[-1.6] = '--'
dashed[-1.75] = '-.'

f_s = 14
#################### SIZES ####################
fig = plt.figure(figsize=(14, 10))
###############################################

#################### PANEL (a) ####################
ax1 = plt.subplot2grid((4,2), (0,0), colspan=1, rowspan=1)

mark_s2 = 5
black_lines = []

########

min_y, max_y = 0, 0
for G in [-2.6, -3.1, -3.6]:
    L = rho_fields['G' + str(G) + 'E6P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E6P4'][x_range]
    rho_niso = rho_fields['G' + str(G) + 'E6P2'][x_range]

    if min_y == 0:
        min_y = np.amin(1 - rho_iso/rho_niso)
    else:
        min_y = min(min_y, np.amin(1 - rho_iso/rho_niso))
    if max_y == 0:
        max_y = np.amax(1 - rho_iso/rho_niso)
    else:
        max_y = max(max_y, np.amax(1 - rho_iso/rho_niso))

    line_swap, = ax1.plot(np.array(range(L//2 - 1)),
                          1 - rho_iso/rho_niso, dashed[G], color = 'black',
                          markersize = mark_s2, label = '$Gc_s^2=' + str(G) + '$')
    black_lines.append(line_swap)

ax1.set_xlabel('$x$', fontsize=f_s)
ax1.set_ylabel('$1 - n_{P4}/n_{P2}$', fontsize=f_s)

_inset_loc = 'lower right'
_label_pos = 0.1
if abs(min_y) < abs(max_y):
    _inset_loc = 'upper right'
    _label_pos = 0.875

ax1.text(0.025, _label_pos, r'$\boldsymbol{E}^{(6)}_{P4,F6}\quad \text{vs}\quad \boldsymbol{E}^{(6)}_{P2,F6}$',
         transform = ax1.transAxes, fontsize=11)
ax1.text(0.95, 1.075, '$(a)$', transform = ax1.transAxes, fontsize=f_s)

### lines legends
black_lines.insert(0, plt.Line2D(rm1_axis, rm1_axis, linestyle='none', label = '$\psi = \exp(-1/n)$'))
lgnd_lines = plt.legend(handles = black_lines, handlelength = 2.,
                        borderaxespad = 0., columnspacing = 1,
                        labelspacing=0.2, ncol = 4,
                        bbox_to_anchor=(1.02, 1.5),
                        frameon=True)

lgnd_lines.get_texts()[0].set_x(-20)
lgnd_lines.get_texts()[0].set_size("large")
lgnd_lines.get_texts()[1].set_size("large")
lgnd_lines.get_texts()[2].set_size("large")
lgnd_lines.get_texts()[3].set_size("large")

ax1.add_artist(lgnd_lines)

### INSET
ax1i = inset_axes(ax1, width='35%', height='70%', loc = _inset_loc)

for G in [-2.6, -3.1, -3.6]:
    L = rho_fields['G' + str(G) + 'E6P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E6P4'][x_range]
    ax1i.plot(np.array(range(L//2 - 1)), rho_iso, dashed[G], color = 'black')

ax1i.set_ylim([-0.5, 3.8])
ax1i.set_xlim([-7, 47])

ax1i.set_ylabel('$n$', fontsize=10)

## Setting tivks size
for tick in ax1i.yaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax1i.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)

## Setting ticks position
ax1i.tick_params(axis="y",direction="in", pad = -9)
ax1i.tick_params(axis="x",direction="in", pad = -9)

# Set ticks distance
ax1i.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax1i.yaxis.set_major_locator(ticker.MultipleLocator(1))
# eliminate first x tick

#################### PANEL (b) ####################
ax2 = plt.subplot2grid((4,2), (1,0), colspan=1, rowspan=1)

mark_s2 = 5

########

min_y, max_y = 0, 0
for G in [-2.6, -3.1, -3.6]:
    L = rho_fields['G' + str(G) + 'E8P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E8P4'][x_range]
    rho_niso = rho_fields['G' + str(G) + 'E8P2'][x_range]

    if min_y == 0:
        min_y = np.amin(1 - rho_iso/rho_niso)
    else:
        min_y = min(min_y, np.amin(1 - rho_iso/rho_niso))
    if max_y == 0:
        max_y = np.amax(1 - rho_iso/rho_niso)
    else:
        max_y = max(max_y, np.amax(1 - rho_iso/rho_niso))
    
    ax2.plot(np.array(range(L//2 - 1)), 1 - rho_iso/rho_niso, dashed[G], color = 'black', markersize = mark_s2)

ax2.set_xlabel('$x$', fontsize=f_s)
ax2.set_ylabel('$1 - n_{P4}/n_{P2}$', fontsize=f_s)

_inset_loc = 'lower right'
_label_pos = 0.1
if abs(min_y) < abs(max_y):
    _inset_loc = 'upper right'
    _label_pos = 0.875

ax2.text(0.025, _label_pos, r'$\boldsymbol{E}^{(8)}_{P4,F6}\quad \text{vs}\quad \boldsymbol{E}^{(8)}_{P2,F8}$',
         transform = ax2.transAxes, fontsize=11)
ax2.text(0.95, 1.075, '$(b)$', transform = ax2.transAxes, fontsize=f_s)

### INSET
ax2i = inset_axes(ax2, width='35%', height='70%', loc = _inset_loc)

for G in [-2.6, -3.1, -3.6]:
    L = rho_fields['G' + str(G) + 'E8P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E8P4'][x_range]
    ax2i.plot(np.array(range(L//2 - 1)), rho_iso, dashed[G], color = 'black')

ax2i.set_ylim([-0.5, 3.8])
ax2i.set_xlim([-7, 47])

ax2i.set_ylabel('$n$', fontsize=10)

## Setting tivks size
for tick in ax2i.yaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax2i.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)

## Setting ticks position
ax2i.tick_params(axis="y",direction="in", pad = -9)
ax2i.tick_params(axis="x",direction="in", pad = -9)

# Set ticks distance
ax2i.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax2i.yaxis.set_major_locator(ticker.MultipleLocator(1))
# eliminate first x tick

#################### PANEL (c) ####################
ax3 = plt.subplot2grid((4,2), (2,0), colspan=1, rowspan=1)

mark_s2 = 5

########

min_y, max_y = 0, 0
for G in [-2.6, -3.1, -3.6]:
    L = rho_fields['G' + str(G) + 'E10P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E10P4'][x_range]
    rho_niso = rho_fields['G' + str(G) + 'E10P2'][x_range]

    if min_y == 0:
        min_y = np.amin(1 - rho_iso/rho_niso)
    else:
        min_y = min(min_y, np.amin(1 - rho_iso/rho_niso))
    if max_y == 0:
        max_y = np.amax(1 - rho_iso/rho_niso)
    else:
        max_y = max(max_y, np.amax(1 - rho_iso/rho_niso))
    
    ax3.plot(np.array(range(L//2 - 1)), 1 - rho_iso/rho_niso, dashed[G], color = 'black', markersize = mark_s2)

ax3.set_xlabel('$x$', fontsize=f_s)
ax3.set_ylabel('$1 - n_{P4}/n_{P2}$', fontsize=f_s)

_inset_loc = 'lower right'
_label_pos = 0.1
if abs(min_y) < abs(max_y):
    _inset_loc = 'upper right'
    _label_pos = 0.875

ax3.text(0.025, _label_pos, r'$\boldsymbol{E}^{(10)}_{P4,F6}\quad \text{vs}\quad \boldsymbol{E}^{(10)}_{P2,F10}$',
         transform = ax3.transAxes, fontsize=11)
ax3.text(0.95, 1.075, '$(c)$', transform = ax3.transAxes, fontsize=f_s)

### INSET
ax3i = inset_axes(ax3, width='35%', height='70%', loc = _inset_loc)

for G in [-2.6, -3.1, -3.6]:
    L = rho_fields['G' + str(G) + 'E10P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E10P4'][x_range]
    ax3i.plot(np.array(range(L//2 - 1)), rho_iso, dashed[G], color = 'black')

ax3i.set_ylim([-0.5, 3.8])
ax3i.set_xlim([-7, 47])

ax3i.set_ylabel('$n$', fontsize=10)

## Setting tivks size
for tick in ax3i.yaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax3i.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)

## Setting ticks position
ax3i.tick_params(axis="y",direction="in", pad = -9)
ax3i.tick_params(axis="x",direction="in", pad = -9)

# Set ticks distance
ax3i.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax3i.yaxis.set_major_locator(ticker.MultipleLocator(1))
# eliminate first x tick

#################### PANEL (d) ####################
ax4 = plt.subplot2grid((4,2), (3,0), colspan=1, rowspan=1)

mark_s2 = 5

########

min_y, max_y = 0, 0
for G in [-2.6, -3.1, -3.6]:
    L = rho_fields['G' + str(G) + 'E12P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E12P4'][x_range]
    rho_niso = rho_fields['G' + str(G) + 'E12P2'][x_range]

    if min_y == 0:
        min_y = np.amin(1 - rho_iso/rho_niso)
    else:
        min_y = min(min_y, np.amin(1 - rho_iso/rho_niso))
    if max_y == 0:
        max_y = np.amax(1 - rho_iso/rho_niso)
    else:
        max_y = max(max_y, np.amax(1 - rho_iso/rho_niso))
    
    ax4.plot(np.array(range(L//2 - 1)), 1 - rho_iso/rho_niso, dashed[G], color = 'black', markersize = mark_s2)

ax4.set_xlabel('$x$', fontsize=f_s)
ax4.set_ylabel('$1 - n_{P4}/n_{P2}$', fontsize=f_s)

_inset_loc = 'lower right'
_label_pos = 0.1
if abs(min_y) < abs(max_y):
    _inset_loc = 'upper right'
    _label_pos = 0.875

ax4.text(0.025, _label_pos, r'$\boldsymbol{E}^{(12)}_{P4,F6}\quad \text{vs}\quad \boldsymbol{E}^{(12)}_{P2,F12}$',
         transform = ax4.transAxes, fontsize=11)
ax4.text(0.95, 1.075, '$(d)$', transform = ax4.transAxes, fontsize=f_s)

### INSET
ax4i = inset_axes(ax4, width='35%', height='70%', loc = _inset_loc)

for G in [-2.6, -3.1, -3.6]:
    L = rho_fields['G' + str(G) + 'E12P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E12P4'][x_range]
    ax4i.plot(np.array(range(L//2 - 1)), rho_iso, dashed[G], color = 'black')

ax4i.set_ylim([-0.5, 3.8])
ax4i.set_xlim([-7, 47])

ax4i.set_ylabel('$n$', fontsize=10)

## Setting tivks size
for tick in ax4i.yaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax4i.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)

## Setting ticks position
ax4i.tick_params(axis="y",direction="in", pad = -9)
ax4i.tick_params(axis="x",direction="in", pad = -9)

# Set ticks distance
ax4i.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax4i.yaxis.set_major_locator(ticker.MultipleLocator(1))
# eliminate first x tick


################################################
################ SECOND COLUMN #################

#################### PANEL (e) ####################
ax5 = plt.subplot2grid((4,2), (0,1), colspan=1, rowspan=1)

black_lines = []

min_y, max_y = 0, 0
for G in [-1.4, -1.6, -1.75]:
    L = rho_fields['G' + str(G) + 'E6P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E6P4'][x_range]
    rho_niso = rho_fields['G' + str(G) + 'E6P2'][x_range]

    if min_y == 0:
        min_y = np.amin(1 - rho_iso/rho_niso)
    else:
        min_y = min(min_y, np.amin(1 - rho_iso/rho_niso))
    if max_y == 0:
        max_y = np.amax(1 - rho_iso/rho_niso)
    else:
        max_y = max(max_y, np.amax(1 - rho_iso/rho_niso))
        
    line_swap, = ax5.plot(np.array(range(L//2 - 1)),
                          1 - rho_iso/rho_niso, dashed[G],
                          color = 'black', markersize = mark_s2,
                          label = '$Gc_s^2=' + str(G) + '$')
    black_lines.append(line_swap)

### lines legends
black_lines.insert(0, plt.Line2D(rm1_axis, rm1_axis, linestyle='none', label = '$\psi = 1 - \exp(-n)$'))
lgnd_lines = plt.legend(handles = black_lines, handlelength = 2.,
                        labelspacing=0.2, ncol = 4,
                        bbox_to_anchor=(1.035, 1.5),
                        frameon=True)

lgnd_lines.get_texts()[0].set_x(-20)
lgnd_lines.get_texts()[0].set_size("large")
lgnd_lines.get_texts()[1].set_size("large")
lgnd_lines.get_texts()[2].set_size("large")
lgnd_lines.get_texts()[3].set_size("large")

ax5.add_artist(lgnd_lines)


ax5.set_xlabel('$x$', fontsize=f_s)
ax5.set_ylabel('$1 - n_{P4}/n_{P2}$', fontsize=f_s)

_inset_loc = 'lower right'
_label_pos = 0.1
if abs(min_y) < abs(max_y):
    _inset_loc = 'upper right'
    _label_pos = 0.875

ax5.text(0.025, _label_pos, r'$\boldsymbol{E}^{(6)}_{P4,F6}\quad \text{vs}\quad \boldsymbol{E}^{(6)}_{P2,F6}$',
         transform = ax5.transAxes, fontsize=11)
ax5.text(0.95, 1.075, '$(e)$', transform = ax5.transAxes, fontsize=f_s)
#ax5.text(39, 3.5e-12, '$(d)$', fontsize=f_s)
    
### INSET
ax5i = inset_axes(ax5, width='35%', height='70%', loc = _inset_loc)

for G in [-1.4, -1.6, -1.75]:
    L = rho_fields['G' + str(G) + 'E6P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E6P4'][x_range]
    ax5i.plot(np.array(range(L//2 - 1)), rho_iso, dashed[G], color = 'black')

ax5i.set_ylim([-0.5, 2.5])
ax5i.set_xlim([-7, 47])

ax5i.set_ylabel('$n$', fontsize=10)

## Setting tivks size
for tick in ax5i.yaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax5i.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)

## Setting ticks position
ax5i.tick_params(axis="y",direction="in", pad = -9)
ax5i.tick_params(axis="x",direction="in", pad = -9)

# Set ticks distance
ax5i.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax5i.yaxis.set_major_locator(ticker.MultipleLocator(1))
# eliminate first x tick

#################### PANEL (f) ####################
ax6 = plt.subplot2grid((4,2), (1,1), colspan=1, rowspan=1)

min_y, max_y = 0, 0
for G in [-1.4, -1.6, -1.75]:
    L = rho_fields['G' + str(G) + 'E8P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E8P4'][x_range]
    rho_niso = rho_fields['G' + str(G) + 'E8P2'][x_range]

    if min_y == 0:
        min_y = np.amin(1 - rho_iso/rho_niso)
    else:
        min_y = min(min_y, np.amin(1 - rho_iso/rho_niso))
    if max_y == 0:
        max_y = np.amax(1 - rho_iso/rho_niso)
    else:
        max_y = max(max_y, np.amax(1 - rho_iso/rho_niso))
    
    ax6.plot(np.array(range(L//2 - 1)), 1 - rho_iso/rho_niso, dashed[G], color = 'black', markersize = mark_s2)

ax6.set_xlabel('$x$', fontsize=f_s)
ax6.set_ylabel('$1 - n_{P4}/n_{P2}$', fontsize=f_s)

_inset_loc = 'lower right'
_label_pos = 0.1
if abs(min_y) < abs(max_y):
    _inset_loc = 'upper right'
    _label_pos = 0.875

ax6.text(0.025, _label_pos, r'$\boldsymbol{E}^{(8)}_{P4,F6}\quad \text{vs}\quad \boldsymbol{E}^{(8)}_{P2,F8}$',
         transform = ax6.transAxes, fontsize=11)
ax6.text(0.95, 1.075, '$(f)$', transform = ax6.transAxes, fontsize=f_s)

### INSET
ax6i = inset_axes(ax6, width='35%', height='70%', loc = _inset_loc)

for G in [-1.4, -1.6, -1.75]:
    L = rho_fields['G' + str(G) + 'E8P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E8P4'][x_range]
    ax6i.plot(np.array(range(L//2 - 1)), rho_iso, dashed[G], color = 'black')

ax6i.set_ylim([-0.5, 2.5])
ax6i.set_xlim([-7, 47])

ax6i.set_ylabel('$n$', fontsize=10)

## Setting tivks size
for tick in ax6i.yaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax6i.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)

## Setting ticks position
ax6i.tick_params(axis="y",direction="in", pad = -9)
ax6i.tick_params(axis="x",direction="in", pad = -9)

# Set ticks distance
ax6i.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax6i.yaxis.set_major_locator(ticker.MultipleLocator(1))
# eliminate first x tick

#################### PANEL (g) ####################
ax7 = plt.subplot2grid((4,2), (2,1), colspan=2, rowspan=1)

min_y, max_y = 0, 0
for G in [-1.4, -1.6, -1.75]:
    L = rho_fields['G' + str(G) + 'E10P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E10P4'][x_range]
    rho_niso = rho_fields['G' + str(G) + 'E10P2'][x_range]

    if min_y == 0:
        min_y = np.amin(1 - rho_iso/rho_niso)
    else:
        min_y = min(min_y, np.amin(1 - rho_iso/rho_niso))
    if max_y == 0:
        max_y = np.amax(1 - rho_iso/rho_niso)
    else:
        max_y = max(max_y, np.amax(1 - rho_iso/rho_niso))
    
    ax7.plot(np.array(range(L//2 - 1)), 1 - rho_iso/rho_niso, dashed[G], color = 'black', markersize = mark_s2)

ax7.set_xlabel('$x$', fontsize=f_s)
ax7.set_ylabel('$1 - n_{P4}/n_{P2}$', fontsize=f_s)

_inset_loc = 'lower right'
_label_pos = 0.1
if abs(min_y) < abs(max_y):
    _inset_loc = 'upper right'
    _label_pos = 0.875

ax7.text(0.025, _label_pos, r'$\boldsymbol{E}^{(10)}_{P4,F6}\quad \text{vs}\quad \boldsymbol{E}^{(10)}_{P2,F10}$', transform = ax7.transAxes, fontsize=11)
ax7.text(0.95, 1.075, '$(g)$', transform = ax7.transAxes, fontsize=f_s)

### INSET
ax7i = inset_axes(ax7, width='35%', height='70%', loc = _inset_loc)

for G in [-1.4, -1.6, -1.75]:
    L = rho_fields['G' + str(G) + 'E10P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E10P4'][x_range]
    ax7i.plot(np.array(range(L//2 - 1)), rho_iso, dashed[G], color = 'black')

ax7i.set_ylim([-0.5, 2.5])
ax7i.set_xlim([-7, 47])

ax7i.set_ylabel('$n$', fontsize=10)

## Setting tivks size
for tick in ax7i.yaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax7i.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)

## Setting ticks position
ax7i.tick_params(axis="y",direction="in", pad = -9)
ax7i.tick_params(axis="x",direction="in", pad = -9)

# Set ticks distance
ax7i.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax7i.yaxis.set_major_locator(ticker.MultipleLocator(1))
# eliminate first x tick


#################### PANEL (h) ####################
ax8 = plt.subplot2grid((4,2), (3,1), colspan=1, rowspan=1)

min_y, max_y = 0, 0
for G in [-1.4, -1.6, -1.75]:
    L = rho_fields['G' + str(G) + 'E12P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E12P4'][x_range]
    rho_niso = rho_fields['G' + str(G) + 'E12P2'][x_range]

    if min_y == 0:
        min_y = np.amin(1 - rho_iso/rho_niso)
    else:
        min_y = min(min_y, np.amin(1 - rho_iso/rho_niso))
    if max_y == 0:
        max_y = np.amax(1 - rho_iso/rho_niso)
    else:
        max_y = max(max_y, np.amax(1 - rho_iso/rho_niso))
    
    ax8.plot(np.array(range(L//2 - 1)), 1 - rho_iso/rho_niso, dashed[G], color = 'black', markersize = mark_s2)

ax8.set_xlabel('$x$', fontsize=f_s)
ax8.set_ylabel('$1 - n_{P4}/n_{P2}$', fontsize=f_s)

_inset_loc = 'lower right'
_label_pos = 0.875
if abs(min_y) < abs(max_y):
    _inset_loc = 'upper right'
    _label_pos = 0.1

ax8.text(0.025, _label_pos, r'$\boldsymbol{E}^{(12)}_{P4,F6}\quad \text{vs}\quad \boldsymbol{E}^{(12)}_{P2,F12}$', transform = ax8.transAxes, fontsize=11)
ax8.text(0.95, 1.075, '$(h)$', transform = ax8.transAxes, fontsize=f_s)

### INSET
ax8i = inset_axes(ax8, width='35%', height='70%', loc = 'lower right')

for G in [-1.4, -1.6, -1.75]:
    L = rho_fields['G' + str(G) + 'E12P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E12P4'][x_range]
    ax8i.plot(np.array(range(L//2 - 1)), rho_iso, dashed[G], color = 'black')

ax8i.set_ylim([-0.5, 2.5])
ax8i.set_xlim([-7, 47])

ax8i.set_ylabel('$n$', fontsize=10)

## Setting tivks size
for tick in ax8i.yaxis.get_major_ticks():
    tick.label.set_fontsize(5)
for tick in ax8i.xaxis.get_major_ticks():
    tick.label.set_fontsize(5)

## Setting ticks position
ax8i.tick_params(axis="y",direction="in", pad = -9)
ax8i.tick_params(axis="x",direction="in", pad = -9)

# Set ticks distance
ax8i.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax8i.yaxis.set_major_locator(ticker.MultipleLocator(1))
# eliminate first x tick

#################### SAVING ####################
fig.tight_layout()

from pathlib import Path
reproduced_figures = Path("reproduced-figures")
if not reproduced_figures.is_dir():
    reproduced_figures.mkdir()

plt.savefig(reproduced_figures / 'figure_4.png',
            bbox_inches = 'tight', dpi = _dpi)
plt.close()
