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

E6_P2F6_sym = sp_symbols("\\bf{E}^{(6)}_{P2\,F6}")
E6_P4F6_sym = sp_symbols("\\bf{E}^{(6)}_{P4\,F6}")
E8_P2F8_sym = sp_symbols("\\bf{E}^{(8)}_{P2\,F8}")
E8_P4F6_sym = sp_symbols("\\bf{E}^{(8)}_{P4\,F6}")
E10_P2F10_sym = sp_symbols("\\bf{E}^{(10)}_{P2\,F10}")
E10_P4F6_sym = sp_symbols("\\bf{E}^{(10)}_{P4\,F6}")
E12_P2F12_sym = sp_symbols("\\bf{E}^{(12)}_{P2\,F12}")
E12_P4F6_sym = sp_symbols("\\bf{E}^{(12)}_{P4\,F6}")

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

laplace_files = {}
for key in stencil_string:
    for _psi in psis:
        laplace_files[StencilPsiKey(key, _psi)] = \
            reproduced_results / LaplaceFileName(key, _psi)


rho_fields = {}
E_sym_dict = {E6_P2F6_sym: 'E6P2', E6_P4F6_sym: 'E6P4'}

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

import matplotlib.ticker as ticker

##################################################
#################### FIGURE 1 ####################
##################################################

from matplotlib import rc, rcParams
from idpy.Utils.Plots import SetAxPanelLabelCoords, SetMatplotlibLatexParamas, CreateFiguresPanels

SetMatplotlibLatexParamas([rc], [rcParams])

if False:
    ##rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    rc('font',**{'family':'STIXGeneral'})
    rc('mathtext', **{'fontset': 'stix'})
    rc('text', usetex=True)
    ## To align latex text and symbols!!!
    ## https://stackoverflow.com/questions/40424249/vertical-alignment-of-matplotlib-legend-labels-with-latex-math
    rcParams['text.latex.preview'] = True
    rcParams['text.latex.preamble']=[r"\usepackage{amsmath, sourcesanspro}"]

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
fig = plt.figure(figsize=(5.2, 12))
###############################################

#################### PANEL (a) ####################
mark_s = 9

ax1 = plt.subplot2grid((5,2), (1,0), colspan=1, rowspan=1)

black_lines = []
black_labels = []
G = -2.6
red_p, = ax1.plot(1./gibbs_rad['G=' + str(G)]['E6_P4F6']['P4Iso=' + 'Yes']['droplet'], 
                  delta_p['G=' + str(G)]['E6_P4F6']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red', 
                  markersize = mark_s, label = r'$\bf{E}^{(6)}_{P4,F6}$')

blue_p, = ax1.plot(1./gibbs_rad['G=' + str(G)]['E6_P2F6']['P4Iso=' + 'No']['droplet'], 
                   delta_p['G=' + str(G)]['E6_P2F6']['P4Iso=' + 'No']['droplet'], '+', color = 'blue', 
                   markersize = mark_s, label = r'$\bf{E}^{(6)}_{P2,F6}$')

line_swap, = ax1.plot(rm1_axis, 
                      rm1_axis * sigma_f['G=' + str(G)]['E6_P2F6']['P4Iso=' + 'No']['droplet'], 
                      label = '$Gc_s^2=' + str(G) + '$', color = 'black')

black_lines.append(line_swap)

min_y, max_y = 0, 0
for G in [-3.1, -3.6]:
    if min_y == 0:
        min_y = np.amin(delta_p['G=' + str(G)]['E6_P4F6']['P4Iso=' + 'Yes']['droplet'])
    else:
        min_y = min(min_y, np.amin(delta_p['G=' + str(G)]['E6_P4F6']['P4Iso=' + 'Yes']['droplet']))
    if max_y == 0:
        max_y = np.amax(delta_p['G=' + str(G)]['E6_P4F6']['P4Iso=' + 'Yes']['droplet'])
    else:
        max_y = max(max_y, np.amax(delta_p['G=' + str(G)]['E6_P4F6']['P4Iso=' + 'Yes']['droplet']))

    ax1.plot(1./gibbs_rad['G=' + str(G)]['E6_P4F6']['P4Iso=' + 'Yes']['droplet'], 
             delta_p['G=' + str(G)]['E6_P4F6']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red', 
             markersize = mark_s)

    ax1.plot(1./gibbs_rad['G=' + str(G)]['E6_P2F6']['P4Iso=' + 'No']['droplet'], 
             delta_p['G=' + str(G)]['E6_P2F6']['P4Iso=' + 'No']['droplet'], '+', color = 'blue', 
             markersize = mark_s)

    line_swap, = ax1.plot(rm1_axis, 
                          rm1_axis * sigma_f['G=' + str(G)]['E6_P2F6']['P4Iso=' + 'No']['droplet'], 
                          dashed[G], color = 'black', label = '$Gc_s^2=' + str(G) + '$')

    black_lines.append(line_swap)

    
#(lines, labels) = plt.gca().get_legend_handles_labels()
#lines.insert(2, plt.Line2D(rm1_axis, rm1_axis, linestyle='none'))
#labels.insert(2,'')

# _{\\mbox{\\tiny{Gibbs}}}
ax1.ticklabel_format(axis='y', style = 'sci', scilimits=(0,0))
ax1.set_xlabel('$R^{-1}$', fontsize=f_s)
ax1.set_ylabel('$\\Delta p$', fontsize=f_s)

ax1.set_xlim([0,x_lim])
### points legend
legend_size = 10
if False:
    lgnd_points = plt.legend(handles = [red_p, blue_p], ncol = 2,
                             handletextpad = 0., columnspacing = 0.75,
                             bbox_to_anchor=(1.55, 1.27), frameon=False)

lgnd_points = plt.legend(handles = [red_p, blue_p], loc = 'upper left', frameon=False)

lgnd_points.get_texts()[0].set_color("red")
lgnd_points.get_texts()[1].set_color("blue")

lgnd_points.get_texts()[0].set_size("large")
lgnd_points.get_texts()[1].set_size("large")

lgnd_points.legend_handles[0]._sizes = [6]
lgnd_points.legend_handles[1]._sizes = [6]

### lines legends
black_lines.insert(0, plt.Line2D(rm1_axis, rm1_axis, linestyle='none', label = '$\psi = \exp(-1/n)$'))
lgnd_lines = plt.legend(handles = black_lines, handlelength = 2,
                        labelspacing=0.2,
                        bbox_to_anchor=(0.925, 1.75),
                        frameon=True)

lgnd_lines.get_texts()[0].set_x(-20)
lgnd_lines.get_texts()[0].set_size("large")
lgnd_lines.get_texts()[1].set_size("large")
lgnd_lines.get_texts()[2].set_size("large")
lgnd_lines.get_texts()[3].set_size("large")

#ml = [method_name for method_name in dir(lgnd_lines.get_texts()[0]) if callable(getattr(lgnd_lines.get_texts()[0], method_name))]
#print(ml)

### adding legents to the plot
ax1.add_artist(lgnd_points)
ax1.add_artist(lgnd_lines)

#lgnd1 = ax1.legend(lines,labels,numpoints=1, loc=4,ncol=2)
#lgnd1 = ax1.legend(lines, labels, loc='upper center', ncol=2, fancybox=True, 
#                   bbox_to_anchor=(0.4, 1.9), frameon=False,  handleheight=1.,
#                   prop={'size': legend_size}, borderpad=0.8, labelspacing=0.5)

# Shrink current axis by 20%
b_height = 1
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width, box.height * b_height])

ax1.text(_panel_label_pos[0], _panel_label_pos[1], '$(a)$',
         transform = ax1.transAxes, fontsize=f_s)

#################### PANEL (b) ####################
mark_s = 9

ax2 = plt.subplot2grid((5,2), (2,0), colspan=1, rowspan=1)

black_lines = []
black_labels = []
G = -2.6
red_p, = ax2.plot(1./gibbs_rad['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet'], 
                  delta_p['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red', 
                  markersize = mark_s, label = r'$\bf{E}^{(8)}_{P4,F6}$')

blue_p, = ax2.plot(1./gibbs_rad['G=' + str(G)]['E8_P2F8']['P4Iso=' + 'No']['droplet'], 
                   delta_p['G=' + str(G)]['E8_P2F8']['P4Iso=' + 'No']['droplet'], '+', color = 'blue', 
                   markersize = mark_s, label = r'$\bf{E}^{(8)}_{P2,F8}$')

line_swap, = ax2.plot(rm1_axis, 
                      rm1_axis * sigma_f['G=' + str(G)]['E8_P2F8']['P4Iso=' + 'No']['droplet'], 
                      label = '$Gc_s^2=' + str(G) + '$', color = 'black')

black_lines.append(line_swap)

min_y, max_y = 0, 0
for G in [-3.1, -3.6]:
    if min_y == 0:
        min_y = np.amin(delta_p['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet'])
    else:
        min_y = min(min_y, np.amin(delta_p['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet']))
    if max_y == 0:
        max_y = np.amax(delta_p['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet'])
    else:
        max_y = max(max_y, np.amax(delta_p['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet']))

    ax2.plot(1./gibbs_rad['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet'], 
             delta_p['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red', 
             markersize = mark_s)

    ax2.plot(1./gibbs_rad['G=' + str(G)]['E8_P2F8']['P4Iso=' + 'No']['droplet'], 
             delta_p['G=' + str(G)]['E8_P2F8']['P4Iso=' + 'No']['droplet'], '+', color = 'blue', 
             markersize = mark_s)

    line_swap, = ax2.plot(rm1_axis, 
                          rm1_axis * sigma_f['G=' + str(G)]['E8_P2F8']['P4Iso=' + 'No']['droplet'], 
                          dashed[G], color = 'black', label = '$Gc_s^2=' + str(G) + '$')

    black_lines.append(line_swap)

    
#(lines, labels) = plt.gca().get_legend_handles_labels()
#lines.insert(2, plt.Line2D(rm1_axis, rm1_axis, linestyle='none'))
#labels.insert(2,'')

# _{\\mbox{\\tiny{Gibbs}}}
ax2.ticklabel_format(axis='y', style = 'sci', scilimits=(0,0))
ax2.set_xlabel('$R^{-1}$', fontsize=f_s)
ax2.set_ylabel('$\\Delta p$', fontsize=f_s)

ax2.set_xlim([0,x_lim])
### points legend
legend_size = 10

lgnd_points = plt.legend(handles = [red_p, blue_p], loc = 'upper left', frameon=False)

lgnd_points.get_texts()[0].set_color("red")
lgnd_points.get_texts()[1].set_color("blue")

lgnd_points.get_texts()[0].set_size("large")
lgnd_points.get_texts()[1].set_size("large")

lgnd_points.legend_handles[0]._sizes = [6]
lgnd_points.legend_handles[1]._sizes = [6]

#ml = [method_name for method_name in dir(lgnd_lines.get_texts()[0]) if callable(getattr(lgnd_lines.get_texts()[0], method_name))]
#print(ml)

### adding legents to the plot
ax2.add_artist(lgnd_points)

#lgnd1 = ax2.legend(lines,labels,numpoints=1, loc=4,ncol=2)
#lgnd1 = ax2.legend(lines, labels, loc='upper center', ncol=2, fancybox=True, 
#                   bbox_to_anchor=(0.4, 1.9), frameon=False,  handleheight=1.,
#                   prop={'size': legend_size}, borderpad=0.8, labelspacing=0.5)

# Shrink current axis by 20%
b_height = 1
box = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width, box.height * b_height])

ax2.text(_panel_label_pos[0], _panel_label_pos[1], '$(b)$',
         transform = ax2.transAxes, fontsize=f_s)


#################### PANEL (c) ####################
mark_s = 9

ax3 = plt.subplot2grid((5,2), (3,0), colspan=1, rowspan=1)

black_lines = []
black_labels = []
G = -2.6
red_p, = ax3.plot(1./gibbs_rad['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet'], 
                  delta_p['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red', 
                  markersize = mark_s, label = r'$\bf{E}^{(10)}_{P4,F6}$')

blue_p, = ax3.plot(1./gibbs_rad['G=' + str(G)]['E8_P2F8']['P4Iso=' + 'No']['droplet'], 
                   delta_p['G=' + str(G)]['E8_P2F8']['P4Iso=' + 'No']['droplet'], '+', color = 'blue', 
                   markersize = mark_s, label = r'$\bf{E}^{(10)}_{P2,F10}$')

line_swap, = ax3.plot(rm1_axis, 
                      rm1_axis * sigma_f['G=' + str(G)]['E8_P2F8']['P4Iso=' + 'No']['droplet'], 
                      label = '$Gc_s^2=' + str(G) + '$', color = 'black')

black_lines.append(line_swap)

min_y, max_y = 0, 0
for G in [-3.1, -3.6]:
    if min_y == 0:
        min_y = np.amin(delta_p['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet'])
    else:
        min_y = min(min_y, np.amin(delta_p['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet']))
    if max_y == 0:
        max_y = np.amax(delta_p['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet'])
    else:
        max_y = max(max_y, np.amax(delta_p['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet']))

    ax3.plot(1./gibbs_rad['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet'], 
             delta_p['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red', 
             markersize = mark_s)

    ax3.plot(1./gibbs_rad['G=' + str(G)]['E8_P2F8']['P4Iso=' + 'No']['droplet'], 
             delta_p['G=' + str(G)]['E8_P2F8']['P4Iso=' + 'No']['droplet'], '+', color = 'blue', 
             markersize = mark_s)

    line_swap, = ax3.plot(rm1_axis, 
                          rm1_axis * sigma_f['G=' + str(G)]['E8_P2F8']['P4Iso=' + 'No']['droplet'], 
                          dashed[G], color = 'black', label = '$Gc_s^2=' + str(G) + '$')

    black_lines.append(line_swap)

    
#(lines, labels) = plt.gca().get_legend_handles_labels()
#lines.insert(2, plt.Line2D(rm1_axis, rm1_axis, linestyle='none'))
#labels.insert(2,'')

# _{\\mbox{\\tiny{Gibbs}}}
ax3.ticklabel_format(axis='y', style = 'sci', scilimits=(0,0))
ax3.set_xlabel('$R^{-1}$', fontsize=f_s)
ax3.set_ylabel('$\\Delta p$', fontsize=f_s)

ax3.set_xlim([0,x_lim])
### points legend
legend_size = 10
lgnd_points = plt.legend(handles = [red_p, blue_p], loc = 'upper left', frameon=False)

lgnd_points.get_texts()[0].set_color("red")
lgnd_points.get_texts()[1].set_color("blue")

lgnd_points.get_texts()[0].set_size("large")
lgnd_points.get_texts()[1].set_size("large")

lgnd_points.legend_handles[0]._sizes = [6]
lgnd_points.legend_handles[1]._sizes = [6]

#ml = [method_name for method_name in dir(lgnd_lines.get_texts()[0]) if callable(getattr(lgnd_lines.get_texts()[0], method_name))]
#print(ml)

### adding legents to the plot
ax3.add_artist(lgnd_points)

#lgnd1 = ax3.legend(lines,labels,numpoints=1, loc=4,ncol=2)
#lgnd1 = ax3.legend(lines, labels, loc='upper center', ncol=2, fancybox=True, 
#                   bbox_to_anchor=(0.4, 1.9), frameon=False,  handleheight=1.,
#                   prop={'size': legend_size}, borderpad=0.8, labelspacing=0.5)

# Shrink current axis by 20%
b_height = 1
box = ax3.get_position()
ax3.set_position([box.x0, box.y0, box.width, box.height * b_height])

ax3.text(_panel_label_pos[0], _panel_label_pos[1], '$(c)$',
         transform = ax3.transAxes, fontsize=f_s)

#################### PANEL (d) ####################
mark_s = 9

ax4 = plt.subplot2grid((5,2), (4,0), colspan=1, rowspan=1)

black_lines = []
black_labels = []
G = -2.6
red_p, = ax4.plot(1./gibbs_rad['G=' + str(G)]['E12_P4F6']['P4Iso=' + 'Yes']['droplet'], 
                  delta_p['G=' + str(G)]['E12_P4F6']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red', 
                  markersize = mark_s, label = r'$\bf{E}^{(12)}_{P4,F6}$')

blue_p, = ax4.plot(1./gibbs_rad['G=' + str(G)]['E12_P2F12']['P4Iso=' + 'No']['droplet'], 
                   delta_p['G=' + str(G)]['E12_P2F12']['P4Iso=' + 'No']['droplet'], '+', color = 'blue', 
                   markersize = mark_s, label = r'$\bf{E}^{(12)}_{P2,F12}$')

line_swap, = ax4.plot(rm1_axis, 
                      rm1_axis * sigma_f['G=' + str(G)]['E12_P2F12']['P4Iso=' + 'No']['droplet'], 
                      label = '$Gc_s^2=' + str(G) + '$', color = 'black')

black_lines.append(line_swap)

min_y, max_y = 0, 0
for G in [-3.1, -3.6]:
    if min_y == 0:
        min_y = np.amin(delta_p['G=' + str(G)]['E12_P4F6']['P4Iso=' + 'Yes']['droplet'])
    else:
        min_y = min(min_y, np.amin(delta_p['G=' + str(G)]['E12_P4F6']['P4Iso=' + 'Yes']['droplet']))
    if max_y == 0:
        max_y = np.amax(delta_p['G=' + str(G)]['E12_P4F6']['P4Iso=' + 'Yes']['droplet'])
    else:
        max_y = max(max_y, np.amax(delta_p['G=' + str(G)]['E12_P4F6']['P4Iso=' + 'Yes']['droplet']))

    ax4.plot(1./gibbs_rad['G=' + str(G)]['E12_P4F6']['P4Iso=' + 'Yes']['droplet'], 
             delta_p['G=' + str(G)]['E12_P4F6']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red', 
             markersize = mark_s)

    ax4.plot(1./gibbs_rad['G=' + str(G)]['E12_P2F12']['P4Iso=' + 'No']['droplet'], 
             delta_p['G=' + str(G)]['E12_P2F12']['P4Iso=' + 'No']['droplet'], '+', color = 'blue', 
             markersize = mark_s)

    line_swap, = ax4.plot(rm1_axis, 
                          rm1_axis * sigma_f['G=' + str(G)]['E12_P2F12']['P4Iso=' + 'No']['droplet'], 
                          dashed[G], color = 'black', label = '$Gc_s^2=' + str(G) + '$')

    black_lines.append(line_swap)

    
#(lines, labels) = plt.gca().get_legend_handles_labels()
#lines.insert(2, plt.Line2D(rm1_axis, rm1_axis, linestyle='none'))
#labels.insert(2,'')

# _{\\mbox{\\tiny{Gibbs}}}
ax4.ticklabel_format(axis='y', style = 'sci', scilimits=(0,0))
ax4.set_xlabel('$R^{-1}$', fontsize=f_s)
ax4.set_ylabel('$\\Delta p$', fontsize=f_s)

ax4.set_xlim([0,x_lim])
### points legend
legend_size = 10
lgnd_points = plt.legend(handles = [red_p, blue_p], loc = 'upper left', frameon=False)

lgnd_points.get_texts()[0].set_color("red")
lgnd_points.get_texts()[1].set_color("blue")

lgnd_points.get_texts()[0].set_size("large")
lgnd_points.get_texts()[1].set_size("large")

lgnd_points.legend_handles[0]._sizes = [6]
lgnd_points.legend_handles[1]._sizes = [6]

#ml = [method_name for method_name in dir(lgnd_lines.get_texts()[0]) if callable(getattr(lgnd_lines.get_texts()[0], method_name))]
#print(ml)

### adding legents to the plot
ax4.add_artist(lgnd_points)

#lgnd1 = ax4.legend(lines,labels,numpoints=1, loc=4,ncol=2)
#lgnd1 = ax4.legend(lines, labels, loc='upper center', ncol=2, fancybox=True, 
#                   bbox_to_anchor=(0.4, 1.9), frameon=False,  handleheight=1.,
#                   prop={'size': legend_size}, borderpad=0.8, labelspacing=0.5)

# Shrink current axis by 20%
b_height = 1
box = ax4.get_position()
ax4.set_position([box.x0, box.y0, box.width, box.height * b_height])

ax4.text(_panel_label_pos[0], _panel_label_pos[1], '$(d)$',
         transform = ax4.transAxes, fontsize=f_s)


###################################################
################## SECOND COLUMN ##################


#################### PANEL (e) ####################
ax5 = plt.subplot2grid((5,2), (1,1), colspan=1, rowspan=1)
black_lines = []

G = -1.4
red_p, = ax5.plot(1./gibbs_rad['G=' + str(G)]['E6_P4F6']['P4Iso=' + 'Yes']['droplet'], 
                  delta_p['G=' + str(G)]['E6_P4F6']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red',
                  markersize = mark_s, label = r'$\bf{E}^{(6)}_{P4,F6}$')

blue_p, = ax5.plot(1./gibbs_rad['G=' + str(G)]['E6_P2F6']['P4Iso=' + 'No']['droplet'], 
                   delta_p['G=' + str(G)]['E6_P2F6']['P4Iso=' + 'No']['droplet'], '+', color = 'blue',
                   markersize = mark_s, label = r'$\bf{E}^{(6)}_{P2,F6}$')

line_swap, = ax5.plot(rm1_axis,
                      rm1_axis * sigma_f['G=' + str(G)]['E6_P2F6']['P4Iso=' + 'No']['droplet'],
                      label = '$Gc_s^2=' + str(G) + '$', color = 'black')
black_lines.append(line_swap)

min_y, max_y = 0, 0
for G in [-1.6, -1.75]:
    if min_y == 0:
        min_y = np.amin(delta_p['G=' + str(G)]['E6_P4F6']['P4Iso=' + 'Yes']['droplet'])
    else:
        min_y = min(min_y, np.amin(delta_p['G=' + str(G)]['E6_P4F6']['P4Iso=' + 'Yes']['droplet']))
    if max_y == 0:
        max_y = np.amax(delta_p['G=' + str(G)]['E6_P4F6']['P4Iso=' + 'Yes']['droplet'])
    else:
        max_y = max(max_y, np.amax(delta_p['G=' + str(G)]['E6_P4F6']['P4Iso=' + 'Yes']['droplet']))

    ax5.plot(1./gibbs_rad['G=' + str(G)]['E6_P4F6']['P4Iso=' + 'Yes']['droplet'], 
             delta_p['G=' + str(G)]['E6_P4F6']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red', 
             markersize = mark_s)

    ax5.plot(1./gibbs_rad['G=' + str(G)]['E6_P2F6']['P4Iso=' + 'No']['droplet'], 
             delta_p['G=' + str(G)]['E6_P2F6']['P4Iso=' + 'No']['droplet'], '+', color = 'blue', 
             markersize = mark_s)

    line_swap, = ax5.plot(rm1_axis,
                          rm1_axis * sigma_f['G=' + str(G)]['E6_P2F6']['P4Iso=' + 'No']['droplet'],
                          dashed[G], color = 'black', label = '$Gc_s^2=' + str(G) + '$')
    black_lines.append(line_swap)

ax5.set_xlabel('$R^{-1}$', fontsize=f_s)
#ax5.set_title('$\psi = 1 - \\exp(-n)$')
ax5.ticklabel_format(axis='y', style = 'sci', scilimits=(0,0))
ax5.set_xlim([0,x_lim])

lgnd_points2 = plt.legend(handles = [red_p, blue_p], loc = 'upper left', frameon=False)

lgnd_points2.get_texts()[0].set_color("red")
lgnd_points2.get_texts()[1].set_color("blue")

lgnd_points2.get_texts()[0].set_size("large")
lgnd_points2.get_texts()[1].set_size("large")

lgnd_points2.legend_handles[0]._sizes = [6]
lgnd_points2.legend_handles[1]._sizes = [6]

### lines legends
black_lines.insert(0, plt.Line2D(rm1_axis, rm1_axis, linestyle='none', label = '$\psi = 1 - \exp(-n)$'))
lgnd_lines2 = plt.legend(handles = black_lines, handlelength = 2.,
                        labelspacing=0.2,
                        bbox_to_anchor=(0.9625, 1.75),
                        frameon=True)

lgnd_lines2.get_texts()[0].set_x(-20)
lgnd_lines2.get_texts()[0].set_size("large")
lgnd_lines2.get_texts()[1].set_size("large")
lgnd_lines2.get_texts()[2].set_size("large")
lgnd_lines2.get_texts()[3].set_size("large")

### adding legents to the plot
ax5.add_artist(lgnd_points2)
ax5.add_artist(lgnd_lines2)

# Shrink current axis by 20%
box = ax5.get_position()
ax5.set_position([box.x0, box.y0, box.width, box.height * b_height])

ax5.text(_panel_label_pos[0], _panel_label_pos[1], '$(e)$',
         transform = ax5.transAxes, fontsize=f_s)

#################### PANEL (f) ####################
ax6 = plt.subplot2grid((5,2), (2,1), colspan=1, rowspan=1)
black_lines = []

G = -1.4
red_p, = ax6.plot(1./gibbs_rad['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet'], 
                  delta_p['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red',
                  markersize = mark_s, label = r'$\bf{E}^{(8)}_{P4,F6}$')

blue_p, = ax6.plot(1./gibbs_rad['G=' + str(G)]['E8_P2F8']['P4Iso=' + 'No']['droplet'], 
                   delta_p['G=' + str(G)]['E8_P2F8']['P4Iso=' + 'No']['droplet'], '+', color = 'blue',
                   markersize = mark_s, label = r'$\bf{E}^{(8)}_{P2,F8}$')

line_swap, = ax6.plot(rm1_axis,
                      rm1_axis * sigma_f['G=' + str(G)]['E8_P2F8']['P4Iso=' + 'No']['droplet'],
                      label = '$Gc_s^2=' + str(G) + '$', color = 'black')
black_lines.append(line_swap)

min_y, max_y = 0, 0
for G in [-1.6, -1.75]:
    if min_y == 0:
        min_y = np.amin(delta_p['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet'])
    else:
        min_y = min(min_y, np.amin(delta_p['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet']))
    if max_y == 0:
        max_y = np.amax(delta_p['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet'])
    else:
        max_y = max(max_y, np.amax(delta_p['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet']))

    ax6.plot(1./gibbs_rad['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet'], 
             delta_p['G=' + str(G)]['E8_P4F6']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red', 
             markersize = mark_s)

    ax6.plot(1./gibbs_rad['G=' + str(G)]['E8_P2F8']['P4Iso=' + 'No']['droplet'], 
             delta_p['G=' + str(G)]['E8_P2F8']['P4Iso=' + 'No']['droplet'], '+', color = 'blue', 
             markersize = mark_s)

    line_swap, = ax6.plot(rm1_axis,
                          rm1_axis * sigma_f['G=' + str(G)]['E8_P2F8']['P4Iso=' + 'No']['droplet'],
                          dashed[G], color = 'black', label = '$Gc_s^2=' + str(G) + '$')
    black_lines.append(line_swap)

ax6.set_xlabel('$R^{-1}$', fontsize=f_s)
#ax6.set_title('$\psi = 1 - \\exp(-n)$')
ax6.ticklabel_format(axis='y', style = 'sci', scilimits=(0,0))
ax6.set_xlim([0,x_lim])

lgnd_points = plt.legend(handles = [red_p, blue_p], loc = 'upper left', frameon=False)

lgnd_points.get_texts()[0].set_color("red")
lgnd_points.get_texts()[1].set_color("blue")

lgnd_points.get_texts()[0].set_size("large")
lgnd_points.get_texts()[1].set_size("large")

lgnd_points.legend_handles[0]._sizes = [6]
lgnd_points.legend_handles[1]._sizes = [6]

#ax6.legend(loc='upper center', ncol=1, fancybox=True,
#           bbox_to_anchor=(0.5, 1.7), frameon=False, prop={'size': legend_size})

# Shrink current axis by 20%
box = ax6.get_position()
ax6.set_position([box.x0, box.y0, box.width, box.height * b_height])

ax6.text(_panel_label_pos[0], _panel_label_pos[1], '$(f)$',
         transform = ax6.transAxes, fontsize=f_s)


#################### PANEL (g) ####################
ax7 = plt.subplot2grid((5,2), (3,1), colspan=1, rowspan=1)
black_lines = []

G = -1.4
red_p, = ax7.plot(1./gibbs_rad['G=' + str(G)]['E10_P4F6']['P4Iso=' + 'Yes']['droplet'], 
                  delta_p['G=' + str(G)]['E10_P4F6']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red',
                  markersize = mark_s, label = r'$\bf{E}^{(10)}_{P4,F6}$')

blue_p, = ax7.plot(1./gibbs_rad['G=' + str(G)]['E10_P2F10']['P4Iso=' + 'No']['droplet'], 
                   delta_p['G=' + str(G)]['E10_P2F10']['P4Iso=' + 'No']['droplet'], '+', color = 'blue',
                   markersize = mark_s, label = r'$\bf{E}^{(10)}_{P2,F10}$')

line_swap, = ax7.plot(rm1_axis,
                      rm1_axis * sigma_f['G=' + str(G)]['E10_P2F10']['P4Iso=' + 'No']['droplet'],
                      label = '$Gc_s^2=' + str(G) + '$', color = 'black')
black_lines.append(line_swap)

min_y, max_y = 0, 0
for G in [-1.6, -1.75]:
    if min_y == 0:
        min_y = np.amin(delta_p['G=' + str(G)]['E10_P4F6']['P4Iso=' + 'Yes']['droplet'])
    else:
        min_y = min(min_y, np.amin(delta_p['G=' + str(G)]['E10_P4F6']['P4Iso=' + 'Yes']['droplet']))
    if max_y == 0:
        max_y = np.amax(delta_p['G=' + str(G)]['E10_P4F6']['P4Iso=' + 'Yes']['droplet'])
    else:
        max_y = max(max_y, np.amax(delta_p['G=' + str(G)]['E10_P4F6']['P4Iso=' + 'Yes']['droplet']))

    ax7.plot(1./gibbs_rad['G=' + str(G)]['E10_P4F6']['P4Iso=' + 'Yes']['droplet'], 
             delta_p['G=' + str(G)]['E10_P4F6']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red', 
             markersize = mark_s)

    ax7.plot(1./gibbs_rad['G=' + str(G)]['E10_P2F10']['P4Iso=' + 'No']['droplet'], 
             delta_p['G=' + str(G)]['E10_P2F10']['P4Iso=' + 'No']['droplet'], '+', color = 'blue', 
             markersize = mark_s)

    line_swap, = ax7.plot(rm1_axis,
                          rm1_axis * sigma_f['G=' + str(G)]['E10_P2F10']['P4Iso=' + 'No']['droplet'],
                          dashed[G], color = 'black', label = '$Gc_s^2=' + str(G) + '$')
    black_lines.append(line_swap)

ax7.set_xlabel('$R^{-1}$', fontsize=f_s)
#ax7.set_title('$\psi = 1 - \\exp(-n)$')
ax7.ticklabel_format(axis='y', style = 'sci', scilimits=(0,0))
ax7.set_xlim([0,x_lim])

lgnd_points = plt.legend(handles = [red_p, blue_p], loc = 'upper left', frameon=False)

lgnd_points.get_texts()[0].set_color("red")
lgnd_points.get_texts()[1].set_color("blue")

lgnd_points.get_texts()[0].set_size("large")
lgnd_points.get_texts()[1].set_size("large")

lgnd_points.legend_handles[0]._sizes = [6]
lgnd_points.legend_handles[1]._sizes = [6]

#ax7.legend(loc='upper center', ncol=1, fancybox=True,
#           bbox_to_anchor=(0.5, 1.7), frameon=False, prop={'size': legend_size})

# Shrink current axis by 20%
box = ax7.get_position()
ax7.set_position([box.x0, box.y0, box.width, box.height * b_height])

ax7.text(_panel_label_pos[0], _panel_label_pos[1], '$(g)$',
         transform = ax7.transAxes, fontsize=f_s)

#################### PANEL (h) ####################
ax8 = plt.subplot2grid((5,2), (4,1), colspan=1, rowspan=1)
black_lines = []

G = -1.4
red_p, = ax8.plot(1./gibbs_rad['G=' + str(G)]['E12_P4F6']['P4Iso=' + 'Yes']['droplet'], 
                  delta_p['G=' + str(G)]['E12_P4F6']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red',
                  markersize = mark_s, label = r'$\bf{E}^{(12)}_{P4,F6}$')

blue_p, = ax8.plot(1./gibbs_rad['G=' + str(G)]['E12_P2F12']['P4Iso=' + 'No']['droplet'], 
                   delta_p['G=' + str(G)]['E12_P2F12']['P4Iso=' + 'No']['droplet'], '+', color = 'blue',
                   markersize = mark_s, label = r'$\bf{E}^{(12)}_{P2,F12}$')

line_swap, = ax8.plot(rm1_axis,
                      rm1_axis * sigma_f['G=' + str(G)]['E12_P2F12']['P4Iso=' + 'No']['droplet'],
                      label = '$Gc_s^2=' + str(G) + '$', color = 'black')
black_lines.append(line_swap)

min_y, max_y = 0, 0
for G in [-1.6, -1.75]:
    if min_y == 0:
        min_y = np.amin(delta_p['G=' + str(G)]['E12_P4F6']['P4Iso=' + 'Yes']['droplet'])
    else:
        min_y = min(min_y, np.amin(delta_p['G=' + str(G)]['E12_P4F6']['P4Iso=' + 'Yes']['droplet']))
    if max_y == 0:
        max_y = np.amax(delta_p['G=' + str(G)]['E12_P4F6']['P4Iso=' + 'Yes']['droplet'])
    else:
        max_y = max(max_y, np.amax(delta_p['G=' + str(G)]['E12_P4F6']['P4Iso=' + 'Yes']['droplet']))

    ax8.plot(1./gibbs_rad['G=' + str(G)]['E12_P4F6']['P4Iso=' + 'Yes']['droplet'], 
             delta_p['G=' + str(G)]['E12_P4F6']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red', 
             markersize = mark_s)

    ax8.plot(1./gibbs_rad['G=' + str(G)]['E12_P2F12']['P4Iso=' + 'No']['droplet'], 
             delta_p['G=' + str(G)]['E12_P2F12']['P4Iso=' + 'No']['droplet'], '+', color = 'blue', 
             markersize = mark_s)

    line_swap, = ax8.plot(rm1_axis,
                          rm1_axis * sigma_f['G=' + str(G)]['E12_P2F12']['P4Iso=' + 'No']['droplet'],
                          dashed[G], color = 'black', label = '$Gc_s^2=' + str(G) + '$')
    black_lines.append(line_swap)

ax8.set_xlabel('$R^{-1}$', fontsize=f_s)
#ax8.set_title('$\psi = 1 - \\exp(-n)$')
ax8.ticklabel_format(axis='y', style = 'sci', scilimits=(0,0))
ax8.set_xlim([0,x_lim])

lgnd_points = plt.legend(handles = [red_p, blue_p], loc = 'upper left', frameon=False)

lgnd_points.get_texts()[0].set_color("red")
lgnd_points.get_texts()[1].set_color("blue")

lgnd_points.get_texts()[0].set_size("large")
lgnd_points.get_texts()[1].set_size("large")

lgnd_points.legend_handles[0]._sizes = [6]
lgnd_points.legend_handles[1]._sizes = [6]

#ax8.legend(loc='upper center', ncol=1, fancybox=True,
#           bbox_to_anchor=(0.5, 1.7), frameon=False, prop={'size': legend_size})

# Shrink current axis by 20%
box = ax8.get_position()
ax8.set_position([box.x0, box.y0, box.width, box.height * b_height])

ax8.text(_panel_label_pos[0], _panel_label_pos[1], '$(h)$',
         transform = ax8.transAxes, fontsize=f_s)



#################### SAVING ####################
fig.tight_layout()

from pathlib import Path
reproduced_figures = Path("reproduced-figures")
if not reproduced_figures.is_dir():
    reproduced_figures.mkdir()

plt.savefig(reproduced_figures / 'figure_3.png', dpi = _dpi)
plt.savefig(reproduced_figures / 'figure_3.pdf', dpi = _dpi)

plt.close()
