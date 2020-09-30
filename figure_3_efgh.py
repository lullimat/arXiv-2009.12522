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
      psis[1]: [-1.4, -1.6]}

Ls = [127, 159, 191, 223, 255, 287, 319, 351]

E8_P2F8_sym = sp_symbols("\\boldsymbol{E}^{(8)}_{P2\,F8}")
E8_P4F6_sym = sp_symbols("\\boldsymbol{E}^{(8)}_{P4\,F6}")

'''
Getting usual weights
'''

S5_E8_P2F8 = SCFStencils(E = BasisVectors(x_max = 2), 
                         len_2s = [1, 2, 4, 5, 8])
S5_E8_P2F8_W = S5_E8_P2F8.FindWeights()

'''
Getting new weights
'''

w1, w2, w4, w5, w8 = sp_symbols("w(1) w(2) w(4) w(5) w(8)")
eps = sp_symbols('\\varepsilon')
w_sym_list = [w1, w2, w4, w5, w8]

eps_expr = (48*w4 + 96*w5 + 96*w8)
eps_expr /= (6*w1 + 12*w2 + 72*w4 + 156*w5 + 144*w8)

chi_i_expr = 2*w4 - 8*w8 - w5

S5_E8_P4F6 = SCFStencils(E = BasisVectors(x_max = 2), 
                         len_2s = [1, 2, 4, 5, 8])

S5_E8_P4F6.GetWolfEqs()

cond_e2 = S5_E8_P4F6.e_expr[2] - 1
cond_e4 = S5_E8_P4F6.e_expr[4] - sp_Rational('4/7')
cond_e6 = S5_E8_P4F6.e_expr[6] - sp_Rational('32/105')
cond_eps = eps_expr - sp_Rational('10/31')
cond_chi_i = chi_i_expr

S5_E8_P4F6_eqs = [cond_e2, cond_e4, cond_e6,
                  cond_eps, cond_chi_i]
    
S5_E8_P4F6_W = S5_E8_P4F6.FindWeights(S5_E8_P4F6_eqs)

'''
File Names
'''
stencil_string = {E8_P2F8_sym: 'E8_P2F8', 
                  E8_P4F6_sym: 'E8_P4F6'}

stencil_dict = {E8_P2F8_sym: S5_E8_P2F8, 
                E8_P4F6_sym: S5_E8_P4F6}

stencil_sym_list = [E8_P2F8_sym, E8_P4F6_sym]


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

flat_files_E8 = \
    {StencilPsiKey(E8_P2F8_sym, psis[0]): reproduced_results / FlatFileName(E8_P2F8_sym, psis[0]), 
     StencilPsiKey(E8_P2F8_sym, psis[1]): reproduced_results / FlatFileName(E8_P2F8_sym, psis[1]), 
     StencilPsiKey(E8_P4F6_sym, psis[0]): reproduced_results / FlatFileName(E8_P4F6_sym, psis[0]), 
     StencilPsiKey(E8_P4F6_sym, psis[1]): reproduced_results / FlatFileName(E8_P4F6_sym, psis[1])}

laplace_files_E8 = \
    {StencilPsiKey(E8_P2F8_sym, psis[0]): reproduced_results / LaplaceFileName(E8_P2F8_sym, psis[0]), 
     StencilPsiKey(E8_P2F8_sym, psis[1]): reproduced_results / LaplaceFileName(E8_P2F8_sym, psis[1]), 
     StencilPsiKey(E8_P4F6_sym, psis[0]): reproduced_results / LaplaceFileName(E8_P4F6_sym, psis[0]), 
     StencilPsiKey(E8_P4F6_sym, psis[1]): reproduced_results / LaplaceFileName(E8_P4F6_sym, psis[1])}

rho_fields = {}
E_sym_dict = {E8_P2F8_sym: 'E8P2', E8_P4F6_sym: 'E8P4'}

for _psi in psis:

    for _stencil in [E8_P2F8_sym, E8_P4F6_sym]:
        _data_swap = ManageData(dump_file = flat_files_E8[StencilPsiKey(_stencil, _psi)])
        _is_file_there = _data_swap.Read()
        if not _is_file_there:
            raise Exception("Could not find file!", 
                            flat_files_E8[StencilPsiKey(_stencil, _psi)])
        
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

E_sym_YN = {E8_P2F8_sym: 'No', E8_P4F6_sym: 'Yes'}

for _psi in psis:
    for _stencil in [E8_P2F8_sym, E8_P4F6_sym]:
        _data_swap = ManageData(dump_file = laplace_files_E8[StencilPsiKey(_stencil, _psi)])
        _is_file_there = _data_swap.Read()
        if not _is_file_there:
            raise Exception("Could not find file!", 
                            laplace_files_E6[StencilPsiKey(_stencil, _psi)])        

        for G in Gs[_psi]:
            _swap_gibbs_rad, _swap_delta_p = [], []
            for L in Ls:
                _data_key = str(G) + "_" + str(L)
                _swap_gibbs_rad.append(_data_swap.PullData(_data_key)['R_Gibbs'])
                _swap_delta_p.append(_data_swap.PullData(_data_key)['delta_p'])

            gibbs_rad['G=' + str(G)]['Belts=2']['P4Iso=' + E_sym_YN[_stencil]]['droplet'] = \
                np.array(_swap_gibbs_rad)
            delta_p['G=' + str(G)]['Belts=2']['P4Iso=' + E_sym_YN[_stencil]]['droplet'] = \
                np.array(_swap_delta_p)
            
sigma_f = defaultdict( # G
    lambda: defaultdict( #  'B2F6'
        lambda: defaultdict(  # 'P4Iso=' + YN
            lambda: defaultdict(dict) # 'droplet'
        )
    )
)

for _psi in psis:
    for _stencil in [E8_P2F8_sym]:
        for G in Gs[_psi]:
            _sc_eq_cache = ShanChanEquilibriumCache(stencil = stencil_dict[_stencil], 
                                                    psi_f = _psi, G = G, 
                                                    c2 = XIStencils['D2Q9']['c2'])
            
            sigma_f['G=' + str(G)]['Belts=2']['P4Iso=' + E_sym_YN[_stencil]]['droplet'] = \
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

a = 0.9
 
b_height = 0.8
legend_size = 10

dashed = {}
dashed[-2.6] = '-'
dashed[-3.1] = '--'
dashed[-3.6] = '-.'

dashed[-1.4] = '-'
dashed[-1.6] = '--'

f_s = 14
#################### SIZES ####################
fig = plt.figure(figsize=(5.2, 9))
###############################################

#################### PANEL (a) ####################
mark_s = 9

ax1 = plt.subplot2grid((3,2), (0,0), colspan=1, rowspan=1)

black_lines = []
black_labels = []
G = -2.6
print(gibbs_rad['G=' + str(G)]['Belts=2']['P4Iso=' + 'Yes']['droplet'])
red_p, = ax1.plot(1./gibbs_rad['G=' + str(G)]['Belts=2']['P4Iso=' + 'Yes']['droplet'], 
                  delta_p['G=' + str(G)]['Belts=2']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red', 
                  markersize = mark_s, label = r'$\boldsymbol{E}^{(8)}_{P4,F6}$')

blue_p, = ax1.plot(1./gibbs_rad['G=' + str(G)]['Belts=2']['P4Iso=' + 'No']['droplet'], 
                   delta_p['G=' + str(G)]['Belts=2']['P4Iso=' + 'No']['droplet'], '+', color = 'blue', 
                   markersize = mark_s, label = r'$\boldsymbol{E}^{(8)}_{P2,F8}$')

line_swap, = ax1.plot(rm1_axis, 
                      rm1_axis * sigma_f['G=' + str(G)]['Belts=2']['P4Iso=' + 'No']['droplet'], 
                      label = '$Gc_s^2=' + str(G) + '$', color = 'black')

black_lines.append(line_swap)

for G in [-3.1, -3.6]:
    ax1.plot(1./gibbs_rad['G=' + str(G)]['Belts=2']['P4Iso=' + 'Yes']['droplet'], 
             delta_p['G=' + str(G)]['Belts=2']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red', 
             markersize = mark_s)

    ax1.plot(1./gibbs_rad['G=' + str(G)]['Belts=2']['P4Iso=' + 'No']['droplet'], 
             delta_p['G=' + str(G)]['Belts=2']['P4Iso=' + 'No']['droplet'], '+', color = 'blue', 
             markersize = mark_s)

    line_swap, = ax1.plot(rm1_axis, 
                          rm1_axis * sigma_f['G=' + str(G)]['Belts=2']['P4Iso=' + 'No']['droplet'], 
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
lgnd_points = plt.legend(handles = [red_p, blue_p],
                         labelspacing=0.5,
                         bbox_to_anchor=(0.13, 1.9), frameon=False,
                         handletextpad=0.)

lgnd_points.get_texts()[0].set_color("red")
lgnd_points.get_texts()[1].set_color("blue")

lgnd_points.get_texts()[0].set_size("large")
lgnd_points.get_texts()[1].set_size("large")

lgnd_points.legendHandles[0]._legmarker.set_markersize(6)
lgnd_points.legendHandles[1]._legmarker.set_markersize(6)

### lines legends
black_lines.insert(0, plt.Line2D(rm1_axis, rm1_axis, linestyle='none', label = '$\psi = \exp(-1/n)$'))
lgnd_lines = plt.legend(handles = black_lines, handlelength = 2.,
                        labelspacing=0.5,
                        bbox_to_anchor=(1.05, 2),
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
b_height = 1.
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width, box.height * b_height])

ax1.text(2.5e-3, 4.7e-3, '$\\psi = \\exp(-1/n)$', fontsize=11)
ax1.text(0.038, 5.8e-3, '$(e)$', fontsize=f_s)

#ax1.text(2.5e-3, 3.8e-3, '$\\psi = \\exp(-1/n)$', fontsize=11)
#ax1.text(0.038, 4.7e-3, '$(a)$', fontsize=f_s)

#################### PANEL (b) ####################
ax2 = plt.subplot2grid((3,2), (0,1), colspan=1, rowspan=1)
black_lines = []

G = -1.4
ax2.plot(1./gibbs_rad['G=' + str(G)]['Belts=2']['P4Iso=' + 'Yes']['droplet'], 
         delta_p['G=' + str(G)]['Belts=2']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red',
         markersize = mark_s)

ax2.plot(1./gibbs_rad['G=' + str(G)]['Belts=2']['P4Iso=' + 'No']['droplet'], 
         delta_p['G=' + str(G)]['Belts=2']['P4Iso=' + 'No']['droplet'], '+', color = 'blue',
         markersize = mark_s)

line_swap, = ax2.plot(rm1_axis,
                      rm1_axis * sigma_f['G=' + str(G)]['Belts=2']['P4Iso=' + 'No']['droplet'],
                      label = '$Gc_s^2=' + str(G) + '$', color = 'black')
black_lines.append(line_swap)

for G in [-1.6]:
    ax2.plot(1./gibbs_rad['G=' + str(G)]['Belts=2']['P4Iso=' + 'Yes']['droplet'], 
             delta_p['G=' + str(G)]['Belts=2']['P4Iso=' + 'Yes']['droplet'], 'x', color = 'red', 
             markersize = mark_s)

    ax2.plot(1./gibbs_rad['G=' + str(G)]['Belts=2']['P4Iso=' + 'No']['droplet'], 
             delta_p['G=' + str(G)]['Belts=2']['P4Iso=' + 'No']['droplet'], '+', color = 'blue', 
             markersize = mark_s)

    line_swap, = ax2.plot(rm1_axis,
                          rm1_axis * sigma_f['G=' + str(G)]['Belts=2']['P4Iso=' + 'No']['droplet'],
                          dashed[G], color = 'black', label = '$Gc_s^2=' + str(G) + '$')
    black_lines.append(line_swap)

ax2.set_xlabel('$R^{-1}$', fontsize=f_s)
#ax2.set_title('$\psi = 1 - \\exp(-n)$')
ax2.ticklabel_format(axis='y', style = 'sci', scilimits=(0,0))
ax2.set_xlim([0,x_lim])

#ax2.legend(loc='upper center', ncol=1, fancybox=True,
#           bbox_to_anchor=(0.5, 1.7), frameon=False, prop={'size': legend_size})

### lines legends
black_lines.insert(0, plt.Line2D(rm1_axis, rm1_axis, linestyle='none', label = '$\psi = 1 - \exp(-n)$'))
lgnd_lines2 = plt.legend(handles = black_lines, handlelength = 2.,
                         labelspacing=0.5,
                         bbox_to_anchor=(1.05, 1.9),
                         frameon=True)

lgnd_lines2.get_texts()[0].set_x(-25)
lgnd_lines2.get_texts()[0].set_size("large")
lgnd_lines2.get_texts()[1].set_size("large")
lgnd_lines2.get_texts()[2].set_size("large")

# Shrink current axis by 20%
box = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width, box.height * b_height])

ax2.text(2.5e-3, 1.385e-3, '$\\psi = 1 - \\exp(-n)$', fontsize=11)
ax2.text(0.038, 1.68e-3, '$(f)$', fontsize=f_s)

#################### PANEL (c) ####################
ax3 = plt.subplot2grid((3,2), (1,0), colspan=2, rowspan=1)

mark_s2 = 5

########

for G in [-2.6, -3.1, -3.6]:
    L = rho_fields['G' + str(G) + 'E8P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E8P4'][x_range]
    rho_niso = rho_fields['G' + str(G) + 'E8P2'][x_range]
    ax3.plot(np.array(range(L//2 - 1)), 1 - rho_iso/rho_niso, dashed[G], color = 'black', markersize = mark_s2)

ax3.set_xlabel('$x$', fontsize=f_s)
ax3.set_ylabel('$1 - n_{P4}/n_{P2}$', fontsize=f_s)

ax3.text(-0.5, 0.8e-11, '$\\psi = \\exp(-1/n)$', fontsize=11)
ax3.text(39, 9.2e-11, '$(g)$', fontsize=f_s)

### INSET
ax3i = inset_axes(ax3, width='35%', height='70%', loc = 'upper right')

for G in [-2.6, -3.1, -3.6]:
    L = rho_fields['G' + str(G) + 'E8P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E8P4'][x_range]
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
ax4 = plt.subplot2grid((3,2), (2,0), colspan=2, rowspan=1)

for G in [-1.4, -1.6]:
    L = rho_fields['G' + str(G) + 'E8P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E8P4'][x_range]
    rho_niso = rho_fields['G' + str(G) + 'E8P2'][x_range]
    ax4.plot(np.array(range(L//2 - 1)), 1 - rho_iso/rho_niso, dashed[G], color = 'black', markersize = mark_s2)

ax4.set_xlabel('$x$', fontsize=f_s)
ax4.set_ylabel('$1 - n_{P4}/n_{P2}$', fontsize=f_s)

ax4.text(-0.5, 0.7e-12, '$\\psi = 1 - \\exp(-n)$', fontsize=11)
ax4.text(39, 7.6e-12, '$(h)$', fontsize=f_s)

### INSET
ax4i = inset_axes(ax4, width='35%', height='70%', loc = 'upper right')

for G in [-1.4, -1.6]:
    L = rho_fields['G' + str(G) + 'E8P4'].shape[0]
    x_range = np.array(range(L)) > L//2
    rho_iso = rho_fields['G' + str(G) + 'E8P4'][x_range]
    ax4i.plot(np.array(range(L//2 - 1)), rho_iso, dashed[G], color = 'black')

ax4i.set_ylim([-0.5, 1.9])
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


#################### SAVING ####################
fig.tight_layout()
from pathlib import Path
reproduced_figures = Path("reproduced-figures")
if not reproduced_figures.is_dir():
    reproduced_figures.mkdir()

plt.savefig(reproduced_figures / 'figure_3_E8.png',
            bbox_inches = 'tight', dpi = _dpi)
plt.close()
