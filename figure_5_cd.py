# Figure 3, panels (c) and (d)

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
      psis[1]: [-1.4, -1.6]}

Ls = [255]

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
     StencilPsiKey(E8_P4F6_sym, psis[0]): reproduced_results / DropletsFileName(E8_P4F6_sym, psis[0])}


laplace_files_E8 = \
    {StencilPsiKey(E8_P2F8_sym, psis[0]): reproduced_results / LaplaceFileName(E8_P2F8_sym, psis[0]), 
     StencilPsiKey(E8_P2F8_sym, psis[1]): reproduced_results / LaplaceFileName(E8_P2F8_sym, psis[1]), 
     StencilPsiKey(E8_P4F6_sym, psis[0]): reproduced_results / LaplaceFileName(E8_P4F6_sym, psis[0]), 
     StencilPsiKey(E8_P4F6_sym, psis[1]): reproduced_results / LaplaceFileName(E8_P4F6_sym, psis[1])}

E_sym_YN = {E8_P2F8_sym: 'No', E8_P4F6_sym: 'Yes'}

gibbs_rad = defaultdict( # G
    lambda: defaultdict( #  'B2F6'
        lambda: defaultdict(  # 'P4Iso=' + YN
            lambda: defaultdict(dict) # 'droplet'
        )
    )
)

from idpy.Utils.ManageData import ManageData

for _psi in [psis[0]]:
    for _stencil in [E8_P2F8_sym, E8_P4F6_sym]:
        _data_swap = ManageData(dump_file = laplace_files_E8[StencilPsiKey(_stencil, _psi)])
        _is_file_there = _data_swap.Read()
        if not _is_file_there:
            raise Exception("Could not find file!", 
                            laplace_files_E8[StencilPsiKey(_stencil, _psi)])        
        
        for G in Gs[_psi]:
            _swap_gibbs_rad = []
            for L in Ls:
                _data_key = str(G) + "_" + str(L)
                _swap_gibbs_rad.append(_data_swap.PullData(_data_key)['R_Gibbs'])

            gibbs_rad['G=' + str(G)]['B2F6']['P4Iso=' + E_sym_YN[_stencil]]['droplet'] = \
                np.array(_swap_gibbs_rad)
            

UFields = {}
E_sym_dict = {E8_P2F8_sym: 'E8P2', E8_P4F6_sym: 'E8P4'}

_psi, G, L = psis[0], -3.6, 255

for _stencil in [E8_P2F8_sym, E8_P4F6_sym]:
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

import matplotlib.ticker as ticker

##################################################
#################### FIGURE 2 ####################
##################################################

from matplotlib import rc, rcParams
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

G = -3.6
LX = 255
LXA = LX**2
c2 = 1/3.
xrange = [LX//2 - LX/25, 13*LX/16]
n_q_lin = 9

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

norm_2iso_min, norm_2iso_max = np.amin(norm_2iso), np.amax(norm_2iso)
norm_2niso_min, norm_2niso_max = np.amin(norm_2niso), np.amax(norm_2niso)
print("norm_2iso_min: ", norm_2iso_min, ", norm_2iso_max: ", norm_2iso_max)
print("norm_2niso_min: ", norm_2niso_min, ", norm_2niso_max: ", norm_2niso_max)

norm_2iso_lin = ((norm_2iso - norm_2iso_min)/((norm_2iso_max - norm_2iso_min)/n_q_lin))
norm_2niso_lin = ((norm_2niso - norm_2niso_min)/((norm_2niso_max - norm_2niso_min)/n_q_lin))

# need to manually select the radius
# print(gibbs_rad['G=' + str(G)]['B2F6']['P4Iso=' + 'Yes']['droplet'])
sim_2iso_rad = gibbs_rad['G=' + str(G)]['B2F6']['P4Iso=' + 'Yes']['droplet'][0]
sim_2niso_rad = gibbs_rad['G=' + str(G)]['B2F6']['P4Iso=' + 'No']['droplet'][0]

circle_2iso = plt.Circle((LX//2, LX//2), sim_2iso_rad, color = 'black', fill = False)
circle_2niso = plt.Circle((LX//2, LX//2), sim_2niso_rad, color = 'black', fill = False)

for x in range(norm_2iso_lin.shape[0]):
    for y in range(norm_2iso_lin.shape[1]):
        norm_2iso_lin[(x,y)] = int((norm_2iso_lin[(x,y)]))
        
for x in range(norm_2niso_lin.shape[0]):
    for y in range(norm_2niso_lin.shape[1]):
        norm_2niso_lin[(x,y)] = int(norm_2niso_lin[(x,y)])


f_s = 16
#################### SIZES ####################
fig = plt.figure(figsize=(4.75, 8))
###############################################

#################### PANEL (a) ####################
ax1 = plt.subplot2grid((2,1), (0,0), colspan=1, rowspan=1)

new_xrange = np.linspace(LX//2, int(xrange[1]), 5)
new_yrange = np.linspace(LX//2, int(xrange[1]), 5)

ax1.set_xticks(new_xrange)
ax1.set_xticklabels([int(elem) for elem in new_xrange - LX//2])
ax1.set_yticks(new_yrange)
ax1.set_yticklabels([int(elem) for elem in new_yrange - LX//2])

ax1.set_xlim(xrange)
ax1.set_ylim(xrange)
ax1.add_artist(circle_2iso)
im1 = ax1.imshow(norm_2iso_lin, cmap = 'rainbow', vmin = 0., interpolation='bicubic')
ax1.plot([LX//2 + 0.5], [LX//2 + 0.5], 'o', color = 'black', markersize = 6)

_panel_label_pos = (0.93, 1.065)
_e_label_pos = (0.42, 1.065)
_psi_label_pos = (0.31, 1.065)
_g_label_pos = (0.85, 1.065)

ax1.text(_e_label_pos[0], _e_label_pos[1], r'$\boldsymbol{E}^{(8)}_{P4,F6}$',
         transform = ax1.transAxes, color = 'red', fontsize = f_s)
ax1.text(_panel_label_pos[0], _panel_label_pos[1], '$(c)$',
         transform = ax1.transAxes, fontsize = f_s)
#ax1.text(_psi_label_pos[0], _psi_label_pos[1], '$\psi = \exp(-1/n),$', transform = ax1.transAxes, fontsize = f_s)
#ax1.text(_g_label_pos[0], _g_label_pos[1], '$Gc_s^2 = -3.6$', transform = ax1.transAxes, fontsize = f_s)

cb1 = fig.colorbar(im1, ticks = np.linspace(0, 9, 4))

cb1.set_label('$\\lfloor 9\; (u - u_{m})/(u_{M} - u_{m}) \\rfloor$', labelpad=+18, rotation=-90, fontsize = f_s)

#################### PANEL (b) ####################
ax2 = plt.subplot2grid((2,1), (1,0), colspan=1, rowspan=1)

ax2.set_xticks(new_xrange)
ax2.set_xticklabels([int(elem) for elem in new_xrange - LX//2])
ax2.set_yticks(new_yrange)
ax2.set_yticklabels([int(elem) for elem in new_yrange - LX//2])

ax2.set_xlim(xrange)
ax2.set_ylim(xrange)
ax2.add_artist(circle_2niso)
im2 = ax2.imshow(norm_2niso_lin, cmap = 'rainbow', vmin = 0., interpolation='bicubic')
ax2.plot([LX//2 - 0.5], [LX//2 + 0.5], 'o', color = 'black', markersize = 6)

ax2.text(_e_label_pos[0], _e_label_pos[1], r'$\boldsymbol{E}^{(8)}_{P2,F8}$', transform = ax2.transAxes,
         color = 'blue', fontsize = f_s)

ax2.text(_panel_label_pos[0], _panel_label_pos[1], '$(d)$',
         transform = ax2.transAxes, fontsize = f_s)

cb2 = fig.colorbar(im2, ticks = np.linspace(0, 9, 4))

cb2.set_label('$\\lfloor 9\; (u - u_{m})/(u_{M} - u_{m}) \\rfloor$', labelpad=+18, rotation=-90, fontsize = f_s)


#################### SAVING ####################
fig.tight_layout()
from pathlib import Path
reproduced_figures = Path("reproduced-figures")
if not reproduced_figures.is_dir():
    reproduced_figures.mkdir()
plt.savefig(reproduced_figures / 'figure_5_cd.png',
            bbox_inches = 'tight', dpi = _dpi)

plt.close()
