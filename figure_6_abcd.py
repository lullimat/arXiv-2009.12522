import sys
sys.path.append("../../")
import numpy as np

from pathlib import Path
reproduced_results = Path("reproduced-results")

device_str, lang, _dpi = sys.argv[1], sys.argv[2], int(sys.argv[3])

import sympy as sp

n = sp.symbols('n')
psis = [sp.exp(-1/n), 1 - sp.exp(-n)]
psi_codes = {psis[0]: 'exp((NType)(-1./ln))', 
             psis[1]: '1. - exp(-(NType)ln)'}

Gs = {psis[0]: [-3.6], 
      psis[1]: [-1.75]}

lw = 2.9

E6_P2F6_sym = sp.symbols("\\bf{E}^{(6)}_{P2\,F6}")
E8_P2F8_sym = sp.symbols("\\bf{E}^{(8)}_{P2\,F8}")
E6_P4F6_sym = sp.symbols("\\bf{E}^{(6)}_{P4\,F6}")
E8_P4F6_sym = sp.symbols("\\bf{E}^{(8)}_{P4\,F6}")
E10_P2F10_sym = sp.symbols("\\bf{E}^{(10)}_{P2\,F10}")
E10_P4F6_sym = sp.symbols("\\bf{E}^{(10)}_{P4\,F6}")
E12_P2F12_sym = sp.symbols("\\bf{E}^{(12)}_{P2\,F12}")
E12_P4F6_sym = sp.symbols("\\bf{E}^{(12)}_{P4\,F6}")

E6_P2F6_str = r'$\bf{E}^{(6)}_{P2,F6}$'
E8_P2F8_str = r'$\bf{E}^{(8)}_{P2,F8}$'
E6_P4F6_str = r'$\bf{E}^{(6)}_{P4,F6}$'
E8_P4F6_str = r'$\bf{E}^{(8)}_{P4,F6}$'
E10_P2F10_str = r'$\bf{E}^{(10)}_{P2,F10}$'
E10_P4F6_str = r'$\bf{E}^{(10)}_{P4,F6}$'
E12_P2F12_str = r'$\bf{E}^{(12)}_{P2,F12}$'
E12_P4F6_str = r'$\bf{E}^{(12)}_{P4,F6}$'


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

def DropletsFileName(stencil_sym, psi):
    psi_str = str(psi).replace("/", "_").replace("-", "_")
    psi_str = psi_str.replace(" ", "_")
    psi_str = psi_str.replace("(", "").replace(")","")
    lang_str = str(lang) + "_" + device_str

    return (lang_str + stencil_string[stencil_sym] + "_" +
            psi_str + "_droplets")

def LaplaceFileName(stencil_sym, psi):
    psi_str = str(psi).replace("/", "_").replace("-", "_")
    psi_str = psi_str.replace(" ", "_")
    psi_str = psi_str.replace("(", "").replace(")","")
    lang_str = str(lang) + "_" + device_str

    return (lang_str + stencil_string[stencil_sym] + "_" +
            psi_str + "_laplace")

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

laplace_files = \
    {StencilPsiKey(E6_P2F6_sym, psis[0]): reproduced_results / LaplaceFileName(E6_P2F6_sym, psis[0]),
     StencilPsiKey(E6_P2F6_sym, psis[1]): reproduced_results / LaplaceFileName(E6_P2F6_sym, psis[1]),
     StencilPsiKey(E6_P4F6_sym, psis[0]): reproduced_results / LaplaceFileName(E6_P4F6_sym, psis[0]),
     StencilPsiKey(E6_P4F6_sym, psis[1]): reproduced_results / LaplaceFileName(E6_P4F6_sym, psis[1]),
     StencilPsiKey(E8_P2F8_sym, psis[0]): reproduced_results / LaplaceFileName(E8_P2F8_sym, psis[0]),
     StencilPsiKey(E8_P2F8_sym, psis[1]): reproduced_results / LaplaceFileName(E8_P2F8_sym, psis[1]),
     StencilPsiKey(E8_P4F6_sym, psis[0]): reproduced_results / LaplaceFileName(E8_P4F6_sym, psis[0]),
     StencilPsiKey(E8_P4F6_sym, psis[1]): reproduced_results / LaplaceFileName(E8_P4F6_sym, psis[1]),
     StencilPsiKey(E10_P2F10_sym, psis[0]): reproduced_results / LaplaceFileName(E10_P2F10_sym, psis[0]),
     StencilPsiKey(E10_P2F10_sym, psis[1]): reproduced_results / LaplaceFileName(E10_P2F10_sym, psis[1]),
     StencilPsiKey(E10_P4F6_sym, psis[0]): reproduced_results / LaplaceFileName(E10_P4F6_sym, psis[0]),
     StencilPsiKey(E10_P4F6_sym, psis[1]): reproduced_results / LaplaceFileName(E10_P4F6_sym, psis[1]),
     StencilPsiKey(E12_P2F12_sym, psis[0]): reproduced_results / LaplaceFileName(E12_P2F12_sym, psis[0]),
     StencilPsiKey(E12_P2F12_sym, psis[1]): reproduced_results / LaplaceFileName(E12_P2F12_sym, psis[1]),
     StencilPsiKey(E12_P4F6_sym, psis[0]): reproduced_results / LaplaceFileName(E12_P4F6_sym, psis[0]),
     StencilPsiKey(E12_P4F6_sym, psis[1]): reproduced_results / LaplaceFileName(E12_P4F6_sym, psis[1])}

E_sym_YN = {E6_P2F6_sym: 'No', E6_P4F6_sym: 'Yes',
            E8_P2F8_sym: 'No', E8_P4F6_sym: 'Yes',
            E10_P2F10_sym: 'No', E10_P4F6_sym: 'Yes',
            E12_P2F12_sym: 'No', E12_P4F6_sym: 'Yes'}


'''
Read data
'''

E_sym_dict = {E6_P2F6_sym: 'E6P2', E6_P4F6_sym: 'E6P4',
              E8_P2F8_sym: 'E8P2', E8_P4F6_sym: 'E8P4',
              E10_P2F10_sym: 'E10P2', E10_P4F6_sym: 'E10P4',
              E12_P2F12_sym: 'E12P2', E12_P4F6_sym: 'E12P4'}

E_str_dict = {E6_P2F6_str: 'E6P2', E6_P4F6_str: 'E6P4',
              E8_P2F8_str: 'E8P2', E8_P4F6_str: 'E8P4',
              E10_P2F10_str: 'E10P2', E10_P4F6_str: 'E10P4',
              E12_P2F12_str: 'E12P2', E12_P4F6_str: 'E12P4'}

E_sym_dict_m1, E_str_dict_m1 = {}, {}
for _ in E_sym_dict:
    E_sym_dict_m1[E_sym_dict[_]] = _

for _ in E_str_dict:
    E_str_dict_m1[E_str_dict[_]] = _


UFields = {}
gibbs_rad = {}


from idpy.Utils.ManageData import ManageData
AveU = ManageData(dump_file = 'AverageProfiles')

if not AveU.Read():
    print("File 'AverageProfiles' not found (!): computing and saving the average profiles")
    for L in [255]:
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

                    '''
                    _data_swap = ManageData(dump_file = laplace_files[StencilPsiKey(_stencil, _psi)])
                    _data_key = str(G) + "_" + str(L)
                    print(_data_key, laplace_files[StencilPsiKey(_stencil, _psi)])
                    gibbs_rad['G' + str(G) + 'LX' + str(L) + E_sym_dict[_stencil]] = \
                        _swap_gibbs_rad.append(_data_swap.PullData(_data_key)['R_Gibbs'])
                    '''



    _u_profile_ave, _u_profile_err = {}, {}

    L, G, stencil_name = 255, -3.1, 'E12P2'

    _r_range = np.arange(0, L//2, 1)
    _n_bins = 2 ** 8
    c2 = 1/3

    '''
    preparing ALL profiles
    '''

    for stencil_name in ['E12P4', 'E12P2', 'E10P4', 'E10P2', 
                         'E8P4', 'E8P2', 'E6P4', 'E6P2']:
        for G in [-1.75, -3.6]:
            _str = 'G' + str(G) + 'LX' + str(L) + stencil_name

            UX = UFields[_str][0: L * L]
            UY = UFields[_str][L * L: 2 * L * L]
            UX = UX.reshape([L, L])
            UY = UY.reshape([L, L])

            UX = UX[L//2: L, L//2: L]
            UY = UY[L//2: L, L//2: L]

            XRange = np.arange(0, L//2 + 1)
            YRange = np.arange(0, L//2 + 1)

            UX.shape, UY.shape

            from scipy import interpolate

            UX_spl = interpolate.interp2d(XRange, YRange, UX, kind='cubic')
            UY_spl = interpolate.interp2d(XRange, YRange, UY, kind='cubic')

            _u_profiles = np.zeros([_n_bins, _r_range.shape[0]])

            _delta_theta = 0.1
            _bin_i = 0
            for _delta_theta in np.linspace(0, 1, _n_bins):
                for _r in _r_range:
                    _x_cut = _r * np.cos(_delta_theta * np.pi/4.)
                    _y_cut = _r * np.sin(_delta_theta * np.pi/4.)
                    _u_profiles[_bin_i, _r] = np.sqrt(((UX_spl(_x_cut, _y_cut)) ** 2 + 
                                                       (UY_spl(_x_cut, _y_cut)) ** 2)/c2)
                _bin_i += 1

            _u_profile_ave[_str] = np.zeros(_r_range.shape[0])
            _u_profile_err[_str] = np.zeros(_r_range.shape[0])

            for _r in _r_range:
                _u_profile_ave[_str][_r] = np.mean(_u_profiles[:, _r])
                _u_profile_err[_str][_r] = np.sqrt(np.var(_u_profiles[:, _r])/(_n_bins - 1))

    AveU.PushData(data = _u_profile_ave, key = '_u_profile_ave')
    AveU.PushData(data = _u_profile_err, key = '_u_profile_err')
    AveU.PushData(data = L, key = 'L')

    AveU.Dump()
else:
    _u_profile_ave = AveU.PullData('_u_profile_ave')
    _u_profile_err = AveU.PullData('_u_profile_err')
    L = AveU.PullData('L')
    _r_range = np.arange(0, L//2, 1)

'''
Plotting
'''
from matplotlib import rc, rcParams
import matplotlib
import matplotlib.pyplot as plt
from idpy.Utils.Plots import SetAxPanelLabelCoords, SetMatplotlibLatexParamas, CreateFiguresPanels, SetAxTicksFont

SetMatplotlibLatexParamas([rc], [rcParams])

if False:
    rc('font',**{'family':'STIXGeneral'})
    rc('mathtext', **{'fontset': 'stix'})
    rc('text', usetex=True)
    rcParams['text.latex.preview'] = True
    rcParams['text.latex.preamble']=[r"\usepackage{amsmath, sourcesanspro}"]

from pathlib import Path
reproduced_results = Path("reproduced-results")

_fs = 20

_l_fs = 17
rc('legend', fontsize = _l_fs)

fig, axs = plt.subplots(2, 2, sharex = True, figsize = (9.5, 8))
fig.subplots_adjust(hspace = 0, wspace = 0.5)

_p_iso_colors = {'Yes': 'red', 'No': 'blue'}
_panel_label_pos = (0.865, 0.9)
_p_iso_lw = {'Yes': lw, 'No': lw * 0.525}
_g_label_pos = (0.6, 1.075)

'''
G = -3.6
'''
if True:
    '''
    E6 G = -3.6
    '''
    if True:
        _stencils_list = ['E6P4', 'E6P2']        
        for stencil_name in _stencils_list:
            _color = _p_iso_colors[E_sym_YN[E_sym_dict_m1[stencil_name]]]
            _lw = _p_iso_lw[E_sym_YN[E_sym_dict_m1[stencil_name]]]            
            L, G = 255, -3.6
            _str = 'G' + str(G) + 'LX' + str(L) + stencil_name
            axs[0, 0].errorbar(x = _r_range, 
                               y = _u_profile_ave[_str], 
                               yerr = _u_profile_err[_str], 
                               color = _color, 
                               label = E_str_dict_m1[stencil_name],
                               linewidth = _lw)

        lgnd_points, _lgnd_i = axs[0, 0].legend(frameon=False, loc = 'upper center'), 0

        for stencil_name in _stencils_list:
            _color = _p_iso_colors[E_sym_YN[E_sym_dict_m1[stencil_name]]]            
            lgnd_points.get_texts()[_lgnd_i].set_color(_color)
            _lgnd_i += 1

        if False:
            for tick in axs[0, 0].yaxis.get_major_ticks():
                tick.label.set_fontsize(_fs)
        else:
            SetAxTicksFont(axs[0, 0], _fs)

        axs[0, 0].set_xlim([(1 - 0.05) * 255/5, 255//2])
        axs[0, 0].set_ylabel('$\\langle u(r) \\rangle /c_s^2$', fontsize=_fs)
        _y_ticks_swap = np.arange(0, axs[0, 0].get_ylim()[1], 0.006)
        axs[0, 0].set_yticks(_y_ticks_swap)
        axs[0, 0].tick_params(axis="x", direction='in', length=8)
        axs[0, 0].text(_panel_label_pos[0], _panel_label_pos[1],
                       '$(a)$', transform = axs[0, 0].transAxes, fontsize = _fs)
        axs[0, 0].text(_g_label_pos[0], _g_label_pos[1],
                       '$\psi = \exp(-1/n), \quad Gc_s^2 = -3.6$', transform = axs[0, 0].transAxes, fontsize = _fs)        

    '''
    E8 G = -3.6
    '''
    if True:
        _stencils_list = ['E8P4', 'E8P2']
        for stencil_name in _stencils_list:
            _color = _p_iso_colors[E_sym_YN[E_sym_dict_m1[stencil_name]]]
            _lw = _p_iso_lw[E_sym_YN[E_sym_dict_m1[stencil_name]]]            
            L, G = 255, -3.6
            _str = 'G' + str(G) + 'LX' + str(L) + stencil_name
            axs[1, 0].errorbar(x = _r_range, 
                               y = _u_profile_ave[_str], 
                               yerr = _u_profile_err[_str], 
                               color = _color, 
                               label = E_str_dict_m1[stencil_name],
                               linewidth = _lw)

        lgnd_points, _lgnd_i = axs[1, 0].legend(frameon=False, loc = 'upper center'), 0

        for stencil_name in _stencils_list:
            _color = _p_iso_colors[E_sym_YN[E_sym_dict_m1[stencil_name]]]            
            lgnd_points.get_texts()[_lgnd_i].set_color(_color)
            _lgnd_i += 1

        if False:
            for tick in axs[1, 0].yaxis.get_major_ticks():
                tick.label.set_fontsize(_fs)
            for tick in axs[1, 0].xaxis.get_major_ticks():
                tick.label.set_fontsize(_fs)
        else:
            SetAxTicksFont(axs[1, 0], _fs)
            
        axs[1, 0].set_xlim([(1 - 0.05) * 255/5, 255//2])
        axs[1, 0].set_ylabel('$\\langle u(r) \\rangle /c_s^2$', fontsize=_fs)
        axs[1, 0].set_xlabel('$r$', fontsize=_fs)        
        _y_ticks_swap = np.arange(0, axs[1, 0].get_ylim()[1], 0.003)
        axs[1, 0].set_yticks(_y_ticks_swap)
        axs[1, 0].set_ylim(axs[1, 0].get_ylim())
        axs[1, 0].tick_params(axis="x", direction='in', length=8)
        axs[1, 0].text(_panel_label_pos[0], _panel_label_pos[1],
                       '$(b)$', transform = axs[1, 0].transAxes, fontsize = _fs)                
'''
G = -3.6
'''
if True:
    '''
    E10 G = -3.6
    '''
    _stencils_list = ['E10P4', 'E10P2']
    if True:
        for stencil_name in _stencils_list:
            _color = _p_iso_colors[E_sym_YN[E_sym_dict_m1[stencil_name]]]
            _lw = _p_iso_lw[E_sym_YN[E_sym_dict_m1[stencil_name]]]            
            L, G = 255, -3.6
            _str = 'G' + str(G) + 'LX' + str(L) + stencil_name
            axs[0, 1].errorbar(x = _r_range, 
                               y = _u_profile_ave[_str], 
                               yerr = _u_profile_err[_str], 
                               color = _color, 
                               label = E_str_dict_m1[stencil_name],
                               linewidth = _lw)

        lgnd_points, _lgnd_i = axs[0, 1].legend(frameon=False, loc = 'upper center'), 0

        for stencil_name in _stencils_list:
            _color = _p_iso_colors[E_sym_YN[E_sym_dict_m1[stencil_name]]]            
            lgnd_points.get_texts()[_lgnd_i].set_color(_color)
            _lgnd_i += 1        


        if False:
            for tick in axs[0, 1].yaxis.get_major_ticks():
                tick.label.set_fontsize(_fs)
        else:
            SetAxTicksFont(axs[0, 1], _fs)

        axs[0, 1].set_xlim([(1 - 0.05) * 255/5, 255//2])
        axs[0, 1].set_ylabel('$\\langle u(r) \\rangle /c_s^2$', fontsize=_fs)
        _y_ticks_swap = np.arange(0, axs[0, 1].get_ylim()[1], 0.003)
        axs[0, 1].set_yticks(_y_ticks_swap)
        axs[0, 1].tick_params(axis="x", direction='in', length=8)
        axs[0, 1].text(_panel_label_pos[0], _panel_label_pos[1],
                       '$(c)$', transform = axs[0, 1].transAxes, fontsize = _fs)

    '''
    E8 G = -3.6
    '''
    if True:
        _stencils_list = ['E12P4', 'E12P2']
        for stencil_name in _stencils_list:
            _color = _p_iso_colors[E_sym_YN[E_sym_dict_m1[stencil_name]]]
            _lw = _p_iso_lw[E_sym_YN[E_sym_dict_m1[stencil_name]]]            
            L, G = 255, -3.6
            _str = 'G' + str(G) + 'LX' + str(L) + stencil_name
            axs[1, 1].errorbar(x = _r_range, 
                               y = _u_profile_ave[_str], 
                               yerr = _u_profile_err[_str], 
                               color = _color, 
                               label = E_str_dict_m1[stencil_name],
                               linewidth = _lw)

        lgnd_points, _lgnd_i = axs[1, 1].legend(frameon=False, loc = 'upper center'), 0

        for stencil_name in _stencils_list:
            _color = _p_iso_colors[E_sym_YN[E_sym_dict_m1[stencil_name]]]            
            lgnd_points.get_texts()[_lgnd_i].set_color(_color)
            _lgnd_i += 1        

        if False:
            for tick in axs[1, 1].yaxis.get_major_ticks():
                tick.label.set_fontsize(_fs)
            for tick in axs[1, 1].xaxis.get_major_ticks():
                tick.label.set_fontsize(_fs)
        else:
            SetAxTicksFont(axs[1, 1], _fs)

        axs[1, 1].set_xlim([(1 - 0.05) * 255/5, 255//2])
        axs[1, 1].set_ylabel('$\\langle u(r) \\rangle /c_s^2$', fontsize=_fs)
        axs[1, 1].set_xlabel('$r$', fontsize=_fs)
        _y_ticks_swap = np.arange(0, axs[1, 1].get_ylim()[1], 0.002)
        axs[1, 1].set_yticks(_y_ticks_swap)
        axs[1, 1].set_ylim(axs[1, 1].get_ylim())
        axs[1, 1].tick_params(axis="x", direction='in', length=8)
        axs[1, 1].text(_panel_label_pos[0], _panel_label_pos[1],
                       '$(d)$', transform = axs[1, 1].transAxes, fontsize = _fs)
        
reproduced_figures = Path("reproduced-figures")
        
plt.savefig(reproduced_figures / 'figure_6_abcd.png',
            bbox_inches = 'tight', dpi = _dpi)
plt.close()
