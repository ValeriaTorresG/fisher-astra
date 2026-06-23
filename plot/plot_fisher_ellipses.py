import argparse, csv, json, os
from pathlib import Path
import numpy as np

if 'MPLCONFIGDIR' not in os.environ:
    mplconfig = Path(os.environ.get('TMPDIR', '/tmp')) / 'matplotlib-cache'
    mplconfig.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(mplconfig)
if 'XDG_CACHE_HOME' not in os.environ:
    os.environ['XDG_CACHE_HOME'] = os.environ['MPLCONFIGDIR']
fontconfig_cache = Path(os.environ['XDG_CACHE_HOME']) / 'fontconfig'
fontconfig_cache.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import LinearLocator
matplotlib.rcParams['text.usetex'] = True
# plt.style.use('dark_background')


DEFAULT_FISHER_DIR = '/pscratch/sd/v/vtorresg/hod-astra-box500/power_spectra/fisher_cosmo_hod'
DEFAULT_PARAMS_FILE = 'params/params_cosmo.csv'
DEFAULT_HOD_PARAMS_FILE = 'params/params_hods_c000_ph000_seed0.csv'
CASES = ('matter', 'void', 'random_void', 'sheet', 'filament', 'knot',
         'combined', #'combined_random_void',
         'combined_all_voids')
CASE_GROUPS = {
    'galaxy_envs': ('matter', 'void', 'sheet', 'filament', 'knot', 'combined'),
    'random_void': ('matter', 'void', 'random_void', 'combined',
                    'combined_random_void', 'combined_all_voids'),
    'combined_only': ('matter', 'combined', 'combined_all_voids'),
    'all_observables': CASES}
CASE_GROUP_ALIASES = {'galaxy': 'galaxy_envs',
                      'gal': 'galaxy_envs',
                      'envs': 'galaxy_envs',
                      'environments': 'galaxy_envs',
                      'random': 'random_void',
                      'rand_void': 'random_void',
                      'randvoid': 'random_void',
                      'combined': 'combined_only',
                      'combineds': 'combined_only',
                      'comb': 'combined_only',
                      'observables': 'all_observables',
                      'all_obs': 'all_observables',
                      'all_cases': 'all_observables'}
CASE_GROUP_CHOICES = list(CASE_GROUPS) + sorted(CASE_GROUP_ALIASES) + ['all']
PARAM_COLUMNS = {'Omega_b': 'omega_b',
                 'omega_cdm': 'omega_cdm',
                 'n_s': 'n_s',
                 'sigma_8m': 'sigma8_m'}
COSMO_PARAMS = tuple(PARAM_COLUMNS)
HOD_PARAMS = ('LOGM_CUT', 'LOGM1', 'SIGMA', 'ALPHA', 'KAPPA')
DEFAULT_PLOT_PARAMS = {'cosmo': COSMO_PARAMS,
                       'hod': HOD_PARAMS}
PARAM_LABELS = {'Omega_b': r'$\omega_b$',
                'omega_cdm': r'$\omega_{\rm cdm}$',
                'n_s': r'$n_s$',
                'sigma_8m': r'$\sigma_8$',
                'LOGM_CUT': r'$\log M_{\rm cut}$',
                'LOGM1': r'$\log M_1$',
                'SIGMA': r'$\sigma$',
                'ALPHA': r'$\alpha$',
                'KAPPA': r'$\kappa$'}
CASE_LABELS = {'matter': r'$P_{\rm \, all}^{\rm \, gal}(k)$',
               'void': r'$P_{\rm \, void}^{\rm \, gal}(k)$',
               'sheet': r'$P_{\rm \, sheet}^{\rm \, gal}(k)$',
               'filament': r'$P_{\rm \, filament}^{\rm \, gal}(k)$',
               'knot': r'$P_{\rm \, knot}^{\rm \, gal}(k)$',
               'combined': r'$\mathbf{d}_{\rm \, combined}^{\rm \, gal}$',
               'random_void': r'$P_{\rm \, void}^{\rm \, rand}(k)$',
               'combined_random_void': r'$\mathbf{d}_{\rm \, combined}^{\rm \, hybrid}$',
               'combined_all_voids': r'$\mathbf{d}_{\rm \, combined}^{\rm \, gal \, + \, rand\ void}$'}

theme_color = 'black'
CASE_STYLES = {'matter': {'color': theme_color, 'lw': 2.},
               'void': {'color': '#3f3f95', 'lw': 2.},
               'sheet': {'color': '#7bb576', 'lw': 2.},
               'filament': {'color': '#e79a2d', 'lw': 2.},
               'knot': {'color': '#df1f24', 'lw': 2.},
               'combined': {'color': '#5aa4b9', 'lw': 2.},
               'random_void': {'color': '#3f3f95', 'lw': 2., 'ls': '--'},
               'combined_random_void': {'color': '#17becf', 'lw': 2.},
               'combined_all_voids': {'color': 'navy', 'lw': 2., 'ls': '-'}}
DEFAULT_TICK_DECIMALS = {'Omega_b': 4,
                         'omega_cdm': 3,
                         'n_s': 3,
                         'sigma_8m': 3,
                         'LOGM_CUT': 3,
                         'LOGM1': 3,
                         'SIGMA': 3,
                         'ALPHA': 3,
                         'KAPPA': 3}
PARAM_ALIASES = {'omega_b': 'Omega_b',
                 'Omega_b': 'Omega_b',
                 'omega_cdm': 'omega_cdm',
                 'n_s': 'n_s',
                 'sigma_8m': 'sigma_8m',
                 'sigma8_m': 'sigma_8m'}
PARAM_ALIASES.update({parameter.lower(): parameter for parameter in HOD_PARAMS})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fisher-dir', type=str, default=DEFAULT_FISHER_DIR)
    parser.add_argument('--params-file', type=str, default=DEFAULT_PARAMS_FILE)
    parser.add_argument('--hod-params-file', type=str, default=DEFAULT_HOD_PARAMS_FILE)
    parser.add_argument('--cosmo', type=str, default='c000')
    parser.add_argument('--fiducial-hod', type=str, default='')
    parser.add_argument('--params', nargs='+', default=['all'])
    parser.add_argument('--cases', nargs='+', default=None, choices=list(CASES) + ['all', 'each'])
    parser.add_argument('--case-groups', nargs='+', default=['all'], choices=CASE_GROUP_CHOICES)
    parser.add_argument('--all-group-params', action='store_true')
    parser.add_argument('--out', type=str, default='')
    parser.add_argument('--dpi', type=int, default=360)
    parser.add_argument('--range-sigma', type=float, default=2.0)
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


def normalize_cosmo(value):
    value = str(value).strip().lower()
    if value.startswith('abacus_cosm'):
        value = value.replace('abacus_cosm', 'c')
    if value.startswith('cosm'):
        value = value.replace('cosm', 'c', 1)
    if value.startswith('c'):
        return f'c{int(value[1:]):03d}'
    return f'c{int(value):03d}'


def normalize_hod(value):
    value = str(value).strip().lower()
    if value.startswith('hod'):
        value = value[3:]
    return f'hod{int(value):03d}'


def normalize_cases(values):
    if 'all' in values or 'each' in values:
        return list(CASES)
    cases = []
    for value in values:
        if value not in cases:
            cases.append(value)
    return cases


def normalize_case_groups(values):
    if 'all' in values:
        return list(CASE_GROUPS)
    groups = []
    for value in values:
        key = CASE_GROUP_ALIASES.get(str(value).strip().lower(),
                                     str(value).strip().lower())
        if key not in groups:
            groups.append(key)
    return groups


def case_specs_from_args(args):
    if args.cases:
        return [('custom', normalize_cases(args.cases))]
    return [(group, list(CASE_GROUPS[group]))
            for group in normalize_case_groups(args.case_groups)]


def load_fisher_metadata(fisher_dir):
    path = Path(fisher_dir).expanduser().resolve() / 'fisher_metadata.json'
    if not path.is_file():
        return {}, None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f), path


def normalize_params(values, metadata):
    available = metadata.get('parameters') or list(COSMO_PARAMS)
    if any(value == 'all' for value in values):
        return list(available)
    params = []
    for value in values:
        token = str(value).strip()
        token_lower = token.lower()
        if token_lower in ('cosmo', 'cosmology', 'cosmological'):
            expanded = [parameter for parameter in COSMO_PARAMS
                        if parameter in available]
        elif token_lower == 'hod':
            expanded = [parameter for parameter in HOD_PARAMS
                        if parameter in available]
        else:
            expanded = [PARAM_ALIASES.get(token, PARAM_ALIASES.get(token_lower, token))]
        for parameter in expanded:
            if parameter not in available:
                raise RuntimeError(f'Parameter {parameter} is not available. '
                                   f'Available: {", ".join(available)}')
            if parameter not in params:
                params.append(parameter)
    return params


def parameter_indices(parameters, metadata):
    available = metadata.get('parameters') or list(COSMO_PARAMS)
    return [available.index(parameter) for parameter in parameters]


def available_parameters_from_group(metadata, group):
    groups = metadata.get('parameter_groups') or {}
    if group in groups and groups[group]:
        return list(groups[group])
    available = metadata.get('parameters') or list(COSMO_PARAMS)
    if group == 'cosmo':
        return [parameter for parameter in COSMO_PARAMS if parameter in available]
    if group == 'hod':
        return [parameter for parameter in HOD_PARAMS if parameter in available]
    raise ValueError(f'Unknown parameter group: {group}')


def parameters_from_group(metadata, group, all_group_params=False):
    available = available_parameters_from_group(metadata, group)
    if all_group_params:
        return available
    defaults = [parameter for parameter in DEFAULT_PLOT_PARAMS[group]
                if parameter in available]
    return defaults if defaults else available[:3]


def plot_specs_from_args(values, metadata, all_group_params=False):
    tokens = [str(value).strip().lower() for value in values]
    if any(token == 'all' for token in tokens):
        specs = []
        for group in ('cosmo', 'hod'):
            params = parameters_from_group(metadata, group, all_group_params)
            if params:
                specs.append((group, params))
        if specs:
            return specs
        return [('all', normalize_params(values, metadata))]

    if tokens and all(token in ('cosmo', 'cosmology', 'cosmological', 'hod')
                      for token in tokens):
        specs = []
        for token in tokens:
            group = 'cosmo' if token in ('cosmo', 'cosmology', 'cosmological') else 'hod'
            params = parameters_from_group(metadata, group, all_group_params)
            if params:
                specs.append((group, params))
        return specs

    return [('custom', normalize_params(values, metadata))]


def load_parameter_covariances(fisher_dir, cases, indices):
    covariances = {}
    fisher_dir = Path(fisher_dir).expanduser().resolve()
    for case in cases:
        path = fisher_dir / f'parameter_cov_{case}.npy'
        if case == 'combined' and not path.is_file():
            path = fisher_dir / 'parameter_cov_all.npy'
        if not path.is_file():
            raise RuntimeError(f'Missing parameter covariance file: {path}')
        cov = np.load(path)
        covariances[case] = cov[np.ix_(indices, indices)]
    return covariances


def load_cosmo_fiducial_values(params_file, cosmo, parameters):
    target = normalize_cosmo(cosmo)
    with open(params_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for raw_row in reader:
            row = {key.strip(): str(value).strip() if value is not None else ''
                   for key, value in raw_row.items() if key is not None}
            root = row.get('root', '')
            if root and normalize_cosmo(root) == target:
                values = []
                for parameter in parameters:
                    column = PARAM_COLUMNS.get(parameter, parameter)
                    if column not in row:
                        raise RuntimeError(f'Missing fiducial column {column} for {parameter}.')
                    values.append(float(row[column]))
                return np.asarray(values, dtype=np.float64)
    raise RuntimeError(f'Fiducial cosmology {cosmo} not found in {params_file}.')


def fiducial_hod_from_metadata(metadata, requested):
    if requested:
        return normalize_hod(requested)
    fiducials = metadata.get('fiducials') or {}
    hod_info = fiducials.get('hod') or {}
    name = hod_info.get('name')
    if name:
        return normalize_hod(name)
    return 'hod000'


def load_hod_fiducial_values(params_file, fiducial_hod, parameters, metadata):
    fiducial_hod = normalize_hod(fiducial_hod)
    fiducials = metadata.get('fiducials') or {}
    hod_info = fiducials.get('hod') or {}
    metadata_values = hod_info.get('parameters') or {}
    if metadata_values and all(parameter in metadata_values for parameter in parameters):
        return np.asarray([float(metadata_values[parameter])
                           for parameter in parameters],
                          dtype=np.float64)

    with open(params_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for raw_row in reader:
            row = {key.strip(): str(value).strip() if value is not None else ''
                   for key, value in raw_row.items() if key is not None}
            hod_value = row.get('hod') or row.get('filename', '').replace('.fits', '')
            if hod_value and normalize_hod(hod_value) == fiducial_hod:
                values = []
                for parameter in parameters:
                    if parameter not in row:
                        raise RuntimeError(f'Missing HOD fiducial column {parameter}.')
                    values.append(float(row[parameter]))
                return np.asarray(values, dtype=np.float64)
    raise RuntimeError(f'Fiducial HOD {fiducial_hod} not found in {params_file}.')


def load_fiducial_values(params_file, hod_params_file, cosmo, fiducial_hod,
                         parameters, metadata):
    values = []
    cosmo_params = [parameter for parameter in parameters if parameter in COSMO_PARAMS]
    hod_params = [parameter for parameter in parameters if parameter in HOD_PARAMS]
    cosmo_values = {}
    hod_values = {}
    if cosmo_params:
        loaded = load_cosmo_fiducial_values(params_file, cosmo, cosmo_params)
        cosmo_values = dict(zip(cosmo_params, loaded))
    if hod_params:
        loaded = load_hod_fiducial_values(
            hod_params_file, fiducial_hod, hod_params, metadata)
        hod_values = dict(zip(hod_params, loaded))

    for parameter in parameters:
        if parameter in cosmo_values:
            values.append(cosmo_values[parameter])
        elif parameter in hod_values:
            values.append(hod_values[parameter])
        else:
            raise RuntimeError(f'No fiducial value loader for parameter {parameter}.')
    return np.asarray(values, dtype=np.float64)


def tick_decimals_for_params(parameters):
    return {i: DEFAULT_TICK_DECIMALS.get(parameter, 3)
            for i, parameter in enumerate(parameters)}


def latex_formatter_hide_edges(ndec):
    def formatter(x, pos):
        ticks = plt.gca().get_xticks()
        if pos == 0 or pos == len(ticks) - 1:
            return ''
        return rf'$\mathdefault{{{x:.{ndec}f}}}$'
    return FuncFormatter(formatter)


def covariance_sigmas(cov):
    diag = np.diag(cov)
    sigmas = np.full(diag.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(diag) & (diag > 0.0)
    sigmas[mask] = np.sqrt(diag[mask])
    return sigmas


def max_sigmas(covariances):
    all_sigmas = np.vstack([covariance_sigmas(cov) for cov in covariances.values()])
    sigmas = np.nanmax(all_sigmas, axis=0)
    if np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
        raise RuntimeError('Cannot determine finite positive parameter errors for plot limits.')
    return sigmas


def confidence_scale_2d(level):
    return np.sqrt(-2.0 * np.log(1.0 - float(level)))


def add_confidence_ellipse(ax, cov2d, mean2d, level, edgecolor, facecolor='none',
                           alpha=1.0, lw=2.5, ls='-'):
    cov2d = np.asarray(cov2d, dtype=np.float64)
    mean2d = np.asarray(mean2d, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov2d)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    scale = confidence_scale_2d(level)
    width, height = 2.0 * scale * np.sqrt(eigvals)
    ellipse = Ellipse(xy=mean2d, width=width, height=height, angle=angle,
                      facecolor=facecolor, edgecolor=edgecolor, alpha=alpha,
                      lw=lw, ls=ls)
    ax.add_patch(ellipse)
    return ellipse


def plot_triangle(covariances, theta_fid, parameters, cases, out, dpi, range_sigma):
    labels = [PARAM_LABELS.get(parameter, parameter.replace('_', r'\_'))
              for parameter in parameters]
    npar = len(parameters)
    tick_decimals = tick_decimals_for_params(parameters)
    limit_sigmas = max_sigmas(covariances)

    fig, axes = plt.subplots(npar, npar, figsize=(3.2 * npar, 3.2 * npar),
                             sharex='col', sharey=False)
    if npar == 1:
        axes = np.array([[axes]])

    for row in range(npar):
        for col in range(npar):
            ax = axes[row, col]
            if col > row:
                ax.axis('off')
                continue

            if row == col:
                mu = theta_fid[row]
                xlim = (mu - range_sigma * limit_sigmas[row],
                        mu + range_sigma * limit_sigmas[row])
                x = np.linspace(xlim[0], xlim[1], 600)
                for case in cases:
                    sig = covariance_sigmas(covariances[case])[row]
                    if not np.isfinite(sig) or sig <= 0.0:
                        continue
                    style = CASE_STYLES[case]
                    y = np.exp(-0.5 * ((x - mu) / sig) ** 2)
                    y /= y.max()
                    ax.plot(x, y, color=style['color'], lw=2.5,
                            ls=style.get('ls', '-'),
                            label=CASE_LABELS[case])
                    if case in ('combined', 'combined_random_void', 'combined_all_voids'):
                        ax.fill_between(x, 0, y, color=style['color'], alpha=0.08)
                ax.axvline(mu, color=theme_color, lw=0.6, alpha=0.9)
                ax.set_xlim(*xlim)
                ax.set_ylim(0.0, 1.1)
                ax.set_yticks([])
                ax.grid(alpha=0.2, lw=0.3)
            else:
                i = row
                j = col
                mean2d = theta_fid[[j, i]]
                for case in cases:
                    style = CASE_STYLES[case]
                    cov2d = covariances[case][np.ix_([j, i], [j, i])]
                    add_confidence_ellipse(ax, cov2d, mean2d, level=0.6827,
                                           edgecolor=style['color'],
                                        #    facecolor=style['color'],
                                           alpha=1 if case in ('combined', 'combined_random_void',
                                                               'combined_all_voids') else 1,
                                        #    lw=style['lw'],
                                           ls=style.get('ls', '-'))
                ax.scatter(mean2d[0], mean2d[1], marker='x', s=22,
                           c='black', linewidths=0.9, zorder=8)
                ax.set_xlim(theta_fid[j] - range_sigma * limit_sigmas[j],
                            theta_fid[j] + range_sigma * limit_sigmas[j])
                ax.set_ylim(theta_fid[i] - range_sigma * limit_sigmas[i],
                            theta_fid[i] + range_sigma * limit_sigmas[i])
                ax.grid(alpha=1.0, lw=0.3)

            if row == npar - 1:
                ax.set_xlabel(labels[col], fontsize=20)
            if col == 0 and row > 0:
                ax.set_ylabel(labels[row], fontsize=20)
            ax.tick_params(axis='both', labelsize=14)

    for row in range(npar):
        for col in range(npar):
            ax = axes[row, col]
            if col > row:
                continue
            ax.xaxis.set_major_locator(LinearLocator(5))
            ax.xaxis.set_major_formatter(
                latex_formatter_hide_edges(tick_decimals[col]))
            if row == col:
                ax.set_yticks([])
            else:
                ax.yaxis.set_major_locator(LinearLocator(5))
                ax.yaxis.set_major_formatter(
                    latex_formatter_hide_edges(tick_decimals[row]))
            if row < npar - 1:
                ax.tick_params(labelbottom=False)
            if col > 0:
                ax.tick_params(labelleft=False)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.0, hspace=0.0, top=0.96)

    legend_handles = [
        Patch(facecolor='none',
              edgecolor=CASE_STYLES[case]['color'],
              linewidth=CASE_STYLES[case].get('lw', 2.5),
              linestyle=CASE_STYLES[case].get('ls', '-'))
        for case in cases
    ]
    legend_labels = [CASE_LABELS[case] for case in cases]
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='upper left',
                   bbox_to_anchor=(0.72, 0.93), ncol=1, frameon=False,
                   fontsize=20, handlelength=2.7, handleheight=1.1,
                   labelspacing=0.8, handletextpad=0.9, borderaxespad=0.0)
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    return fig


def output_path_for_spec(args_out, fisher_dir, case_group, spec_name, multiple):
    if args_out:
        out = Path(args_out).expanduser().resolve()
        if multiple:
            suffix = out.suffix or '.png'
            return out.with_name(f'{out.stem}_{case_group}_{spec_name}{suffix}')
        return out
    return fisher_dir / f'conf_ellip_{case_group}_{spec_name}.png'


def main():
    args = parse_args()
    fisher_dir = Path(args.fisher_dir).expanduser().resolve()
    metadata, metadata_path = load_fisher_metadata(fisher_dir)
    case_specs = case_specs_from_args(args)
    specs = plot_specs_from_args(args.params, metadata, args.all_group_params)
    fiducial_hod = fiducial_hod_from_metadata(metadata, args.fiducial_hod)

    print(f'---> fisher directory: {fisher_dir}')
    if metadata_path:
        print(f'---> metadata: {metadata_path}')
    print(f'---> case groups: {case_specs}')
    print(f'---> fiducial HOD: {fiducial_hod}')

    multiple = len(specs) * len(case_specs) > 1
    for case_group, cases in case_specs:
        for spec_name, parameters in specs:
            if len(parameters) < 1:
                continue
            indices = parameter_indices(parameters, metadata)
            theta_fid = load_fiducial_values(
                Path(args.params_file).expanduser().resolve(),
                Path(args.hod_params_file).expanduser().resolve(),
                args.cosmo, fiducial_hod, parameters, metadata)
            covariances = load_parameter_covariances(fisher_dir, cases, indices)
            out = output_path_for_spec(args.out, fisher_dir, case_group,
                                       spec_name, multiple)
            out.parent.mkdir(parents=True, exist_ok=True)

            print(f'---> case group: {case_group}')
            print(f'---> cases: {cases}')
            print(f'---> plot group: {spec_name}')
            print(f'---> parameters: {parameters}')
            print(f'---> fiducial theta: {theta_fid}')
            fig = plot_triangle(covariances, theta_fid, parameters, cases,
                                out, args.dpi, args.range_sigma)
            print(f'---> wrote: {out}')
            if args.show:
                plt.show()
            plt.close(fig)


if __name__ == '__main__':
    main()