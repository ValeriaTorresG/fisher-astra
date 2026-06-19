import argparse, csv, json, os
from pathlib import Path

import numpy as np

if 'MPLCONFIGDIR' not in os.environ:
    mplconfig = Path(os.environ.get('TMPDIR', '/tmp')) / 'matplotlib-cache'
    mplconfig.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(mplconfig)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
from matplotlib.ticker import FuncFormatter, LinearLocator


DEFAULT_FISHER_DIR = '/pscratch/sd/v/vtorresg/hod-astra-box500/power_spectra/fisher_cosmo'
DEFAULT_PARAMS_FILE = 'params/params_cosmo.csv'
CASES = ('matter', 'void', 'sheet', 'filament', 'knot', 'combined')
PARAM_COLUMNS = {'Omega_b': 'omega_b',
                 'omega_cdm': 'omega_cdm',
                 'n_s': 'n_s',
                 'sigma_8m': 'sigma8_m'}
PARAM_LABELS = {'Omega_b': r'$\omega_b$',
                'omega_cdm': r'$\omega_{\rm cdm}$',
                'n_s': r'$n_s$',
                'sigma_8m': r'$\sigma_8$'}
CASE_LABELS = {'matter': r'$P_{\rm matter}$',
               'void': r'$P_{\rm void}$',
               'sheet': r'$P_{\rm sheet}$',
               'filament': r'$P_{\rm filament}$',
               'knot': r'$P_{\rm knot}$',
               'combined': r'$P_{\rm combined}$'}
CASE_STYLES = {'matter': {'color': 'black', 'lw': 2.8},
               'void': {'color': '#3f3f95', 'lw': 2.2},
               'sheet': {'color': '#7bb576', 'lw': 2.2},
               'filament': {'color': '#e79a2d', 'lw': 2.2},
               'knot': {'color': '#df1f24', 'lw': 2.2},
               'combined': {'color': '#5aa4b9', 'lw': 2.8}}
DEFAULT_TICK_DECIMALS = {'Omega_b': 4,
                         'omega_cdm': 3,
                         'n_s': 3,
                         'sigma_8m': 3}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fisher-dir', type=str, default=DEFAULT_FISHER_DIR)
    parser.add_argument('--params-file', type=str, default=DEFAULT_PARAMS_FILE)
    parser.add_argument('--cosmo', type=str, default='c000')
    parser.add_argument('--params', nargs='+', default=['all'])
    parser.add_argument('--cases', nargs='+', default=['all'],
                        choices=list(CASES) + ['each'])
    parser.add_argument('--out', type=str, default='')
    parser.add_argument('--dpi', type=int, default=360)
    parser.add_argument('--range-sigma', type=float, default=2.0)
    parser.add_argument('--show', action='store_true')
    return parser.parse_args()


def setup_matplotlib():
    # plt.style.use('dark_background')
    matplotlib.rcParams['text.usetex'] = True


def normalize_cosmo(value):
    value = str(value).strip().lower()
    if value.startswith('abacus_cosm'):
        value = value.replace('abacus_cosm', 'c')
    if value.startswith('cosm'):
        value = value.replace('cosm', 'c', 1)
    if value.startswith('c'):
        return f'c{int(value[1:]):03d}'
    return f'c{int(value):03d}'


def normalize_cases(values):
    if 'all' in values or 'each' in values:
        return list(CASES)
    cases = []
    for value in values:
        if value not in cases:
            cases.append(value)
    return cases


def load_fisher_metadata(fisher_dir):
    path = Path(fisher_dir).expanduser().resolve() / 'fisher_metadata.json'
    if not path.is_file():
        return {}, None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f), path


def normalize_params(values, metadata):
    available = metadata.get('parameters') or list(PARAM_COLUMNS)
    if any(value == 'all' for value in values):
        return list(available)
    missing = [value for value in values if value not in available]
    if missing:
        raise RuntimeError(f'Parameters not available: {", ".join(missing)}. '
                           f'Available: {", ".join(available)}')
    return list(values)


def parameter_indices(parameters, metadata):
    available = metadata.get('parameters') or list(PARAM_COLUMNS)
    return [available.index(parameter) for parameter in parameters]


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


def load_fiducial_values(params_file, cosmo, parameters):
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
                    ax.plot(x, y, color=style['color'], lw=2.5,#style['lw'],
                            label=CASE_LABELS[case])
                    if case == 'combined':
                        ax.fill_between(x, 0, y, color=style['color'], alpha=0.08)
                ax.axvline(mu, color='black', lw=0.6, alpha=0.9)
                ax.set_xlim(*xlim)
                ax.set_ylim(0.0, 1.1)
                ax.set_yticks([])
                ax.grid(alpha=0.3, lw=0.3)
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
                                           alpha=1 if case == 'combined' else 1,
                                        #    lw=style['lw'],
                                           ls='-')
                # ax.scatter(mean2d[0], mean2d[1], marker='x', s=55,
                #            c='white', lw=1.5)
                ax.set_xlim(theta_fid[j] - range_sigma * limit_sigmas[j],
                            theta_fid[j] + range_sigma * limit_sigmas[j])
                ax.set_ylim(theta_fid[i] - range_sigma * limit_sigmas[i],
                            theta_fid[i] + range_sigma * limit_sigmas[i])
                ax.grid(alpha=0.3, lw=0.3)

            if row == npar - 1:
                ax.set_xlabel(labels[col], fontsize=17)
            if col == 0 and row > 0:
                ax.set_ylabel(labels[row], fontsize=17)
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
              linewidth=CASE_STYLES[case].get('lw', 2.5))
        for case in cases
    ]
    legend_labels = [CASE_LABELS[case] for case in cases]
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='upper left',
                   bbox_to_anchor=(0.76, 0.86), ncol=1, frameon=False,
                   fontsize=17, handlelength=2.7, handleheight=1.1,
                   labelspacing=0.8, handletextpad=0.9, borderaxespad=0.0)
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    return fig


def main():
    args = parse_args()
    setup_matplotlib()
    fisher_dir = Path(args.fisher_dir).expanduser().resolve()
    metadata, metadata_path = load_fisher_metadata(fisher_dir)
    cases = normalize_cases(args.cases)
    parameters = normalize_params(args.params, metadata)
    indices = parameter_indices(parameters, metadata)
    theta_fid = load_fiducial_values(Path(args.params_file).expanduser().resolve(),
                                     args.cosmo, parameters)
    covariances = load_parameter_covariances(fisher_dir, cases, indices)

    if args.out:
        out = Path(args.out).expanduser().resolve()
    else:
        out = fisher_dir / 'conf_ellip_envs.png'
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f'---> fisher directory: {fisher_dir}')
    if metadata_path:
        print(f'---> metadata: {metadata_path}')
    print(f'---> cases: {cases}')
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