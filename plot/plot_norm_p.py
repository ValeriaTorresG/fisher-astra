import argparse, os, re
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
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
plt.style.use('dark_background')


DEFAULT_PK_ROOT = '/pscratch/sd/v/vtorresg/hod-astra-box500/power_spectra'
FIELDS = ('matter', 'void', 'sheet', 'filament', 'knot', 'random_void')
FIELD_ALIASES = {'random_voids': 'random_void',
                 'rand_void': 'random_void',
                 'randvoid': 'random_void'}
FIELD_CHOICES = ['all'] + list(FIELDS) + sorted(FIELD_ALIASES)
PK_FILE_RE = re.compile(r'^pk_(?P<field_request>.+?)_(?P<tag>zone_.+)\.csv$')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='*')
    parser.add_argument('--pk-root', type=str, default=DEFAULT_PK_ROOT)
    parser.add_argument('--discover', action='store_true')
    parser.add_argument('--hod', nargs='+', default=['all'])
    parser.add_argument('--cosmo', nargs='+', default=['all'])
    parser.add_argument('--fields', nargs='+', default=['all'], choices=FIELD_CHOICES)
    parser.add_argument('--pk-kind', type=str, default='pk_used', choices=['pk_used', 'pk_raw'])
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--skip-missing', action='store_true')
    return parser.parse_args()


def normalize_cosmo(value):
    value = str(value).strip().lower()
    if value in ('all', '*'):
        return 'all'
    if value.startswith('c'):
        return f'c{int(value[1:]):03d}'
    return f'c{int(value):03d}'


def normalize_hod(value):
    value = str(value).strip().lower()
    if value in ('all', '*'):
        return 'all'
    if value.startswith('hod'):
        value = value[3:]
    return f'hod{int(value):03d}'


def normalize_fields(values):
    if any(value == 'all' for value in values):
        return list(FIELDS)
    fields = []
    for value in values:
        value = FIELD_ALIASES.get(str(value).strip().lower(), value)
        if value not in fields:
            fields.append(value)
    return fields


def parse_pk_name(path):
    match = PK_FILE_RE.match(Path(path).name)
    if not match:
        return None
    info = match.groupdict()
    tag = info['tag']
    parts = tag.split('_')
    cosmo = next((part for part in parts if re.fullmatch(r'c\d{3}', part)), 'unknown')
    hod_part = next((part for part in parts if part.startswith('hod')), 'hodunknown')
    hod = normalize_hod(hod_part[3:]) if hod_part != 'hodunknown' else hod_part
    return {'field_request': info['field_request'], 'tag': tag, 'cosmo': cosmo, 'hod': hod}


def discover_pk_files(pk_root, hod_values, cosmo_values):
    hod_filter = {normalize_hod(value) for value in hod_values}
    cosmo_filter = {normalize_cosmo(value) for value in cosmo_values}
    paths = []
    for path in sorted(Path(pk_root).expanduser().resolve().rglob('pk_*.csv')):
        info = parse_pk_name(path)
        if info is None:
            continue
        if info['field_request'] != 'all':
            continue
        if 'all' not in hod_filter and info['hod'] not in hod_filter:
            continue
        if 'all' not in cosmo_filter and info['cosmo'] not in cosmo_filter:
            continue
        paths.append(path)
    return paths


def load_pk_csv(path):
    data = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding=None)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def get_column(data, name, path):
    if name not in data.dtype.names:
        raise KeyError(f'Missing column {name!r} in {path}')
    return np.asarray(data[name], dtype=np.float64)


def make_output_path(csv_path, outdir):
    csv_path = Path(csv_path)
    if outdir:
        plot_dir = Path(outdir).expanduser().resolve()
    else:
        plot_dir = csv_path.parent / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir / f'pk_{csv_path.stem}.png'


def plot_power(csv_path, fields, pk_kind, outdir, show=False):
    data = load_pk_csv(csv_path)
    k = get_column(data, 'k_h_mpc', csv_path)

    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    plotted_fields = []

    for field in fields:
        column = f'{pk_kind}_{field}'
        pk = get_column(data, column, csv_path)
        mask = (k > 0.0) & np.isfinite(k) & np.isfinite(pk) & (pk > 0.0)
        if np.any(mask):
            ax.loglog(k[mask], pk[mask], lw=1.7, label=field)
            plotted_fields.append(field)

    if not plotted_fields:
        raise RuntimeError(f'No finite positive P(k) values found in {csv_path}.')

    ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$P(k)\ [(\mathrm{Mpc}/h)^3]$')
    ax.grid(alpha=0.4, lw=0.2, which='both')
    ax.legend()
    ax.set_title(' | '.join([Path(csv_path).stem, pk_kind, 'raw P(k)']))

    fig.tight_layout()
    output_path = make_output_path(csv_path, outdir)
    fig.savefig(output_path, dpi=220)
    print(f'---> saved plot to {output_path}')
    if show:
        plt.show()
    plt.close(fig)

    return {'input': str(csv_path),
            'output': str(output_path),
            'fields': plotted_fields}


def main():
    args = parse_args()
    fields = normalize_fields(args.fields)

    paths = [Path(path).expanduser().resolve() for path in args.input]
    if args.discover:
        paths.extend(discover_pk_files(args.pk_root, args.hod, args.cosmo))
    paths = sorted(dict.fromkeys(paths))

    for path in paths:
        try:
            result = plot_power(path, fields, args.pk_kind, args.outdir, args.show)
        except Exception as exc:
            if args.skip_missing:
                print(f'---> skipped {path}: {exc}')
                continue
            raise
        print(f"---> wrote: {result['output']}")
        print(f"---> plotted fields: {', '.join(result['fields'])}")


if __name__ == '__main__':
    main()