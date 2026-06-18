import argparse, json, os, re
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('dark_background')


DEFAULT_PK_ROOT = '/pscratch/sd/v/vtorresg/hod-astra-box500/power_spectra'
FIELDS = ('matter', 'void', 'sheet', 'filament', 'knot')
ENVIRONMENTS = ('void', 'sheet', 'filament', 'knot')
PK_FILE_RE = re.compile(r'^pk_(?P<field_request>.+?)_(?P<tag>zone_.+)\.csv$')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='*')
    parser.add_argument('--pk-root', type=str, default=DEFAULT_PK_ROOT)
    parser.add_argument('--discover', action='store_true')
    parser.add_argument('--hod', nargs='+', default=['all'])
    parser.add_argument('--cosmo', nargs='+', default=['all'])
    parser.add_argument('--fields', nargs='+', default=['all'], choices=['all'] + list(FIELDS))
    parser.add_argument('--pk-kind', type=str, default='pk_used', choices=['pk_used', 'pk_raw'])
    parser.add_argument('--metadata', type=str, default='')
    parser.add_argument('--counts', nargs='*', default=[])
    parser.add_argument('--n-total', type=float, default=0.0)
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
        if value not in fields:
            fields.append(value)
    return fields


def parse_counts(values):
    counts = {}
    for item in values:
        name, raw_value = item.split('=', 1)
        name = name.strip().lower()
        if name not in FIELDS:
            raise ValueError(f'Unknown count field {name!r}. Expected one of {FIELDS}.')
        counts[name] = float(raw_value)
    return counts


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


def infer_metadata_path(csv_path):
    csv_path = Path(csv_path)
    info = parse_pk_name(csv_path)
    if info is None:
        return None
    return csv_path.parent / 'meta' / f"run_metadata_pk_{info['field_request']}_{info['tag']}.json"


def load_metadata(csv_path, explicit_metadata=''):
    meta_path = Path(explicit_metadata).expanduser().resolve() if explicit_metadata else infer_metadata_path(csv_path)
    if meta_path is None or not meta_path.is_file():
        return None, meta_path
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f), meta_path


def counts_from_metadata(metadata):
    if not metadata:
        return {}, 0.0

    field_info = metadata.get('field_info') or {}
    raw_selection = metadata.get('raw_selection') or {}
    n_total = float(raw_selection.get('n_selected') or 0.0)

    counts = {}
    for field in FIELDS:
        info = field_info.get(field) or {}
        if 'n_selected' in info:
            counts[field] = float(info['n_selected'])

    if 'matter' in counts and n_total <= 0.0:
        n_total = counts['matter']

    astra_info = metadata.get('astra_classification') or {}
    class_counts = astra_info.get('class_counts') or {}
    for field in ENVIRONMENTS:
        if field not in counts and field in class_counts:
            counts[field] = float(class_counts[field])

    return counts, n_total


def load_pk_csv(path):
    data = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding=None)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)
    return data


def get_column(data, name, path):
    if name not in data.dtype.names:
        raise KeyError(f'Missing column {name!r} in {path}')
    return np.asarray(data[name], dtype=np.float64)


def environment_fraction(field, counts, n_total):
    if field == 'matter':
        return 1.0
    n_field = counts.get(field)
    return float(n_field) / float(n_total)


def make_output_path(csv_path, outdir):
    csv_path = Path(csv_path)
    if outdir:
        plot_dir = Path(outdir).expanduser().resolve()
    else:
        plot_dir = csv_path.parent / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir / f'pk_norm_{csv_path.stem}.png'


def plot_normalized_power(csv_path, fields, pk_kind, metadata_path, cli_counts, cli_n_total,
                          outdir, show=False):
    metadata, inferred_meta_path = load_metadata(csv_path, metadata_path)
    meta_counts, meta_n_total = counts_from_metadata(metadata)
    counts = {**meta_counts, **cli_counts}
    n_total = float(cli_n_total) if cli_n_total > 0.0 else float(meta_n_total)

    data = load_pk_csv(csv_path)
    k = get_column(data, 'k_h_mpc', csv_path)

    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    fractions = {}

    for field in fields:
        column = f'{pk_kind}_{field}'
        pk = get_column(data, column, csv_path)
        fraction = environment_fraction(field, counts, n_total)
        pk_norm = pk if field == 'matter' else (fraction ** 2) * pk
        fractions[field] = fraction

        mask = (k > 0.0) & np.isfinite(k) & np.isfinite(pk_norm) & (pk_norm > 0.0)
        if np.any(mask):
            if field == 'matter':
                label = 'matter'
            else:
                label = rf'{field}: $f_\alpha={fraction:.4g}$'
            ax.loglog(k[mask], pk_norm[mask], lw=1.7, label=label)

    ax.set_xlabel(r'$k\ [h\,\mathrm{Mpc}^{-1}]$')
    ax.set_ylabel(r'$f_\alpha^2 P_{\alpha\alpha}(k)\ [(\mathrm{Mpc}/h)^3]$')
    ax.grid(alpha=0.4, lw=0.2, which='both')
    ax.legend()

    title_bits = [Path(csv_path).stem, pk_kind]
    if inferred_meta_path and Path(inferred_meta_path).is_file():
        title_bits.append('fractions from metadata')
    ax.set_title(' | '.join(title_bits))

    fig.tight_layout()
    output_path = make_output_path(csv_path, outdir)
    fig.savefig(output_path, dpi=220)
    print(f'---> saved plot to {output_path}')
    if show:
        plt.show()
    plt.close(fig)

    return {'input': str(csv_path),
            'metadata': str(inferred_meta_path) if inferred_meta_path else None,
            'output': str(output_path),
            'n_total': n_total,
            'fractions': fractions}


def main():
    args = parse_args()
    fields = normalize_fields(args.fields)
    counts = parse_counts(args.counts)

    paths = [Path(path).expanduser().resolve() for path in args.input]
    if args.discover:
        paths.extend(discover_pk_files(args.pk_root, args.hod, args.cosmo))
    paths = sorted(dict.fromkeys(paths))

    for path in paths:
        try:
            result = plot_normalized_power(path, fields, args.pk_kind, args.metadata,
                                           counts, args.n_total, args.outdir, args.show)
        except Exception as exc:
            if args.skip_missing:
                print(f'---> skipped {path}: {exc}')
                continue
            raise
        print(f"---> wrote: {result['output']}")
        print(f"---> N_total = {result['n_total']:.0f}")
        for field in fields:
            fraction = result['fractions'][field]
            print(f'------------- f_{field} = {fraction:.8g}')


if __name__ == '__main__':
    main()