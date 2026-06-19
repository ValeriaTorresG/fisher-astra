import argparse, csv, json
from pathlib import Path

import numpy as np


DEFAULT_FISHER_DIR = '/pscratch/sd/v/vtorresg/hod-astra-box500/power_spectra/fisher_cosmo'
ENV_CASES = ('matter', 'void', 'sheet', 'filament', 'knot', 'combined')
DEFAULT_PARAMS = ('Omega_b', 'omega_cdm', 'n_s', 'sigma_8m')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fisher-dir', type=str, default=DEFAULT_FISHER_DIR)
    parser.add_argument('--cases', nargs='+', default=['all'], choices=list(ENV_CASES) + ['each'])
    parser.add_argument('--params', nargs='+', default=['all'])
    parser.add_argument('--out', type=str, default='')
    return parser.parse_args()


def normalize_cases(values):
    if 'all' in values or 'each' in values:
        return list(ENV_CASES)
    cases = []
    for value in values:
        if value not in cases:
            cases.append(value)
    return cases


def load_metadata(fisher_dir):
    path = Path(fisher_dir).expanduser().resolve() / 'fisher_metadata.json'
    if not path.is_file():
        return {}, None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f), path


def normalize_params(values, metadata):
    available = metadata.get('parameters') or list(DEFAULT_PARAMS)
    if any(value == 'all' for value in values):
        return list(available)
    return list(values)


def parameter_indices(parameters, metadata):
    available = metadata.get('parameters') or list(DEFAULT_PARAMS)
    return [available.index(parameter) for parameter in parameters]


def load_parameter_covariance(fisher_dir, case):
    path = Path(fisher_dir).expanduser().resolve() / f'parameter_cov_{case}.npy'
    if case == 'combined' and not path.is_file():
        path = Path(fisher_dir).expanduser().resolve() / 'parameter_cov_all.npy'
    if not path.is_file():
        raise RuntimeError(f'Missing parameter covariance file: {path}')
    cov = np.load(path)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise RuntimeError(f'Invalid parameter covariance shape in {path}: {cov.shape}')
    return cov, path


def marginalized_errors(cov, indices):
    subcov = cov[np.ix_(indices, indices)]
    diag = np.diag(subcov)
    sigma = np.full(diag.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(diag) & (diag >= 0.0)
    sigma[mask] = np.sqrt(diag[mask])
    return sigma


def print_table(rows, parameters):
    headers = ['case'] + [f'sigma_marg({parameter})' for parameter in parameters]
    widths = [max(len(headers[0]), max(len(row[0]) for row in rows))]
    for i, header in enumerate(headers[1:], start=1):
        widths.append(max(len(header), max(len(f'{row[i]:.8g}') for row in rows)))

    print('  '.join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print('  '.join('-' * width for width in widths))
    for row in rows:
        values = [row[0]] + [f'{value:.8g}' for value in row[1:]]
        print('  '.join(value.ljust(widths[i]) for i, value in enumerate(values)))


def write_csv(path, rows, parameters):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['case'] + [f'sigma_marg_{parameter}' for parameter in parameters])
        writer.writerows(rows)


def main():
    args = parse_args()
    fisher_dir = Path(args.fisher_dir).expanduser().resolve()
    metadata, metadata_path = load_metadata(fisher_dir)
    cases = normalize_cases(args.cases)
    parameters = normalize_params(args.params, metadata)
    indices = parameter_indices(parameters, metadata)

    if metadata_path:
        print(f'---> metadata: {metadata_path}')
    print(f'---> fisher directory: {fisher_dir}')

    rows = []
    for case in cases:
        cov, path = load_parameter_covariance(fisher_dir, case)
        sigma = marginalized_errors(cov, indices)
        rows.append([case] + list(sigma))
        print(f'---> loaded: {path}')

    print_table(rows, parameters)

    if args.out:
        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        write_csv(out, rows, parameters)
        print(f'---> wrote: {out}')


if __name__ == '__main__':
    main()