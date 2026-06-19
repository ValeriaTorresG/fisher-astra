import argparse, csv, json, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from calc_deriv_cosmo import (DEFAULT_PK_ROOT, ENVIRONMENT_FIELDS, FIELDS,
                              OBSERVABLE_KINDS, add_bool_argument,
                              align_pk_to_reference, load_pk_csv,
                              normalize_cosmo, normalize_hod,
                              observable_description, parse_pk_file)


DEFAULT_COSMO = 'c000'
DEFAULT_FIELDS = list(FIELDS)
PK_KINDS = ('pk_used', 'pk_raw')

Component = SimpleNamespace


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pk-root', type=str, default=DEFAULT_PK_ROOT)
    parser.add_argument('--outdir', type=str, default='')
    parser.add_argument('--cosmo', type=str, default=DEFAULT_COSMO)
    parser.add_argument('--fields', nargs='+', default=['all'],
                        choices=['all'] + list(FIELDS))
    parser.add_argument('--hod', nargs='+', default=['all'])
    parser.add_argument('--pk-kind', type=str, default='pk_used', choices=PK_KINDS)
    parser.add_argument('--observable', type=str, default='f2pk', choices=OBSERVABLE_KINDS)
    parser.add_argument('--pk-file-field', type=str, default='all')
    parser.add_argument('--kmin', type=float, default=None)
    parser.add_argument('--kmax', type=float, default=None)
    parser.add_argument('--ddof', type=int, default=1)
    add_bool_argument(parser, '--strict-bins', True,
                      'Require matching k bins between all fiducial mocks')
    add_bool_argument(parser, '--drop-nonfinite', True,
                      'Drop mocks with any non-finite data-vector component')
    parser.add_argument('--write-data-matrix', action='store_true',
                        help='Also write the mock-by-component data matrix.')
    return parser.parse_args()


def normalize_fields(values):
    if any(value == 'all' for value in values):
        return list(FIELDS)
    fields = []
    for value in values:
        if value not in fields:
            fields.append(value)
    return fields


def normalize_hod_filter(values):
    hod_values = {normalize_hod(value) for value in values}
    if 'all' in hod_values:
        return None
    return hod_values


def discover_fiducial_pk_files(pk_root, fields, pk_kind, pk_file_field,
                               cosmo, hod_filter):
    candidates = []
    pk_root = Path(pk_root).expanduser().resolve()
    for path in sorted(pk_root.rglob('pk_*.csv')):
        pk_file = parse_pk_file(path, fields, pk_kind)
        if pk_file is None:
            continue
        if pk_file.cosmo != cosmo:
            continue
        if pk_file.coverage_score < len(fields):
            continue
        if pk_file_field and pk_file.field_request != pk_file_field:
            continue
        if hod_filter and pk_file.hod not in hod_filter:
            continue
        candidates.append(pk_file)

    best_by_realization = {}
    for pk_file in candidates:
        key = (pk_file.hod, pk_file.phase, pk_file.seed)
        current = best_by_realization.get(key)
        score = (pk_file.coverage_score,
                 pk_file.field_request == 'all',
                 len(pk_file.available_fields),
                 str(pk_file.path))
        if current is None or score > current[0]:
            best_by_realization[key] = (score, pk_file)

    return sorted((pk_file for _, pk_file in best_by_realization.values()),
                  key=lambda item: (int(item.hod[3:]), int(item.phase),
                                    int(item.seed), str(item.path)))


def k_selection_mask(k, kmin, kmax):
    mask = np.isfinite(k)
    if kmin is not None:
        mask &= k >= float(kmin)
    if kmax is not None:
        mask &= k <= float(kmax)
    if not np.any(mask):
        raise RuntimeError('No k bins remain after applying the requested k range.')
    return mask


def make_components(fields, k, k_min, k_max):
    components = []
    component_index = 0
    for field in fields:
        for k_index in range(k.size):
            components.append(Component(component_index=component_index,
                                        global_component_index=component_index,
                                        field=field,
                                        k_index=k_index,
                                        k_h_mpc=float(k[k_index]),
                                        k_min_h_mpc=float(k_min[k_index]),
                                        k_max_h_mpc=float(k_max[k_index]),
                                        label=f'{field}_k{k_index:03d}'))
            component_index += 1
    return components


def concatenate_fields(pk, fields, mask):
    return np.concatenate([np.asarray(pk['values'][field], dtype=np.float64)[mask]
                           for field in fields])


def load_data_matrix(pk_files, fields, pk_kind, observable, strict_bins,
                     drop_nonfinite, kmin, kmax):
    metadata_cache = {}
    reference = None
    mask = None
    rows = []
    samples = []
    dropped = []

    for pk_file in pk_files:
        pk = load_pk_csv(pk_file.path, fields, pk_kind, observable, metadata_cache)
        if reference is None:
            reference = pk
            mask = k_selection_mask(reference['k'], kmin, kmax)
        else:
            pk = align_pk_to_reference(pk, reference, fields, strict_bins)

        vector = concatenate_fields(pk, fields, mask)
        finite = np.isfinite(vector)
        if not np.all(finite):
            info = {'hod': pk_file.hod,
                    'cosmo': pk_file.cosmo,
                    'phase': pk_file.phase,
                    'seed': pk_file.seed,
                    'path': str(pk_file.path),
                    'n_nonfinite': int(np.sum(~finite))}
            if drop_nonfinite:
                dropped.append(info)
                continue
            raise RuntimeError(f'Non-finite data-vector components in {pk_file.path}: '
                               f"{info['n_nonfinite']}")

        samples.append({'sample_index': len(samples),
                        'hod': pk_file.hod,
                        'cosmo': pk_file.cosmo,
                        'phase': pk_file.phase,
                        'seed': pk_file.seed,
                        'path': str(pk_file.path)})
        rows.append(vector)

    if not rows:
        raise RuntimeError('No finite fiducial data vectors were loaded.')

    matrix = np.vstack(rows).astype(np.float64)
    k = np.asarray(reference['k'], dtype=np.float64)[mask]
    k_min = np.asarray(reference['k_min'], dtype=np.float64)[mask]
    k_max = np.asarray(reference['k_max'], dtype=np.float64)[mask]
    components = make_components(fields, k, k_min, k_max)
    return matrix, samples, dropped, components


def sample_covariance(matrix, ddof):
    matrix = np.asarray(matrix, dtype=np.float64)
    n_samples, n_components = matrix.shape
    denominator = n_samples - int(ddof)
    if denominator <= 0:
        raise RuntimeError(f'Need n_samples > ddof to compute covariance; '
                           f'got n_samples={n_samples}, ddof={ddof}.')
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    cov = centered.T @ centered / float(denominator)
    return mean, cov


def correlation_from_covariance(cov):
    cov = np.asarray(cov, dtype=np.float64)
    diag = np.diag(cov)
    scale = np.sqrt(np.outer(diag, diag))
    corr = np.full_like(cov, np.nan, dtype=np.float64)
    mask = np.isfinite(scale) & (scale > 0.0)
    corr[mask] = cov[mask] / scale[mask]
    return corr


def subset_components(components, indices):
    selected = []
    for local_index, global_index in enumerate(indices):
        component = components[global_index]
        selected.append(Component(component_index=local_index,
                                  global_component_index=component.global_component_index,
                                  field=component.field,
                                  k_index=component.k_index,
                                  k_h_mpc=component.k_h_mpc,
                                  k_min_h_mpc=component.k_min_h_mpc,
                                  k_max_h_mpc=component.k_max_h_mpc,
                                  label=component.label))
    return selected


def field_slices(fields, n_k):
    slices = {}
    start = 0
    for field in fields:
        stop = start + n_k
        slices[field] = slice(start, stop)
        start = stop
    return slices


def write_components_csv(path, components):
    headers = ['component_index', 'global_component_index', 'field', 'k_index',
               'k_h_mpc', 'k_min_h_mpc', 'k_max_h_mpc', 'label']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for component in components:
            writer.writerow([component.component_index,
                             component.global_component_index,
                             component.field,
                             component.k_index,
                             component.k_h_mpc,
                             component.k_min_h_mpc,
                             component.k_max_h_mpc,
                             component.label])


def write_mean_csv(path, mean, cov, components):
    headers = ['component_index', 'global_component_index', 'field', 'k_index',
               'k_h_mpc', 'k_min_h_mpc', 'k_max_h_mpc', 'mean', 'variance',
               'std']
    variance = np.diag(cov)
    std = np.sqrt(np.where(variance >= 0.0, variance, np.nan))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, component in enumerate(components):
            writer.writerow([component.component_index,
                             component.global_component_index,
                             component.field,
                             component.k_index,
                             component.k_h_mpc,
                             component.k_min_h_mpc,
                             component.k_max_h_mpc,
                             mean[i],
                             variance[i],
                             std[i]])


def write_square_matrix_csv(path, matrix, components, value_prefix):
    headers = ['component_index', 'global_component_index', 'field', 'k_index',
               'k_h_mpc', 'k_min_h_mpc', 'k_max_h_mpc']
    headers += [f'{value_prefix}_{component.label}' for component in components]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i, component in enumerate(components):
            row = [component.component_index,
                   component.global_component_index,
                   component.field,
                   component.k_index,
                   component.k_h_mpc,
                   component.k_min_h_mpc,
                   component.k_max_h_mpc]
            row.extend(matrix[i])
            writer.writerow(row)


def write_samples_csv(path, samples):
    headers = ['sample_index', 'hod', 'cosmo', 'phase', 'seed', 'path']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(samples)


def write_data_matrix_csv(path, matrix, samples, components):
    headers = ['sample_index', 'hod', 'cosmo', 'phase', 'seed']
    headers += [component.label for component in components]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for sample, row_values in zip(samples, matrix):
            row = [sample['sample_index'], sample['hod'], sample['cosmo'],
                   sample['phase'], sample['seed']]
            row.extend(row_values)
            writer.writerow(row)


def matrix_diagnostics(cov, n_samples, ddof):
    n_components = int(cov.shape[0])
    rank = int(np.linalg.matrix_rank(cov))
    condition_number = None
    if n_components > 0 and rank == n_components:
        condition_number = float(np.linalg.cond(cov))
    hartlap = None
    if n_samples > n_components + 2:
        hartlap = float((n_samples - n_components - 2) / (n_samples - 1))
    return {'n_components': n_components,
            'n_samples': int(n_samples),
            'ddof': int(ddof),
            'rank': rank,
            'is_singular': bool(rank < n_components),
            'condition_number': condition_number,
            'hartlap_factor': hartlap}


def write_covariance_product(outdir, name, mean, cov, components):
    corr = correlation_from_covariance(cov)
    paths = {'components': str(outdir / f'components_{name}.csv'),
             'mean': str(outdir / f'mean_{name}.csv'),
             'covariance_csv': str(outdir / f'cov_{name}.csv'),
             'covariance_npy': str(outdir / f'cov_{name}.npy'),
             'correlation_csv': str(outdir / f'corr_{name}.csv'),
             'correlation_npy': str(outdir / f'corr_{name}.npy')}
    write_components_csv(paths['components'], components)
    write_mean_csv(paths['mean'], mean, cov, components)
    write_square_matrix_csv(paths['covariance_csv'], cov, components, 'cov')
    write_square_matrix_csv(paths['correlation_csv'], corr, components, 'corr')
    np.save(paths['covariance_npy'], cov)
    np.save(paths['correlation_npy'], corr)
    return paths


def write_metadata(path, args, outdir, fields, components, samples, dropped,
                   products, diagnostics, field_slice_metadata):
    metadata = {'pk_root': str(Path(args.pk_root).expanduser().resolve()),
                'outdir': str(outdir),
                'cosmo': normalize_cosmo(args.cosmo),
                'fields': fields,
                'pk_kind': args.pk_kind,
                'observable': args.observable,
                'observable_description': observable_description(args.observable),
                'pk_file_field': args.pk_file_field,
                'kmin': args.kmin,
                'kmax': args.kmax,
                'strict_bins': args.strict_bins,
                'drop_nonfinite': args.drop_nonfinite,
                'ddof': args.ddof,
                'n_samples': len(samples),
                'n_dropped_nonfinite': len(dropped),
                'n_components_all': len(components),
                'samples': samples,
                'dropped_nonfinite': dropped,
                'field_slices': field_slice_metadata,
                'diagnostics': diagnostics,
                'outputs': products}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def main():
    args = parse_args()
    t0 = time.time()

    fields = normalize_fields(args.fields)
    hod_filter = normalize_hod_filter(args.hod)
    cosmo = normalize_cosmo(args.cosmo)
    pk_root = Path(args.pk_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else pk_root / 'covariances' / cosmo
    outdir.mkdir(parents=True, exist_ok=True)

    pk_files = discover_fiducial_pk_files(pk_root, fields, args.pk_kind,
                                          args.pk_file_field, cosmo,
                                          hod_filter)
    if not pk_files:
        raise RuntimeError(f'No fiducial P(k) files found for cosmo={cosmo}, '
                           f'fields={fields}, pk_file_field={args.pk_file_field}.')

    print(f'---> pk root: {pk_root}')
    print(f'---> output directory: {outdir}')
    print(f'---> fiducial cosmology: {cosmo}')
    print(f'---> fields: {fields}')
    print(f'---> observable: {args.observable} ({observable_description(args.observable)})')
    print(f'---> candidate mocks: {len(pk_files)}')

    matrix, samples, dropped, components = load_data_matrix(
        pk_files, fields, args.pk_kind, args.observable, args.strict_bins,
        args.drop_nonfinite, args.kmin, args.kmax)
    mean_all, cov_all = sample_covariance(matrix, args.ddof)
    n_samples, n_components = matrix.shape
    n_k = n_components // len(fields)
    slices = field_slices(fields, n_k)

    print(f'---> finite mocks used: {n_samples}')
    if dropped:
        print(f'---> dropped non-finite mocks: {len(dropped)}')
    print(f'---> loaded vector length: {n_components} '
          f'({len(fields)} fields x {n_k} k bins)')

    products = {}
    diagnostics = {}
    field_slice_metadata = {}
    for field, slc in slices.items():
        indices = np.arange(slc.start, slc.stop)
        field_components = subset_components(components, indices)
        field_mean = mean_all[indices]
        field_cov = cov_all[np.ix_(indices, indices)]
        products[field] = write_covariance_product(outdir, field, field_mean,
                                                   field_cov, field_components)
        diagnostics[field] = matrix_diagnostics(field_cov, n_samples, args.ddof)
        field_slice_metadata[field] = {'start': int(slc.start),
                                       'stop': int(slc.stop),
                                       'n_components': int(slc.stop - slc.start)}

    if all(field in slices for field in ENVIRONMENT_FIELDS):
        env_indices = np.concatenate(
            [np.arange(slices[field].start, slices[field].stop)
             for field in ENVIRONMENT_FIELDS])
        combined_components = subset_components(components, env_indices)
        combined_mean = mean_all[env_indices]
        combined_cov = cov_all[np.ix_(env_indices, env_indices)]
        products['combined'] = write_covariance_product(
            outdir, 'combined', combined_mean, combined_cov, combined_components)
        diagnostics['combined'] = matrix_diagnostics(combined_cov, n_samples, args.ddof)
        field_slice_metadata['combined'] = {'fields': list(ENVIRONMENT_FIELDS),
                                            'global_component_indices': env_indices.astype(int).tolist(),
                                            'n_components': int(env_indices.size)}

        products['all'] = write_covariance_product(
            outdir, 'all', combined_mean, combined_cov, combined_components)
        diagnostics['all'] = diagnostics['combined']
        field_slice_metadata['all'] = field_slice_metadata['combined']

    samples_path = outdir / 'samples.csv'
    write_samples_csv(samples_path, samples)
    products['samples'] = str(samples_path)

    if args.write_data_matrix:
        data_matrix_path = outdir / 'data_matrix_all.csv'
        write_data_matrix_csv(data_matrix_path, matrix, samples, components)
        products['data_matrix_all'] = str(data_matrix_path)

    metadata_path = outdir / 'cov_metadata.json'
    products['metadata'] = str(metadata_path)
    write_metadata(metadata_path, args, outdir, fields, components, samples, dropped,
                   products, diagnostics, field_slice_metadata)

    summary = {'elapsed_sec': time.time() - t0,
               'pk_root': str(pk_root),
               'outdir': str(outdir),
               'cosmo': cosmo,
               'fields': fields,
               'observable': args.observable,
               'n_samples': n_samples,
               'n_components_loaded': n_components,
               'outputs': products}
    summary_path = outdir / f'cov_summary_{int(time.time())}.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    for field in fields:
        print(f'---> wrote: {products[field]["covariance_csv"]}')
    if 'combined' in products:
        print(f'---> wrote: {products["combined"]["covariance_csv"]}')
        print(f'---> wrote alias: {products["all"]["covariance_csv"]}')
    print(f'---> wrote: {metadata_path}')
    print(f'---> wrote: {summary_path}')
    print(f"---> elapsed: {summary['elapsed_sec']:.2f} s")


if __name__ == '__main__':
    main()