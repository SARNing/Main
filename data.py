"""JPG paper-split loader, immutable image cache and angle-aware episodes."""
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


CLASSES = ['2S1', 'BMP2', 'BRDM2', 'BTR60', 'BTR70', 'D7', 'T62', 'T72', 'ZIL131', 'ZSU234']
PREPROCESS = 'jpg_grayscale_bilinear128_align_corners_false_then_minmax_v1'
PRETRAIN_ROLE = 'pretrain_17_all'
SAME_QUERY_ROLE = 'query_same_17_az180_360'
SUPPORT_ROLE = 'support_17_az000_180'
CROSS_QUERY_ROLE = 'query_cross_15_az180_360'
NOT_USED_ROLE = 'not_used_15_az000_180'
PROTOCOL_ROLES = {
    PRETRAIN_ROLE, SAME_QUERY_ROLE, SUPPORT_ROLE, CROSS_QUERY_ROLE, NOT_USED_ROLE,
}
CANONICAL_ROLES = {PRETRAIN_ROLE, CROSS_QUERY_ROLE, NOT_USED_ROLE}


def read_chip(path):
    """Read a legacy Phoenix chip; retained for raw-data validation utilities."""
    raw = Path(path).read_bytes()
    try:
        header = raw[:4096].split(b'[EndofPhoenixHeader]')[0].decode('ascii')
        def field(key):
            match = re.search(r'^' + key + r'\s*=\s*(\d+)', header, re.M)
            if match is None:
                raise ValueError('missing ' + key)
            return int(match.group(1))
        offset, rows, cols = field('PhoenixHeaderLength'), field('NumberOfRows'), field('NumberOfColumns')
        if min(rows, cols, offset) <= 0 or offset + 8 * rows * cols != len(raw):
            raise ValueError('header dimensions do not match magnitude and phase payload')
        image = np.frombuffer(raw, dtype='>f4', count=rows * cols, offset=offset).astype(np.float32).reshape(rows, cols)
        if not np.isfinite(image).all() or (image < 0).any():
            raise ValueError('amplitude must be finite and nonnegative')
        return image
    except (UnicodeDecodeError, KeyError, OverflowError) as exc:
        raise ValueError('invalid Phoenix chip: ' + str(path)) from exc


def read_jpg(path):
    """Read a JPEG as a two-dimensional float32 grayscale array."""
    try:
        with Image.open(path) as image:
            image = image.convert('L')
            array = np.asarray(image, dtype=np.float32)
    except (OSError, ValueError) as exc:
        raise ValueError('invalid JPG image: ' + str(path)) from exc
    if array.ndim != 2 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError('JPG image must be a non-empty finite grayscale array: ' + str(path))
    return array


def _read_paper_manifest(root):
    manifest_path = root / 'manifest.csv'
    with manifest_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {
            'protocol_split', 'category', 'source_split', 'azimuth_deg',
            'depression_deg', 'source_path', 'output_path',
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError('paper split manifest has missing columns')
        rows = list(reader)
    if not rows:
        raise ValueError('empty paper split manifest')

    dataset_root = root.parent.resolve()
    canonical, source_roles = [], {}
    for row in rows:
        role, category, source_split = (
            row['protocol_split'], row['category'], row['source_split'],
        )
        if role not in PROTOCOL_ROLES:
            raise ValueError('unknown protocol split: ' + role)
        if category not in CLASSES:
            raise ValueError('unknown category: ' + category)
        if source_split not in {'train', 'test'}:
            raise ValueError('unknown source split: ' + source_split)
        try:
            azimuth = float(row['azimuth_deg'])
            depression = float(row['depression_deg'])
        except ValueError as exc:
            raise ValueError('invalid angle in paper split manifest') from exc
        if not np.isfinite(azimuth) or not 0.0 <= azimuth <= 360.0:
            raise ValueError('azimuth outside [0,360]')
        expected_depression = 17.0 if source_split == 'train' else 15.0
        if depression != expected_depression:
            raise ValueError('incorrect depression split')
        if source_split == 'train':
            expected_roles = {
                PRETRAIN_ROLE,
                SUPPORT_ROLE if azimuth < 180.0 else SAME_QUERY_ROLE,
            }
        else:
            expected_roles = {
                NOT_USED_ROLE if azimuth < 180.0 else CROSS_QUERY_ROLE,
            }
        if role not in expected_roles:
            raise ValueError('protocol split does not match source split and azimuth')

        relative = Path(row['output_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('JPG output path escapes paper split root')
        link = root / relative
        if not link.exists():
            raise FileNotFoundError('missing JPG from paper split: ' + str(link))
        path = link.resolve()
        if path != dataset_root and dataset_root not in path.parents:
            raise ValueError('JPG link target escapes dataset root')
        if path.suffix.lower() not in {'.jpg', '.jpeg'}:
            raise ValueError('paper split entry is not a JPEG: ' + str(link))

        source_name = Path(row['source_path']).name.casefold()
        if not source_name:
            raise ValueError('empty source path in paper split manifest')
        identity = (source_split, category, source_name)
        roles = source_roles.setdefault(identity, set())
        if role in roles:
            raise ValueError('duplicate paper split entry: ' + repr(identity + (role,)))
        roles.add(role)
        if role in CANONICAL_ROLES:
            selected = dict(row)
            selected['_path'] = path
            selected['_identity'] = identity
            canonical.append(selected)

    for identity, roles in source_roles.items():
        source_split = identity[0]
        if source_split == 'train':
            if PRETRAIN_ROLE not in roles or len(roles) != 2:
                raise ValueError('incomplete train protocol roles: ' + repr(identity))
        elif len(roles) != 1:
            raise ValueError('duplicate test protocol roles: ' + repr(identity))
    identities = [row['_identity'] for row in canonical]
    if len(identities) != len(set(identities)) or len(canonical) != len(source_roles):
        raise ValueError('canonical JPG records are not one-to-one with source images')
    return manifest_path, canonical


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    metadata, records = _read_paper_manifest(root)
    paths = []
    digest = hashlib.sha256(metadata.read_bytes() + PREPROCESS.encode())
    for record in records:
        path = record['_path']
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        paths.append(path)
    fingerprint = digest.hexdigest()
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['fingerprint'] != fingerprint:
            raise ValueError('cache fingerprint differs; choose a new cache directory')
        for name in ['images', 'labels', 'split', 'azimuth']:
            array = np.load(output / (name + '.npy'), mmap_mode='r')
            if len(array) != len(records):
                raise ValueError('incomplete cache: ' + name)
        return manifest
    output.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(output / 'images.npy', mode='w+', dtype=np.float32, shape=(len(records), 1, 128, 128))
    labels, split, azimuth = [], [], []
    for index, (record, path) in enumerate(zip(records, paths)):
        image = torch.from_numpy(read_jpg(path)).unsqueeze(0).unsqueeze(0)
        image = F.interpolate(image, size=(128, 128), mode='bilinear', align_corners=False)
        low, high = image.min(), image.max()
        image = (image - low) / (high - low).clamp_min(1e-8)
        images[index] = image[0].numpy()
        labels.append(CLASSES.index(record['category']))
        split.append(0 if record['source_split'] == 'train' else 1)
        expected_angle = 17 if split[-1] == 0 else 15
        if float(record['depression_deg']) != expected_angle:
            raise ValueError('incorrect depression split')
        azimuth.append(float(record['azimuth_deg']))
    images.flush()
    del images
    np.save(output / 'labels.npy', np.asarray(labels, dtype=np.int64))
    np.save(output / 'split.npy', np.asarray(split, dtype=np.int64))
    np.save(output / 'azimuth.npy', np.asarray(azimuth, dtype=np.float64))
    public_records = [{key: value for key, value in record.items() if not key.startswith('_')} for record in records]
    manifest = dict(fingerprint=fingerprint, count=len(records), classes=CLASSES, preprocess=PREPROCESS,
                    data_root=str(root), records=public_records)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def load_cache(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    arrays = {name: np.load(path / (name + '.npy'), mmap_mode='r') for name in ['images', 'labels', 'split', 'azimuth']}
    return manifest, arrays


def make_episodes(labels, split, azimuth, *, k, episodes, protocol, seed, query_per_class=15):
    if protocol not in ['same', 'cross'] or k not in [1, 3, 5, 10] or episodes < 1 or query_per_class < 1:
        raise ValueError('invalid episode configuration')
    labels, split, azimuth = map(np.asarray, (labels, split, azimuth))
    if labels.ndim != 1 or labels.shape != split.shape or labels.shape != azimuth.shape:
        raise ValueError('episode arrays must have matching 1D shapes')
    if not np.isfinite(azimuth).all() or ((azimuth < 0) | (azimuth > 360)).any():
        raise ValueError('azimuth outside [0,360]')
    if set(np.unique(labels).tolist()) != set(range(10)):
        raise ValueError('all ten classes are required')
    rng = np.random.default_rng(seed)
    support_pools, query_pools = [], []
    for category in range(10):
        support_pools.append(np.flatnonzero((labels == category) & (split == 0) & (azimuth < 180)))
        query_pools.append(np.flatnonzero((labels == category) & (split == int(protocol == 'cross')) & (azimuth >= 180)))
        if len(support_pools[-1]) < k or len(query_pools[-1]) < query_per_class:
            raise ValueError('insufficient support/query pool for class ' + str(category))
    support, query = [], []
    for _ in range(episodes):
        support.append(np.concatenate([rng.choice(pool, k, replace=False) for pool in support_pools]))
        query.append(np.concatenate([rng.choice(pool, query_per_class, replace=False) for pool in query_pools]))
    return dict(support=np.stack(support), query=np.stack(query))
