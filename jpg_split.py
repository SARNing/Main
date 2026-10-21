"""Create a non-destructive JPG view for the FDPE-Net paper protocol."""
import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


FOLDER_TO_CLASS = {
    '0': 'BMP2',
    '1': 'BTR70',
    '2': 'T72',
    '3': '2S1',
    '4': 'BRDM2',
    '5': 'D7',
    '6': 'BTR60',
    '7': 'T62',
    '8': 'ZIL131',
    '9': 'ZSU234',
}
SOURCE_SPLITS = {'TRAINDATA': 'train', 'TESTDATA': 'test'}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_jpg_index(dataset):
    dataset = Path(dataset).resolve()
    index = {}
    modes, sizes = Counter(), Counter()
    for source_directory, split in SOURCE_SPLITS.items():
        root = dataset / source_directory
        if not root.is_dir():
            raise FileNotFoundError('missing JPG source directory: ' + str(root))
        for folder, category in FOLDER_TO_CLASS.items():
            class_root = root / folder
            if not class_root.is_dir():
                raise FileNotFoundError('missing JPG class directory: ' + str(class_root))
            for path in sorted(class_root.iterdir()):
                if not path.is_file() or path.suffix.lower() != '.jpg':
                    continue
                key = (split, category, path.stem.casefold())
                if key in index:
                    raise ValueError('duplicate JPG identity: ' + repr(key))
                with Image.open(path) as image:
                    image.verify()
                    modes[image.mode] += 1
                    sizes['%dx%d' % image.size] += 1
                index[key] = path.resolve()
    return index, dict(sorted(modes.items())), dict(sorted(sizes.items()))


def read_metadata(path):
    path = Path(path).resolve()
    records = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            split, category = row['split'], row['category']
            if split not in {'train', 'test'} or category not in FOLDER_TO_CLASS.values():
                raise ValueError('unsupported metadata split/category: %s/%s' % (split, category))
            depression = float(row['desired_depression_deg'])
            expected_depression = 17.0 if split == 'train' else 15.0
            if depression != expected_depression:
                raise ValueError('metadata depression does not match split: ' + row['path'])
            azimuth = float(row['azimuth_deg'])
            if not 0.0 <= azimuth <= 360.0:
                raise ValueError('metadata azimuth outside [0, 360]: ' + row['path'])
            key = (split, category, Path(row['path']).stem.casefold())
            if key in records:
                raise ValueError('duplicate metadata identity: ' + repr(key))
            records[key] = dict(row, azimuth_deg=azimuth, desired_depression_deg=depression)
    if not records:
        raise ValueError('metadata is empty')
    return records


def protocol_roles(record):
    azimuth = record['azimuth_deg']
    if record['split'] == 'train':
        return [
            'pretrain_17_all',
            'support_17_az000_180' if azimuth < 180.0 else 'query_same_17_az180_360',
        ]
    return [
        'not_used_15_az000_180' if azimuth < 180.0 else 'query_cross_15_az180_360'
    ]


def build_entries(dataset, metadata):
    jpgs, modes, sizes = read_jpg_index(dataset)
    records = read_metadata(metadata)
    if set(jpgs) != set(records):
        missing_jpg = sorted(set(records) - set(jpgs))[:5]
        extra_jpg = sorted(set(jpgs) - set(records))[:5]
        raise ValueError(
            'JPG files do not match metadata; missing_jpg=%r extra_jpg=%r'
            % (missing_jpg, extra_jpg)
        )
    entries = []
    hashes = {}
    for key in sorted(records):
        record, source = records[key], jpgs[key]
        digest = file_sha256(source)
        hashes[source] = digest
        numeric_label = next(folder for folder, category in FOLDER_TO_CLASS.items() if category == record['category'])
        for role in protocol_roles(record):
            output_path = Path(role) / record['category'] / source.name
            entries.append(
                dict(
                    protocol_split=role,
                    category=record['category'],
                    numeric_label=numeric_label,
                    source_split=record['split'],
                    azimuth_deg=record['azimuth_deg'],
                    depression_deg=record['desired_depression_deg'],
                    source_path=str(source),
                    output_path=str(output_path),
                    sha256=digest,
                )
            )
    return entries, len(jpgs), hashes, modes, sizes


def write_split(dataset, metadata, output):
    dataset, metadata, output = map(lambda value: Path(value).resolve(), (dataset, metadata, output))
    if output.exists():
        raise FileExistsError('output already exists: ' + str(output))
    entries, source_count, hashes, modes, sizes = build_entries(dataset, metadata)
    temporary = output.with_name('.' + output.name + '.building')
    if temporary.exists():
        raise FileExistsError('temporary output already exists: ' + str(temporary))
    temporary.mkdir(parents=True)
    try:
        for entry in entries:
            link = temporary / entry['output_path']
            link.parent.mkdir(parents=True, exist_ok=True)
            relative_target = os.path.relpath(entry['source_path'], start=str(link.parent))
            link.symlink_to(relative_target)
        fields = [
            'protocol_split', 'category', 'numeric_label', 'source_split',
            'azimuth_deg', 'depression_deg', 'source_path', 'output_path', 'sha256',
        ]
        with (temporary / 'manifest.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(entries)
        split_counts = Counter(entry['protocol_split'] for entry in entries)
        category_counts = defaultdict(Counter)
        for entry in entries:
            category_counts[entry['protocol_split']][entry['category']] += 1
        source_fingerprint = hashlib.sha256()
        for path, digest in sorted((str(path), digest) for path, digest in hashes.items()):
            source_fingerprint.update(path.encode())
            source_fingerprint.update(digest.encode())
        summary = {
            'dataset_root': str(dataset),
            'metadata_csv': str(metadata),
            'unique_source_images': source_count,
            'logical_entries': len(entries),
            'split_counts': dict(sorted(split_counts.items())),
            'category_counts': {key: dict(sorted(value.items())) for key, value in sorted(category_counts.items())},
            'image_modes': modes,
            'image_sizes': sizes,
            'source_fingerprint_sha256': source_fingerprint.hexdigest(),
            'boundary_rule': 'support/not-used: 0 <= azimuth < 180; query: 180 <= azimuth <= 360',
            'storage': 'relative symbolic links; source JPG files are unchanged',
        }
        (temporary / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
        (temporary / 'README.md').write_text(
            '# MSTAR paper protocol split\n\n'
            'This directory is generated from the JPG dataset and the verified Phoenix metadata. '
            'Images are relative symbolic links; the source JPG files are unchanged.\n\n'
            '- `pretrain_17_all`: all 17-degree images used for supervised pretraining.\n'
            '- `support_17_az000_180`: 17-degree few-shot support pool with azimuth in `[0, 180)`.\n'
            '- `query_same_17_az180_360`: 17-degree same-depression query pool with azimuth in `[180, 360]`.\n'
            '- `query_cross_15_az180_360`: 15-degree cross-depression query pool with azimuth in `[180, 360]`.\n'
            '- `not_used_15_az000_180`: 15-degree images outside the paper evaluation pools.\n\n'
            'The same-depression query images are also present in `pretrain_17_all`, following the paper protocol; '
            'this is not an unseen-image test split. K-shot episodes must sample K support images and 15 query images '
            'per class without replacement from the corresponding pools.\n'
        )
        temporary.rename(output)
    except BaseException:
        # Keep any partial build for diagnosis; it never replaces the requested output.
        raise
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    summary = write_split(args.dataset, args.metadata, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
