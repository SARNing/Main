"""Sequential pretrain -> evaluation -> ablation experiment queue."""
import argparse
import json
import os
import subprocess
import sys
import time
import shutil
from pathlib import Path

from .model import MODEL_NAMES
from .train import atomic_json
from .runtime import queue_lock, source_fingerprint
from .data import prepare
from .evaluate import checkpoint_hash
import numpy as np


def completed_model(output, *, model, epochs, seed, max_batches, episodes, steps, head_init,
                    expected_fingerprint=None, expected_source=None):
    output = Path(output)
    complete = output / 'training_complete.json'
    checkpoint = output / 'last.pt'
    result_file = output / 'evaluation' / ('results_%s.json' % head_init)
    if not complete.exists() or not result_file.exists() or not checkpoint.exists():
        return False
    training = json.loads(complete.read_text())
    for key, value in dict(model=model, epochs=epochs, seed=seed, max_batches=max_batches).items():
        if training['config'].get(key) != value:
            return False
    if expected_fingerprint is not None and training['fingerprint'] != expected_fingerprint:
        return False
    if expected_source is not None and training['config'].get('source_sha256') != expected_source:
        return False
    rows = json.loads(result_file.read_text())
    wanted = {(p, k) for p in ['same', 'cross'] for k in [1, 3, 5, 10]}
    if len(rows) != 8 or {(r['protocol'], r['k']) for r in rows} != wanted:
        return False
    sha = checkpoint_hash(checkpoint)
    expected = dict(model=model, episodes=episodes, steps=steps, seed=seed, head_init=head_init,
                    checkpoint_epoch=epochs, max_pretrain_batches=max_batches,
                    fingerprint=training['fingerprint'], checkpoint_sha256=sha)
    for row in rows:
        if any(row.get(key) != value for key, value in expected.items()):
            return False
        scores = output / 'evaluation' / ('%s_k%d_%s_scores.npy' % (row['protocol'], row['k'], head_init))
        if not scores.exists():
            return False
        values = np.load(scores)
        if values.shape != (episodes,) or not np.isfinite(values).all():
            return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('../Dataset/paper_split'))
    parser.add_argument('--cache', type=Path, default=Path('cache_jpg_paper'))
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=MODEL_NAMES)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-batches', type=int, default=0)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--head-init', choices=['random', 'pretrained'], default='random')
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    with queue_lock(args.output / 'queue.lock', os.environ.get('MSTAR_QUEUE_LOCK_FD')) as lock:
        run_queue(args, lock)


def run_queue(args, lock):
    manifest = prepare(args.data, args.cache)
    source_sha = source_fingerprint()
    snapshot = args.output / 'source_snapshot'
    if snapshot.exists():
        saved = json.loads((args.output / 'source.json').read_text())
        if saved['source_sha256'] != source_sha:
            raise ValueError('source changed since queue start; choose a new output')
    else:
        snapshot.mkdir()
        for path in sorted(Path(__file__).parent.glob('*.py')):
            shutil.copy2(path, snapshot / path.name)
        atomic_json(args.output / 'source.json', dict(source_sha256=source_sha))
    queue = dict(pid=os.getpid(), status='running', seed=args.seed, models=args.models,
                 epochs=args.epochs, episodes=args.episodes, steps=args.steps, head_init=args.head_init,
                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 python=sys.executable, started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                 fingerprint=manifest['fingerprint'], source_sha256=source_sha,
                 completed=[], current=None, stage=None)
    status_path = args.output / 'queue_status.json'
    atomic_json(status_path, queue)
    try:
        for name in args.models:
            if completed_model(args.output / name, model=name, epochs=args.epochs, seed=args.seed,
                               max_batches=args.max_batches, episodes=args.episodes, steps=args.steps,
                               head_init=args.head_init, expected_fingerprint=manifest['fingerprint'],
                               expected_source=source_sha):
                queue['completed'].append(name)
                print(json.dumps(dict(stage='skip_complete', model=name)), flush=True)
                atomic_json(status_path, queue)
                continue
            queue.update(current=name, stage='pretraining')
            atomic_json(status_path, queue)
            model_output = args.output / name
            train_cmd = [sys.executable, '-m', 'mstar.train', '--model', name, '--output', str(model_output),
                         '--data', str(args.data), '--cache', str(args.cache), '--epochs', str(args.epochs),
                         '--seed', str(args.seed), '--device', args.device, '--max-batches', str(args.max_batches)]
            if (model_output / 'last.pt').exists():
                train_cmd.append('--resume')
            print(json.dumps(dict(stage='train', model=name, command=train_cmd)), flush=True)
            subprocess.run(train_cmd, check=True, pass_fds=[lock.fileno()])
            queue.update(stage='evaluation')
            atomic_json(status_path, queue)
            evaluate_cmd = [sys.executable, '-m', 'mstar.evaluate', '--checkpoint', str(model_output / 'last.pt'),
                            '--cache', str(args.cache), '--output', str(model_output / 'evaluation'),
                            '--seed', str(args.seed), '--episodes', str(args.episodes), '--steps', str(args.steps),
                            '--device', args.device, '--head-init', args.head_init]
            print(json.dumps(dict(stage='evaluate', model=name, command=evaluate_cmd)), flush=True)
            subprocess.run(evaluate_cmd, check=True, pass_fds=[lock.fileno()])
            queue['completed'].append(name)
            atomic_json(status_path, queue)
        combined = []
        for name in args.models:
            combined.extend(json.loads((args.output / name / 'evaluation' / ('results_%s.json' % args.head_init)).read_text()))
        atomic_json(args.output / 'comparison.json', combined)
        queue.update(status='complete', current=None, stage=None, finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    except BaseException as exc:
        queue.update(status='failed', error=repr(exc))
        raise
    finally:
        atomic_json(status_path, queue)


if __name__ == '__main__':
    main()
