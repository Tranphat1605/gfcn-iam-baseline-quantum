"""Run full test or resume training in an isolated model subprocess."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', choices=['baseline','quantum'], required=True)
parser.add_argument('--mode', choices=['test','train'], default='test')
parser.add_argument('--dataset-path', type=Path, help='Formatted IAM directory containing train, valid, test, params.pkl')
parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
args = parser.parse_args()
project = root/args.model
dataset = (args.dataset_path or project/'Datasets/iam_lines/iam_lines_64H_NKR').resolve()
required = ['params.pkl','train','valid','test']
if not all((dataset/part).exists() for part in required):
    parser.error('Full IAM dataset is not included. Copy your formatted dataset or supply --dataset-path. See README.md.')
folder = 'GFCN_iam_lignes_64H_NKR_baseline' if args.model=='baseline' else 'QGFCN_iam_4q_filter_only'
command = [sys.executable,'-X','utf8','main.py','--mode',args.mode,'--dataset-path',str(dataset),
           '--output-folder',folder,'--load-epoch','last' if args.mode=='train' else 'best']
if args.model=='quantum':
    command += ['--architecture','quanv-only','--device',args.device]
env = dict(os.environ, PYTHONUTF8='1')
if args.device=='cpu':
    env['CUDA_VISIBLE_DEVICES'] = '-1'
elif args.device=='cuda':
    import torch
    if not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu.')
raise SystemExit(subprocess.call(command, cwd=project/'source', env=env))
