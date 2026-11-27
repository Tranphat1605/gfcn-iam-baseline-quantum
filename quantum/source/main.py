#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#
#  This software is a computer program written in XXX whose purpose is XXX.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source.ocr import OCR
from source.quantum_models import QuantumGFCN, QuanvolutionOnlyGFCN
from source.ocr_dataset import OCRDataset
from torch.optim import Adam


if __name__ == "__main__":

    import argparse
    import math
    import json
    import random
    import xml.etree.ElementTree as ET
    import numpy as np
    import torch

    project_root = Path(__file__).resolve().parents[1]
    default_dataset = project_root / "Datasets/iam_lines/iam_lines_64H_NKR"
    parser = argparse.ArgumentParser(description="GFCN with fixed 4-qubit quanvolution (local trained model)")
    parser.add_argument("--mode", choices=("train", "test", "smoke"), default="test")
    parser.add_argument("--architecture", choices=("quanv-vqc", "quanv-only"), default="quanv-only")
    parser.add_argument("--output-folder", help="Separate experiment/checkpoint directory")
    parser.add_argument("--load-epoch", choices=("best", "last"), help="Checkpoint to resume")
    parser.add_argument("--dataset-path", default=str(default_dataset))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-index", type=int, help="Training sample index; default picks a short IAM line")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1000, help="Total epoch target, including completed epochs when resuming")
    parser.add_argument("--max-session-hours", type=float, default=0, help="Pause after a saved epoch once this session budget is reached; 0 disables")
    args = parser.parse_args()
    if args.epochs < 1 or args.max_session_hours < 0:
        parser.error("Epoch target must be positive and session hours nonnegative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.mode == "train" and args.device != "cpu" and torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free < 3 * 1024 ** 3:
            parser.error("Less than 3 GiB GPU memory is free. Finish/pause baseline before quantum GPU training.")

    params = {
        "dataset_path": args.dataset_path,
        "output_folder": "QGFCN_iam_4q_vqc4_l2" if args.architecture == "quanv-vqc" else "QGFCN_iam_4q_filter_only",
        "set_names": ["train", "valid", "test"],
        "train_set_name": "train",
        "valid_set_names": ["valid", ],
        "test_set_names": ["test", ],
        "models": {
            "end_to_end_model": QuantumGFCN if args.architecture == "quanv-vqc" else QuanvolutionOnlyGFCN,
        },
        "augmentation": {
            "activated": False,
        },
        "normalization": {
            "type": "instance",
            "eps": 0.001,
            "momentum": 0.99,
            "num_group": 32,
            "track_running_stats": False,
        },
        "dropout": 0.4,  # dropout probability
        "height": 64,  # image height
        "nb_epochs": args.epochs,
        "session_max_seconds": args.max_session_hours * 3600,
        "load_epoch": "best",  # ["best", "last"]
        "batch_size": 2,
        "lr": 0.0001,
        "optimizer": Adam,
        "eval_on_valid": True,  # whether to compute metrics on valid set during training or not
        "eval_on_valid_interval": 2,  # epochs interval
        "input_channels": 1,  # 1 for grayscaled image, 3 for RGB
        "nb_gate": 6,  # number of GateBlocks to use in the GFCN model
        "reduce_len": 4,  # width division through model
        "dataset_class": OCRDataset,
        "losses": ["loss_ctc"],
        "metrics": ["edit"],
        "quantum_enabled": True,
        "architecture": args.architecture,
        "quanvolution_depth": 2,
        "quanvolution_seed": 0,
        "vqc_depth": 2,
        "quantum_chunk_size": 4096,
        "seed": args.seed,
    }
    if args.device != "auto":
        params["device"] = args.device

    # Isolate the one-batch check from actual baseline checkpoints/results.
    if args.output_folder:
        params["output_folder"] = args.output_folder
    if args.load_epoch:
        params["load_epoch"] = args.load_epoch
    if args.mode == "smoke":
        params["output_folder"] += "_smoke"
        params["batch_size"] = 1

    model = OCR(params)

    model.load_dataset()
    params["quantum_input_mean"] = float(model.dataset.params["mean"][0])
    params["quantum_input_std"] = float(model.dataset.params["std"][0])
    model.load_model()

    try:
        if args.mode == "train":
            model.train()
        elif args.mode == "smoke":
            sample_index = args.smoke_index
            if sample_index is None and Path(args.dataset_path).resolve() == default_dataset.resolve():
                lines = [line for page in ET.parse(Path(__file__).resolve().parents[1] /
                         "preformat iam/xml/train.xml").getroot() for line in page[2]]
                sample_index = min(range(len(lines)), key=lambda i:
                    int(lines[i].attrib["Right"]) - int(lines[i].attrib["Left"]))
            sample_index = 0 if sample_index is None else sample_index
            sample = model.dataset.files["train"][sample_index]
            model.dataset.files["train"] = [sample]
            generator = model.dataset.get_generator(
                "train", params["batch_size"], shuffle=False,
                augmentation=params["augmentation"])
            batch, _ = next(generator)
            losses, metrics = model.train_batch(model.ctc_generator_to_values(batch))
            if not math.isfinite(losses["loss_ctc"]):
                raise RuntimeError("Non-finite CTC loss")
            quantum_head = model.models["end_to_end_model"].end_conv
            gradient = quantum_head.theta.grad if args.architecture == "quanv-vqc" else quantum_head.weight.grad
            if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
                raise RuntimeError("Classification head did not receive finite nonzero gradients")
            result = {"device": str(model.device), "sample": sample, "losses": losses,
                      "metrics": metrics, "architecture": args.architecture,
                      "head_gradient_norm": gradient.norm().item(),
                      "input_shape": list(batch[0].shape), "smoke_check_only": True}
            if args.architecture == "quanv-vqc":
                result["vqc_gradient_norm"] = gradient.norm().item()
            reports = Path(__file__).resolve().parents[1] / "reports"
            reports.mkdir(exist_ok=True)
            (reports / "quantum_iam_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2))
        else:
            if model.latest_epoch < 0:
                raise RuntimeError("No trained checkpoint found; run --mode train first")
            model.predict("test", ["cer", "wer"])
    finally:
        model.writer.close()
