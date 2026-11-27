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
sys.path.append("..")
from source.ocr import OCR
from source.models import GFCN
from source.ocr_dataset import OCRDataset
from torch.optim import Adam


if __name__ == "__main__":

    import argparse
    import math

    parser = argparse.ArgumentParser(description="Original GFCN IAM baseline")
    parser.add_argument("--mode", choices=("train", "test", "smoke"), default="test")
    parser.add_argument("--output-folder", help="Separate experiment/checkpoint directory")
    parser.add_argument("--load-epoch", choices=("best", "last"), help="Checkpoint to resume")
    parser.add_argument("--dataset-path", default="../Datasets/iam_lines/iam_lines_64H_NKR")
    args = parser.parse_args()

    params = {
        "dataset_path": args.dataset_path,
        "output_folder": "GFCN_iam_lignes_64H_NKR",
        "set_names": ["train", "valid", "test"],
        "train_set_name": "train",
        "valid_set_names": ["valid", ],
        "test_set_names": ["test", ],
        "models": {
            "end_to_end_model": GFCN,
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
        "nb_epochs": 1000,
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
    }

    # Isolate the one-batch check from actual baseline checkpoints/results.
    if args.output_folder:
        params["output_folder"] = args.output_folder
    if args.load_epoch:
        params["load_epoch"] = args.load_epoch
    if args.mode == "smoke":
        params["output_folder"] += "_smoke"

    model = OCR(params)

    model.load_dataset()
    model.load_model()

    try:
        if args.mode == "train":
            model.train()
        elif args.mode == "smoke":
            generator = model.dataset.get_generator(
                "train", params["batch_size"], shuffle=False,
                augmentation=params["augmentation"])
            batch, _ = next(generator)
            losses, metrics = model.train_batch(model.ctc_generator_to_values(batch))
            if not math.isfinite(losses["loss_ctc"]):
                raise RuntimeError("Non-finite CTC loss")
            print({"device": str(model.device), "losses": losses, "metrics": metrics})
        else:
            if model.latest_epoch < 0:
                raise RuntimeError("No trained checkpoint found; run --mode train first")
            model.predict("test", ["cer", "wer"])
    finally:
        model.writer.close()
