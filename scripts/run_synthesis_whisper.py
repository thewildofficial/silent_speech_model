import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from absl import flags

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from align import align_from_distances
from architecture import Model, MODEL_SIZE, DROPOUT, NUM_LAYERS
from asr_evaluation import evaluate as whisper_evaluate
from data_utils import TextTransform
from read_emg import PreprocessedEMGDataset
from vocoder import Vocoder

FLAGS = flags.FLAGS


class MelModel(Model):
    """
    Wrapper around architecture.Model that skips the final log_softmax so we can
    treat the outputs as mel-like features for vocoding.
    """

    def forward(self, x_raw):
        if self.training:
            r = np.random.randint(0, 8)
            if r > 0:
                x_raw[:, :-r, :] = x_raw[:, r:, :]
                x_raw[:, -r:, :] = 0

        x_raw = x_raw.transpose(1, 2)
        x_raw = self.conv_blocks(x_raw)
        x_raw = x_raw.transpose(1, 2)
        x_raw = self.w_raw_in(x_raw)

        x = x_raw
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)

        if self.has_aux_out:
            aux_out = self.w_aux(x)
            return self.w_out(x), aux_out
        return self.w_out(x)


def synthesize_and_transcribe(
    base_dir: str,
    model_path: str,
    normalizers_file: str,
    hifigan_checkpoint: str,
    output_dir: str,
    max_examples: int,
    lm_directory: str,
    use_dev_split: bool,
    raw_gain: float = 1.0,
):
    os.makedirs(output_dir, exist_ok=True)

    # Ensure Vocoder sees the checkpoint flag.
    FLAGS(["prog", f"--hifigan_checkpoint={hifigan_checkpoint}"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PreprocessedEMGDataset(
        base_dir=base_dir,
        train=False,
        dev=use_dev_split,
        test=not use_dev_split,
        normalizers_file=normalizers_file,
    )

    # Build mel model wrapper and load checkpoint
    state_dict = torch.load(model_path, map_location="cpu")
    num_outs = state_dict["w_out.weight"].shape[0]
    text_transform = TextTransform(togglePhones=False)
    model = MelModel(
        MODEL_SIZE,
        DROPOUT,
        NUM_LAYERS,
        num_outs,
        text_transform,
        steps_per_epoch=1000,
        epochs=1,
        lm_directory=lm_directory,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    audio_normalizer = dataset.mfcc_norm
    vocoder = Vocoder(device=device)

    n = min(max_examples, len(dataset))
    subset = [dataset[i] for i in range(n)]

    for i, dp in enumerate(subset):
        x_raw = torch.tensor(dp["raw_emg"], dtype=torch.float32, device=device).unsqueeze(
            0
        )
        if raw_gain != 1.0:
            x_raw = x_raw * raw_gain

        with torch.no_grad():
            pred = model(x_raw)
            if isinstance(pred, tuple):
                pred = pred[0]
            pred = pred.squeeze(0).cpu()

            # Align to parallel voiced audio when available (silent examples)
            if dp["silent"]:
                y_ref = torch.tensor(dp["parallel_voiced_audio_features"], dtype=torch.float32)
                costs = torch.cdist(pred, y_ref)
                alignment = align_from_distances(costs.T.numpy())
                pred = pred[alignment]

            pred = audio_normalizer.inverse(pred)
            audio = vocoder(torch.tensor(pred, device=device)).cpu().numpy()

        sf.write(os.path.join(output_dir, f"example_output_{i}.wav"), audio, 22050)

    wer = whisper_evaluate(subset, output_dir)
    print(f"Whisper WER (subset of {n}): {wer:.3f}")
    return wer


def parse_args():
    parser = argparse.ArgumentParser(description="Synthesize audio with HiFi-GAN and score with Whisper.")
    parser.add_argument("--base_dir", required=True, help="Processed data base directory")
    parser.add_argument("--model_path", required=True, help="Path to 80-dim mel predictor checkpoint")
    parser.add_argument("--normalizers_file", default="normalizers.pkl")
    parser.add_argument("--hifigan_checkpoint", required=True, help="Path to HiFi-GAN generator checkpoint")
    parser.add_argument("--raw_gain", type=float, default=1.0, help="Scalar to multiply raw EMG inputs before model forward (input boost probe)")
    parser.add_argument("--output_dir", default="output_synthesis")
    parser.add_argument("--max_examples", type=int, default=10)
    parser.add_argument("--lm_directory", default="artifacts/lm")
    parser.add_argument("--use_dev_split", action="store_true", help="Use dev split instead of test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    synthesize_and_transcribe(
        base_dir=args.base_dir,
        model_path=args.model_path,
        normalizers_file=args.normalizers_file,
        hifigan_checkpoint=args.hifigan_checkpoint,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        lm_directory=args.lm_directory,
        use_dev_split=args.use_dev_split,
        raw_gain=args.raw_gain,
    )

