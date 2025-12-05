import sys
import os
import logging

import torch
from torch import nn

from architecture import Model, S4Model, MODEL_SIZE, DROPOUT, NUM_LAYERS
from transduction_model import test
from read_emg import PreprocessedEMGDataset
from asr_evaluation import evaluate
from data_utils import TextTransform, phoneme_inventory, print_confusion

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_list('models', [], 'identifiers of models to evaluate')
flags.DEFINE_boolean('dev', False, 'evaluate dev insead of test')
flags.DEFINE_string('lm_directory', 'artifacts/lm', 'Path to LM directory containing lexicon/LM')
flags.DEFINE_string('hifigan_checkpoint', None, 'Path to HiFi-GAN generator checkpoint (optional)')

class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, x_raw, sess):
        ys = []
        ps = []
        for model in self.models:
            out = model(x_raw)
            if isinstance(out, tuple):
                y, p = out
            else:
                y = out
                p = torch.zeros_like(y)
            ys.append(y)
            ps.append(p)
        return torch.stack(ys, 0).mean(0), torch.stack(ps, 0).mean(0)

def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, 'eval_log.txt'), 'w'),
            logging.StreamHandler()
            ], level=logging.INFO, format="%(message)s")

    dev     = FLAGS.dev
    testset = PreprocessedEMGDataset(
        base_dir = FLAGS.base_dir,
        train = False,
        dev = FLAGS.dev,
        test = not FLAGS.dev,
        normalizers_file = FLAGS.normalizers_file,
    )
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'

    models = []
    for fname in FLAGS.models:
        state_dict = torch.load(fname, map_location=device)
        text_transform = TextTransform(togglePhones=False)
        num_outs_ckpt = state_dict["w_out.weight"].shape[0]
        vocab_size = len(text_transform.chars)
        expected_outs = vocab_size + 1  # account for CTC blank
        num_outs = num_outs_ckpt
        if num_outs_ckpt != expected_outs:
            logging.warning(
                "Checkpoint outputs (%s) do not match expected vocab+blank size (%s). "
                "Proceeding with checkpoint size; verify checkpoint/vocab pairing.",
                num_outs_ckpt,
                expected_outs,
            )
        steps_per_epoch = 1000  # placeholder; not used in forward
        epochs = 1
        lm_directory = FLAGS.lm_directory
        if FLAGS.S4:
            model = S4Model(
                testset.num_features,
                num_outs,
            ).to(device)
        else:
            model = Model(
                MODEL_SIZE,
                DROPOUT,
                NUM_LAYERS,
                num_outs,
                text_transform,
                steps_per_epoch,
                epochs,
                lm_directory,
            ).to(device)
        model.load_state_dict(state_dict, strict=False)
        models.append(model)
    ensemble = EnsembleModel(models)

    _, _, confusion = test(ensemble, testset, device)
    print_confusion(confusion)

    # Skip audio synthesis / Whisper transcription for text-only DTW evaluation

if __name__ == "__main__":
    FLAGS(sys.argv)
    main()
