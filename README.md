# Towards Learning Universal Audio Representations

In [Towards Learning Universal Audio Representations] (to appear at
[ICASSP 2022]), we introduce a Holistic Audio Representation Evaluation Suite
(HARES), containing 12 downstream tasks spanning the speech, music, and
environmental sound domains, with the hope that this will spur research on
developing better models for universal audio representations. Together with the
benchmark, we also propose a new Slowfast NFNet architecture in the paper.


## HARES tasks

Below is a summary of all 12 HARES tasks, with the links to obtaining these
freely available datasets. Note that the lables of original test sets of
[Birdsong] and [TUT18] are not publicly availabe - therefore we use the splits
created by the authors of [Pre-Training Audio Representations with Self-Supervision],
which is based on the original training dataset. For more details about how to
assemble these tasks, please refer to Appendix A of the arXiv version of
[our paper].

| Dataset   |      Task      |  #Samples | #Classes | Domain |
|----------|:-------------|------:|------:|:------|
| [AudioSet] | audio tagging | 1.9m | 527 | environment |
| [Birdsong] | animal sound | 36k | 2 | environment |
| [TUT18] | acoustic scenes | 8.6k | 10 | environment |
| [ESC-50] | acoustic scenes | 2.0k | 50 | environment |
| [Speech Commands v1] | keyword | 90k | 12 | speech |
| [Speech Commands v2] | keyword | 96k | 35 | speech |
| [Fluent Speech Commands] | intention | 27k | 31 | speech |
| [VoxForge] | languge id | 145k | 6 | speech |
| [VoxCeleb] | speaker id | 147k | 1251 | speech |
| [NSynth-instrument] | instrument id | 293k | 11 | music |
| [NSynth-pitch] | pitch estimation | 293k | 128 | music |
| [MagnaTagATune] | music tagging | 26k | 50 | music |


## Audio Slowfast NFNets, a JAX implementation

We provide a [JAX]/[Haiku] implementation of the Slowfast NfNet-F0. This
convolutional neural network combines Slowfast networks' ability to model both
transient and long-range signals in audio, and NFNets' strong performance
optimized for hardware accelerators. It achieves the state-of-the-art score on
the HARES benchmark.

You may use our unit tests to test your development environment and to know more
about the usage of the models, which can be executed using `pytest`:

```bash
$ pip install -r requirements.txt
$ python -m pytest [-n <NUMCPUS>] slowfast_nfnets
```

### Usage

The unit tests provided together with the model shows a few use cases of how the
model can be run.


## Citing this work

BibTex for citing the paper:

```bibtex
@inproceedings{wang2021towards,
  title={Towards Learning Universal Audio Representations},
  author={Wang, Luyu and Luc, Pauline and Wu, Yan and Recasens, Adria and Smaira, Lucas and Brock, Andrew and Jaegle, Andrew and Alayrac, Jean-Baptiste and Dieleman, Sander and Carreira, Joao and van den Oord, Aaron},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2022},
  organization={IEEE}
}
```

## Disclaimer

This is not an official Google product.

[ICASSP 2022]: https://2022.ieeeicassp.org/
[JAX]: https://github.com/google/jax "JAX on GitHub"
[Haiku]: https://github.com/deepmind/dm-haiku
[Towards Learning Universal Audio Representations]: https://arxiv.org/abs/2111.12124
[AudioSet]: http://research.google.com/audioset/
[Birdsong]: http://dcase.community/challenge2018/task-bird-audio-detection
[TUT18]: http://dcase.community/challenge2018/task-acoustic-scene-classification
[ESC-50]: http://github.com/karolpiczak/ESC-50
[Speech Commands v1]: http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
[Speech Commands v2]: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
[Fluent Speech Commands]: http://fluent.ai/research/fluent-speech-commands/
[VoxForge]: http://tensorflow.org/datasets/catalog/voxforge
[VoxCeleb]: http://tensorflow.org/datasets/catalog/voxceleb
[NSynth-instrument]: http://tensorflow.org/datasets/catalog/nsynth
[NSynth-pitch]: http://tensorflow.org/datasets/catalog/nsynth
[MagnaTagATune]: http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
[Pre-Training Audio Representations with Self-Supervision]: https://ieeexplore.ieee.org/abstract/document/9060816
[our paper]: https://arxiv.org/abs/2111.12124
