# Third-Party Notices

AlpaJudge depends on the third-party Python packages listed below. They are installed from PyPI
by `uv sync` and are not redistributed in this repository. Each package is governed by its own
license. Versions are the ones pinned in [uv.lock](uv.lock). License identifiers are taken from
each package's published metadata.

## Runtime dependencies

These packages are installed with `alpajudge`, either directly or through other dependencies.
The `nvidia-*` CUDA libraries are installed by the Linux `torch` wheel.

| Package                  | Version   | License                                            |
| ------------------------ | --------- | -------------------------------------------------- |
| accelerate               | 1.14.0    | Apache-2.0                                         |
| certifi                  | 2026.7.22 | MPL-2.0                                            |
| charset-normalizer       | 3.5.1     | MIT                                                |
| colorama                 | 0.4.6     | BSD-3-Clause                                       |
| filelock                 | 3.32.5    | MIT                                                |
| fsspec                   | 2026.7.0  | BSD-3-Clause                                       |
| hf-xet                   | 1.6.0     | Apache-2.0                                         |
| huggingface-hub          | 0.36.2    | Apache-2.0                                         |
| idna                     | 3.19      | BSD-3-Clause                                       |
| jinja2                   | 3.1.6     | BSD-3-Clause                                       |
| markupsafe               | 3.0.3     | BSD-3-Clause                                       |
| mpmath                   | 1.3.0     | BSD-3-Clause                                       |
| networkx                 | 3.6.1     | BSD-3-Clause                                       |
| numpy                    | 2.5.2     | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| nvidia-cublas-cu12       | 12.8.4.1  | NVIDIA Software License Agreement (proprietary)    |
| nvidia-cuda-cupti-cu12   | 12.8.90   | NVIDIA Software License Agreement (proprietary)    |
| nvidia-cuda-nvrtc-cu12   | 12.8.93   | NVIDIA Software License Agreement (proprietary)    |
| nvidia-cuda-runtime-cu12 | 12.8.90   | NVIDIA Software License Agreement (proprietary)    |
| nvidia-cudnn-cu12        | 9.10.2.21 | NVIDIA Software License Agreement (proprietary)    |
| nvidia-cufft-cu12        | 11.3.3.83 | NVIDIA Software License Agreement (proprietary)    |
| nvidia-cufile-cu12       | 1.13.1.3  | NVIDIA Software License Agreement (proprietary)    |
| nvidia-curand-cu12       | 10.3.9.90 | NVIDIA Software License Agreement (proprietary)    |
| nvidia-cusolver-cu12     | 11.7.3.90 | NVIDIA Software License Agreement (proprietary)    |
| nvidia-cusparse-cu12     | 12.5.8.93 | NVIDIA Software License Agreement (proprietary)    |
| nvidia-cusparselt-cu12   | 0.7.1     | NVIDIA Software License Agreement (proprietary)    |
| nvidia-nccl-cu12         | 2.27.3    | BSD-3-Clause                                       |
| nvidia-nvjitlink-cu12    | 12.8.93   | NVIDIA Software License Agreement (proprietary)    |
| nvidia-nvtx-cu12         | 12.8.90   | Apache-2.0                                         |
| packaging                | 26.3      | Apache-2.0 OR BSD-2-Clause                         |
| pillow                   | 12.3.0    | MIT-CMU                                            |
| psutil                   | 7.2.2     | BSD-3-Clause                                       |
| pyyaml                   | 6.0.3     | MIT                                                |
| regex                    | 2026.9.3  | Apache-2.0 AND CNRI-Python                         |
| requests                 | 2.34.2    | Apache-2.0                                         |
| safetensors              | 0.8.0     | Apache-2.0                                         |
| setuptools               | 84.0.0    | MIT                                                |
| sympy                    | 1.14.0    | BSD-3-Clause                                       |
| tokenizers               | 0.22.2    | Apache-2.0                                         |
| torch                    | 2.8.0     | BSD-3-Clause                                       |
| torchvision              | 0.23.0    | BSD-3-Clause                                       |
| tqdm                     | 4.70.0    | MPL-2.0 AND MIT                                    |
| transformers             | 4.57.6    | Apache-2.0                                         |
| triton                   | 3.4.0     | MIT                                                |
| typing-extensions        | 4.16.0    | PSF-2.0                                            |
| urllib3                  | 2.7.0     | MIT                                                |

## Development and example dependencies

The `dev` dependency group, which is used for the tests and the example notebook, also installs
`ipykernel`, `ipywidgets`, `jupyterlab`, `matplotlib`, `nbclient`, `nbformat`, `numpy`,
`physical-ai-av`, `pre-commit`, `pytest`, and their dependencies. See [uv.lock](uv.lock) for the
complete list.
