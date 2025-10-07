
<p align="center">
    <img alt="Demo output image" src="https://raw.githubusercontent.com/seatim/seldovia/main/static/seldovia-demo1.png"/>
</p>

# seldovia

Image segmentation demo using
[Mask2Former](https://huggingface.co/models?other=mask2former) models and the
[transformers](https://github.com/huggingface/transformers) library.

## Setup

Create a virtual environment:

    virtualenv VENV_DIR

Activate virtualenv:

    . VENV_DIR/bin/activate

Install dependencies:

    pip install -r requirements.txt

## Usage

To perform semantic segmentation on an image, run:

    python demo.py PATH_TO_IMAGE

To perform instance segmentation on an image, run:

    python demo.py PATH_TO_IMAGE -m instance

To perform panoptic segmentation on an image, run:

    python demo.py PATH_TO_IMAGE -m panoptic

To get a list of the vocabulary of the segmentation model, run:

    python demo.py PATH_TO_IMAGE -l

To get a list of the vocabulary of the backbone segmentation model, run:

    python demo.py PATH_TO_IMAGE -L

Tip: PATH_TO_IMAGE can be a URL.

## License

Source code in this repository is released under the
[Mozilla Public License version 2.0](https://www.mozilla.org/en-US/MPL/2.0/).

Image files and other static data in this repository are release under the
[CC-BY-SA](https://creativecommons.org/licenses/by-sa/4.0/) license.

