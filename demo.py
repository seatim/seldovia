
import sys
from textwrap import fill

import click
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import torch

from transformers import \
    AutoImageProcessor, Mask2FormerForUniversalSegmentation

from PIL import Image
from skimage.measure import regionprops

CHECKPOINTS = {
    "semantic": "facebook/mask2former-swin-large-ade-semantic",
    "instance": "facebook/mask2former-swin-large-coco-instance",
    "panoptic": "facebook/mask2former-swin-base-coco-panoptic",
}
INDENT_KWARGS = dict(initial_indent="    ", subsequent_indent="    ")


def load_image(path):
    if path.startswith('http'):
        return Image.open(requests.get(path, stream=True).raw)
    else:
        return Image.open(path)


def visualize_segmentation(ax, results, image, model, alpha=0.5):
    """Visualize segmentation results by overlaying alpha mask with unique
    color for each object.
    """
    segmentation = results['segmentation'].numpy()
    segments_info = results['segments_info']
    height, width = segmentation.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign random color to each instance
    instance_colors = {
        segment['id']: np.random.randint(0, 255, size=3)
        for segment in segments_info
    }

    for segment in segments_info:
        mask = segmentation == segment['id']
        color_mask[mask] = instance_colors[segment['id']]

    # Overlay mask on image with alpha blending
    image_np = np.array(image).astype(np.uint8)
    overlay = (1 - alpha) * image_np + alpha * color_mask
    overlay = overlay.astype(np.uint8)

    ax.imshow(overlay)
    ax.axis('off')

    # Draw color patches and names on the right edge
    line_height = 35
    spacing = 10
    y0 = spacing

    for segment in segments_info:
        label_text = model.config.id2label[segment['label_id']]
        if 'score' in segment:
            label_text += f" ({segment['score']:.2f})"

        color = instance_colors[segment['id']] / 255

        # Draw color patch
        rect = mpatches.Rectangle(
            (width - 30, y0), 20, line_height,
            linewidth=0, edgecolor=None, facecolor=color, alpha=1.0,
            transform=ax.transData, clip_on=False
        )
        ax.add_patch(rect)

        # Draw label text
        ax.text(
            width - 35, y0 + line_height / 2, label_text,
            va='center', ha='right', fontsize=11, color='white',
            bbox=dict(facecolor=(0, 0, 0, 0.2), edgecolor='none',
                      boxstyle='round,pad=0.2')
        )
        y0 += line_height + spacing


def draw_binary_mask(ax, results, model):
    segmentation = results['segmentation']
    segments_info = results['segments_info']
    try:
        seg_np = segmentation.numpy()
    except AttributeError:
        seg_np = np.array(segmentation)

    ax.imshow(seg_np)
    ax.axis('off')

    # Map segment id to label id
    segment_to_label = {
        segment['id']: segment['label_id'] for segment in segments_info}

    # For each segment, find centroid and plot label
    for segment in segments_info:
        segment_id = segment['id']
        label_id = segment['label_id']
        label_name = model.config.id2label[label_id]
        mask = (seg_np == segment_id)
        props = regionprops(mask.astype(np.uint8))
        if props:
            y, x = props[0].centroid
            ax.text(
                x, y, label_name,
                color='white', fontsize=8, weight='bold',
                ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5,
                          boxstyle='round,pad=0.2')
            )


def run_segmentation(image, mode):
    """Run a segmentation process on an image.
    """
    try:
        checkpoint = CHECKPOINTS[mode]
    except KeyError:
        raise ValueError("mode must be 'semantic', 'instance', or 'panoptic'")

    processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint)

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    method = getattr(processor, f"post_process_{mode}_segmentation")
    results = method(outputs, target_sizes=[image.size[::-1]])[0]

    return results, model


def semantic_to_instance(results, model):
    """Convert a semantic segmentation map to an instance map by giving each
    label present in the map a unique instance number.  This lets us see what
    the semantic labels are using the same visualization function that is used
    for instance maps.
    """
    semantic_map = results.numpy()
    unique_labels = np.unique(semantic_map)

    seg_info = [{"id": k, "label_id": label_id}
                for k, label_id in enumerate(unique_labels)]

    label_id_map = {label_id: k for k, label_id in enumerate(unique_labels)}
    segmentation = torch.tensor(np.vectorize(label_id_map.get)(semantic_map))

    return {"segmentation": segmentation, "segments_info": seg_info}


@click.command()
@click.argument('path')
@click.option('-m', '--mode', type=click.Choice(list(CHECKPOINTS)),
              default="semantic", show_default=True)
@click.option('-l', '--list-vocabulary', is_flag=True)
@click.option('-L', '--list-backbone-vocabulary', is_flag=True)
def main(path, mode, list_vocabulary, list_backbone_vocabulary):

    image = load_image(path)
    results, model = run_segmentation(image, mode)

    if list_vocabulary:
        print('Vocabulary of model:')
        vocabulary = ', '.join(model.config.label2id)
        print(fill(vocabulary, **INDENT_KWARGS))
        print()

    if list_backbone_vocabulary:
        print('Vocabulary of backbone model:')
        vocabulary = ', '.join(model.config.backbone_config.label2id)
        print(fill(vocabulary, **INDENT_KWARGS))
        print()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    axes = axes.flat
    axes[0].imshow(image)
    axes[0].axis('off')

    if mode == "semantic":
        results = semantic_to_instance(results, model)

    draw_binary_mask(axes[1], results, model)
    visualize_segmentation(axes[2], results, image, model)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
