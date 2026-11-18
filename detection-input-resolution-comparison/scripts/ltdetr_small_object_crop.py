"""Training-only object-centered crops; DINOv3 and validation are untouched."""
import numpy as np


def crop_sample(sample, rng, probability=0.5, target_size=640):
    """YOLO-normalized boxes; small means projected area < 32² at target size."""
    boxes = np.asarray(sample['bboxes'], dtype=np.float64).reshape(-1, 4)
    if not len(boxes) or rng.random() >= probability:
        return sample
    eligible = np.flatnonzero(
        (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
        & (boxes[:, 2] * boxes[:, 3] * target_size**2 < 32**2)
    )
    if not len(eligible):
        return sample
    image = sample['image']
    height, width = image.shape[:2]
    anchor = boxes[int(rng.choice(eligible))]
    fraction = float(rng.uniform(0.35, 0.60))
    crop_w = min(width, max(2, int(round(width * fraction)), int(np.ceil(anchor[2] * width)) + 2))
    crop_h = min(height, max(2, int(round(height * fraction)), int(np.ceil(anchor[3] * height)) + 2))
    # Choose an integer origin that keeps the entire anchor inside the crop.
    ax1, ay1 = (anchor[:2] - anchor[2:] / 2) * [width, height]
    ax2, ay2 = (anchor[:2] + anchor[2:] / 2) * [width, height]
    xmin, xmax = max(0, int(np.ceil(ax2 - crop_w))), min(width - crop_w, int(np.floor(ax1)))
    ymin, ymax = max(0, int(np.ceil(ay2 - crop_h))), min(height - crop_h, int(np.floor(ay1)))
    if xmin > xmax or ymin > ymax:
        return sample
    left = int(rng.integers(xmin, xmax + 1))
    top = int(rng.integers(ymin, ymax + 1))
    xy1 = (boxes[:, :2] - boxes[:, 2:] / 2) * [width, height]
    xy2 = (boxes[:, :2] + boxes[:, 2:] / 2) * [width, height]
    xy1 = np.maximum(xy1, [left, top]) - [left, top]
    xy2 = np.minimum(xy2, [left + crop_w, top + crop_h]) - [left, top]
    sizes = xy2 - xy1
    # Keep every intersecting annotation, including partially cropped objects.
    keep = (sizes[:, 0] > 0) & (sizes[:, 1] > 0)
    cropped_boxes = np.concatenate(((xy1 + xy2) / 2, sizes), axis=1)[keep]
    cropped_boxes /= [crop_w, crop_h, crop_w, crop_h]
    return {
        **sample,
        'image': image[top:top + crop_h, left:left + crop_w].copy(),
        'bboxes': cropped_boxes,
        'class_labels': np.asarray(sample['class_labels'])[keep],
    }


class WorkerRandom:
    """Use NumPy's worker-seeded RNG, including on forked data-loader workers."""
    random = staticmethod(np.random.random)
    choice = staticmethod(np.random.choice)
    uniform = staticmethod(np.random.uniform)
    integers = staticmethod(np.random.randint)


def enable_small_object_crops():
    from lightly_train._task_models.ltdetr_object_detection.transforms import (
        LTDETRObjectDetectionTrainTransform,
    )
    cls = LTDETRObjectDetectionTrainTransform
    if getattr(cls, '_small_object_crop_enabled', False):
        raise RuntimeError('Crop wrapper already enabled')
    original = cls.__call__

    def training_call(self, sample):
        return original(self, crop_sample(sample, WorkerRandom()))

    cls.__call__ = training_call
    cls._small_object_crop_enabled = True
    print('Training-only small-object crops: p=0.5, area at 640 < 32^2; '
          'crop dimensions 35%-60% of full image. Validation unchanged.', flush=True)
