import os
import numpy as np
from tifffile import imread, imwrite
from cellpose import models, metrics
import concurrent
from pathlib import Path


class CellPose:
    def __init__(
        self,
        base_dir,
        model_name,
        model_dir,
        raw_dir,
        real_mask_dir,
        test_raw_dir,
        test_real_mask_dir,
        n_epochs=400,
        diam_mean=30,
        learning_rate=0.0001,
        weight_decay=1.0e-4,
        channels=1,
        min_train_masks=1,
        gpu=True,
        real_train_3D=False,
        save_masks=True,
    ):

        self.base_dir = base_dir
        self.model_name = model_name
        self.model_dir = model_dir
        self.raw_dir = os.path.join(base_dir, raw_dir)
        self.real_mask_dir = os.path.join(base_dir, real_mask_dir)
        self.test_raw_dir = os.path.join(base_dir, test_raw_dir)
        self.test_real_mask_dir = os.path.join(base_dir, test_real_mask_dir)
        self.min_train_masks = min_train_masks
        self.save_raw_dir = os.path.join(
            self.base_dir, (raw_dir).replace("/", "") + "_sliced"
        )

        self.save_real_mask_dir = os.path.join(
            self.base_dir,
            (real_mask_dir).replace("/", "") + "_sliced",
        )
        self.save_test_raw_dir = os.path.join(
            self.base_dir,
            (test_raw_dir).replace("/", "") + "_sliced",
        )
        self.save_test_real_mask_dir = os.path.join(
            self.base_dir,
            (test_real_mask_dir).replace("/", "") + "_sliced",
        )

        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.real_train_3D = real_train_3D
        self.channels = channels
        self.diam_mean = diam_mean
        self.save_masks = save_masks
        if model_dir and model_name is not None:
            self.pretrained_cellpose_model_path = os.path.join(
                model_dir, model_name
            )
        else:
            self.pretrained_cellpose_model_path = False
        self.gpu = gpu
        self.acceptable_formats = [".tif", ".TIFF", ".TIF", ".png"]
        if self.save_masks:
            Path(self.save_raw_dir).mkdir(exist_ok=True)
            Path(self.save_real_mask_dir).mkdir(exist_ok=True)
            Path(self.save_test_raw_dir).mkdir(exist_ok=True)
            Path(self.save_test_real_mask_dir).mkdir(exist_ok=True)

    def create_train(self):
        files_labels = os.listdir(self.real_mask_dir)
        (
            self.train_images,
            self.train_labels,
            self.train_names,
        ) = self._load_data(files_labels)

        files_test_labels = os.listdir(self.test_real_mask_dir)
        self.test_images, self.test_labels, self.test_names = self._load_data(
            files_test_labels
        )
        if self.save_masks:

            for i in range(len(self.train_images)):
                imwrite(
                    os.path.join(
                        self.save_raw_dir,
                        self.train_names[i] + "_" + str(i) + ".tif",
                    ),
                    self.train_images[i].astype(np.float32),
                )
                imwrite(
                    os.path.join(
                        self.save_real_mask_dir,
                        self.train_names[i] + "_" + str(i) + ".tif",
                    ),
                    self.train_labels[i].astype(np.uint16),
                )
            for i in range(len(self.test_images)):
                imwrite(
                    os.path.join(
                        self.save_test_raw_dir,
                        self.test_names[i] + "_" + str(i) + ".tif",
                    ),
                    self.test_images[i].astype(np.float32),
                )
                imwrite(
                    os.path.join(
                        self.save_test_real_mask_dir,
                        self.test_names[i] + "_" + str(i) + ".tif",
                    ),
                    self.test_labels[i].astype(np.uint16),
                )

    def get_data(self):
        files_labels = os.listdir(self.save_real_mask_dir)
        (
            self.train_images,
            self.train_labels,
            self.train_names,
        ) = self._load_saved_data(files_labels)

        files_test_labels = os.listdir(self.save_test_real_mask_dir)
        (
            self.test_images,
            self.test_labels,
            self.test_names,
        ) = self._load_saved_test_data(files_test_labels)

    def train(self):

        self.cellpose_model = models.CellposeModel(
            gpu=self.gpu,
            pretrained_model=self.pretrained_cellpose_model_path,
            diam_mean=self.diam_mean,
        )

        self.new_cellpose_model_path = self.cellpose_model.train(
            self.train_images,
            self.train_labels,
            test_data=self.test_images,
            test_labels=self.test_labels,
            save_path=self.model_dir,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            channels=self.channels,
            weight_decay=self.weight_decay,
            model_name=self.model_name,
            min_train_masks=self.min_train_masks,
        )
        self.diam_labels = self.cellpose_model.diam_labels.copy()

    def evaluate(self):

        self.masks = self.cellpose_model.eval(
            self.test_images, diameter=self.diam_labels
        )[0]
        ap = metrics.average_precision(self.test_labels, self.masks)[0]
        print("")
        print(
            f">>> average precision at iou threshold 0.5 = {ap[:,0].mean():.3f}"
        )

    def _load_saved_data(self, files_labels):

        images = []
        labels = []
        names = []
        for fname in files_labels:
            if any(fname.endswith(f) for f in self.acceptable_formats):
                name = os.path.splitext(fname)[0]
                labelimage = imread(
                    os.path.join(self.save_real_mask_dir, fname)
                ).astype(np.uint16)
                image = imread(os.path.join(self.save_raw_dir, fname))

                labels.append(labelimage)
                images.append(image)
                names.append(name)

        return images, labels, names

    def _load_saved_test_data(self, files_labels):

        images = []
        labels = []
        names = []
        for fname in files_labels:
            if any(fname.endswith(f) for f in self.acceptable_formats):
                name = os.path.splitext(fname)[0]
                labelimage = imread(
                    os.path.join(self.save_test_real_mask_dir, fname)
                ).astype(np.uint16)
                image = imread(os.path.join(self.save_test_raw_dir, fname))

                labels.append(labelimage)
                images.append(image)
                names.append(name)

        return images, labels, names

    def _load_data(self, files_labels):

        images = []
        labels = []
        names = []
        for fname in files_labels:
            if any(fname.endswith(f) for f in self.acceptable_formats):
                name = os.path.splitext(fname)[0]
                labelimage = imread(
                    os.path.join(self.real_mask_dir, fname)
                ).astype(np.uint16)
                image = imread(os.path.join(self.raw_dir, fname))
                if not self.real_train_3D:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=os.cpu_count() - 1
                    ) as executor:
                        future_labels = [
                            executor.submit(slicer, labelimage, i)
                            for i in range(labelimage.shape[0])
                        ]
                        future_raw = [
                            executor.submit(slicer, image, i)
                            for i in range(image.shape[0])
                        ]
                        current_labels = [
                            r.result()
                            for r in concurrent.futures.as_completed(
                                future_labels
                            )
                        ]
                        current_raw = [
                            r.result()
                            for r in concurrent.futures.as_completed(
                                future_raw
                            )
                        ]
                    for i in range(len(current_labels)):
                        if current_labels[i].max() > 0:
                            labels.append(current_labels[i])
                            images.append(current_raw[i])
                            current_name = name + str(i)
                            names.append(current_name)

                else:
                    labels.append(labelimage)
                    images.append(image)
                    names.append(name)

        return images, labels, names


def slicer(image, i):

    return image[i, :]
