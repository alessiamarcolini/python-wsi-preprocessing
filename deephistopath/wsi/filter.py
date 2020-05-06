# encoding: utf-8

# ------------------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------

"""Provides the Filter class.

Filter is the main API class for applying filters to WSI slides images.
"""
import math
import multiprocessing
import numpy as np
import os

import scipy.ndimage.morphology as sc_morph
import skimage.color as sk_color
import skimage.exposure as sk_exposure
import skimage.feature as sk_feature
import skimage.filters as sk_filters
import skimage.future as sk_future
import skimage.morphology as sk_morphology
import skimage.segmentation as sk_segmentation

from deephistopath.wsi import slide
from deephistopath.wsi import util
from deephistopath.wsi.util import Time


class Filter(object):
    def __init__(self, slide):
        self._slide = slide

    # ---public interface methods and properties---

    def filter_rgb_to_grayscale(self, output_type="uint8"):
        """
        Convert an RGB NumPy array to a grayscale NumPy array.

        Shape (h, w, c) to (h, w).

        Returns:
            Grayscale image as NumPy array with shape (h, w).
        """
        # Another common RGB ratio possibility: [0.299, 0.587, 0.114]
        grayscale = np.dot(
            self._slide.resampled_array[..., :3], [0.2125, 0.7154, 0.0721]
        )
        grayscale = (
            grayscale.astype(output_type)
            if output_type == "float"
            else grayscale.astype("uint8")
        )
        return grayscale

    def filter_greyscale_complement(self, output_type):
        """
        Obtain the complement of an image as a NumPy array.

        Args:
            type: Type of array to return (float or uint8).

        Returns:
            Complement image as Numpy array.
        """
        greyscale_image_array = self.filter_rgb_to_grayscale(output_type)
        complement = (
            1.0 - greyscale_image_array
            if output_type == "float"
            else 255 - greyscale_image_array
        )
        return complement

    def filter_hysteresis_threshold(self, low=50, high=100, output_type="uint8"):
        """
        Apply two-level (hysteresis) threshold to an image as a NumPy array, returning a binary image.

        Args:
            low: Low threshold.
            high: High threshold.
            output_type: Type of array to return (bool, float, or uint8).

        Returns:
            NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above hysteresis threshold.
        """
        hyst = sk_filters.apply_hysteresis_threshold(
            self._slide.resampled_array, low, high
        )
        return self._type_dispatcher(hyst, output_type)

    def filter_otsu_threshold(self, output_type="uint8"):
        """
        Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.

        Args:
            np_img: Image as a NumPy array.
            output_type: Type of array to return (bool, float, or uint8).

        Returns:
            NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
        """
        otsu_thresh_value = sk_filters.threshold_otsu(self._slide.resampled_array)
        otsu = self._slide.resampled_array > otsu_thresh_value
        return self._type_dispatcher(otsu, output_type)

    def filter_local_otsu_threshold(self, disk_size=3, output_type="uint8"):
        """
        Compute local Otsu threshold for each pixel and return binary image based on pixels being less than the
        local Otsu threshold.

        Args:
            disk_size: Radius of the disk structuring element used to compute the Otsu threshold for each pixel.
            output_type: Type of array to return (bool, float, or uint8).

        Returns:
            NumPy array (bool, float, or uint8) where local Otsu threshold values have been applied to original image.
        """
        local_otsu = sk_filters.rank.otsu(
            self._slide.resampled_array, sk_morphology.disk(disk_size)
        )
        return self._type_dispatcher(local_otsu, output_type)

    def filter_entropy(self, neighborhood=9, threshold=5, output_type="uint8"):
        """
        Filter image based on entropy (complexity).

        Args:
            neighborhood: Neighborhood size (defines height and width of 2D array of 1's).
            threshold: Threshold value.
            output_type: Type of array to return (bool, float, or uint8).

        Returns:
            NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a measure of complexity.
        """
        entropy = (
            sk_filters.rank.entropy(
                self._slide.resampled_array, np.ones((neighborhood, neighborhood))
            )
            > threshold
        )
        return self._type_dispatcher(entropy, output_type)

    def filter_canny(
        self, sigma=1, low_threshold=0, high_threshold=25, output_type="uint8"
    ):
        """
        Filter image based on Canny algorithm edges.

        Args:
            sigma: Width (std dev) of Gaussian.
            low_threshold: Low hysteresis threshold value.
            high_threshold: High hysteresis threshold value.
            output_type: Type of array to return (bool, float, or uint8).

        Returns:
            NumPy array (bool, float, or uint8) representing Canny edge map (binary image).
        """
        filter_canny = sk_feature.canny(
            self._slide.resampled_array,
            sigma=sigma,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        return self._type_dispatcher(filter_canny, output_type)

    def mask_percent(self, np_array):
        """
        Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).

        Returns:
            The percentage of the NumPy array that is masked.
        """
        if _is_rgb(np_array):
            squashed = np.sum(np_array, axis=(2))
            mask_percentage = 100 - np.count_nonzero(squashed) / squashed.size * 100
        else:
            mask_percentage = 100 - np.count_nonzero(np_array) / np_array.size * 100
        return mask_percentage

    def tissue_percent(self, np_array):
        """
        Determine the percentage of a NumPy array that is tissue (not masked).

        Returns:
            The percentage of the NumPy array that is tissue.
        """
        return 100 - self.mask_percent(np_array)

    def filter_remove_small_objects(
        self,
        np_array,
        min_size=3000,
        avoid_overmask=True,
        overmask_thresh=95,
        output_type="uint8",
    ):
        """
        Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
        is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
        reduce the amount of masking that this filter performs.

        Args:
            min_size: Minimum size of small object to remove.
            avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
            overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
            output_type: Type of array to return (bool, float, or uint8).

        Returns:
            NumPy array (bool, float, or uint8).
        """
        rem_sm = np_array.astype(bool)  # make sure mask is boolean
        rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
        if (
            (self.mask_percent(rem_sm) >= overmask_thresh)
            and (min_size >= 1)
            and (avoid_overmask is True)
        ):
            new_min_size = min_size / 2
            rem_sm = self.filter_remove_small_objects(
                np_array, new_min_size, avoid_overmask, overmask_thresh, output_type,
            )
        return self._type_dispatcher(rem_sm, output_type)

    def filter_remove_small_holes(self, np_array, min_size=3000, output_type="uint8"):
        """
        Filter image to remove small holes less than a particular size.

            min_size: Remove small holes below this size.
            output_type: Type of array to return (bool, float, or uint8).

        Returns:
            NumPy array (bool, float, or uint8).
        """

        rem_sm = sk_morphology.remove_small_holes(np_array, min_size=min_size)

        return self._type_dispatcher(rem_sm, output_type)

    def filter_contrast_stretch(self, np_array, low=40, high=60):
        """
        Filter image (gray or RGB) using contrast stretching to increase contrast in image based on the intensities in
        a specified range.

        Args:
            np_img: Image as a NumPy array (gray or RGB).
            low: Range low value (0 to 255).
            high: Range high value (0 to 255).

        Returns:
            Image as NumPy array with contrast enhanced.
        """
        low_p, high_p = np.percentile(np_array, (low * 100 / 255, high * 100 / 255))
        contrast_stretch = sk_exposure.rescale_intensity(
            np_array, in_range=(low_p, high_p)
        )
        return contrast_stretch

    # ---private interface methods and properties---

    @staticmethod
    def _is_rgb(np_array):
        if np_array.ndim == 3 and np_array.ndim.shape[2] == 3:
            return True
        return False

    def _type_dispatcher(self, np_array, output_type):
        _map = {"bool": np_array.astype("bool"), "float": np_array.astype("float")}
        return _map.get(output_type, (255 * np_array).astype("uint8"))
