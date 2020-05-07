# encoding: utf-8

import pytest

from .unitutil import property_mock, initializer_mock, ANY
from deephistopath.wsi.slide import Slide


class DescribeSlide(object):
    def it_constructs_from_args(self, request):
        _init_ = initializer_mock(request, Slide)
        _wsi_path = "/foo/bar/myslide.svs"
        _scale_factor = 22

        slide = Slide(_wsi_path, _scale_factor)

        _init_.assert_called_once_with(ANY, _wsi_path, _scale_factor)
        assert isinstance(slide, Slide)

    def but_it_has_wrong_wsi_path_type(self):
        with pytest.raises(TypeError) as err:
            slide = Slide(None, scale_factor=33)
            wsi_name = slide.wsi_name

        assert isinstance(err.value, TypeError)
        assert (
            str(err.value) == "expected str, bytes or os.PathLike object, not NoneType"
        )

    def it_generates_the_correct_breadcumb(self, request, breadcumb_fixture):
        resampled_dims, dir_path, wsi_path, scale_factor, expected_path = (
            breadcumb_fixture
        )
        _resampled_dimensions = property_mock(request, Slide, "_resampled_dimensions")
        _resampled_dimensions.return_value = resampled_dims
        directpry_path = dir_path
        _wsi_path = wsi_path
        _scale_factor = scale_factor
        slide = Slide(_wsi_path, _scale_factor)

        _breadcumb = slide._breadcumb(directpry_path)

        assert _breadcumb == expected_path

    def it_knows_its_wsi_name(self, wsi_name_fixture):
        _wsi_path, expected_value = wsi_name_fixture
        slide = Slide(_wsi_path, scale_factor=32)

        wsi_name = slide.wsi_name

        assert wsi_name == expected_value

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            (
                (245, 123, 145, 99),
                "/foo/bar/b/0/9",
                "/foo/bar/myslide.svs",
                64,
                "/foo/bar/b/0/9/myslide-64x-245x123-145x99.png",
            ),
            (
                (245, 123, 145, 99),
                "/foo/bar/b/0/9",
                "/foo/bar/myslide.svs",
                32,
                "/foo/bar/b/0/9/myslide-32x-245x123-145x99.png",
            ),
            (
                (None, None, None, None),
                "/foo/bar/b/0/9",
                "/foo/bar/myslide.svs",
                64,
                "/foo/bar/b/0/9/myslide*.png",
            ),
            (
                (None, 234, 192, None),
                "/foo/bar/b/0/9",
                "/foo/bar/myslide.svs",
                64,
                "/foo/bar/b/0/9/myslide-64x-Nonex234-192xNone.png",
            ),
            (
                (123, 234, 192, None),
                "/foo/bar/b/0/9",
                "/foo/bar/myslide.svs",
                64,
                "/foo/bar/b/0/9/myslide-64x-123x234-192xNone.png",
            ),
            (
                (None, None, 192, None),
                "/foo/bar/b/0/9",
                "/foo/bar/myslide.svs",
                64,
                "/foo/bar/b/0/9/myslide-64x-NonexNone-192xNone.png",
            ),
        ]
    )
    def breadcumb_fixture(self, request):
        resampled_dims, dir_path, wsi_path, scale_factor, expected_path = request.param
        return resampled_dims, dir_path, wsi_path, scale_factor, expected_path

    @pytest.fixture(
        params=[("/foo/bar/myslide.svs", "myslide"), ("/foo/myslide.svs", "myslide")]
    )
    def wsi_name_fixture(self, request):
        wsi_path, expceted_value = request.param
        return wsi_path, expceted_value
