from auxiliary.image_handler import ImageHandler

image_handler = ImageHandler(image2Ddir="/home/kaorikizuna/Dot Vision/dot-vision/coordinates/nick/image_2D.png",
                             image3Ddir="/home/kaorikizuna/Dot Vision/dot-vision/coordinates/nick/image_3D.png")


def test_image_handler():
    assert image_handler.image2d is not None
    assert image_handler.image3d is not None


def test_image_width_height():
    assert image_handler.image2d.width == 590
    assert image_handler.image2d.height == 543


def test_image_max_total():
    assert image_handler.max_height == 543
    assert image_handler.total_width == 590 + 678
