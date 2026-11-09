import unittest

import numpy as np
from PIL import Image

from src.tool.combine_layers import _resize_premultiplied, combine_layers


class CombineLayersTest(unittest.TestCase):
    def test_mismatched_sizes_use_first_layer_canvas(self):
        base = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
        layer = Image.new("RGBA", (16, 8), (0, 0, 255, 128))
        out = combine_layers([base, layer])
        self.assertEqual(out.size, (32, 32))
        self.assertEqual(out.mode, "RGBA")

    def test_composite_order_is_bottom_to_top(self):
        bottom = Image.new("RGBA", (8, 8), (255, 0, 0, 255))
        top = Image.new("RGBA", (8, 8), (0, 0, 255, 255))
        out = combine_layers([bottom, top])
        self.assertEqual(out.getpixel((4, 4)), (0, 0, 255, 255))

    def test_transparent_green_does_not_bleed_during_resize(self):
        layer = Image.new("RGBA", (4, 4), (0, 255, 0, 0))
        for y in range(4):
            for x in range(2):
                layer.putpixel((x, y), (255, 0, 0, 255))
        out = np.asarray(_resize_premultiplied(layer, (16, 16)))
        self.assertLess(int(out[..., 1].max()), 50)


if __name__ == "__main__":
    unittest.main()
