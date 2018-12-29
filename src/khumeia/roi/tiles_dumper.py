import os

from khumeia.utils import io_utils


class ItemTileDumper(object):
    def __init__(self, item, output_dir, save_format="jpg"):
        self.item = item
        self.image = item.image
        self.output_dir = output_dir
        self.save_format = save_format

    def dump_tiles_for_item(self, tile):
        if tile.item_id == self.item.key:

            try:
                os.makedirs(os.path.join(self.output_dir, tile.label))
            except OSError:
                pass

            tile_data = tile.get_data(self.image)
            tile_basename = "{}_{}.{}".format(self.item.key, tile.key, self.save_format)
            io_utils.imsave(os.path.join(self.output_dir, tile.label, tile_basename), tile_data)

            return os.path.join(self.output_dir, tile.label, tile_basename), tile.label
        else:
            return None

    def __call__(self, tile):
        return self.dump_tiles_for_item(tile)
