import json

from khumeia.roi.tile import Tile, LabelledTile


class SlidingWindow:
    """

    Sliding windows play an integral role in object classification, as they allow us to localize exactly “where” in an
    image an object resides.

    Sliding window approaches are simple in concept, a bounding box of the desired size(s) slides across the test image
    and at each location applies an image classifier to the current window

    ![example1](https://cdn-images-1.medium.com/max/800/1*FHEOyHm1BTWyygQcgvNSXQ.png)

    Sample cutouts of a sliding window iterating from top to bottom (Imagery Courtesy of DigitalGlobe)

    ![example2](https://cdn-images-1.medium.com/max/800/1*BkQLxT_FVz6XqHul5qezEw.gif)

    Sliding window shown iterating across an image (left).
    An image classifier is applied to these cutouts and anything resembling a boat is saved as a positive (right)
    (Imagery Courtesy of DigitalGlobe)

    """
    def __init__(self,
                 tile_size=64,
                 padding=0,
                 stride=64,
                 label_assignment_mode="center",
                 intersection_over_area_threshold=0.5,
                 margin_from_bounds=0,
                 discard_background=False,
                 data_transform_fn=None):
        """

        Args:
            tile_size(int): tile size (h,w) in pixels
            padding(int):  padding in pixels. best keep it to 0
            stride(int): Stride ("pas") in pixels
            label_assignment_mode: "center" or "ioa",
                if center: If a tile contains a groundtruth's center it gets its label
                if ioa: Calculates the intersection over min(area_tile,area_groundtruth), if the ioa > threshold, then
                assigns
            margin_from_bounds: internal margin to use if "center" is selected
            intersection_over_area_threshold(float): threshold
            data_transform_fn: Useful to generate augmented samples or to apply a specific preprocessing
        """
        self.tile_size = tile_size
        self.stride = stride
        self.padding = padding
        self.label_assignment_mode = label_assignment_mode
        self.ioa_threshold = intersection_over_area_threshold
        self.margin_from_bounds = margin_from_bounds
        self.discard_background = discard_background
        self.data_transform_fn = data_transform_fn

    def get_tiles_for_item(self, item):
        """
            Apply the sliding window on a full satellite images to generate a list of tiles
            Compares the tiles to the item's groundtruths
            Tiles that matches the declared conditions are assigned with the groundtruth's label

        Args:
            item(tp_isae_helpers.satellite_image.SatelliteImage): input item

        Returns:
            list[LabelledTile]: A list of tile with their labels

        """

        labels = item.labels

        tiles = Tile.get_tiles_for_item(item.key,
                                        item.shape,
                                        tile_shape=(self.tile_size, self.tile_size),
                                        padding=self.padding,
                                        stride=float(self.stride) / self.tile_size,
                                        data_transform_fn=self.data_transform_fn)

        tiles_with_labels = map(
            lambda tile: LabelledTile.from_tile_and_groundtruths(tile,
                                                                 labels,
                                                                 label_assignment_mode=self.label_assignment_mode,
                                                                 ioa_threshold=self.ioa_threshold,
                                                                 margin_from_bounds=self.margin_from_bounds), tiles)

        if self.discard_background:
            tiles_with_labels = filter(lambda tile: tile.label != "background", tiles_with_labels)

        return tiles_with_labels

    def __call__(self, item):
        return self.get_tiles_for_item(item)

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4)

    def __str__(self):
        d = dict()
        d['class'] = self.__class__.__name__
        d.update(self.__dict__)
        return json.dumps(d, indent=4)
