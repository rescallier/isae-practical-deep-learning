class Predictor:
    """
    A predictor is simply a wrapper over a model to predict aircraft or background
    FN predict_on_tile should return either [aircraft or background]
    use __init__ to load your model !

    It supports batching if you set your batch_size as > 1 (it will then call `predict_on_tiles`)
    """
    def __init__(self, batch_size=1):
        self.batch_size = batch_size

    def predict_on_batch(self, tiles_data):
        """
        If you want to implement batching
        This should encapsulate preprocessing + self.model.predict_on_batch() where self.model is the loaded keras model
        Args:
            tiles_data(Iterator[np.ndarray]): A list of images

        Returns:
            A list of labels "aircraft" or "background"

        """
        raise NotImplementedError
