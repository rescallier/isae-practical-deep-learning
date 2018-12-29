import itertools

from tqdm.autonotebook import tqdm

try:
    import joblib
except ImportError:
    joblib = None


class Dataset(object):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self.items[item]

    def map(self, func, desc=None, n_jobs=1):
        if n_jobs > 1 and joblib is not None:
            items = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(func)(item) for item in tqdm(self.items, desc=desc))
        else:
            items = map(func, tqdm(self.items, desc=desc))

        return Dataset(items=list(items))

    def flatmap(self, func, desc=None, n_jobs=1):
        if n_jobs > 1 and joblib is not None:
            items = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(func)(item) for item in tqdm(self.items, desc=desc))
        else:
            items = map(func, tqdm(self.items, desc=desc))

        items = itertools.chain.from_iterable(items)

        return Dataset(items=list(items))

    def filter(self, func, desc=None):
        return Dataset(items=list(filter(func, tqdm(self.items, desc=desc))))

    def sample(self, sampler):
        return Dataset(items=sampler(self.items))

    def extend(self, dataset):
        return Dataset(items=self.items + dataset.items)
