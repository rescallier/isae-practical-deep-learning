import os

try:
    from joblib import Memory
    if os.environ.get("TP_ISAE_DATA") is not None:
        cache_dir = os.path.join(os.path.expandvars(os.environ.get("TP_ISAE_DATA")), "cache")
    else:
        cache_dir = "/tmp/cache"

    memory = Memory(cache_dir, verbose=0)
except ImportError:
    memory = None
