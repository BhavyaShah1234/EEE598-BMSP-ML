def make_temporal_feature_names(base_features, window=1):
    offsets = list(range(-window, window+1))   # e.g. [-1, 0, 1]
    names = []
    for off in offsets:
        suffix = f"t{off:+d}"     # t-1, t+1
        for f in base_features:
            names.append(f"{f}_{suffix}")
    return names