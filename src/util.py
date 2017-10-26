""" Useful helper functions """

def dot_product(vec1, vec2):
    """
    @param dict vec1: a feature vector represented by a mapping from a feature
        (string) to a weight (float).
    @param dict vec2: same as vec1
    @return float: the dot product between vec1 and vec2
    """
    if len(vec1) < len(vec2):
        return dot_product(vec2, vec1)
    else:
        return sum(vec1.get(f, 0) * v for f, v in vec2.items())

def increment(vec1, scale, vec2):
    """
    Implements vec1 += scale * vec2 for sparse vectors.
    @param dict vec1: the feature vector which is mutated.
    @param float scale
    @param dict vec2: a feature vector.
    """
    for feature, val in vec2.items():
        vec1[feature] = vec1.get(feature, 0) + val * scale
