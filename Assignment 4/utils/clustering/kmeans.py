import numpy as np
from utils.functions import get_features



def kmeans(image, k=3, sample_size=5000, max_iters=500, include_position=False):
    image = image.copy().astype(float)
    H, W, C = image.shape
    features = get_features(image, include_position)
    print(features)
    sample = features[np.random.choice(np.arange(len(features)), sample_size)]
    centers = sample[np.random.choice(np.arange(len(sample)), k)]

    for _ in range(max_iters):
        print(_)
        distances = []
        for i in range(k):
            print(i)
            center = centers[i]
            diff = center - sample
            distance = np.sum(np.power(diff, 2), axis=1)
            distances.append(distance)
        distances = np.array(distances)

        groups = distances.argmin(axis=0)

        for i in range(k):
            subgroup = sample[groups == i,:]
            if len(subgroup) > 0:
                new_center = None
                new_center = np.mean(subgroup, axis=0)
            else:
                new_center = centers[i]

            centers[i] = new_center

    distances = []
    for i in range(k):
        center = centers[i]
        diff = center - features
        distance = np.sum(np.power(diff, 2), axis=1)
        distances.append(distance)

    distances = np.array(distances)
    groups = distances.argmin(axis=0)

    groups = groups.reshape((H, W))

    return centers, groups

