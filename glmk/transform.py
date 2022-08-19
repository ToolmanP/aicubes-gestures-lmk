import numpy as np

def normalize(img:np.ndarray):
    img = img.astype(dtype=np.float32)
    maxx = np.amax(img,axis=tuple(range(len(img.shape)-1)))
    minn = np.amin(img,axis=tuple(range(len(img.shape)-1)))
    img = (img-minn)/(maxx-minn)
    return img