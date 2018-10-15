import os
import sys

# For Python2 backward comability:
from builtins import range

# import facerec modules
from facerec.feature import PCA
from facerec.distance import EuclideanDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.visual import subplot
from facerec.util import minmax_normalize
from facerec.serialization import save_model, load_model
import numpy as np

try:
    from PIL import Image
except ImportError:
    import Image

def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes 

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                X.append(read_image(os.path.join(subject_path, filename), sz=sz))
                y.append(c)
            c = c+1
    return [X,y]

def read_image(fn, sz=None):
    try:
        im = Image.open(fn)
        im = im.convert("I")
        # resize to given size (if given)
        if sz is not None:
            im = im.resize(sz, Image.ANTIALIAS)
        return np.asarray(im, dtype=np.uint16)
    except IOError as e:
        print("I/O error: {0}".format(e))
        raise e
    except:
        print("Unexpected error: {0}".format(sys.exc_info()[0]))
        raise

def model_build(path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "res", "train"), feature=PCA(), dist_metric=EuclideanDistance(), k=1, sz=None):
    model_fn = os.path.join(path, "mdl.pkl")
    if not os.path.isfile(model_fn):
        [X,y] = read_images(path, sz=sz)
        classifier = NearestNeighbor(dist_metric=dist_metric, k=k)
        model = PredictableModel(feature=feature, classifier=classifier)
        model.compute(X, y)
        save_model(model_fn, model)
    return load_model(model_fn)

def model_rebuild(path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "res", "train"), feature=PCA(), dist_metric=EuclideanDistance(), k=1, sz=None):
    model_fn = os.path.join(path, "mdl.pkl")
    if os.path.isfile(model_fn):
        os.remove(model_fn)

    [X,y] = read_images(path, sz=sz)
    classifier = NearestNeighbor(dist_metric=dist_metric, k=k)
    model = PredictableModel(feature=feature, classifier=classifier)
    model.compute(X, y)
    save_model(model_fn, model)
    return load_model(model_fn)

def find_feature_similarity(model, frames, x, y, sz):
    distances = np.ndarray(x.shape)
    for i in range(x.shape[0]):
        fr = frames[i]
        x_fr = x.T[i]
        y_fr = y.T[i]
        for j in range(x_fr.size):
            if not np.isnan(x_fr[j]) and not np.isnan(y_fr[j]):
                xl = x_fr[j].astype(np.int) - (sz[0] // 2)
                xh = xl + sz[0]
                yl = y_fr[j].astype(np.int) - (sz[1] // 2)
                yh = yl + sz[1]

                xl = max(xl, 0)
                xh = min(xh, fr.shape[1])
                yl = max(yl, 0)
                yh = min(yh, fr.shape[0])

                sub_fr = fr[yl:yh,xl:xh].astype(np.uint32)

                if sub_fr.shape != sz:
                    sub_fr_img = Image.fromarray(sub_fr, "I").resize(sz, Image.ANTIALIAS)
                    sub_fr = np.asarray(sub_fr_img, dtype=np.uint16)

                distances[i,j] = model.predict(sub_fr)[1]['distances']
            else:
                distances[i,j] = np.nan
    return distances
