
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import networkx as nx
import random

from io import BytesIO
from itertools import chain
from collections import namedtuple, OrderedDict


Sentence = namedtuple("Sentence", "words tags")

def read_data(filename):
    """Read tagged sentence data"""
    with open(filename, 'r') as f:
        sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
    return OrderedDict(((s[0], Sentence(*zip(*[l.strip().split("\t")
                        for l in s[1:]]))) for s in sentence_lines if s[0]))


def read_tags(filename):
    """Read a list of word tag classes"""
    with open(filename, 'r') as f:
        tags = f.read().split("\n")
    return frozenset(tags)


def model2png(model, filename="", overwrite=False, show_ends=False):
    """Convert a Pomegranate model into a PNG image

    The conversion pipeline extracts the underlying NetworkX graph object,
    converts it to a PyDot graph, then writes the PNG data to a bytes array,
    which can be saved as a file to disk or imported with matplotlib for display.

        Model -> NetworkX.Graph -> PyDot.Graph -> bytes -> PNG

    Parameters
    ----------
    model : Pomegranate.Model
        The model object to convert. The model must have an attribute .graph
        referencing a NetworkX.Graph instance.

    filename : string (optional)
        The PNG file will be saved to disk with this filename if one is provided.
        By default, the image file will NOT be created if a file with this name
        already exists unless overwrite=True.

    overwrite : bool (optional)
        overwrite=True allows the new PNG to overwrite the specified file if it
        already exists

    show_ends : bool (optional)
        show_ends=True will generate the PNG including the two end states from
        the Pomegranate model (which are not usually an explicit part of the graph)
    """
    nodes = model.graph.nodes()
    if not show_ends:
        nodes = [n for n in nodes if n not in (model.start, model.end)]
    g = nx.relabel_nodes(model.graph.subgraph(nodes), {n: n.name for n in model.graph.nodes()})
    pydot_graph = nx.drawing.nx_pydot.to_pydot(g)
    pydot_graph.set_rankdir("LR")
    png_data = pydot_graph.create_png(prog='dot')
    img_data = BytesIO()
    img_data.write(png_data)
    img_data.seek(0)
    if filename:
        if os.path.exists(filename) and not overwrite:
            raise IOError("File already exists. Use overwrite=True to replace existing files on disk.")
        with open(filename, 'wb') as f:
            f.write(img_data.read())
        img_data.seek(0)
    return mplimg.imread(img_data)


def show_model(model, figsize=(5, 5), **kwargs):
    """Display a Pomegranate model as an image using matplotlib

    Parameters
    ----------
    model : Pomegranate.Model
        The model object to convert. The model must have an attribute .graph
        referencing a NetworkX.Graph instance.

    figsize : tuple(int, int) (optional)
        A tuple specifying the dimensions of a matplotlib Figure that will
        display the converted graph

    **kwargs : dict
        The kwargs dict is passed to the model2png program, see that function
        for details
    """
    plt.figure(figsize=figsize)
    plt.imshow(model2png(model, **kwargs))
    plt.axis('off')


class Subset(namedtuple("BaseSet", "sentences keys vocab X tagset Y N stream")):
    def __new__(cls, sentences, keys):
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        wordset = frozenset(chain(*word_sequences))
        tagset = frozenset(chain(*tag_sequences))
        N = sum(1 for _ in chain(*(sentences[k].words for k in keys)))
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        return super().__new__(cls, {k: sentences[k] for k in keys}, keys, wordset, word_sequences,
                               tagset, tag_sequences, N, stream.__iter__)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences.items())


class Dataset(namedtuple("_Dataset", "sentences keys vocab X tagset Y training_set testing_set N stream")):
    def __new__(cls, tagfile, datafile, train_test_split=0.8, seed=112890):
        tagset = read_tags(tagfile)
        sentences = read_data(datafile)
        keys = tuple(sentences.keys())
        wordset = frozenset(chain(*[s.words for s in sentences.values()]))
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        N = sum(1 for _ in chain(*(s.words for s in sentences.values())))
        
        # split data into train/test sets
        _keys = list(keys)
        if seed is not None: random.seed(seed)
        random.shuffle(_keys)
        split = int(train_test_split * len(_keys))
        training_data = Subset(sentences, _keys[:split])
        testing_data = Subset(sentences, _keys[split:])
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        return super().__new__(cls, dict(sentences), keys, wordset, word_sequences, tagset,
                               tag_sequences, training_data, testing_data, N, stream.__iter__)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences.items())
