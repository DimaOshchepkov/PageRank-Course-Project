import pandas as pd
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import norm
from typing import Tuple, Mapping, Any, Literal
from tqdm import tqdm

MappingIntToObject = Mapping[int, Tuple[Any, Literal['source', 'target']]]


