import pandas as pd
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import norm
from typing import Tuple, Mapping, Any, Literal
from tqdm import tqdm

MappingIntToObject = Mapping[int, Tuple[Any, Literal['source', 'target']]]


class EdgeListDataInputMixin():
    
    def preprocess_from_edgelist(self, data:pd.DataFrame, source: str, target:str, edge_attr:str):
        pass
    
class EdgeListDataInputMixin2():
    
    def preprocessing2(self, data:np.ndarray):
        pass
    
    
class PageRank():
    
    def __init__(self, data:pd.DataFrame, source: str, target:str, edge_attr:str,
                 personalize_by: Any, a: float = 0.8, b: float = 0.15, c: float=0.05,
                 max_iter: int = 100) -> None:
        # Decide values of a,b,c. For this tutorial, let's use a = 0.8, b = 0.15, c = 0.05.
        self.a = a
        self.b = b
        self.c = c
        self.target = target
        self.source = source
        self.max_iter = max_iter
        
        (self.__transition_matrix, self.__user_person_vec,
         self.__N_to_obj) = self.__create_transition_matrix_from_pd(data, source,
                                                             target, edge_attr, personalize_by)
         
        
    def __create_transition_matrix_from_pd(self, data: pd.DataFrame,
               source: str, target: str, edge_attr: str,
               personalization: Any) -> Tuple[sparse.csr_matrix, MappingIntToObject]:
    
        obj_source_to_N = {}
        obj_target_to_N = {}
        N_to_obj = {}
        
        source_uniq = set(data[source])
        for index, obj in zip(range(len(source_uniq)), source_uniq):
            obj_source_to_N[obj] = index
            N_to_obj[index] = (obj, 'source')
            
        
        target_uniq = set(data[target])
        for index, obj in zip(range(len(source_uniq), len(source_uniq) + len(target_uniq)), target_uniq):
            obj_target_to_N[obj] = index
            N_to_obj[index] = (obj, 'target')
        
        count_top = len(N_to_obj)
        transition_matrix = sparse.lil_matrix((count_top, count_top))
        
        # Функция для отображения значений согласно словарю
        def map_source_to_N(value):
            return obj_source_to_N[value]

        def map_target_to_N(value):
            return obj_target_to_N[value]

        # Создаем векторизованную версию функций
        vectorized_source_to_N = np.vectorize(map_source_to_N)
        vectorized_target_to_N = np.vectorize(map_target_to_N)

        # Применяем векторизованные функции к массивам
        start_nodes = vectorized_source_to_N(data[source].to_numpy())  # Начальные вершины ребер
        end_nodes = vectorized_target_to_N(data[target].to_numpy())    # Конечные вершины ребер
        weights = np.array(data[edge_attr])
        
        # Заполнение матрицы значениями из массивов
        transition_matrix[start_nodes, end_nodes] = weights

        # Если граф неориентированный, добавьте обратные ребра
        transition_matrix[end_nodes, start_nodes] = weights
        
        # Получаем суммы элементов каждой строки в виде numpy массива
        
        transition_matrix_csr = sparse.csr_matrix(transition_matrix)
        del transition_matrix
        
        row_sums = transition_matrix_csr.sum(axis=1)
        transition_matrix_csr /= row_sums
        
        user_person_vec = np.zeros(transition_matrix_csr.shape[1])
        #user_person_vec[obj_source_to_N[personalization]] = 1
        pers_data = data.loc[data[source] == personalization]
        for ind, row in pers_data.iterrows():
            user_person_vec[obj_target_to_N[row[target]]] = row[edge_attr]
        user_person_vec /= user_person_vec.sum()

        # transition_matrix_csr.getrow(obj_source_to_N[personalization]).transpose()
        return (transition_matrix_csr,
                transition_matrix_csr.getrow(obj_source_to_N[personalization]).transpose(),
                N_to_obj)
    
    def get_pagerank(self) -> pd.DataFrame:
        
        pagerank = self.__get_pagerank(self.__transition_matrix,
                                       self.__user_person_vec, self.max_iter)
         
         
        target_column = []
        pagerank_column = []
        for ind, el_pagerank in enumerate(pagerank):
            if self.__N_to_obj[ind][1] == 'target':
                target_column.append(self.__N_to_obj[ind][0])
                pagerank_column.append(el_pagerank)

        return pd.DataFrame({self.target: target_column , 'pagerank': pagerank_column})
    
    def __get_pagerank(self, transition_matrix: sparse.csr_matrix,
                       user_person_vec: sparse.csr_matrix, max_iter: int = 200) -> np.ndarray:
        
        # Initialize PageRank vector: height x 1 vector of all 1/height 's
        # We also represent the PageRank vector as a sparse csr matrix so that every arithmetic operation result
        # is stored as a sparse csr matrix throughout the algorithm... again dealing with sparsity issue & memory issue
        
        transition_matrix_T_mult_a = self.a * transition_matrix.transpose()
        user_person_vec_mult_b = self.b* user_person_vec
        height = transition_matrix.shape[0]
        user_pagerank_vec = sparse.csr_matrix(np.ones((height,1)) / height)
        sup_vec = self.c*sparse.csr_matrix((1/height)*np.ones((height,1)))

        # Инициализация прогресс-бара
        progress_bar = tqdm(total=max_iter)
        # Update the PageRank vector until convergence! Our convergence criterion is to see whether
        # the magnitude of the difference of the PageRank vector after an update is 0
        for i in range(max_iter):
            # (deep) copy the current PageRank vector to compare with the updated vector
            old_user_pagerank_vec = user_pagerank_vec.copy()
            
            # Update!
            user_pagerank_vec = transition_matrix_T_mult_a.dot(user_pagerank_vec) \
                                + user_person_vec_mult_b.multiply(user_pagerank_vec) \
                                + sup_vec 
                                
            if i % 10 == 0:
                user_pagerank_vec /= user_pagerank_vec.sum()
            
            magnitude = norm(user_pagerank_vec-old_user_pagerank_vec)
            progress_bar.set_description("magnitude: {:.6f}".format(magnitude))
            progress_bar.update()
            
            if magnitude == 0:
                break
        progress_bar.close()
        return np.concatenate(user_pagerank_vec.toarray())
    
    def set_params(self, a: float = None, b: float = None, c: float = None,
                   max_iter: int = None) -> None:
        self.a = a if a is not None else self.a
        self.b = b if b is not None else self.b
        self.c = c if c is not None else self.c
        self.max_iter = max_iter if max_iter is not None else self.max_iter