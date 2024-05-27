from pathlib import Path
from typing import Any, Optional
import pandas as pd
from scipy import sparse
from .common import (ReadFromTxtToTransition, split_by_target_column_value,
                     BaseGraphContext,
                     ReadFromPandasToTransition,
                     MappingIntToObject)   
import numpy as np
from tqdm import tqdm


class PageRank():
    """ 
    Код класса является переработкой для конкретной задачи кода из
    https://github.com/danieljunhee/Tutorial-on-Personalized-PageRank/blob/master/Personalized_PageRank_Tutorial.ipynb
    """
    
    def __init__(self,
                transition_matrix_csr: sparse.csr_matrix,
                personalized_vec: Optional[sparse.csr_matrix],
                index_to_object: MappingIntToObject) -> None:
  
        # Запоминаем матрицу переходов, вероятности переходов для персонализируемой вершины,
        # и отображение целых чисел (индексов) в исходные объекты
        (self.__transition_matrix,
         self.__user_person_vec,
         self.__N_to_obj) = (transition_matrix_csr,
                            personalized_vec,
                            index_to_object)
         
          
    def get_pagerank(self,
                     a: float = 0.8,
                     b: float = 0.15,
                     c: float = 0.05,
                     max_iter: int = 100) -> pd.DataFrame:
        """ Вычисляет PageRank для графа.

        Метод итеративно вычисляет персонализированный PageRank с учетом 
        вектора предпочтений `user_person_vec` (если он есть), пока не будет достигнуто 
        максимальное количество итераций (`max_iter`).
        
        Важно: если нет персонализации, то коэфициент при телепортирующей матрице равен b + c

        Args:
            transition_matrix: Матрица переходов графа в формате 
                            разреженной матрицы CSR.
            user_person_vec: Вектор предпочтений пользователя, 
                            представленный разреженной матрицей CSR.
            max_iter: Максимальное количество итераций. По умолчанию 100.
            a: Параметр затухания (damping factor). По умолчанию 0.8.
            b: Параметр телепортации для персонализированного PageRank. 
            По умолчанию 0.15.
            c: Дополнительный параметр, используемый для выравнивания 
            вероятностей. По умолчанию 0.05.

        Returns:
            DataFrame, содержащий две колонки:
                - target: Целевые вершины графа.
                - pagerank: Значения PageRank, соответствующие 
                            целевым вершинам.
        """
        
        if self.__user_person_vec is not None:
            pagerank = self.__get_personalized_pagerank(
                self.__transition_matrix,
                self.__user_person_vec,
                max_iter,
                a, b, c
            )
        else:
            pagerank = self.__get_pagerank(
                self.__transition_matrix,
                max_iter,
                a, b + c
            )
            
         
        target_column = []
        pagerank_column = []
        for ind, el_pagerank in enumerate(pagerank):
            if self.__N_to_obj[ind][1] == BaseGraphContext.NodeType.TARGET:
                target_column.append(self.__N_to_obj[ind][0])
                pagerank_column.append(el_pagerank)
                
        

        return pd.DataFrame({BaseGraphContext.NodeType.TARGET.value: target_column , 'pagerank': pagerank_column})
    
    def __get_personalized_pagerank(self,
                       transition_matrix: sparse.csr_matrix,
                       user_person_vec: sparse.csr_matrix,
                       max_iter: int = 100,
                       a: float = 0.8,
                       b: float = 0.15,
                       c: float = 0.05) -> np.ndarray:
        """ Вычисляет персонализированный PageRank для графа.

        Метод итеративно вычисляет персонализированный PageRank с учетом 
        вектора предпочтений `user_person_vec`, пока не будет достигнуто 
        максимальное количество итераций (`max_iter`).

        Args:
            transition_matrix: Матрица переходов графа в формате 
                            разреженной матрицы CSR.
            user_person_vec: Вектор предпочтений пользователя, 
                            представленный разреженной матрицей CSR.
            max_iter: Максимальное количество итераций. По умолчанию 100.
            a: Параметр затухания (damping factor). По умолчанию 0.8.
            b: Параметр телепортации для персонализированного PageRank. 
            По умолчанию 0.15.
            c: Дополнительный параметр, используемый для выравнивания 
            вероятностей. По умолчанию 0.05.

        Returns:
            Одномерный массив NumPy, содержащий значения 
            персонализированного PageRank для каждой вершины графа.
        """
        
        # Initialize PageRank vector: height x 1 vector of all 1/height 's
        # We also represent the PageRank vector as a sparse csr matrix so that every arithmetic operation result
        # is stored as a sparse csr matrix throughout the algorithm... again dealing with sparsity issue & memory issue
        transition_matrix_T_mult_a = a * transition_matrix.transpose()
        user_person_vec_mult_b = b * user_person_vec
        height = transition_matrix.shape[0]
        user_pagerank_vec = sparse.csr_matrix(np.ones((height,1)) / height)
        sup_vec = c*sparse.csr_matrix((1/height)*np.ones((height,1)))


        for i in tqdm(range(max_iter)):         
            # Update!
            user_pagerank_vec = transition_matrix_T_mult_a.dot(user_pagerank_vec) \
                                + user_person_vec_mult_b.multiply(user_pagerank_vec) \
                                + sup_vec 
                                
            if i % 10 == 0:
                user_pagerank_vec /= user_pagerank_vec.sum()
            
        return np.concatenate(user_pagerank_vec.toarray())
    
    
    def __get_pagerank(self,
                       transition_matrix: sparse.csr_matrix,
                       max_iter: int = 100,
                       a: float = 0.85,
                       b: float = 0.15) -> np.ndarray:
        """ Вычисляет PageRank для графа, представленного матрицей переходов.

            Метод итеративно вычисляет PageRank, пока не будет достигнуто 
            максимальное количество итераций (`max_iter`).

            Args:
                transition_matrix: Матрица переходов графа в формате 
                                разреженной матрицы CSR.
                max_iter: Максимальное количество итераций. По умолчанию 100.
                a: Параметр затухания (damping factor). По умолчанию 0.85.
                b: Параметр телепортации для персонализированного PageRank. 
                По умолчанию 0.15.

            Returns:
                Одномерный массив NumPy, содержащий значения PageRank 
                для каждой вершины графа.
        """
        
        # Initialize PageRank vector: height x 1 vector of all 1/height 's
        # We also represent the PageRank vector as a sparse csr matrix so that every arithmetic operation result
        # is stored as a sparse csr matrix throughout the algorithm... again dealing with sparsity issue & memory issue
        
        transition_matrix_T_mult_a = a * transition_matrix.transpose()
        height = transition_matrix.shape[0]
        user_pagerank_vec = sparse.csr_matrix(np.ones((height,1)) / height)
        sup_vec = b * sparse.csr_matrix((1/height)*np.ones((height,1)))

        for i in tqdm(range(max_iter)):         
            # Update!
            user_pagerank_vec = transition_matrix_T_mult_a.dot(user_pagerank_vec) \
                                + sup_vec 
                                
            if i % 10 == 0:
                user_pagerank_vec /= user_pagerank_vec.sum()
            

        return np.concatenate(user_pagerank_vec.toarray())
    
    
class PagerankFactory():
    
    def read_pd(self,
            data: pd.DataFrame,
            source: str, target: str,
            edge_attr: str,
            personalization: Optional[Any] = None) -> PageRank:
        """ Создает объект PageRank из данных Pandas DataFrame.

        Args:
            data: DataFrame, содержащий ребра графа.
            source: Название столбца с исходными вершинами ребер.
            target: Название столбца с целевыми вершинами ребер.
            edge_attr: Название столбца с весами ребер.
            personalization: Опциональный объект, для которого будет 
                            персонализирован PageRank. Если None персонализации не будет

        Returns:
            Объект PageRank, инициализированный на основе данных DataFrame.
        """
        
        (transition_matrix_csr,
            pers_vec,
            N_to_obj) = ReadFromPandasToTransition()._read_pd(
                data,
                source,
                target,
                edge_attr,
                personalization)
            
        return PageRank(
            transition_matrix_csr,
            pers_vec,
            N_to_obj)
        
    def read_txt(self,
            file_name: Path,
            personalization: Optional[Any] = None):
        (transition_matrix_csr,
            pers_vec,
            N_to_obj) = ReadFromTxtToTransition()._read_txt(file_name, personalization)
            
        return PageRank(
            transition_matrix_csr,
            pers_vec,
            N_to_obj)
        
    

    
if __name__ == '__main__':
    data_test = pd.DataFrame({
        'source': ['A', 'B', 'B', 'B'],
        'target': ['X', 'X', 'Z', 'C'],
        'edge_attr': [4, 4, 5, 3]
    })  
        
    pg = PagerankFactory().read_pd(data=data_test, source='source', target='target',
                        edge_attr='edge_attr', personalization='A')
    print(pg.get_pagerank())