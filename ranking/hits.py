from pathlib import Path
from typing import Any, Optional, Tuple
import pandas as pd
from scipy import sparse
import scipy
from .common import (ReadFromTxtToTransition, split_by_target_column_value,
                     BaseGraphContext,
                     ReadFromPandasToTransition,
                     MappingIntToObject)                  
from tqdm import tqdm
import numpy as np


# https://dspace.mit.edu/bitstream/handle/1721.1/86737/49317146-MIT.pdf?sequence=2&isAllowed=y
class HITS():
    
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
         
    
    def get_rank(self,
                     max_iter: int = 100) -> pd.DataFrame:
        
        authoritativeness, hubness = self.__get_rank(self.__transition_matrix,
                                                    user_rank_vec=self.__user_person_vec,
                                                    max_iter=max_iter)
         
         
        target_column = []
        rank_column_auth = []
        rank_column_hub = []
        for ind, (a, h) in enumerate(zip(authoritativeness, hubness)):
            if self.__N_to_obj[ind][1] == BaseGraphContext.NodeType.TARGET:
                target_column.append(self.__N_to_obj[ind][0])
                rank_column_auth.append(a)
                rank_column_hub.append(h)

        return pd.DataFrame({BaseGraphContext.NodeType.TARGET.value: target_column,
                             'authoritativeness': rank_column_auth,
                             'hubness': rank_column_hub})
    
    def __get_rank(self,
                       transition_matrix: sparse.csr_matrix,
                       user_rank_vec: sparse.csr_matrix,
                       max_iter: int) -> Tuple[np.ndarray, np.ndarray]:
        
        
        # transition_T = transition_matrix.transpose()
        # hub = user_rank_vec.copy()
        # auth = user_rank_vec.copy()

        # # Update the PageRank vector until convergence! Our convergence criterion is to see whether
        # # the magnitude of the difference of the PageRank vector after an update is 0
        # for i in tqdm(range(max_iter)):
        #     # (deep) copy the current PageRank vector to compare with the updated vector
            
        #     old_hub = hub.copy() 
        #     old_auth = auth.copy()
                   
        #     old_hub /= old_hub.sum()
        #     old_auth /= old_auth.sum()
            
        #     # Update!
        #     hub = sparse.csr_matrix(transition_matrix.multiply(old_auth.T).sum(axis=1))
        #     auth = sparse.csr_matrix(transition_T.multiply(old_hub.T).sum(axis=1))
        _, _, vt = scipy.sparse.linalg.svds(transition_matrix, k=1, maxiter=max_iter)
        a = vt.flatten().real
        h = transition_matrix @ a                    
        h /= h.sum()
        a /= a.sum()
        
        return a.flatten(), h.flatten()
    
class HITSFactory():
    
    def read_pd(self,
            data: pd.DataFrame,
            source: str, target: str,
            edge_attr: str,
            personalization: Optional[Any] = None,
            directional: bool = False) -> HITS:
        """ Создает объект HITS из данных Pandas DataFrame.

        Args:
            data: DataFrame, содержащий ребра графа.
            source: Название столбца с исходными вершинами ребер.
            target: Название столбца с целевыми вершинами ребер.
            edge_attr: Название столбца с весами ребер.
            personalization: Опциональный объект, для которого будет 
                            персонализирован HITS. Если None персонализации не будет

        Returns:
            Объект HITS, инициализированный на основе данных DataFrame.
        """
        
        (transition_matrix_csr,
            pers_vec,
            N_to_obj) = ReadFromPandasToTransition()._read_pd(
                data,
                source,
                target,
                edge_attr,
                personalization,
                directional)
            
        return HITS(
            transition_matrix_csr,
            pers_vec,
            N_to_obj)
        
    def read_txt(self,
            file_name: Path,
            personalization: Optional[Any] = None):
        
        (transition_matrix_csr,
            pers_vec,
            N_to_obj) = ReadFromTxtToTransition()._read_txt(file_name, personalization)
            
        return HITS(
            transition_matrix_csr,
            pers_vec,
            N_to_obj)

  
def main():
    data_test = pd.DataFrame({
        'source': ['A', 'B', 'B', 'B'],
        'target': ['X', 'X', 'Z', 'C'],
        'edge_attr': [4, 4, 5, 3]
    })  
        
    pg = HITSFactory().read_pd(data_test, 'source', 'target', 'edge_attr', 'A')
    print(pg.get_rank())
    
if __name__ == '__main__':
    main()