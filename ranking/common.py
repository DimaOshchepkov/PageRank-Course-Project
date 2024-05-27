import numpy as np
import pandas as pd
from scipy import sparse
from typing import Tuple, Mapping, Any
from scipy.sparse import coo_matrix
from enum import Enum
from typing import Optional
from sklearn.model_selection import train_test_split
from pathlib import Path


class BaseGraphContext:
    """ Базовый класс, определяющий константы для типов узлов графа. """

    class NodeType(Enum):
        SOURCE = "source"
        TARGET = "target"
        
MappingIntToObject = Mapping[int, Tuple[Any, BaseGraphContext.NodeType]]

class ReadFromTxtToTransition():
    
    def _read_txt(self,
            file_name: Path,
            personalization: Optional[Any] = None) -> Tuple[sparse.csr_matrix,
                                                            sparse.csr_matrix,
                                                            MappingIntToObject]:
        """ Создает объект Tuple[sparse.csr_matrix,sparse.csr_matrix,MappingIntToObject]
        из данных txt.

        Args:
            data: DataFrame, содержащий ребра графа.
            source: Название столбца с исходными вершинами ребер.
            target: Название столбца с целевыми вершинами ребер.
            edge_attr: Название столбца с весами ребер.
            personalization: Опциональный объект, для которого будет 
                            персонализирован PageRank. Если None персонализации не будет

        Returns:
            Объект Tuple[sparse.csr_matrix,sparse.csr_matrix,MappingIntToObject],
            инициализированный на основе данных txt.
        """

        skiprows = 0
        with open(file_name, 'r') as f:
            for line in f:
                if line[0] != "#":
                    break
                skiprows += 1
        
        data = pd.read_csv(file_name, sep='\t', header=None, skiprows=skiprows, dtype=int)
        print(data)
        
        source_uniq = data.iloc[:,0].unique()
        target_uniq = data.iloc[:,1].unique()

        obj_source_to_N = dict(zip(source_uniq, range(len(source_uniq))))
        obj_target_to_N = dict(zip(target_uniq,
                                   range(len(source_uniq), len(source_uniq) + len(target_uniq))))

        N_to_obj = {}
        N_to_obj.update({i: (obj, BaseGraphContext.NodeType.SOURCE) for i, obj in enumerate(source_uniq)})
        N_to_obj.update({i: (obj, BaseGraphContext.NodeType.TARGET) for i, obj in enumerate(target_uniq, start=len(source_uniq))})
        
        count_top = len(source_uniq) + len(target_uniq)
        start_node = data.iloc[:,0].apply(lambda x: obj_source_to_N[x])
        end_node = data.iloc[:,1].apply(lambda x: obj_target_to_N[x])

        transition_matrix = coo_matrix((np.ones(len(data)), (start_node, end_node)),
                                        shape=(count_top, count_top))

        transition_matrix_csr = transition_matrix.tocsr()
        
        row_sums = transition_matrix_csr.sum(axis=1)
        transition_matrix_csr /= row_sums
        
        if personalization is None:
            return (
                transition_matrix_csr,
                None,
                N_to_obj)


        return (
            transition_matrix_csr,
            transition_matrix_csr\
                .getrow(obj_source_to_N[personalization])\
                .transpose(),
            N_to_obj)

class ReadFromPandasToTransition():
    
    def _read_pd(self,
            data: pd.DataFrame,
            source: str, target: str,
            edge_attr: str,
            personalization: Optional[Any] = None,
            directional: bool = False) -> Tuple[sparse.csr_matrix,
                                                            sparse.csr_matrix,
                                                            MappingIntToObject]:
        """ Создает объект Tuple[sparse.csr_matrix,sparse.csr_matrix,MappingIntToObject]
        из данных Pandas DataFrame.

        Args:
            data: DataFrame, содержащий ребра графа.
            source: Название столбца с исходными вершинами ребер.
            target: Название столбца с целевыми вершинами ребер.
            edge_attr: Название столбца с весами ребер.
            personalization: Опциональный объект, для которого будет 
                            персонализирован PageRank. Если None персонализации не будет

        Returns:
            Объект Tuple[sparse.csr_matrix,sparse.csr_matrix,MappingIntToObject],
            инициализированный на основе данных DataFrame.
        """
        
        source_uniq = data[source].unique()
        target_uniq = data[target].unique()

        obj_source_to_N = dict(zip(source_uniq, range(len(source_uniq))))
        obj_target_to_N = dict(zip(target_uniq,
                                   range(len(source_uniq), len(source_uniq) + len(target_uniq))))

        N_to_obj = {}
        N_to_obj.update({i: (obj, BaseGraphContext.NodeType.SOURCE) for i, obj in enumerate(source_uniq)})
        N_to_obj.update({i: (obj, BaseGraphContext.NodeType.TARGET) for i, obj in enumerate(target_uniq, start=len(source_uniq))})
        
        count_top = len(source_uniq) + len(target_uniq)
        start_node = data[source].apply(lambda x: obj_source_to_N[x])
        end_node = data[target].apply(lambda x: obj_target_to_N[x])

        transition_matrix = coo_matrix((data[edge_attr], (start_node, end_node)),
                                        shape=(count_top, count_top))

        # Для неориентированного графа
        if directional:
            transition_matrix += transition_matrix.transpose() 
        transition_matrix_csr = transition_matrix.tocsr()
        
        row_sums = transition_matrix_csr.sum(axis=1)
        transition_matrix_csr /= row_sums
        
        if personalization is None:
            return (
                transition_matrix_csr,
                None,
                N_to_obj)


        return (
            transition_matrix_csr,
            transition_matrix_csr\
                .getrow(obj_source_to_N[personalization])\
                .transpose(),
            N_to_obj)
        

       
def split_by_target_column_value(df: pd.DataFrame,
                                 test_size: float,
                                 target_column: str,
                                 target_value: Any,
                                 random_state: int = 42) -> Tuple[pd.DataFrame,
                                                                  pd.DataFrame,
                                                                  pd.DataFrame]:
    """
    Разделяет DataFrame на обучающий и тестовый наборы на основе определенного значения целевого столбца.
    Выдает в качестве тестового набора test_size от набора, в котором значение в target_column == target_value
    В тренировочном все остальное
    Параметры:
        df (pd.DataFrame): DataFrame для разделения.
        test_size: Пропорция данных, которые следует включить в тестовый набор.
        target_column (str): Название целевого столбца для разделения.
        target_value (Any): Значение целевого столбца, используемое для разделения.
        random_state (int): Зерно для генератора случайных чисел.

    Возвращает:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Данные для обучения; тестовые данные для целевой колонки, которые не попали в data,
        тренировочные данные для целевой колоники, которые попали в data
    """
    grouped = df.groupby(by=target_column)
    train_df, test_df = train_test_split(grouped.get_group(target_value), test_size=test_size,
                                         random_state=random_state)
    data = pd.concat([group for ind, group in grouped if ind != target_value], ignore_index=True)
    data = pd.concat([train_df, data], ignore_index=True)
    return data, test_df, train_df

