import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import norm
from sklearn.model_selection import train_test_split
from typing import Tuple, Mapping, Any, Literal, List
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MappingIntToObject = Mapping[int, Tuple[Any, Literal['source', 'target']]]

# https://github.com/danieljunhee/Tutorial-on-Personalized-PageRank/blob/master/Personalized_PageRank_Tutorial.ipynb
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
                       user_person_vec: sparse.csr_matrix, max_iter: int = 100) -> np.ndarray:
        
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

def split_by_target_column_value(df: pd.DataFrame, test_size: float, target_column: str, target_value: Any,
                                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

class O:
    
    def __init__(self, delta: Any) -> None:
        self.delta = delta
        
    def __call__(self, true_score_value: Any) -> int:
        return int(true_score_value > self.delta)
    
class T():
    def __init__(self, delta: Any) -> None:
        self.delta = delta
        
    def __call__(self, pred_value: Any) -> int:
        return int(pred_value > self.delta)

class I():
    
    def __init__(self, pagerankT:T, valuesO: O) -> None:
        self.pagerankT = pagerankT
        self.valuesO = valuesO
        
    def __call__(self, pred_p: Any, true_p: Any,
                 pred_q: Any, true_q: Any) -> int:
        if pred_p >= pred_q and \
            self.valuesO(true_p) < self.valuesO(true_q):
            return 1
        if pred_p <= pred_q and \
            self.valuesO(true_p) > self.valuesO(true_q):
            return 1
        return 0
    
    
class Pairord():
    
    def __init__(self, i: I) -> None:
        self.i = i
        
    def __call__(self, pred_pagerank: List[Any], true_score: List[Any]) -> float:
        metric = len(pred_pagerank) * len(true_score)
        for p in range(len(pred_pagerank)):
            for q in range(len(true_score)):
                metric -= self.i(pred_pagerank[p], true_score[q],
                                        pred_pagerank[q], true_score[p])
            
        metric /= (len(pred_pagerank) * len(true_score))
        
        return metric
    
class Prec():
    
    def __init__(self, o: O, t: T) -> None:
        self.o = o
        self.t = t       
        
    def __call__(self, pred_pagerank: List[Any], true_score: List[Any]) -> float:
        numerator = 0
        denominator = 0
        for p in range(len(pred_pagerank)):
            numerator += self.t(pred_pagerank[p]) * self.o(true_score[p])
            denominator += self.t(pred_pagerank[p])
        
        return numerator / denominator
    
    
class Rec():
    
    def __init__(self, o: O, t: T) -> None:
        self.o = o
        self.t = t       
        
    def __call__(self, pred_pagerank: List[Any], true_score: List[Any]) -> float:
        numerator = 0
        denominator = 0
        for p in range(len(pred_pagerank)):
            numerator += self.t(pred_pagerank[p]) * self.o(true_score[p])
            denominator += self.o(true_score[p])
        
        return numerator / denominator
    
class F1():
    
    def __init__(self, prec: Prec, rec: Rec) -> None:
        self.prec = prec
        self.rec = rec
        
    def __call__(self, pred_pagerank: List[Any], true_score: List[Any]) -> float:
        p = self.prec(pred_pagerank, true_score)
        r = self.rec(pred_pagerank, true_score)
            
        return 2*p*r/(p + r)
    
def write(file_name, list):
    with open(file_name, 'w') as f:
        # Записываем все элементы списка в файл одним вызовом
        f.writelines("%s " % item for item in list)

df = pd.read_csv("res\\rating.csv")
movie_df = pd.read_csv('res/movie.csv')

precision_list = []
accuracy_list = []
recall_list = []
pairroid_list = []
f1_list = []

for user_name in tqdm(np.unique(df['userId'])[:120], desc='Процесс обработки'):
    data, test, train = split_by_target_column_value(df, test_size=0.5, target_column='userId', target_value=user_name)
    
    pg = PageRank(data, 'userId', 'movieId', 'rating', user_name, max_iter=100)

    prediction_movie_for_user = pg.get_pagerank()
        
    for_score = prediction_movie_for_user.merge(test, how='inner', on='movieId')
    
    o = O(test['rating'].mean())
    t = T(prediction_movie_for_user['pagerank'].mean())
    i = I(t, o)
    pairroid = Pairord(i)
    pairroid_ = pairroid(for_score['pagerank'], for_score['rating'])
    pairroid_list.append(pairroid_)


    pred = np.array(for_score['pagerank'])
    true = np.array(for_score['rating'])
    pred_map = np.vectorize(lambda x: t(x))
    true_map = np.vectorize(lambda x: o(x))

    pred_trasformed = pred_map(pred)
    true_trasformed = true_map(true)
    
    # Вычисляем метрики
    accuracy = accuracy_score(true_trasformed, pred_trasformed)
    precision = precision_score(true_trasformed, pred_trasformed)
    recall = recall_score(true_trasformed, pred_trasformed)
    f1 = f1_score(true_trasformed, pred_trasformed)

    # Выводим результаты
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    
write('metric_result/accuracy.txt', accuracy_list)
write('metric_result/precision.txt', precision_list)
write('metric_result/recall.txt', recall_list)
write('metric_result/f1.txt', f1_list)
write('metric_result/pairroid.txt', pairroid_list)



