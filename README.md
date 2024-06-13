# Курсовой проект на тему "Разработка системы ссылочного ранжирования"

Этот курсовой проект был создан в рамках дисциплины "Терория алгоритмов". В нем рассматриваются 2 наиболее распространенных алгоритма ссылочного ранжирования: Pagrank и HITS.

## Быстрый старт
Пример использования HITS ([подробнее](./handmaid_hits_result_demonsration.ipynb)):
```python
df = pd.read_csv("path/to/dataframe")
hits: HITS = HITSFactory().read_pd(data=data, source='userId', 
target='movieId', edge_attr='rating', personalization=1)
prediction_movie_for_user1 = hits.get_rank()
print(prediction_movie_for_user1.sort_values(by='hubness', ascending=False).head())
```

Результат будет выглядеть примерно так:
```
       target  authoritativeness  hubness
0        2288       2.207916e-04      0.0
17760   97709       1.086575e-07      0.0
17836  107408       3.046013e-08      0.0
17835  104780       7.874532e-08      0.0
17834  103593       2.398938e-08      0.0
```

Пример использования Pagerank ([подробнее](./handmaid_pagerank_result_demonsration.ipynb)):
```python

pg = PagerankFactory().read_pd(data, 'userId', 'movieId', 'rating', 1)
prediction_movie_for_user1 = pg.get_pagerank()
print(prediction_movie_for_user1.sort_values(by='pagerank',
 ascending=False).head())

```

Результат будет выглядеть примерно так:
```
     target  pagerank
153     318  0.000767
592     296  0.000750
305     356  0.000708
49      593  0.000704
92      260  0.000579

```

Основной целью проекта было [сравнение производительности](./plot_comparision.ipynb).
Также в проекте есть сравнения с [C++ реализациями](https://github.com/DimaOshchepkov/LinkRanking).
