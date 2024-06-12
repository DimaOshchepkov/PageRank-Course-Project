import json
import pandas as pd

def json_to_table(path_to_json):
    
    # Загружаем JSON данные из файла
    with open(path_to_json, 'r') as file:
        data = json.load(file)

    # Извлекаем список бенчмарков
    benchmarks = data["benchmarks"]

    # Создаем DataFrame из списка бенчмарков
    df = pd.DataFrame(benchmarks, columns=["name",  "real_time", "cpu_time"])

    df[['algorithm_dataset', 'sample', 'iterations']] = df['name'].str.split('/', expand=True)
    df = df.drop(['name', 'iterations'], axis=1)

    df['sample'] = df['sample'].str.split('-', expand=True)[1]

    df[['algorithm', 'dataset']] = df['algorithm_dataset'].str.rsplit('_', expand=True, n=1)
    df = df.drop(['algorithm_dataset'], axis=1)
    df['algorithm'] = df['algorithm'] .str.split('_', expand=True)[1]

    df['real_time'] = df['real_time'].apply(lambda x : round(x, 2))
    df['cpu_time'] = df['cpu_time'].apply(lambda x : round(x, 2))

    column_order = ['algorithm', 'dataset', 'sample', 'real_time', 'cpu_time']
    df = df[column_order]
    
    return df

df = json_to_table("BM_iterative_google_web.json")

for alg, table in df.groupby(by='algorithm'):
    table.sort_values(by='sample', ascending=True)
    if alg == "Hits":
        table.to_excel(f"./hits/my_cpp_hits_results_web.xlsx", index=False)
    elif alg == "Pagerank":
        table.to_excel(f"./pagerank/my_cpp_pagerank_results_web.xlsx", index=False)
        
        
df = json_to_table("BM_iterative_movielens.json")

for alg, table in df.groupby(by='algorithm'):
    table.sort_values(by='sample', ascending=True)
    if alg == "Hits":
        table.to_excel(f"./hits/my_cpp_hits_results_movielens.xlsx", index=False)
    elif alg == "Pagerank":
        table.to_excel(f"./pagerank/my_cpp_pagerank_results_movielens.xlsx", index=False)
        
