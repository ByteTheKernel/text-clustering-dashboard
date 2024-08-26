import time
from memory_profiler import memory_usage


# Функция для измерения времени выполнения и использования памяти алгоритмов кластеризации
def time_measure(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_elapsed = end_time - start_time
        print(f"{func.__name__} Time: {time_elapsed} seconds")
        return result, time_elapsed

    return wrapper


def memory_measure(func):
    def wrapper(*args, **kwargs):
        mem_usage_before = memory_usage(-1, interval=0.1, timeout=1)
        result = func(*args, **kwargs)
        mem_usage_after = memory_usage(-1, interval=0.1, timeout=1)
        memory_used = max(mem_usage_after) - mem_usage_before[0]
        print(f"{func.__name__} Memory: {memory_used} MiB")
        return result, memory_used

    return wrapper


# Объединение декораторов
def measure_resources(func):
    @time_measure
    @memory_measure
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result

    def combined_wrapper(*args, **kwargs):
        result, time_elapsed = time_measure(func)(*args, **kwargs)
        result, memory_used = memory_measure(func)(*args, **kwargs)
        return result, time_elapsed, memory_used

    return combined_wrapper


# Функция для кластеризации с декораторами
@measure_resources
def perform_clustering(clusterer, data):
    labels = clusterer.fit_predict(data)
    return labels
