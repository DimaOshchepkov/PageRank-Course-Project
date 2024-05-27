from typing import Any, List

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
        metric: float = len(pred_pagerank) * len(true_score)
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