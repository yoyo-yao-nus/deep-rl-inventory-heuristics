from utils import grid_search_basestock,grid_search_capped,grid_search_constant,eval_myopic
import json

param_grid = [(4, 2), (4, 3), (4, 4),\
              (9, 2), (9, 3), (9, 4)]
BaseStock_results = {}
CappedBasedStock_results = {}
Constant_results = {}

for (p, l) in param_grid:
    print(f" \033[1mHeuristic search for p={p}, l={l}\033[0m")

    best_S, val_bs = grid_search_basestock(p, l)
    print(f" Basestock: best S={best_S:2d}, value={val_bs:.1f}")

    (best_Sc, best_q), val_cap = grid_search_capped(p, l)
    print(f" Cappedbasestock: best S={best_S:2d}, q_max={best_q:2d}, value={val_cap:.1f}")

    best_r,   val_con = grid_search_constant(p, l)
    print(f" Constantorder: best r={best_r:2d}, value={val_con:.1f}")

    val_myo = eval_myopic(p, l)
    print(f" Myopic1: value={val_myo:.1f}")

    BaseStock_results[(p,l)] = best_S
    CappedBasedStock_results[(p,l)] = (best_Sc,best_q)
    Constant_results[(p,l)] = best_r
    
results = {
    'BaseStock' : str(BaseStock_results),
    'CappedBaseStock' : str(CappedBasedStock_results),
    'ConstantQuantity' : str(Constant_results) 
    }

file_path = 'results/heuristics_results.json'
with open(file_path, 'w') as file:
    json.dump(results, file)