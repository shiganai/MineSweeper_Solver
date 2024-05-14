import numpy as np
from pprint import pprint
import itertools
import copy

def gen_coeff_array(bombs_array):
    # 端の条件はまだ考慮していない。だから端が０じゃないとエラー起きる。
    coeff_array = []
    for index1 in range(bombs_array.shape[0]):
        for index2 in range(bombs_array.shape[1]):
            # When the position shows bomb indication number, create an equation.
            if bombs_array[index1,index2] > 0:
                # Initialize all coefficient with 0.
                coeff_array_each_non_zero = np.zeros_like(bombs_array)
                
                # Set 1 to "target for simulation" pos.
                for coeff_index1 in range(index1-1,index1+2):
                    for coeff_index2 in range(index2-1,index2+2):
                        if bombs_array[coeff_index1, coeff_index2] == -1:
                            coeff_array_each_non_zero[coeff_index1, coeff_index2] = 1
                
                # Flatten the array. Now each position have index as a valuable.
                _dict = {}
                _dict["coeff"] = coeff_array_each_non_zero.flatten()
                _dict["sum"] = bombs_array[index1,index2]
                _dict["sum_index"] = [index1,index2]
                coeff_array.append(_dict)
    
    return coeff_array

# assumed_array. [-1,-1,0,1,0,-1,-1, ...]
#   -1: unassumed, 0: assumed no bomb, 1: assumed a bomb

def assume_recursively(coeff_array, max_depth, target_unclear_indecies, assumed_arrays=None, depth=None):
    if assumed_arrays is None:
        assumed_arrays=[np.zeros_like(coeff_array[0]['coeff'])-1]
        depth = 0
    
    if depth > max_depth:
        print(f"depth:{depth} > max_depth:{max_depth}")
        return False
    
    possible_patterns = []
    for assumed_array in assumed_arrays:
        possible_patterns.extend(
            attempt_all_pattern(
                coeff_array=coeff_array,
                assumed_array=assumed_array,
                target_unclear_indecies=target_unclear_indecies
            )
        )
    possible_patterns = np.array(possible_patterns)
    
    pprint_patterns(possible_patterns)
    
    # Pick up assumed indecies
    target_unclear_indecies = np.where(np.any(possible_patterns!=-1, axis=0))[0]
    
    array_to_detect_must_be_zeros = possible_patterns==0
    array_to_detect_must_be_zeros[possible_patterns==-1] = True # Ignore -1
    must_be_zeros = np.all(array_to_detect_must_be_zeros, axis=0)
    must_be_zeros = must_be_zeros[target_unclear_indecies]
    
    array_to_detect_must_be_ones = possible_patterns==1
    array_to_detect_must_be_ones[possible_patterns==-1] = True # Ignore -1
    must_be_ones = np.all(array_to_detect_must_be_ones, axis=0)
    must_be_ones = must_be_ones[target_unclear_indecies]
    
    return target_unclear_indecies, must_be_zeros, must_be_ones

def attempt_all_pattern(coeff_array, assumed_array, target_unclear_indecies):
    
    # Pick up all equations related to targeting unclear indecies
    new_assuming_eqs = pick_up_equations(coeff_array, target_unclear_indecies)
    
    # Apply current assumption to equations. This will update "sum" when any index is equal to 1 in assumed_array.
    assumption_considered_eqs = consider_assumption(new_assuming_eqs, assumed_array)
    
    # Pick up all possible value pattern based on each equations. 
    possible_patterns_each_eq = detect_possible_patterns_each_eq(assumption_considered_eqs)
    
    # Consider all equations at the same time.
    possible_patterns_all_eq = consider_eq_combination(coeff_array=assumption_considered_eqs, patterns=possible_patterns_each_eq)
    
    # Apply the assumption to patterns
    possible_patterns_assumption_applied = apply_assumption_on_possible_patterns(patterns=possible_patterns_all_eq, assumed_array=assumed_array)
    
    return possible_patterns_assumption_applied

def pick_up_equations(coeff_array, target_unclear_indecies):
    picked_up = []
    for coeff in coeff_array:
        if np.any(coeff['coeff'][target_unclear_indecies] == 1):
            picked_up.append(coeff)
    return picked_up

def consider_assumption(coeff_array, assumed_array):
    considered = []
    for coeff in coeff_array:
        assumed_prod = coeff['coeff'] * assumed_array
        assumed_prod[assumed_array == -1] = 0
        assumed_sum = np.sum(assumed_prod)
        
        considered_coeff = copy.deepcopy(coeff)
        considered_coeff['coeff'][assumed_array != -1] = 0
        considered_coeff['sum'] -= assumed_sum
        considered.append(considered_coeff)
    return considered

def detect_possible_patterns_each_eq(coeff_array):
    # Detect possible bomb patterns.
    possible_patterns = []
    for coeff in coeff_array:
        # Detect non-zero coeff valuables.
        non_zero_indexes = np.where(coeff["coeff"]==1)[0]
        
        for pair in itertools.combinations(non_zero_indexes, coeff["sum"]):
            _tmp = np.zeros_like(coeff["coeff"]) - 1
            _tmp[non_zero_indexes] = 0
            for index in pair:
                _tmp[index] = 1
            possible_patterns.append(_tmp)
    possible_patterns = np.array(possible_patterns)
    return possible_patterns

def consider_eq_combination(coeff_array, patterns):
    valid_patterns = []
    for pattern in patterns:
        is_valid = True
        for coeff in coeff_array:
            _prod = coeff['coeff'] * pattern
            _prod[pattern == -1] = 0
            _sum = np.sum(_prod)
            
            if coeff['sum'] != _sum:
                is_valid = False
                break
        
        if is_valid:
            valid_patterns.append(pattern)
    valid_patterns = np.array(valid_patterns)
    return valid_patterns

def apply_assumption_on_possible_patterns(patterns, assumed_array):
    for pattern in patterns:
        pattern[assumed_array == 0] = 0
        pattern[assumed_array == 1] = 1
    
    return patterns

# -1: target of simulation
# 0: Inside or too far
# >0: number of surrounding bombs
bombs_array = np.array([
    [-1,-1,-1,-1,-1,-1,-1],
    [-1,-1, 3, 1, 2,-1,-1],
    [-1,-1, 1, 0, 1,-1,-1],
    [-1, 2, 1, 0, 1, 2,-1],
    [-1, 2, 0, 0, 0, 1,-1],
    [-1, 1, 0, 0, 1, 2,-1],
    [-1, 2, 0, 0, 1,-1,-1],
    [-1, 1, 1, 2, 3,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1]
    ])
#  pattern: 1 1 1
bombs_array = np.array([
    [-1,-1,-1,-1,-1,-1,-1],
    [-1,-1, 1, 1, 1,-1,-1],
    [-1,-1, 1, 0, 1,-1,-1],
    [-1,-1, 1, 0, 1,-1,-1],
    [-1,-1, 1, 0, 1,-1,-1],
    [-1,-1, 1, 1, 1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1],
    [-1,-1,-1,-1,-1,-1,-1]
    ])
# #  pattern: 1 2 2 1
# bombs_array = np.array([
#     [-1,-1,-1,-1,-1,-1,-1],
#     [-1,-1, 1, 2, 2, 1,-1],
#     [-1,-1, 1, 0, 1,-1,-1],
#     [-1,-1, 1, 0, 1,-1,-1],
#     [-1,-1, 1, 0, 1,-1,-1],
#     [-1,-1, 1, 1, 1,-1,-1],
#     [-1,-1,-1,-1,-1,-1,-1],
#     [-1,-1,-1,-1,-1,-1,-1],
#     [-1,-1,-1,-1,-1,-1,-1]
#     ])

print("Bomb indications. -1: assumable, >=0: number of surrounding bombs")
pprint(bombs_array)

def pprint_patterns(patterns):
    print("-> pprint_patterns()")
    for pattern in patterns:
        pprint(np.reshape(pattern, bombs_array.shape))
    
    print("<- pprint_patterns()")

coeff_array = gen_coeff_array(bombs_array)

target_unclear_indecies=[3]
assumed_array=np.zeros_like(coeff_array[0]['coeff'])-1

target_unclear_indecies, must_be_zeros, must_be_ones = assume_recursively(
    coeff_array=coeff_array,
    target_unclear_indecies=target_unclear_indecies,
    max_depth=2)

pprint(must_be_ones)
pprint(must_be_zeros)

assumed_bombs_array = copy.deepcopy(bombs_array)

for target_unclear_index, must_be_zero, must_be_one in zip(target_unclear_indecies, must_be_zeros, must_be_ones):
    detected_sub = np.unravel_index(target_unclear_index, bombs_array.shape)
    if must_be_zero:
        assumed_bombs_array[detected_sub[0],detected_sub[1]] = -10
    elif must_be_one:
        assumed_bombs_array[detected_sub[0],detected_sub[1]] = 10
    
print("Detected by the following pattern. -1:unclear, 10:detected bomb, -10:dectect no bomb, 0~9: number of bombs around it.")
pprint(assumed_bombs_array)