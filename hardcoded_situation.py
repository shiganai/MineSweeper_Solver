import numpy as np
from pprint import pprint
import itertools
import copy

is_debugging = False

def gen_coeff_array(bombs_indicator_array):
    # 端の条件はまだ考慮していない。だから端が０じゃないとエラー起きる。
    coeff_array = []
    for index1 in range(bombs_indicator_array.shape[0]):
        for index2 in range(bombs_indicator_array.shape[1]):
            # When the position shows bomb indication number, create an equation.
            if bombs_indicator_array[index1,index2] > 0:
                # Initialize all coefficient with 0.
                coeff_array_each_non_zero = np.zeros_like(bombs_indicator_array)
                
                # Set 1 to "target for simulation" pos.
                for coeff_index1 in range(index1-1,index1+2):
                    for coeff_index2 in range(index2-1,index2+2):
                        if bombs_indicator_array[coeff_index1, coeff_index2] == -1:
                            coeff_array_each_non_zero[coeff_index1, coeff_index2] = 1
                
                # Flatten the array. Now each position have index as a valuable.
                _dict = {}
                _dict["coeff"] = coeff_array_each_non_zero.flatten()
                _dict["sum"] = bombs_indicator_array[index1,index2]
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
    
    # Pick up assumed indecies
    target_unclear_indecies = np.where(np.any(possible_patterns!=-1, axis=0))[0]
    
    # Determine position must not be bombs
    array_to_detect_must_not_be_bombs = possible_patterns==0
    must_not_be_bombs = np.all(array_to_detect_must_not_be_bombs, axis=0)
    
    # Determine position must be bombs
    array_to_detect_must_be_bombs = possible_patterns==1
    must_be_bombs = np.all(array_to_detect_must_be_bombs, axis=0)
    
    detected_bombs = np.zeros_like(coeff_array[0]['coeff']) - 1
    detected_bombs[must_not_be_bombs] = 0
    detected_bombs[must_be_bombs] = 1
    
    return detected_bombs, possible_patterns

def attempt_all_pattern(coeff_array, assumed_array, target_unclear_indecies):
    
    # Pick up all equations related to targeting unclear indecies
    new_assuming_eqs = pick_up_equations(coeff_array, target_unclear_indecies)
    
    # Apply current assumption to equations. This will update "sum" when any index is equal to 1 in assumed_array.
    assumption_considered_eqs = consider_assumption(new_assuming_eqs, assumed_array)
    
    # Pick up all possible value pattern based on each equations. 
    possible_patterns = detect_possible_patterns(assumption_considered_eqs, assumed_array)
    
    # Apply the assumption to patterns
    possible_patterns_assumption_applied = apply_assumption_on_possible_patterns(patterns=possible_patterns, assumed_array=assumed_array)
    
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

def detect_possible_patterns(coeff_array, assumed_array):
    # Detect possible bomb patterns.
    return _detect_possible_patterns(coeff_array, assumed_array)
    

def _detect_possible_patterns(coeff_array, assumed_array):
    possible_patterns = []
    
    # Consider the first equation
    coeff = coeff_array[0]
    
    # Detect non-zero coeff valuables.
    non_zero_indexes = np.where(coeff["coeff"]==1)[0]
    
    # When the sum is negative, it means the so far assumption does not fullfill this equations.
    if coeff['sum'] < 0:
        if is_debugging:
            print("assumed_array was confirmed INVALID.")
            print("coeff['coeff']")
            pprint_coeffs(coeff)
            print("assumed_array")
            pprint_patterns([assumed_array])
            print("========================")
        return []
    
    tuple_pairs = list(itertools.combinations(non_zero_indexes, coeff["sum"]))
    
    # When there are no patterns such as coeff["sum"] is too big, return empty.
    if tuple_pairs.__len__() == 0:
        if is_debugging:
            print("assumed_array was confirmed INVALID.")
            print("coeff['coeff']")
            pprint_coeffs(coeff)
            print("assumed_array")
            pprint_patterns([assumed_array])
            print("========================")
        return []
    
    for tuple_pair in tuple_pairs:
        further_assumed_array = copy.deepcopy(assumed_array)
        further_assumed_array[non_zero_indexes] = 0
        further_assumed_array[list(tuple_pair)] = 1
        
        # When there're no more equations to verify, it means this pattern is valid.
        if coeff_array.__len__()==1:
            if is_debugging:
                print("assumed_array was confirmed VALID.")
                print("coeff['coeff']")
                pprint_coeffs(coeff)
                print("further_assumed_array")
                pprint_patterns([further_assumed_array])
                print("========================")
            possible_patterns.append(further_assumed_array)
        # When there're additional equations to verify, move on to them with adding more assumption.
        else:    
            # Go to the next equation
            if is_debugging:
                print("assumed_array")
                pprint_patterns([assumed_array])
                print("coeff['coeff']")
                pprint_coeffs(coeff)
                print("further_assumed_array")
                pprint_patterns([further_assumed_array])
                print(f"tuple_pair: {tuple_pair}")
            further_considred_eqs = consider_assumption(coeff_array[1:], further_assumed_array)
            further_considered_possible_patterns = _detect_possible_patterns(further_considred_eqs, further_assumed_array)
            possible_patterns.extend(further_considered_possible_patterns)
            
    possible_patterns = np.array(possible_patterns)
    return possible_patterns

def apply_assumption_on_possible_patterns(patterns, assumed_array):
    for pattern in patterns:
        pattern[assumed_array == 0] = 0
        pattern[assumed_array == 1] = 1
    
    return patterns

# -1: target of simulation
# 0: Inside or too far
# >0: number of surrounding bombs
bombs_indicator_array = np.array([
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
# #  pattern: 1 1 1
# bombs_indicator_array = np.array([
#     [-1,-1,-1,-1,-1,-1,-1],
#     [-1,-1, 1, 1, 1,-1,-1],
#     [-1,-1, 1, 0, 1,-1,-1],
#     [-1,-1, 1, 0, 1,-1,-1],
#     [-1,-1, 1, 0, 1,-1,-1],
#     [-1,-1, 1, 1, 1,-1,-1],
#     [-1,-1,-1,-1,-1,-1,-1],
#     [-1,-1,-1,-1,-1,-1,-1],
#     [-1,-1,-1,-1,-1,-1,-1]
#     ])
# #  pattern: 1 2 2 1
# bombs_indicator_array = np.array([
#     [-1,-1,-1,-1,-1,-1,-1],
#     [-1,-1, 1, 2, 2, 1,-1],
#     [-1,-1, 1, 0, 0, 1,-1],
#     [-1,-1, 1, 0, 0, 1,-1],
#     [-1,-1, 1, 0, 0, 1,-1],
#     [-1,-1, 1, 1, 1, 1,-1],
#     [-1,-1,-1,-1,-1,-1,-1],
#     [-1,-1,-1,-1,-1,-1,-1],
#     [-1,-1,-1,-1,-1,-1,-1]
#     ])


def pprint_patterns(patterns):
    for pattern in patterns:
        pattern = copy.deepcopy(pattern)
        pattern = np.reshape(pattern, bombs_indicator_array.shape)
        pattern[bombs_indicator_array>=0] = bombs_indicator_array[bombs_indicator_array>=0]
        pprint(pattern)

def pprint_coeffs(coeff):
    pattern = copy.deepcopy(coeff['coeff'])
    pattern = np.reshape(pattern, bombs_indicator_array.shape)
    pattern[bombs_indicator_array>=0] = bombs_indicator_array[bombs_indicator_array>=0]
    pattern[coeff['sum_index'][0], coeff['sum_index'][1]] *= 10
    pprint(pattern)

def pprint_bombs_string(bombs_indicator_array, detected_boms=None):
    strings_to_show = np.empty_like(bombs_indicator_array, dtype=str)
    if detected_boms is None:
        detected_boms = np.zeros_like(bombs_indicator_array) - 1
    
    for index1 in range(strings_to_show.shape[0]):
        for index2 in range(strings_to_show.shape[1]):
            _indicator = bombs_indicator_array[index1,index2]
            _dectedted = detected_boms[index1,index2]
            
            if _indicator >= 0:
                strings_to_show[index1,index2] = str(_indicator)
            elif _dectedted == 0:
                strings_to_show[index1,index2] = 'N'
            elif _dectedted == 1:
                strings_to_show[index1,index2] = 'B'
            else:
                strings_to_show[index1,index2] = '-'
    
    strings_to_show = strings_to_show.tolist()
    
    for row in strings_to_show:
        print(",".join(row))

print("Bomb indications")
pprint_bombs_string(bombs_indicator_array)    

coeff_array = gen_coeff_array(bombs_indicator_array)

target_unclear_indecies=[3]
assumed_array=np.zeros_like(coeff_array[0]['coeff'])-1

detected_bombs, possible_patterns = assume_recursively(
    coeff_array=coeff_array,
    target_unclear_indecies=target_unclear_indecies,
    max_depth=2)


print("Possible patterns")
print("=================")
for pattern in possible_patterns:
    pattern = np.reshape(pattern, bombs_indicator_array.shape)
    pprint_bombs_string(bombs_indicator_array, pattern)
    print("=================")

detected_bombs = np.reshape(detected_bombs, bombs_indicator_array.shape)
print("Thus, the following pattern must be the case.")
pprint_bombs_string(bombs_indicator_array, detected_bombs)
