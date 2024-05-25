# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:53:27 2024

@author: 12900k
"""

import numpy as np
from pprint import pprint
import itertools
import copy

# Explanation of values.
# ==================================================
# bombs_indicator_array = np.array([
#     [-1,-1,-1,-1,-1,-1,-1],
#     [-1,-1, 3, 1, 2,-1,-1],
#     [-1,-1, 1, 0, 1,-1,-1],
#     [-1, 2, 1, 0, 1, 2,-1],
#     [-1, 2, 0, 0, 0, 1,-1],
#     [-1, 1, 0, 0, 1, 2,-1],
#     [-1, 2, 0, 0, 1,-1,-1],
#     [-1, 1, 1, 2, 3,-1,-1],
#     [-1,-1,-1,-1,-1,-1,-1]
#     ])
# -1: target of simulation
# 0: Inside or too far
# >0: number of surrounding bombs
# ==================================================
# coeff_array is an array for the following formatted dictionaries.
#   "sum_index": [int,int]. The position of the indicating value, such as 1,2,3 ...
#   "coeff": [0,0,1,1,1, ...]. This is a 1x? array. The positions of 1 means that is not yet assumed and it next to the indicating value.
#   "sum": int. This is the indicating value.
#   The following equation stands true using assumed_arrays desribed next.
#   sum("coeff" * assumed_arrays = "sum")
# ==================================================
# assumed_arrays, possible_patterns: [[0,0,1,1,0, ...], [0,0,0,1,1, ...]]. They are arrays of possible patterns.
#  -1: Not yet assumed.
#   0: Assumed no bomb.
#   1: Assumed a bomb.
# ==================================================


def example():
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
    #  pattern: 1 2 2 1
    bombs_indicator_array = np.array([
        [-1,-1,-1,-1,-1,-1,-1],
        [-1,-1, 1, 2, 2, 1,-1],
        [-1,-1, 1, 0, 0, 1,-1],
        [-1,-1, 1, 0, 0, 1,-1],
        [-1,-1, 1, 0, 0, 1,-1],
        [-1,-1, 1, 1, 1, 1,-1],
        [-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1,-1,-1]
        ])
    
    _solver = solver(bombs_indicator_array=bombs_indicator_array)
    
    print("Bomb indications")
    _solver.pprint_bombs_string(bombs_indicator_array)
    target_unclear_indecies = [15]

        
    detected_bombs, possible_patterns = _solver.assume_recursively(
        target_unclear_indecies=target_unclear_indecies,
        max_depth=6)
    
    
    print("Possible patterns")
    print("=================")
    for pattern in possible_patterns:
        pattern = np.reshape(pattern, bombs_indicator_array.shape)
        _solver.pprint_bombs_string(pattern)
        print("=================")
    
    detected_bombs = np.reshape(detected_bombs, bombs_indicator_array.shape)
    print("Thus, the following pattern must be the case.")
    _solver.pprint_bombs_string(detected_bombs)

class solver:
    def __init__(self, bombs_indicator_array, is_debugging=False):
        self.bombs_indicator_array = bombs_indicator_array
        self.is_debugging = is_debugging
        self.gen_coeff_array()

    def gen_coeff_array(self):
        # This function picks up positions of possible bombs around indicating values.
        # This funciton will define coeff_array.
        
        bombs_indicator_array = self.bombs_indicator_array
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
        
        self.coeff_array = coeff_array
        return coeff_array
    
    def assume_recursively(self, max_depth, target_unclear_indecies):
        # This function is an API for outer script to kick-start assumption.
        
        # Initialize
        # assumed_arrays is initialized with -1 meaning "not yet assumed"
        assumed_arrays=[np.zeros_like(self.coeff_array[0]['coeff'])-1]
        depth = 0
        
        # Call recursive function for 
        possible_patterns = self._assume_recursively(coeff_array=self.coeff_array, depth=depth, max_depth=max_depth, target_unclear_indecies=target_unclear_indecies, assumed_arrays=assumed_arrays)
        
        # Pick up assumed indecies
        target_unclear_indecies = np.where(np.any(possible_patterns!=-1, axis=0))[0]
        
        # Determine position must not be bombs
        array_to_detect_must_not_be_bombs = possible_patterns==0
        must_not_be_bombs = np.all(array_to_detect_must_not_be_bombs, axis=0)
        
        # Determine position must be bombs
        array_to_detect_must_be_bombs = possible_patterns==1
        must_be_bombs = np.all(array_to_detect_must_be_bombs, axis=0)
        
        detected_bombs = np.zeros_like(self.coeff_array[0]['coeff']) - 1
        detected_bombs[must_not_be_bombs] = 0
        detected_bombs[must_be_bombs] = 1
        
        return detected_bombs, possible_patterns
    
    def _assume_recursively(self, coeff_array, depth, max_depth, target_unclear_indecies, assumed_arrays):
        # This function is a recursive function to detect all possible patterns until the process reaches to a certain depth or reaches to where no more assumption can be made.
        # The algorithm for each recursive process:
        #   1. Picks up all possible patterns for a given assumed_array.
        #       This step will consider all patterns for non -1 position of assumed_array.
        #       This may pick up new positions to consider.
        #   2. Define updated_target_unclear_indecies based on the possible patterns. 
        #       Non -1 positions means some assumptions are made for the positions and all of them are targets.
        #   3. Update depth by 1. 
        #       If the updated depth exceeds the given depth,it means the user does not want to waste resource anymore.
        #       Return possible patterns.
        #   4. Verify if any positions are assumed newly. 
        #       If not, it means the process reach to where no more assumption can be made. 
        #       Return possible patterns.
        #   5. Proceed to next depth if no hit to step 3 and 4
        
        target_unclear_indecies = np.array(target_unclear_indecies)
        possible_patterns = []
        
        for assumed_array in assumed_arrays:
            possible_patterns.extend(
                self.attempt_all_pattern(
                    coeff_array=coeff_array,
                    assumed_array=assumed_array,
                    target_unclear_indecies=target_unclear_indecies
                )
            )
        possible_patterns = np.array(possible_patterns)
        # Update assuming valuables.
        updated_target_unclear_indecies = np.where(np.any(possible_patterns!=-1, axis=0))[0]
        
        depth += 1
        
        # When depth > max_depth, it means this is the manual limit.
        if depth > max_depth:
            print(f"depth:{depth} > max_depth:{max_depth}")
            return possible_patterns
        
        # When target_unclear_indecies == updated_target_unclear_indecies,
        #   it means next assumption won't change anything.
        elif target_unclear_indecies.tolist() == updated_target_unclear_indecies.tolist():
            print(f"All assumption are made at depth:{depth}")
            return possible_patterns
        
        # Proceed to next step with updated_target_unclear_indecies and possible patterns
        else:
            return self._assume_recursively(
                coeff_array=self.coeff_array, 
                depth=depth, 
                max_depth=max_depth, 
                target_unclear_indecies=updated_target_unclear_indecies,
                assumed_arrays=possible_patterns)
    
    def attempt_all_pattern(self, coeff_array, assumed_array, target_unclear_indecies):
        # This function returns all possible patterns which releated to target_unclear_indecies.
        # The assumption consider all existing conditions, meaning it will pick up new equations to consider for border assumed positions.
        
        # Pick up all equations related to targeting unclear indecies
        new_assuming_eqs = self.pick_up_equations(coeff_array, target_unclear_indecies)
        
        # Apply current assumption to equations. This will update "sum" when any index is equal to 1 in assumed_array.
        assumption_considered_eqs = self.consider_assumption(new_assuming_eqs, assumed_array)
        
        # Pick up all possible value pattern based on each equations. 
        possible_patterns = self.detect_possible_patterns(assumption_considered_eqs, assumed_array)
        
        # Apply the assumption to patterns
        possible_patterns_assumption_applied = self.apply_assumption_on_possible_patterns(patterns=possible_patterns, assumed_array=assumed_array)
        
        return possible_patterns_assumption_applied
    
    def pick_up_equations(self, coeff_array, target_unclear_indecies):
        # This function returns all equations related to given target_unclear_indecies.
        
        picked_up = []
        for coeff in coeff_array:
            if np.any(coeff['coeff'][target_unclear_indecies] == 1):
                picked_up.append(coeff)
        return picked_up
    
    def consider_assumption(self, coeff_array, assumed_array):
        # This function will apply current assumption to equations.
        # Will convert 1 into 0 for where assumed_array is not -1, so that it is easy to pick up which position to assume new.
        # When equations value is 1 and assumed_array is 1 for a position, it will reduce "sum" by 1.
        
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
        
    
    def detect_possible_patterns(self, coeff_array, assumed_array):
        # This function returns all possible patterns based on given assumption and a (first) equation.
        # This function is a recursive function.
        # The algorithm:
        #   1: Verify "sum" is not negative.
        #       It means the given assumption are wrong if it is negative.
        #       This case can occur when the patterns are valid without considering this specific equation.
        #   2: Get nCm patterns where n means the total of possible bomb positions and m means the total of bombs (="sum").
        #   3: Verify nCm patterns exists. If no, it means no patterns can satisfy this equation.
        #   4: Generate new assumption based on nCm patterns.
        #   5: Verify if there are more euqations to consider. If no, it means the assumption is valid.
        #   6: Consider next equation when there are more euqations to consider.
        
        possible_patterns = []
        
        # Consider the first equation
        coeff = coeff_array[0]
        
        # Detect non-zero coeff valuables.
        non_zero_indexes = np.where(coeff["coeff"]==1)[0]
        
        # When the sum is negative, it means the so far assumption does not fullfill this equations.
        if coeff['sum'] < 0:
            if self.is_debugging:
                print("assumed_array was confirmed INVALID.")
                print("coeff['coeff']")
                self.pprint_coeffs(coeff)
                print("assumed_array")
                self.pprint_patterns([assumed_array])
                print("========================")
            return []
        
        tuple_pairs = list(itertools.combinations(non_zero_indexes, coeff["sum"]))
        
        # When there are no patterns such as coeff["sum"] is too big, return empty.
        if tuple_pairs.__len__() == 0:
            if self.is_debugging:
                print("assumed_array was confirmed INVALID.")
                print("coeff['coeff']")
                self.pprint_coeffs(coeff)
                print("assumed_array")
                self.pprint_patterns([assumed_array])
                print("========================")
            return []
        
        for tuple_pair in tuple_pairs:
            further_assumed_array = copy.deepcopy(assumed_array)
            further_assumed_array[non_zero_indexes] = 0
            further_assumed_array[list(tuple_pair)] = 1
            
            # When there're no more equations to verify, it means this pattern is valid.
            if coeff_array.__len__()==1:
                if self.is_debugging:
                    print("assumed_array was confirmed VALID.")
                    print("coeff['coeff']")
                    self.pprint_coeffs(coeff)
                    print("further_assumed_array")
                    self.pprint_patterns([further_assumed_array])
                    print("========================")
                possible_patterns.append(further_assumed_array)
            # When there're additional equations to verify, move on to them with adding more assumption.
            else:    
                # Go to the next equation
                if self.is_debugging:
                    print("assumed_array")
                    self.pprint_patterns([assumed_array])
                    print("coeff['coeff']")
                    self.pprint_coeffs(coeff)
                    print("further_assumed_array")
                    self.pprint_patterns([further_assumed_array])
                    print(f"tuple_pair: {tuple_pair}")
                further_considred_eqs = self.consider_assumption(coeff_array[1:], further_assumed_array)
                further_considered_possible_patterns = self.detect_possible_patterns(further_considred_eqs, further_assumed_array)
                possible_patterns.extend(further_considered_possible_patterns)
                
        possible_patterns = np.array(possible_patterns)
        return possible_patterns
    
    def apply_assumption_on_possible_patterns(self, patterns, assumed_array):
        # This function merges possible patterns and assumed array.
        # This is because possible patterns only considers non assumed positions.
        for pattern in patterns:
            pattern[assumed_array == 0] = 0
            pattern[assumed_array == 1] = 1
        
        return patterns
    
    
    def pprint_patterns(self, patterns):
        bombs_indicator_array = self.bombs_indicator_array
        for pattern in patterns:
            pattern = copy.deepcopy(pattern)
            pattern = np.reshape(pattern, bombs_indicator_array.shape)
            pattern[bombs_indicator_array>=0] = bombs_indicator_array[bombs_indicator_array>=0]
            pprint(pattern)
    
    def pprint_coeffs(self, coeff):
        bombs_indicator_array = self.bombs_indicator_array
        pattern = copy.deepcopy(coeff['coeff'])
        pattern = np.reshape(pattern, bombs_indicator_array.shape)
        pattern[bombs_indicator_array>=0] = bombs_indicator_array[bombs_indicator_array>=0]
        pattern[coeff['sum_index'][0], coeff['sum_index'][1]] *= 10
        pprint(pattern)
    
    def pprint_bombs_string(self, detected_boms=None):
        bombs_indicator_array = self.bombs_indicator_array
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

example()