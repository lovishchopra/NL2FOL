import ast
import re

def label_values(input_string,map):
    values = input_string.split(',')
    if isinstance(map,str):
        map = ast.literal_eval(map)
    labeled_values = [f"{v}: {map[v]}" for v in values]
    return ', '.join(labeled_values)

def first_non_empty_line(string):
  """Extracts the first non-empty line from a string.

  Args:
    string: The string to extract the first non-empty line from.

  Returns:
    The first non-empty line from the string.
  """
  lines = string.splitlines()
  for line in lines:
    if line:
      return line

  return None

def extract_propositional_symbols(logical_form):
    symbols = re.findall(r'\b[a-z]\b', logical_form)
    return set(symbols)

def split_string_except_in_brackets(string, delimiter):
    result = []
    stack = []
    current = ''

    for char in string:
        if char == '(':
            stack.append('(')
        elif char == ')':
            if stack:
                stack.pop()
        if char == delimiter and not stack:
            result.append(current)
            current = ''
        else:
            current += char

    if current:
        result.append(current)

    return result

def remove_text_after_last_parenthesis(input_string):
    last_paren_index = input_string.rfind(')')  # Find the index of the last ')'
    if last_paren_index != -1:  # If ')' is found
        result = input_string[:last_paren_index+1]  # Get the substring before the last ')'
        return result.strip()  # Return the result after removing any extra spaces
    else:
        return input_string  # Return the original string if no ')' is found

def fix_inconsistent_arities(clauses1, clauses2):
    all_clauses = clauses1 + clauses2
    predicates = {}
    
    # Extract predicates and their arities
    for clause in all_clauses:
        predicate = clause.split('(')[0]  # Extract predicate
        arity = len(clause.split(','))  # Calculate arity
        
        if predicate in predicates:
            # Update predicate arity if different
            if predicates[predicate] != arity:
                predicates[predicate] = min(predicates[predicate], arity)
        else:
            predicates[predicate] = arity
    
    # Fix inconsistency by dropping extra arguments
    fixed_clauses1 = []
    fixed_clauses2 = []
    for clause in clauses1:
        predicate = clause.split('(')[0]
        args = clause.split('(')[1].split(')')[0].split(',')
        arity = len(args)
        
        if arity > predicates[predicate]:
            print(f"updating {predicate} to arity {predicates[predicate]}")
            # Keep only necessary arguments based on predicate arity
            new_clause = f"{predicate}({', '.join(args[:predicates[predicate]])})"
            fixed_clauses1.append(new_clause)
        else:
            fixed_clauses1.append(clause)
    
    for clause in clauses2:
        predicate = clause.split('(')[0]
        args = clause.split('(')[1].split(')')[0].split(',')
        arity = len(args)
        
        if arity > predicates[predicate]:
            # Keep only necessary arguments based on predicate arity
            new_clause = f"{predicate}({', '.join(args[:predicates[predicate]])})"
            fixed_clauses2.append(new_clause)
        else:
            fixed_clauses2.append(clause)
    
    return ', '.join(fixed_clauses1), ', '.join(fixed_clauses2)

def replace_variables(mapping, clause):
    # Reverse the mapping for character to string replacement
    reversed_mapping = {v: k for k, v in mapping.items()}

    # Replace characters with corresponding strings from the reversed mapping
    replaced_clause = clause
    predicate = clause.split('(')[0]
    args = clause.split('(')[1][:-1].split(',')
    replaced_args=[]
    for arg in args:
        if arg in reversed_mapping:
            replaced_args.append(reversed_mapping[arg])
        else:
            replaced_args.append(arg)
    args=','.join(replaced_args)
    
    replaced_clause=predicate+'('+args+')'
    print("replaced_clause=",replaced_clause)
    return replaced_clause