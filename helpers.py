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