"""
Python parser to parse a logical formula and create a CVC file for the negation of the logical formula.
It also checks and makes sure that all predicates have consistent sorts.
"""
import re
import sys

bound_variables = set()
unbound_variables = {}
predicate_to_sort_map = {}

class Sort:
    """
    Sort class. It is used to define and handle sorts.
    """
    def __init__(self, sort=None):
        self.sort = sort
    
    def getSort(self):
        return self.sort
    
    def setSort(self, sort):
        self.sort = sort
    
    def __repr__(self):
        return str(self.sort)

class Operator:
    """
    Operator class, used to store operator, its arity, whether it is a quantifier and its priority w.r.t. other operators, quantified variable
    """
    def __init__(self, operator):
        self.operator = operator
        self.quantifier = ("exists" in operator or "forall" in operator)
        self.arity = 1 if operator == "not" or self.quantifier else 2
        self.priority = Operator.priority_values(operator)
        self.quanified_variable = self.operator.split(" ")[1].replace("(", "") if self.quantifier else None
    
    def getOperatorArity(self):
        """
        Return operator arity. It is 1 for exists, forall, not and 2 otherwise.
        """
        return self.arity
    
    def __repr__(self):
        """
        Representation of the operator is simply by its name
        """
        return self.operator

    def getPriority(self):
        """
        Priority of the operator
        """
        return self.priority
    
    @staticmethod
    def priority_values(op):
        """
        Not is considered to have the highest priority, then exists, then forall, then and, then or, then implies and iff 
        """
        if op == "not":
            return 5
        elif "exists" in op or "forall" in op:
            return 4
        elif op == "and":
            return 3
        elif op == "or":
            return 2
        elif op in ["=>", "="]:
            return 1
        else:
            return 0

class Predicate:
    """
    Predicate class. This is a recursive class that is used to store predicates and its arguments
    It is recursive because the individual arguments of the predicate class may be formulas or predicates in themselves
    We store the name of the predicate, the various terms (comma separated formulas), the bounded variables in the 
    predicate and the prefix form of the terms in the predicate.
    """
    def __init__(self, predicate_name):
        self.name = predicate_name
        self.terms = []
        self.prefix_form = []
        self.sort = []

    def set_terms(self, terms):
        """
        This function is used to process and set the terms of a specific predicate.
        For example, if the predicate is L(x, y), then the terms are x and y.
        Another example: If the predicate is L(x, g(y) ^ h(a, b)), then the terms are x and g(y) ^ h(a, b) and we
        then need a mechanism to recurse on g(y) ^ h(a, b).
        """
        i = 0
        n_parenthesis = 0
        running_tokens = []     # The tokens in the current term being processed
        while i < len(terms):
            if terms[i] == ',':
                # If we encounter a comma, we end a term if there are no pending open paranthesis.
                # If there are pending open parantheses, then it means that the term is not over.
                # For example, in the function doc above, the comma between a and b does not define end of term in L.
                if n_parenthesis == 0:
                    # If term has ended, then send all of these tokens to process_tokens to recursively process them
                    self.terms.append(CVCGenerator.process_tokens(running_tokens))
                    running_tokens = []
                else:
                    running_tokens.append(terms[i])
            else:
                # Keep track of running open paranthesis
                if terms[i] == '(':
                    n_parenthesis += 1
                elif terms[i] == ')':
                    n_parenthesis -= 1
                running_tokens.append(terms[i])
            i += 1
        
        # When all tokens are done, then we simply process the last set of tokens
        self.terms.append(CVCGenerator.process_tokens(running_tokens))

        # Once all terms are created and processed, we convert each of them separately to postfix.
        for term in self.terms:
            self.prefix_form.append(CVCGenerator.generatePrefixFormula(term))
        
        # Then, we will also find the sort of the terms
        self.find_sort()

    def find_sort(self):
        """
        Function to find the sort of the terms
        """
        self.sort = []
        for term in self.terms:
            # If we have only one token in the term, then it has to be a variable. It cannot be an operator
            # or another predicate.
            if len(term) == 1:
                # If variable is bounded, the sort is BoundSet
                if term[0] in bound_variables:
                    self.sort.append(Sort("BoundSet"))
                else:
                    # Otherwise the variable is unbounded. In that case, we check if the variable already has a sort
                    # if not, we create a new sort.
                    if term[0] not in unbound_variables:
                        unbound_variables[term[0]] = Sort()
                    self.sort.append(unbound_variables[term[0]])
            else:
                # If there are more than one tokens, then it is either an operator applied to multiple predicates or variables.
                # In all cases, the sort is simply bool.
                self.sort.append(Sort("Bool"))
        
        # Unify the sort. This is important in case there are multiple uses of the same predicate. We want to make sure
        # that the sorts are consistent.
        self.unify_sort()
    
    def unify_sort(self):
        """
        Function to unify the sort of the class object with the sort of predicate_to_sort_map
        """
        # If the sort does not exist for the predicate
        if self.name not in predicate_to_sort_map:
            predicate_to_sort_map[self.name] = self.sort
        else:
            # If sort exists, then check compare the two sorts
            cur_sort = predicate_to_sort_map[self.name]
            if len(cur_sort) != len(self.sort):
                raise Exception("Sorts of {0} is not consistent.".format(self.name))
            for i in range(len(cur_sort)):
                cur_sort[i] = self.unify(cur_sort[i], self.sort[i])

    def unify(self, sort1, sort2):
        """
        Unify two specific instances of sort
        """
        sort1_sort = sort1.getSort() if sort1 else None
        sort2_sort = sort2.getSort() if sort2 else None
        if sort1_sort == sort2_sort:   # If they are equal, simply return any of them
            return sort1
        elif sort1_sort is None:       # If one of them is None and other is not, set the one that's not None and return
            sort1.setSort(sort2_sort)
            return sort2
        elif sort2_sort is None:
            sort2.setSort(sort1_sort)
            return sort1
        else:                          # If the sort is not equal
            raise Exception("Sorts of {0} are not consistent".format(self.name))
            
    def __repr__(self):
        """
        Return the prefix form of the predicate
        """
        return  "(" + self.name + " " + " ".join(self.prefix_form) + ")"

def isOperator(op):
    """
    Check if something is an operator. It is an operator if it is one of the standard operator strings or is an instance of
    Operator class.
    """
    operators = ["not", "and", "or", "->", "<->", "=>", "<=>", "="]
    return not isinstance(op, Predicate) and \
           (isinstance(op, Operator) or op in operators or "exists" in op or "forall" in op)


class CVCGenerator:
    """
    Class to generate CVC code for a formula.
    """
    def __init__(self, formula):
        bound_variables.clear()
        unbound_variables.clear()
        predicate_to_sort_map.clear()
        self.formula = formula
        self.tokens = self.tokenize()   # Generate the tokens and process them for a given formula

    def tokenize(self):
        """
        Function to tokenize the code
        """
        # Split the input formula through various operators and paranthesis as tokens.
        tokens = re.split(r'(\(|\)|\s|\bexists\b|\band\b|\bor\b|\bnot\b|\bforall\b|\->|<->|,|<=>|=>|=)', self.formula)
        result = []
        # Remove empty tokens. Replace relevant implies and iff tokens with => and = respectively because they are 
        # supported by CVC
        for token in tokens:
            if token not in ['', ' ']:
                if token == '->':
                    result.append("=>")
                elif token in ['<->', '<=>']:
                    result.append("=")
                else:
                    result.append(token)
        return CVCGenerator.process_tokens(result)
    
    @staticmethod
    def process_tokens(tokens):
        """
        Function to process the tokens
        """
        result = []
        i = 0
        while i < len(tokens):
            # Go through each token
            token = tokens[i]
            if isOperator(token):
                # If it is an operator
                if token in ["exists", "forall"]:
                    # If it is a quantifier, find the bounded variable and add it to the set of bounded variables
                    token = token + " " + tokens[i + 1]
                    bound_variables.add(tokens[i + 1])
                    i += 1

                # Create operator class instance for the token
                result.append(Operator(token))
            else:
                # If it is not an operator
                if i + 1 < len(tokens) and tokens[i + 1] == '(':
                    # If it is a predicate.
                    pred = Predicate(token)
                    i += 2
                    pred_tokens = []
                    n_paranthesis = 1   # Keep track of running paranthesis
                    while i < len(tokens):
                        if tokens[i] == '(':
                            n_paranthesis += 1
                        if tokens[i] == ')':
                            n_paranthesis -= 1         
                        if n_paranthesis == 0:
                            # When the running paranthesis become 0, means that we haev reached the end of the predicate.
                            # In that case, we call set_terms to process the predicate terms
                            pred.set_terms(pred_tokens)   
                            break
                        else:
                            pred_tokens.append(tokens[i])
                        i += 1
                    result.append(pred)
                else:
                    # If it is not a predicate, simply append it
                    result.append(token)
            i += 1
        return result

    @staticmethod
    def infixToPostfix(infix):
        """
        Convert a reversed infix expression to postfix
        """
        if len(infix) == 1:
            return infix

        # Add surrounding paranthesis
        infix = ["("] + infix + [")"]
        l = len(infix)

        # Keep track of output and operator stack
        op_stack = []
        output = []
        
        i = 0
        while i < len(infix):
            # If it is an opening paranthesis, simply push it to stack
            if infix[i] == "(":
                op_stack.append(infix[i])

            # If it is a closing paranthesis, simply keep popping until you find the opening paranthesis
            # and pushing them to output
            elif infix[i] == ")":
                while op_stack[-1] != "(":
                    operator = op_stack.pop()
                    # Also take care of quantified variables if we are getting rid of them
                    if operator.quantifier:
                        bound_variables.remove(operator.quanified_variable)
                    output.append(operator)
                # Pop the opening paranthesis
                op_stack.pop()

            # If it is an operator
            elif isOperator(infix[i]):
                # Pop until the current priority does not become higher than the stack top priority
                while op_stack[-1] != '(' and infix[i].getPriority() <= op_stack[-1].getPriority():
                    operator = op_stack.pop()
                    # Also take care of quantified variables if we are getting rid of them
                    if operator.quantifier:
                        bound_variables.remove(operator.quanified_variable)
                    output.append(operator)

                # If current operator is a quantifier, take care of bound variables
                if infix[i].quantifier:
                    op, variable = str(infix[i]).split(" ")
                    op_stack.append(Operator(op + " ((" + variable + " BoundSet" + "))"))
                    bound_variables.add(variable)
                else:
                    op_stack.append(infix[i])

            # If it is a predicate or a variable, simply push it to output
            else:
                output.append(infix[i])
            i += 1
        
        while len(op_stack) != 0:
            # Once all tokens are processed, simply get done with the op_stack
            operator = op_stack.pop()
            # Also take care of quantified variables if we are getting rid of them
            if operator != '(' and operator.quantifier:
                bound_variables.remove(operator.quanified_variable)
            output.append(operator)
        
        return output            

    @staticmethod
    def generatePrefixFormula(tokens):
        """
        Function to generate the assert statement
        """
        # Reverse the infix formula
        infix = tokens[::-1]  
        for i in range(len(infix)):
            if infix[i] == "(":
                infix[i] = ")"
            elif infix[i] == ")":
                infix[i] = "("
        # Get the reverse postfix formula
        reverse_postfix = CVCGenerator.infixToPostfix(infix)

        # Paranthesize the expression
        stack = []
        for token in reverse_postfix:
            # Push operands to stack
            if not isOperator(token):
                stack.append(token)
            else:
                # For operators, find their arity. Accordingly, pop elements from stack and push the 
                # paranthesized expression on stack
                arity = Operator.getOperatorArity(token)
                if arity == 1:
                    operand = stack.pop()
                    parenthesized_expr = "("+ str(token) + " " + str(operand) +")"
                else: 
                    operand1 = stack.pop()
                    operand2 = stack.pop()
                    parenthesized_expr = "(" + str(token) + " " + str(operand1) + " " + str(operand2) + ")"
                stack.append(parenthesized_expr)
        
        # Stack top is our final formula
        return stack[0]

    def generateCVCScript(self, finite_model_finding=False):
        """
        Function to generate the CVC Script for a given formula
        """
        # Initial declarations in CVC
        cvc_str = "(set-logic ALL)\n(set-option :produce-models true)\n(declare-sort BoundSet 0)\n(declare-sort UnboundSet 0)"
        if finite_model_finding:
            cvc_str += ("\n(set-option :finite-model-find true)"   # Finite model finding, enabled only in selective cases
                        
        prefix_formula = CVCGenerator.generatePrefixFormula(self.tokens)
        
        # Declarations for unbound variables
        for variable in unbound_variables:
            if not unbound_variables[variable].getSort():
                # For unbound variables where the sort is not available, assign them a different sort
                unbound_variables[variable].setSort("UnboundSet")
            cvc_str += "\n(declare-fun {0} () {1})".format(variable, unbound_variables[variable])

        # Declarations for predicates
        for predicate in predicate_to_sort_map:
            sort = " ".join([str(s) for s in predicate_to_sort_map[predicate]])
            cvc_str += "\n(declare-fun {0} ({1}) Bool)".format(predicate, sort)

        # Assert not of the prefix formula
        cvc_str += "\n(assert (not {0}))".format(prefix_formula)

        # Add check-sat and get-model
        cvc_str += "\n(check-sat)\n(get-model)"
        return cvc_str

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python cvc.py "<fol>"')
        sys.exit(1)

    script = CVCGenerator(sys.argv[1].replace("ForAll", "forall").replace("ThereExists", "exists").replace("&", "and").replace("~", "not ").generateCVCScript()
    print(script)

