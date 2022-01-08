import itertools
import random
import re
import string
from collections import defaultdict, Counter
from functools import reduce
from operator import eq, neg
import time

# from sortedcontainers import SortedSet

import search
from utils import argmin_random_tie, count, first, extend

variables = []
domains = {}
domtemp = {}
template = []
conlist = []
neighbors = {}
with open('dom2-f24.txt','r') as file: #diabazoyme to arxeio
    # reading each line     
    for line in file: 
        # reading each word
        for word in line.split():
            template.append(int(word)) #prosthetoyme se mia lista gia thn eykolh xthsh toy
template.pop(0)
x=0
j=0
for i in range(1,len(template)): #Dhmioyrgiea dictionary me kleidi to prwto noymero kai value ola ta ypoloipa
    if i==x+1:
        j = template[i]
        continue
    if j==0:
        x = i
        continue
    if template[x] in domtemp:
        domtemp[template[x]].append(template[i])
    else:
        domtemp[template[x]] = [template[i]]
    j-=1
file.close()#kleisimo arxeioy
template.clear()
i=0
with open('var2-f24.txt','r') as file: #lista metavlhtwn 
    # reading each line     
    for line in file: 
        # reading each word         
        for word in line.split():
            i+=1
            template.append(int(word))
            if i%2==0:
                variables.append(int(word))
file.close()
template.pop(0)

for i in range(1,len(template)): #Dhmioyrgia toy dicitonary domains
    if i%2 == 1:
        domains[template[i-1]] = domtemp[template[i]]


i=0
with open('ctr2-f24.txt','r') as file:#diavazoyme arxeio
    # reading each line     
    for line in file: 
        # reading each word
        if i==0 :
            i+=1
            continue
        for word in line.split():
            conlist.append(word) #Lista me tiw le3eis gia thn eykolh diaxeirhsh toy
            if i%4==1: #Dhmioyrgia twn goitwnwn
                temp = int(word)
                if int(word) not in neighbors:
                    neighbors[int(word)] = []
            if i%4==2:
                if int(word) not in neighbors:
                    neighbors[int(word)] = [temp]
                else:
                    neighbors[int(word)].append(temp)
                neighbors[temp].append(int(word))
            i+=1
file.close()

condict = {}

for i in range(len(conlist)): #Dictionary me ena kleidi mia lista apo 2 metavlhtes
        if i%4==0:
            condict[(int(conlist[i]),int(conlist[i+1]))] = [conlist[i+2],int(conlist[i+3]),1]

cDict = defaultdict(dict)

for i in condict: #Dictionary gia na psaxnei h synarthsh constraints
    cDict[i[0]] [i[1]] = [condict[i][0],condict[i][1], 1]
    cDict[i[1]] [i[0]] = [condict[i][0],condict[i][1], 1]

def constraints(A,a,B,b): #Synarthsh poy koitaei an 2 metavlhtes A kai B paroyn tis times a kai b antistoixa ikanopoiei kathe periorismo
    constraints.num+=1
    if cDict[A][B][0] == '>' and abs(a-b)>cDict[A][B][1]:
        return True
    elif cDict[A][B][0] == '=' and abs(a-b)==cDict[A][B][1]:
        return True
    return False
constraints.num = 0 #global timh
class CSP(search.Problem):
    """This class describes finite-domain Constraint Satisfaction Problems.
    A CSP is specified by the following inputs:
        variables   A list of variables; each is atomic (e.g. int or string).
        domains     A dict of {var:[possible_value, ...]} entries.
        neighbors   A dict of {var:[var,...]} that for each variable lists
                    the other variables that participate in constraints.
        constraints A function f(A, a, B, b) that returns true if neighbors
                    A, B satisfy the constraint when they have values A=a, B=b

    In the textbook and in most mathematical definitions, the
    constraints are specified as explicit pairs of allowable values,
    but the formulation here is easier to express and more compact for
    most cases (for example, the n-Queens problem can be represented
    in O(n) space using this notation, instead of O(n^4) for the
    explicit representation). In terms of describing the CSP as a
    problem, that's all there is.

    However, the class also supports data structures and methods that help you
    solve CSPs by calling a search function on the CSP. Methods and slots are
    as follows, where the argument 'a' represents an assignment, which is a
    dict of {var:val} entries:
        assign(var, val, a)     Assign a[var] = val; do other bookkeeping
        unassign(var, a)        Do del a[var], plus other bookkeeping
        nconflicts(var, val, a) Return the number of other variables that
                                conflict with var=val
        curr_domains[var]       Slot: remaining consistent values for var
                                Used by constraint propagation routines.
    The following methods are used only by graph_search and tree_search:
        actions(state)          Return a list of actions
        result(state, action)   Return a successor of state
        goal_test(state)        Return true if all constraints satisfied
    The following are just for debugging purposes:
        nassigns                Slot: tracks the number of assignments made
        display(a)              Print a human-readable representation
    """

    def __init__(self, variables, domains, neighbors, constraints):
        """Construct a CSP problem. If variables is empty, it becomes domains.keys()."""
        super().__init__(())
        variables = variables or list(domains.keys())
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.curr_domains = None
        self.nassigns = 0
        self.i=0
    def assign(self, var, val, assignment):
        """Add {var: val} to assignment; Discard the old value if any."""
        assignment[var] = val
        self.nassigns += 1

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            del assignment[var]

    def nconflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""

        # Subclasses may implement this more efficiently
        def conflict(var2):
            return var2 in assignment and not self.constraints(var, val, var2, assignment[var2])

        return count(conflict(v) for v in self.neighbors[var])

    def display(self, assignment):
        """Show a human-readable representation of the CSP."""
        # Subclasses can print in a prettier way, or display with a GUI
        print(assignment)

    # These methods are for the tree and graph-search interface:

    def actions(self, state):
        """Return a list of applicable actions: non conflicting
        assignments to an unassigned variable."""
        if len(state) == len(self.variables):
            return []
        else:
            assignment = dict(state)
            var = first([v for v in self.variables if v not in assignment])
            return [(var, val) for val in self.domains[var]
                    if self.nconflicts(var, val, assignment) == 0]

    def result(self, state, action):
        """Perform an action and return the new state."""
        (var, val) = action
        return state + ((var, val),)

    def goal_test(self, state):
        """The goal is to assign all variables, with all constraints satisfied."""
        assignment = dict(state)
        return (len(assignment) == len(self.variables)
                and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                        for variables in self.variables))

    # These are for constraint propagation

    def support_pruning(self):
        """Make sure we can prune values from domains. (We want to pay
        for this only if we use it.)"""
        if self.curr_domains is None:
            self.curr_domains = {v: list(self.domains[v]) for v in self.variables}

    def suppose(self, var, value):
        """Start accumulating inferences from assuming var=value."""
        self.support_pruning()
        removals = [(var, a) for a in self.curr_domains[var] if a != value]
        self.curr_domains[var] = [value]
        return removals

    def prune(self, var, value, removals):
        """Rule out var=value."""
        self.curr_domains[var].remove(value)
        if removals is not None:
            removals.append((var, value))

    def choices(self, var):
        """Return all values for var that aren't currently ruled out."""
        return (self.curr_domains or self.domains)[var]

    def infer_assignment(self):
        """Return the partial assignment implied by the current inferences."""
        self.support_pruning()
        return {v: self.curr_domains[v][0]
                for v in self.variables if 1 == len(self.curr_domains[v])}

    def restore(self, removals):
        """Undo a supposition and all inferences from it."""
        for B, b in removals:
            self.curr_domains[B].append(b)

    # This is for min_conflicts search

    def conflicted_vars(self, current):
        """Return a list of variables in current assignment that are in conflict"""
        return [var for var in self.variables
                if self.nconflicts(var, current[var], current) > 0]


# ______________________________________________________________________________
# Constraint Propagation with AC3


def no_arc_heuristic(csp, queue):
    return queue


# def dom_j_up(csp, queue):
#     return SortedSet(queue, key=lambda t: neg(len(csp.curr_domains[t[1]])))


def AC3(csp, queue=None, removals=None, arc_heuristic=no_arc_heuristic):
    """[Figure 6.3]"""
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    checks = 0
    while queue:
        (Xi, Xj) = queue.pop()
        revised, checks = revise(csp, Xi, Xj, removals, checks)
        if revised:
            if not csp.curr_domains[Xi]:
                cDict[Xi][Xj][2]+=1 #Au3anoyme to varos toy constraint kata 1 
                cDict[Xj][Xi][2]+=1
                return False, checks  # CSP is inconsistent
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
    return True, checks  # CSP is satisfiable


def revise(csp, Xi, Xj, removals, checks=0):
    """Return true if we remove a value."""
    revised = False
    for x in csp.curr_domains[Xi][:]:
        # If Xi=x conflicts with Xj=y for every possible y, eliminate Xi=x
        # if all(not csp.constraints(Xi, x, Xj, y) for y in csp.curr_domains[Xj]):
        conflict = True
        for y in csp.curr_domains[Xj]:
            if csp.constraints(Xi, x, Xj, y):
                conflict = False
            checks += 1
            if not conflict:
                break
        if conflict:
            csp.prune(Xi, x, removals)
            revised = True
    return revised, checks


# Constraint Propagation with AC3b: an improved version
# of AC3 with double-support domain-heuristic

def AC3b(csp, queue=None, removals=None, arc_heuristic=no_arc_heuristic):
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    checks = 0
    while queue:
        (Xi, Xj) = queue.pop()
        # Si_p values are all known to be supported by Xj
        # Sj_p values are all known to be supported by Xi
        # Dj - Sj_p = Sj_u values are unknown, as yet, to be supported by Xi
        Si_p, Sj_p, Sj_u, checks = partition(csp, Xi, Xj, checks)
        if not Si_p:
            return False, checks  # CSP is inconsistent
        revised = False
        for x in set(csp.curr_domains[Xi]) - Si_p:
            csp.prune(Xi, x, removals)
            revised = True
        if revised:
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
        if (Xj, Xi) in queue:
            if isinstance(queue, set):
                # or queue -= {(Xj, Xi)} or queue.remove((Xj, Xi))
                queue.difference_update({(Xj, Xi)})
            else:
                queue.difference_update((Xj, Xi))
            # the elements in D_j which are supported by Xi are given by the union of Sj_p with the set of those
            # elements of Sj_u which further processing will show to be supported by some vi_p in Si_p
            for vj_p in Sj_u:
                for vi_p in Si_p:
                    conflict = True
                    if csp.constraints(Xj, vj_p, Xi, vi_p):
                        conflict = False
                        Sj_p.add(vj_p)
                    checks += 1
                    if not conflict:
                        break
            revised = False
            for x in set(csp.curr_domains[Xj]) - Sj_p:
                csp.prune(Xj, x, removals)
                revised = True
            if revised:
                for Xk in csp.neighbors[Xj]:
                    if Xk != Xi:
                        queue.add((Xk, Xj))
    return True, checks  # CSP is satisfiable


def partition(csp, Xi, Xj, checks=0):
    Si_p = set()
    Sj_p = set()
    Sj_u = set(csp.curr_domains[Xj])
    for vi_u in csp.curr_domains[Xi]:
        conflict = True
        # now, in order to establish support for a value vi_u in Di it seems better to try to find a support among
        # the values in Sj_u first, because for each vj_u in Sj_u the check (vi_u, vj_u) is a double-support check
        # and it is just as likely that any vj_u in Sj_u supports vi_u than it is that any vj_p in Sj_p does...
        for vj_u in Sj_u - Sj_p:
            # double-support check
            if csp.constraints(Xi, vi_u, Xj, vj_u):
                conflict = False
                Si_p.add(vi_u)
                Sj_p.add(vj_u)
            checks += 1
            if not conflict:
                break
        # ... and only if no support can be found among the elements in Sj_u, should the elements vj_p in Sj_p be used
        # for single-support checks (vi_u, vj_p)
        if conflict:
            for vj_p in Sj_p:
                # single-support check
                if csp.constraints(Xi, vi_u, Xj, vj_p):
                    conflict = False
                    Si_p.add(vi_u)
                checks += 1
                if not conflict:
                    break
    return Si_p, Sj_p, Sj_u - Sj_p, checks


# Constraint Propagation with AC4

def AC4(csp, queue=None, removals=None, arc_heuristic=no_arc_heuristic):
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    support_counter = Counter()
    variable_value_pairs_supported = defaultdict(set)
    unsupported_variable_value_pairs = []
    checks = 0
    # construction and initialization of support sets
    while queue:
        (Xi, Xj) = queue.pop()
        revised = False
        for x in csp.curr_domains[Xi][:]:
            for y in csp.curr_domains[Xj]:
                if csp.constraints(Xi, x, Xj, y):
                    support_counter[(Xi, x, Xj)] += 1
                    variable_value_pairs_supported[(Xj, y)].add((Xi, x))
                checks += 1
            if support_counter[(Xi, x, Xj)] == 0:
                csp.prune(Xi, x, removals)
                revised = True
                unsupported_variable_value_pairs.append((Xi, x))
        if revised:
            if not csp.curr_domains[Xi]:
                return False, checks  # CSP is inconsistent
    # propagation of removed values
    while unsupported_variable_value_pairs:
        Xj, y = unsupported_variable_value_pairs.pop()
        for Xi, x in variable_value_pairs_supported[(Xj, y)]:
            revised = False
            if x in csp.curr_domains[Xi][:]:
                support_counter[(Xi, x, Xj)] -= 1
                if support_counter[(Xi, x, Xj)] == 0:
                    csp.prune(Xi, x, removals)
                    revised = True
                    unsupported_variable_value_pairs.append((Xi, x))
            if revised:
                if not csp.curr_domains[Xi]:
                    return False, checks  # CSP is inconsistent
    return True, checks  # CSP is satisfiable


# ______________________________________________________________________________
# CSP Backtracking Search

# Variable ordering


def first_unassigned_variable(assignment, csp):
    """The default variable order."""
    return first([var for var in csp.variables if var not in assignment])


def mrv(assignment, csp):
    """Minimum-remaining-values heuristic."""
    return argmin_random_tie([v for v in csp.variables if v not in assignment],
                             key=lambda var: num_legal_values(csp, var, assignment))


def num_legal_values(csp, var, assignment):
    if csp.curr_domains:
        return len(csp.curr_domains[var])
    else:
        return count(csp.nconflicts(var, val, assignment) == 0 for val in csp.domains[var])

def domwdeg(assignment, csp):
    csp.support_pruning()
    fddg = 100
    for var in csp.variables:
        if var not in assignment:
            dom =  len(csp.curr_domains[var])
            wdeg = 0 
            for nei in neighbors[var]:
               wdeg+=cDict[var][nei][2]
            ddg = dom / wdeg
            if fddg > ddg:
                fddg = ddg
                fvar = var
    return fvar
    
# Value ordering


def unordered_domain_values(var, assignment, csp):
    """The default value order."""
    return csp.choices(var)


def lcv(var, assignment, csp):
    """Least-constraining-values heuristic."""
    return sorted(csp.choices(var), key=lambda val: csp.nconflicts(var, val, assignment))


# Inference


def no_inference(csp, var, value, assignment, removals):
    return True


def forward_checking(csp, var, value, assignment, removals):
    """Prune neighbor values inconsistent with var=value."""
    csp.support_pruning()
    for B in csp.neighbors[var]:
        if B not in assignment:
            for b in csp.curr_domains[B][:]:
                if not csp.constraints(var, value, B, b):
                    csp.prune(B, b, removals)
            if not csp.curr_domains[B]:
                #var me B weight++ sto dict
                cDict[var][B][2]+=1 #Au3anoyme to varos kata 1
                cDict[B][var][2]+=1
                return False
    return True


def mac(csp, var, value, assignment, removals, constraint_propagation=AC3):
    """Maintain arc consistency."""
    return constraint_propagation(csp, {(X, var) for X in csp.neighbors[var]}, removals)


# The search, proper

#fc-dbk einai idio me backtracking
def backtracking_search(csp, select_unassigned_variable=first_unassigned_variable,
                        order_domain_values=unordered_domain_values, inference=no_inference):
    """[Figure 6.5]"""
    def backtrack(assignment):
        global start_time
        if time.time()-start_time>3000: #An parei perissotero apo 5 lepta stamata to
            return None
        backtracking_search.num +=1
        if len(assignment) == len(csp.variables):
            return assignment
        var = select_unassigned_variable(assignment, csp)
        for value in order_domain_values(var, assignment, csp):
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                if inference(csp, var, value, assignment, removals):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                csp.restore(removals)
        csp.unassign(var, assignment)
        return None

    result = backtrack({})
    assert result is None or csp.goal_test(result)
    return result

backtracking_search.num = 0

# ______________________________________________________________________________
# Min-conflicts Hill Climbing search for CSPs


def min_conflicts(csp, max_steps=50):
    """Solve a CSP by stochastic Hill Climbing on the number of conflicts."""
    # Generate a complete assignment for all variables (probably with conflicts)
    csp.current = current = {}
    for var in csp.variables:
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    # Now repeatedly choose a random conflicted variable and change it
    for _ in range(max_steps):
        conflicted = csp.conflicted_vars(current)
        if not conflicted:
            return current
        var = random.choice(conflicted)
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    return None

min_conflicts.num = 0

def min_conflicts_value(csp, var, current):
    min_conflicts_value.num +=1
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose at random."""
    return argmin_random_tie(csp.domains[var], key=lambda val: csp.nconflicts(var, val, current))

min_conflicts_value.num  = 0

neighdict={}
for i in variables:
    neighdict[i]=set()

def FC(csp, var, value, assignment, removals): #FC algorithm
    """Prune neighbor values inconsistent with var=value."""
    csp.support_pruning()
    for B in csp.neighbors[var]:
        for b in csp.curr_domains[B][:]:
            if not csp.constraints(var, value, B, b):
                if B in assignment:
                    neighdict[var].add(B) #Prosthetoyme ton geitona sthn lista apo set gia to synolo sygkroysewn
                csp.prune(B, b, removals)
        if not csp.curr_domains[B]:
            if var<B:
                cDict[var][B][2]+=1
            else:
                cDict[B][var][2]+=1
            return False
    return True

def fc_cbj(csp, select_unassigned_variable=domwdeg, #Vasismeno sto backtracking_search
                        order_domain_values=unordered_domain_values, inference=FC):
    """[Figure 6.5]"""

    def backtrack(assignment,myvar=None):
        fc_cbj.num+=1
        if len(assignment) == len(csp.variables):
            return assignment
        if myvar==None:
            var = select_unassigned_variable(assignment, csp)
        else:
            var=myvar
        for value in order_domain_values(var, assignment, csp):
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                if inference(csp, var, value, assignment, removals):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                csp.restore(removals)
        csp.unassign(var, assignment)
        if neighdict[var]: #An den parei kanena value tote prepei na kanei backjump se sygkekrimenh metavlhth 
            proo=max(neighdict[var]) #Vriskei thn megalyterh metavlhth giati einai kai ayth poy phge teleytaia
            neighdict[var].remove(proo) #Afairoyme to set
            neighdict[proo].update(neighdict[var]) #Vazoyme to proigomyeno set ston proorismo
            neighdict[var].clear()
            backtrack(assignment,proo)
        return None
    result = backtrack({})
    assert result is None or csp.goal_test(result)
    return result


fc_cbj.num =0


mycsp = CSP(variables,domains,neighbors,constraints)


global start_time

print("\n\nRunning with: forward_checking\n")
start_time = time.time()
if backtracking_search(mycsp,domwdeg,unordered_domain_values,forward_checking):
    print("SAT")
else:
    print("UNSAT")
print("--- %s seconds ---" % (time.time() - start_time))
print("CONSTRAINT",constraints.num)
print("NODES",backtracking_search.num)
print("\n\nRunning with: MAC\n")
backtracking_search.num = 0
constraints.num=0
start_time = time.time()
if backtracking_search(mycsp,domwdeg,unordered_domain_values,mac):
    print("SAT")
else:
    print("UNSAT")
print("--- %s seconds ---" % (time.time() - start_time))
print("CONSTRAINT",constraints.num)
print("NODES",backtracking_search.num)
print("\n\nRunning with: min_conflict\n")
backtracking_search.num = 0
constraints.num=0
start_time = time.time()
if min_conflicts(mycsp):
    print("SAT")
else:
    print("UNSAT")
print("--- %s seconds ---" % (time.time() - start_time))
print("CONSTRAINT",constraints.num)
print("NODES",min_conflicts_value.num)
print("\n\nRunning with: FC-CBJ\n")
start_time = time.time()
constraints.num=0
if fc_cbj(mycsp,domwdeg,unordered_domain_values,FC):
    print("SAT")
else:
    print("UNSAT")
print("--- %s seconds ---" % (time.time() - start_time))
print("CONSTRAINT",constraints.num)
print("NODES",fc_cbj.num)


