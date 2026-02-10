from typing import List, Set

from py_ltl.formula import LTLFormula, Globally, Implies, AtomicProposition, Not, Top, Bottom, Eventually, And, Or, \
    Next, Prev, Until
from functools import reduce


def is_ednf_implies_ednf(f: LTLFormula) -> bool:
    # Check if f is Implies(lhs, rhs) and lhs, rhs are DNF
    if not isinstance(f, Implies):
        return False
    return is_ednf(f.left) and is_ednf(f.right)


def is_g_ednf(f: LTLFormula) -> bool:
    # Check if f is Globally(formula) and formula is DNF
    if not isinstance(f, Globally):
        return False
    return is_ednf(f.formula)


def is_g_ednf_implies_ednf(f: LTLFormula) -> bool:
    # Check if f is Globally(Implies(lhs, rhs)) and lhs, rhs are DNF
    if not isinstance(f, Globally):
        return False
    if not isinstance(f.formula, Implies):
        return False
    return is_ednf(f.formula.left) and is_ednf(f.formula.right)


def is_g_ednf_implies_f_ednf(f: LTLFormula) -> bool:
    # Check if f is Globally(Implies(lhs, Eventually(rhs))) and lhs, rhs are DNF
    if not isinstance(f, Globally):
        return False
    if not isinstance(f.formula, Implies):
        return False
    if not is_ednf(f.formula.left):
        return False
    if not (isinstance(f.formula.right, Eventually) and is_ednf(f.formula.right.formula)):
        return False
    return True


def is_gf_dnf(f: LTLFormula) -> bool:
    # Check if f is Globally(Eventually(formula)) and formula is DNF
    if not isinstance(f, Globally):
        return False
    if not isinstance(f.formula, Eventually):
        return False
    return is_ednf(f.formula.formula)


def is_literal(f: LTLFormula) -> bool:
    # Literal = atomic prop or negation of atomic prop (maybe true/false too)
    if isinstance(f, AtomicProposition):
        return True
    if isinstance(f, Not) and isinstance(f.formula, AtomicProposition):
        return True
    if isinstance(f, Top) or isinstance(f, Bottom):
        return True
    return False


def is_conjunction_of_literals(f: LTLFormula) -> bool:
    if is_literal(f):
        return True
    if isinstance(f, And):
        return is_conjunction_of_literals(f.left) and is_conjunction_of_literals(f.right)
    return False


def is_disjunction_of_conjunctions(f: LTLFormula) -> bool:
    if is_conjunction_of_literals(f):
        return True
    if isinstance(f, Or):
        return is_disjunction_of_conjunctions(f.left) and is_disjunction_of_conjunctions(f.right)
    return False


def to_dnf(f: LTLFormula) -> LTLFormula:
    # Base cases
    if is_literal(f):
        return f
    if isinstance(f, Not):
        # Push negation inward (De Morgan)
        formula = f.formula
        if isinstance(formula, AtomicProposition) or isinstance(formula, Top) or isinstance(formula, Bottom):
            return f
        if isinstance(formula, Not):
            return to_dnf(formula.formula)
        if isinstance(formula, And):
            return to_dnf(Or(Not(formula.left), Not(formula.right)))
        if isinstance(formula, Or):
            return to_dnf(And(Not(formula.left), Not(formula.right)))
        raise NotImplementedError("Negation push-down for this formula not implemented")
    if isinstance(f, And):
        left = to_dnf(f.left)
        right = to_dnf(f.right)
        # Distribute OR over AND:
        if isinstance(left, Or):
            # (A or B) and C => (A and C) or (B and C)
            return to_dnf(Or(And(left.left, right), And(left.right, right)))
        if isinstance(right, Or):
            # A and (B or C) => (A and B) or (A and C)
            return to_dnf(Or(And(left, right.left), And(left, right.right)))
        return And(left, right)
    if isinstance(f, Or):
        left = to_dnf(f.left)
        right = to_dnf(f.right)
        return Or(left, right)
    if isinstance(f, Implies):
        return to_dnf(Or(Not(f.left), f.right))
    # Temporal operators and others we don't convert to DNF:
    # Return as-is or raise
    return f


def normalize_to_pattern(formula: LTLFormula) -> LTLFormula:
    if is_pattern(formula):
        return formula
    # Otherwise, we must convert
    if isinstance(formula, Globally):
        inner_formula = normalize_inner_formula_to_pattern(formula.formula)
        formula = Globally(inner_formula)
    else:
        formula = normalize_inner_formula_to_pattern(formula)
    # TODO: eventually this should become unnecessary, but for now we need to check again
    if is_pattern(formula):
        return formula
    # Otherwise, raise ValueError
    raise ValueError(f"Formula {formula} cannot be converted to G(F(f)) or G(f→g) form")


def is_pattern(formula: LTLFormula) -> bool:
    # If already matches one of the six, done
    pattern_checks = [
        is_ednf,
        is_ednf_implies_ednf,
        is_g_ednf,
        is_g_ednf_implies_ednf,
        is_g_ednf_implies_f_ednf,
        is_gf_dnf
    ]

    for check in pattern_checks:
        if check(formula):
            return True
    return False


def normalize_inner_formula_to_pattern(inner):
    if isinstance(inner, Implies):
        lhs_dnf = to_ednf(inner.left)
        rhs = inner.right
        # if rhs can be turned into Eventually(DNF), do so
        if isinstance(rhs, Eventually):
            rhs_dnf = to_ednf(rhs.formula)
            return Implies(lhs_dnf, Eventually(rhs_dnf))
        else:
            rhs_dnf = to_ednf(rhs)
            return Implies(lhs_dnf, rhs_dnf)
    elif isinstance(inner, Eventually):
        inner = inner.formula
        inner_dnf = to_ednf(inner)
        return Eventually(inner_dnf)
    else:
        # convert to GF(DNF)
        inner_dnf = to_ednf(inner)
        return inner_dnf


def is_conjunction_of_literals_and_temporals(f: LTLFormula) -> bool:
    # Checks if it's a conjunction of literals and at most one Next(...) and one Prev(...)
    if is_literal(f):
        return True

    if isinstance(f, And):
        items = list(flatten_and(f))
    else:
        items = [f]

    next_count = 0
    prev_count = 0

    for item in items:
        if is_literal(item):
            continue
        elif isinstance(item, Next):
            if not is_conjunction_of_literals(item.formula):
                return False
            next_count += 1
        elif isinstance(item, Prev):
            if not is_conjunction_of_literals(item.formula):
                return False
            prev_count += 1
        else:
            return False

    return next_count <= 1 and prev_count <= 1


def flatten_and(f: LTLFormula):
    if isinstance(f, And):
        yield from flatten_and(f.left)
        yield from flatten_and(f.right)
    else:
        yield f


def flatten_or(f: LTLFormula):
    if isinstance(f, Or):
        yield from flatten_or(f.left)
        yield from flatten_or(f.right)
    else:
        yield f


def is_dnf(f: LTLFormula) -> bool:
    return is_disjunction_of_conjunctions(f)


def is_ednf(f: LTLFormula) -> bool:
    # EDNF is a disjunction of conjunctions with grouped temporals
    disjuncts = list(flatten_or(f))
    return all(is_conjunction_of_literals_and_temporals(d) for d in disjuncts)


def fold_or(formulas):
    return reduce(lambda x, y: Or(x, y), formulas)


def group_temporals_in_and(f: LTLFormula) -> LTLFormula:
    items = list(flatten_and(f))
    literals = []
    nexts = []
    prevs = []

    for item in items:
        if isinstance(item, Next):
            nexts.append(item.formula)
        elif isinstance(item, Prev):
            prevs.append(item.formula)
        else:
            literals.append(item)

    if nexts:
        literals.append(Next(conjoin(nexts)) if len(nexts) > 1 else Next(nexts[0]))
    if prevs:
        literals.append(Prev(conjoin(prevs)) if len(prevs) > 1 else Prev(prevs[0]))

    if not literals:
        return Top()
    elif len(literals) == 1:
        return literals[0]
    else:
        return conjoin(literals)

def conjoin(formulas: List[LTLFormula]):
    """
    Return a conjunction of the given formulas.
    """
    return reduce(lambda x, y: And(x, y), formulas)

def disjoin(formulas: List[LTLFormula]):
    """
    Return a disjunction of the given formulas.
    """
    return reduce(lambda x, y: Or(x, y), formulas)

def to_ednf(f: LTLFormula) -> LTLFormula:
    # Apply to_dnf first
    f = to_dnf(f)
    disjuncts = list(flatten_or(f))
    grouped = [group_temporals_in_and(d) for d in disjuncts]
    return fold_or(grouped) if len(grouped) > 1 else grouped[0]


def satisfies_ltl_formula(this_formula: LTLFormula, trace: List[Set[str]], t: int = 0) -> bool:
    match this_formula:
        case AtomicProposition(name=name, value=value):
            return name in trace[t]
        case Not(formula=formula):
            return not satisfies_ltl_formula(formula, trace, t)
        case And(left=lhs, right=rhs):
            return all(satisfies_ltl_formula(sub, trace, t) for sub in [lhs, rhs])
        case Or(left=lhs, right=rhs):
            return any(satisfies_ltl_formula(sub, trace, t) for sub in [lhs, rhs])
        case Implies(left=lhs, right=rhs):
            return any(satisfies_ltl_formula(sub, trace, t) for sub in [Not(formula=lhs), rhs])
        case Next(formula=formula):
            if t + 1 >= len(trace):
                return False
            return satisfies_ltl_formula(formula, trace, t + 1)
        case Prev(formula=formula):
            if t - 1 >= 0:
                satisfies_ltl_formula(formula, trace, t - 1)
            return False
        case Eventually(formula=formula):
            return any(satisfies_ltl_formula(formula, trace, j) for j in range(t, len(trace)))
        case Until(left=lhs, right=rhs):
            for j in range(t, len(trace)):
                if satisfies_ltl_formula(rhs, trace, j):
                    return all(satisfies_ltl_formula(lhs, trace, k) for k in range(t, j))
            return False
        case Globally(formula=formula):
            return all(satisfies_ltl_formula(formula, trace, j) for j in range(t, len(trace)))
        case Top():
            return True
        case Bottom():
            return False
        case _:
            raise NotImplementedError(f"Unsupported operator: {formula}")
