from spec_repair.components.new_spec_encoder import get_violated_expression_names_of_type
from spec_repair.components.optimising_final_spec_learner import OptimisingSpecLearner
from spec_repair.enums import Learning
from spec_repair.helpers.spectra_specification import SpectraSpecification
from spec_repair.ltl_types import GR1FormulaType
from spec_repair.util.set_util import first_minimal_hitting_set, all_minimal_hitting_sets
from spec_repair.util.spec_util import run_all_unrealisable_cores_raw, run_all_unrealisable_cores


def get_trivial_solution(spec: SpectraSpecification, violation_trace: list[str]) -> SpectraSpecification:
    """
    Generate a trivial solution for a given specification and violation trace.

    The function works by:
    1. Removing violated assumptions from the specification
    2. Finding unrealizable cores in the remaining specification
    3. Removing minimal set of guarantees to make spec realizable

    :param spec: The original specification to be modified
    :param violation_trace: Execution trace that violates the specification
    :return: Modified specification with removed assumptions and guarantees
    :raises ValueError: If spec or violation_trace is None
    """
    # Input validation
    if spec is None or violation_trace is None:
        raise ValueError("Specification and violation trace must not be None")

    # Step 1: Remove violated assumptions
    learner = OptimisingSpecLearner()
    violated_assumptions = get_violated_expression_names_of_type(
        learner.get_spec_violations(spec, violation_trace, [], Learning.ASSUMPTION_WEAKENING),
        'assumption'
    )
    print("Violated assumptions:", violated_assumptions)
    new_spec = spec.extract_sub_specification(
        lambda x: (x['type'] == GR1FormulaType.GAR) | (~x['name'].isin(violated_assumptions))
    )

    # Step 2: Find unrealizable cores
    unrealisable_cores = run_all_unrealisable_cores(new_spec.to_str(is_to_compile=True))
    if not (unrealisable_cores):
        print("No unrealizable cores found, new spec actually realizable.")
        return new_spec

    # Step 3: Remove minimal set of guarantees
    guarantees_to_remove = first_minimal_hitting_set(unrealisable_cores)
    print("Guarantees to remove:", guarantees_to_remove)
    trivial_spec = new_spec.extract_sub_specification(
        lambda x: (x['type'] == GR1FormulaType.ASM) | (~x['name'].isin(guarantees_to_remove))
    )

    return trivial_spec

def get_all_trivial_solution(spec: SpectraSpecification, violation_trace: list[str]) -> list[SpectraSpecification]:
    """
    Generate a trivial solution for a given specification and violation trace.

    The function works by:
    1. Removing violated assumptions from the specification
    2. Finding unrealizable cores in the remaining specification
    3. Removing minimal set of guarantees to make spec realizable

    :param spec: The original specification to be modified
    :param violation_trace: Execution trace that violates the specification
    :return: Modified specification with removed assumptions and guarantees
    :raises ValueError: If spec or violation_trace is None
    """
    # Input validation
    if spec is None or violation_trace is None:
        raise ValueError("Specification and violation trace must not be None")

    # Step 1: Remove violated assumptions
    learner = OptimisingSpecLearner()
    violated_assumptions = get_violated_expression_names_of_type(
        learner.get_spec_violations(spec, violation_trace, [], Learning.ASSUMPTION_WEAKENING),
        'assumption'
    )
    print("Violated assumptions:", violated_assumptions)
    new_spec = spec.extract_sub_specification(
        lambda x: (x['type'] == GR1FormulaType.GAR) | (~x['name'].isin(violated_assumptions))
    )

    # Step 2: Find unrealizable cores
    unrealisable_cores = run_all_unrealisable_cores(new_spec.to_str(is_to_compile=True))
    if not (unrealisable_cores):
        print("No unrealizable cores found, new spec actually realizable.")
        return [new_spec]

    # Step 3: Remove minimal set of guarantees
    trivial_specs = []
    guarantees_to_remove_list = all_minimal_hitting_sets(unrealisable_cores)
    for i, guarantees_to_remove in enumerate(guarantees_to_remove_list):
        print("Guarantees to remove:", guarantees_to_remove)
        trivial_spec = new_spec.extract_sub_specification(
            lambda x: (x['type'] == GR1FormulaType.ASM) | (~x['name'].isin(guarantees_to_remove))
        )
        trivial_specs.append(trivial_spec)

    return trivial_specs
