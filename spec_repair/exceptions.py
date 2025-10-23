class NoViolationException(Exception):
    pass


class NoWeakeningException(Exception):
    pass


class NoAssumptionWeakeningException(NoWeakeningException):
    pass


class NoGuaranteeWeakeningException(NoWeakeningException):
    pass


class DeadlockRequiredException(Exception):
    pass


class LearningException(Exception):
    pass
