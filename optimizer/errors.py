class OptimizerError(Exception):
    pass


class ParameterValidationError(OptimizerError):
    pass


class SafeExpressionError(OptimizerError):
    pass


class RunnerContractError(OptimizerError):
    pass


class StorageError(OptimizerError):
    pass


class FingerprintMismatchError(StorageError):
    pass


class UnsupportedFeatureError(OptimizerError):
    pass
