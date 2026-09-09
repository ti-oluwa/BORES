"""BORES-specific error classes."""

__all__ = [
    "ActionError",
    "BORESError",
    "BoundaryConditionCompilationError",
    "CompilationError",
    "ComputationError",
    "DeserializationError",
    "EventError",
    "ModelCompilationError",
    "PreconditionerError",
    "ScheduleError",
    "SerializableError",
    "SerializationError",
    "SimulationError",
    "SolverError",
    "StopSimulation",
    "StorageError",
    "StreamError",
    "TimingError",
    "ValidationError",
    "WellCompilationError",
]


class BORESError(Exception):
    """Base class for all BORES-related errors."""

    pass


class ValidationError(BORESError, ValueError):
    """Raised when input data fails validation checks."""

    pass


class NotSupportedError(BORESError, NotImplementedError):
    """Raised when a specific feature is currently unsupported"""

    pass


# Solver Errors
class PreconditionerError(BORESError):
    """Raised when there is an error related to preconditioners."""

    pass


class SolverError(BORESError):
    """Raised when a solver fails to solve the given matrix system either due to convergence or other issues."""

    pass


class ComputationError(BORESError):
    """Raised when there is an error during numerical computations."""

    pass


# Simulation Errors
class SimulationError(BORESError):
    """Base class for simulation-related errors."""

    pass


class TimingError(SimulationError):
    """Raised when there is an error related to simulation timing."""

    pass


class StopSimulation(Exception):
    """Raised to signal that the simulation should stop gracefully."""

    pass


class StreamError(SimulationError):
    """Raised when there is an error related to streaming operations."""

    pass


# Serialization Errors
class SerializableError(BORESError):
    """Raised for errors related to the `Serializable` API."""

    pass


class SerializationError(SerializableError):
    """Raised for errors related to serialization of objects."""

    pass


class DeserializationError(SerializableError):
    """Raised for errors related to deserialization of objects."""

    pass


# Storage Errors
class StorageError(BORESError):
    """Raised when there is an error related to data storage operations."""

    pass


# Gridding Errors
class GridError(BORESError):
    """
    Base exception for all grid-related errors.
    """


class InvalidGridError(GridError, ValidationError):
    """
    Raised when a grid definition is invalid.
    """


class InvalidPointArrayError(InvalidGridError):
    """
    Raised when the point coordinate array is invalid.
    """


class InvalidCellConnectivityError(InvalidGridError):
    """
    Raised when cell connectivity is invalid.
    """


class InvalidFaceConnectivityError(InvalidGridError):
    """
    Raised when face connectivity is invalid.
    """


class InvalidGeometryError(InvalidGridError):
    """
    Raised when derived geometry is inconsistent.
    """


class InvalidVolumeError(InvalidGeometryError):
    """
    Raised when one or more cells have invalid volumes.
    """


class InvalidFaceAreaError(InvalidGeometryError):
    """
    Raised when one or more faces have invalid areas.
    """


class InvalidNormalVectorError(InvalidGeometryError):
    """
    Raised when one or more face normals are invalid.
    """


class CellNotFoundError(GridError):
    """
    Raised when a requested cell does not exist.
    """


class FaceNotFoundError(GridError):
    """
    Raised when a requested face does not exist.
    """


class PointNotFoundError(GridError):
    """
    Raised when a requested point does not exist.
    """


class GridIOError(GridError):
    """
    Base exception for grid import/export failures.
    """


class GridImportError(GridIOError):
    """
    Raised when a grid cannot be imported.
    """


class GridExportError(GridIOError):
    """
    Raised when a grid cannot be exported.
    """


class UnsupportedGridFormatError(GridIOError):
    """
    Raised when a grid format is unsupported.
    """


# Compilation Errors
class CompilationError(BORESError):
    """Base class for errors raised while compiling a rich model into its compiled form."""

    pass


class ModelCompilationError(CompilationError):
    """Raised when a `BlackOilModel` fails to compile."""

    pass


class WellCompilationError(CompilationError):
    """Raised when a well system fails to compile."""

    pass


class BoundaryConditionCompilationError(CompilationError):
    """Raised when boundary conditions fail to compile."""

    pass


# Schedule Errors
class ScheduleError(BORESError):
    """Base class for errors raised while building or applying a schedule."""

    pass


class EventError(ScheduleError):
    """Raised when an event fails to evaluate."""

    pass


class ActionError(ScheduleError):
    """Raised when an action fails to apply."""

    pass
