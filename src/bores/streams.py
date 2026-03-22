"""
Stream model state with optional persistence for memory-efficient simulation workflows.
"""

import logging
import queue
import threading
import time
import typing

import numpy as np
from typing_extensions import Self

from bores.errors import StorageError, StreamError
from bores.states import ModelState, validate_state
from bores.stores import DataStore, EntryMeta
from bores.types import NDimension

__all__ = ["StateStream", "StreamProgress"]

logger = logging.getLogger(__name__)


class StreamProgress(typing.TypedDict):
    """Progress statistics for state streaming."""

    yield_count: int
    saved_count: int
    checkpoints_count: int
    batch_pending: int
    store_backend: typing.Optional[str]
    memory_usage: float


_stop_io = 0  # Signal for stopping I/O thread


class StateStream(typing.Generic[NDimension]):
    """
    Memory-efficient stream for model state iteration with optional persistence.

    Wraps a state generator/iterator and optionally persists states to disk as they're yielded,
    immediately freeing memory. Supports batching for I/O efficiency and async I/O for
    non-blocking disk writes.

    **Why stream states?**
    - Low memory overhead (states persisted immediately/eventually after yield)
    - Batch persistence for I/O efficiency
    - Optional async I/O (2-3x speedup when I/O slower than simulation)
    - Optional validation before save
    - Progress tracking and logging
    - Auto-save on context exit (no lost data)
    - Replay from store (load previously saved states)
    - Selective persistence (save only states matching predicate)
    - Checkpointing for crash recovery
    - Memory monitoring with automatic flushing

    **Async/Background I/O**
    When `background_io=True` disk writes happen in a background thread.  The
    simulation fills a queue; the I/O worker drains it.  `max_queue_size`
    applies back-pressure so memory stays bounded when the simulation is faster
    than disk.

    **Persistence model**
    States are accumulated in a local batch buffer.  When the buffer reaches
    `batch_size` (or the memory limit), `flush(...)` is called:

    * **Synchronous path** — each item in the batch is appended to the store directly.
    * **Asynchronous/background path** — the batch list is enqueued; the I/O worker appends each
      item to the store and then discards the list.

    The store's `append(...)` method is used (not `dump`) so existing entries are
    never overwritten.  The store must have `supports_append=True`.

    **Replay**
    After the generator is exhausted, `replay(...)` loads all saved states back
    from the store.  `__iter__` does this automatically when
    `auto_replay=True` (the default).

    Example Usage:

    ```python
    store = HDF5Store("run01.h5")
    with StateStream(states=run(), store=store, background_io=True) as stream:
        for state in stream.until(some_condition):
            analyse(state)          # background thread writes while we analyse

    # Replay the whole run later
    for state in stream.replay():
        plot(state)

    # Load only specific entries
    for state in stream.replay(indices=[0, 50, 99]):
        ...

    # Load entries matching a predicate on EntryMeta
    for state in stream.replay(predicate=lambda e: e.index % 10 == 0):
        ...
    ```
    """

    def __init__(
        self,
        states: typing.Optional[typing.Iterable[ModelState[NDimension]]] = None,
        store: typing.Optional[DataStore[ModelState[NDimension], typing.Any]] = None,
        batch_size: int = 50,
        validate: bool = False,
        auto_save: bool = True,
        auto_replay: bool = True,
        save: typing.Union[
            typing.Callable[[ModelState[NDimension]], bool], bool
        ] = True,
        checkpoint_store: typing.Optional[
            DataStore[ModelState[NDimension], typing.Any]
        ] = None,
        checkpoint_interval: typing.Optional[int] = None,
        max_batch_memory_usage: typing.Optional[float] = None,
        background_io: bool = False,
        max_queue_size: int = 100,
        io_thread_name: str = "stream-io-worker",
        queue_timeout: float = 1.0,
    ) -> None:
        """
        Initialize state stream.

        :param states: Generator or iterator of `ModelState` instances
        :param store: Optional `DataStore` for persistence. If None, states only yielded (no persistence).
            The data store must support appending new states. that is, `store.supports_append` must be True.
        :param batch_size: Number of states to accumulate before flushing to disk (default: 50)
        :param validate: Validate states before persisting (default: False)
        :param auto_save: Automatically flush remaining states on context exit (default: True)
        :param auto_replay: If True, automatically replay from store when iterating after consumption.
            If False, raises `StreamError` instead (default: True)
        :param lazy_load: When replaying from store, use lazy loading if supported (default: True)
        :param save: Usually a function to filter which states to save. Can be a boolean flag that will determined whether to save states or not.
            If provided, only states where save(state) returns True are saved.
            Example: ```lambda s: s.step % 10 == 0``` (save every 10th state)
        :param checkpoint_interval: Optional interval for checkpointing. If provided,
            creates a checkpoint every N states for crash recovery. Example: 100
        :param checkpoint_store: Optional `DataStore` for checkpointing. This must be provide if `checkpoint_interval` is set.
            The data store must support appending new states. that is, `store.supports_append` must be True.
        :param max_batch_memory_usage: Maximum batch memory in MB before forcing flush.
            Estimated by sampling first state's memory footprint. Batch flushes when either
            `batch_size` or `max_batch_memory_usage` threshold is reached. Example: 50.0 MB
        :param background_io: Enable asynchronous I/O for non-blocking disk writes (default: False).
            When enabled, disk writes happen in a background thread, allowing simulation to continue.
            Provides 2-3x speedup when I/O is slower than simulation timesteps.
        :param max_queue_size: Maximum states/batches in I/O queue before blocking (default: 100).
            Acts as backpressure to prevent unbounded memory growth when I/O can't keep up.
            Higher values allow more buffering but use more memory. Use a negative value for unbounded growth.
        :param io_thread_name: Name for I/O worker thread, useful for debugging (default: "state-io-worker")
        :param queue_timeout: Timeout in seconds for queue operations (default: 1.0).
            Used for responsive shutdown and error checking.
        """
        self.states = iter(states) if states is not None else None
        self.store = store
        self.batch_size = batch_size
        self.validate = validate
        self.auto_save = auto_save
        self.auto_replay = auto_replay
        self.save = save
        self.checkpoint_store = checkpoint_store
        self.checkpoint_interval = checkpoint_interval
        self.max_batch_memory_usage = max_batch_memory_usage

        self.background_io = background_io
        self.max_queue_size = max_queue_size
        self.io_thread_name = io_thread_name
        self.queue_timeout = queue_timeout

        # Incompatible option warnings
        if self.store is None:
            if self.validate:
                logger.warning(
                    "Validation is enabled but no store provided. States will be validated "
                    "but not persisted."
                )
            if self.auto_save:
                logger.debug(
                    "`auto_save=True` but no store provided. This setting has no effect."
                )
            if self.save is not None:
                logger.warning(
                    "`save` provided but no store configured. Predicate will be ignored."
                )
            if self.max_batch_memory_usage is not None:
                logger.warning(
                    "`max_batch_memory_usage` provided but no store configured. Memory-based flushing "
                    "will not occur without persistence."
                )
            if self.background_io:
                logger.warning(
                    "`background_io=True` but no store provided. Async I/O disabled."
                )
                self.background_io = False

        if store is not None and not store.supports_append:
            raise StreamError(
                f"Store {store!r} does not support appending. {self.__class__.__name__} requires `supports_append=True`."
            )

        if checkpoint_interval is not None and checkpoint_store is None:
            raise StreamError(
                "`checkpoint_store` must be provided when `checkpoint_interval` is set."
            )

        if checkpoint_store is not None and not checkpoint_store.supports_append:
            raise StreamError(
                f"`checkpoint_store` {checkpoint_store!r} does not support appending."
            )

        # Internal state
        self._batch: typing.List[ModelState[NDimension]] = []
        self._yield_count: int = 0
        self._saved_count: int = 0
        self._checkpoints_count: int = 0
        self._started: bool = False
        self._uses_save_func = callable(save)
        self._consumed: bool = False
        self._state_size_mb: typing.Optional[float] = None

        # Async I/O infrastructure
        self._io_queue: typing.Optional[queue.Queue] = None
        self._io_thread: typing.Optional[threading.Thread] = None
        self._io_error: typing.Optional[Exception] = None
        self._shutdown_event: typing.Optional[threading.Event] = None
        self._saved_count_lock = threading.Lock()  # Protects _saved_count in async mode

        # Store-only (replay) mode
        if self.states is None and self.store is None:
            raise StreamError("Either `states` or `store` must be provided.")

        if self.states is None and self.store is not None:
            # Store-only mode is intended for replay
            if not self.auto_replay:
                logger.warning(
                    "Creating stream with `store` but no `states`. forcing `auto_replay=True`."
                )
                self.auto_replay = True
            # Mark as already consumed since there's no states to iterate
            self._consumed = True

        if self.background_io:
            self._start_io_worker()

    def _start_io_worker(self) -> None:
        """Start the background I/O worker thread."""
        self._io_queue = queue.Queue(maxsize=self.max_queue_size)
        self._shutdown_event = threading.Event()
        self._io_thread = threading.Thread(
            target=self._io_worker,
            name=self.io_thread_name,
            daemon=False,
        )
        self._io_thread.start()
        logger.info(
            f"Started I/O worker thread '{self.io_thread_name}' "
            f"(max_queue_size={self.max_queue_size if self.max_queue_size >= 0 else 'infinite'})"
        )

    def _io_worker(self) -> None:
        """
        Background thread worker that handles all I/O operations.

        Continuously pulls batches from queue and writes to store.
        Exits when stop IO signal is received or shutdown event is set.
        """
        logger.debug(f"I/O worker thread started (thread_id={threading.get_ident()})")
        if self._io_queue is None or self.store is None or self._shutdown_event is None:
            logger.error("I/O infrastructure not properly initialized")
            return

        try:
            with self.store(mode="a") as store:
                while not self._shutdown_event.is_set():
                    try:
                        # Get batch from queue (timeout to check shutdown periodically)
                        item = self._io_queue.get(timeout=self.queue_timeout)
                        if item is _stop_io:
                            logger.debug("I/O worker received shutdown signal")
                            self._io_queue.task_done()
                            break

                        # Process batch
                        batch: typing.List[ModelState[NDimension]] = item
                        logger.debug(f"I/O worker writing batch of {len(batch)} states")

                        try:
                            count = 0
                            for state in batch:
                                store.append(
                                    state,
                                    validator=validate_state if self.validate else None,
                                    meta=lambda s: {"step": s.step},
                                )
                                count += 1

                            with self._saved_count_lock:
                                self._saved_count += count

                            batch.clear()
                            del batch  # Free memory immediately
                            logger.debug(
                                f"I/O worker completed batch (total saved: {self._saved_count})"
                            )
                        except Exception as exc:
                            logger.error(f"I/O worker error during write: {exc}")
                            self._io_error = exc
                            raise
                        finally:
                            self._io_queue.task_done()

                    except queue.Empty:
                        continue

        except Exception as exc:
            logger.error(f"I/O worker thread crashed: {exc}")
            self._io_error = exc
        finally:
            logger.debug("I/O worker thread exiting")

    def _check_io_error(self) -> None:
        """Check if I/O thread encountered an error and raise it."""
        with self._saved_count_lock:
            if self._io_error is not None:
                raise StreamError(
                    f"Background I/O thread failed: {self._io_error}"
                ) from self._io_error

    def _wait_for_queue(self) -> None:
        """Wait for all pending I/O operations to complete."""
        if not self.background_io or self._io_queue is None:
            return

        logger.debug("Waiting for I/O queue to drain...")
        t0 = time.perf_counter()
        self._io_queue.join()
        logger.debug(f"Queue drain took {time.perf_counter() - t0:.2f}s")

        # Check for errors that occurred during drain
        self._check_io_error()
        logger.debug("I/O queue drained successfully")

    def _stop_io_thread(self) -> None:
        """Stop the I/O worker thread gracefully."""
        if not self.background_io or self._io_thread is None:
            return

        logger.debug("Stopping I/O worker thread...")

        if self._io_queue is None or self._shutdown_event is None:
            logger.error("I/O infrastructure not properly initialized")
            return

        # Signal shutdown
        self._shutdown_event.set()
        self._io_queue.put(_stop_io)
        # Wait for thread to finish
        self._io_thread.join(timeout=30.0)

        if self._io_thread.is_alive():
            logger.error(
                "I/O worker thread did not exit within 30s timeout. "
                "Some data may not have been written."
            )
        else:
            logger.info("I/O worker thread stopped successfully")

        # Final error check
        self._check_io_error()

    def __iter__(self) -> typing.Iterator[ModelState[NDimension]]:
        """
        Iterate over states, optionally persisting as we go.

        Yields states one at a time, accumulating in a batch buffer.
        When batch is full, flush to store and clear buffer.

        Memory pattern:
            1. Yield state to user → User processes it
            2. Add to batch buffer (small memory cost)
            3. When batch full → Flush to disk, clear buffer
            4. Net effect: Only `batch_size` states stay in memory at once

        Note: If the underlying iterable is a generator, it can only be consumed once.
        After the first iteration:
            - If `auto_replay=True` and a store exists, automatically replays from store
            - If `auto_replay=False`, raises `StreamError` (use `replay()` explicitly)
            - If no store exists, raises `StreamError` (create fresh stream)

        :return: Iterator over `ModelState` instances
        :raises `StreamError`: If trying to iterate again after exhaustion (when `auto_replay=False` or no store)
        """
        # Check if we've already consumed the iterable
        if self._consumed:
            if self.auto_replay and self.store is not None:
                logger.debug(
                    "Stream already consumed. Auto-replaying from store. "
                    "Set `auto_replay=False` to disable this behavior."
                )
                yield from self.replay()
                return

            elif self.store is not None:
                raise StreamError(
                    "Stream already consumed. The underlying iterable has been exhausted. "
                    "Use `replay()` or set `auto_replay=True`."
                )
            else:
                raise StreamError(
                    "Stream already consumed and no store available for replay."
                )

        # No states provided, this shouldn't happen as `_consumed` should already be set to false
        # but still handle it
        if self.states is None:
            raise StreamError("No states provided and stream not consumed.")

        if self.store is None:
            logger.info("No store provided, streaming without persistence")
            for state in self.states:
                self._yield_count += 1
                yield state
            self._consumed = True
            return

        io_mode = "async" if self.background_io else "sync"
        logger.debug(
            f"Streaming -> {self.store} ({io_mode}, batch_size={self.batch_size})"
        )

        for state in self.states:
            self._yield_count += 1

            # Surface any background I/O errors before continuing/yielding
            if self.background_io:
                self._check_io_error()

            yield state

            if self._should_save(state=state):
                self._batch.append(state)

                if self._should_flush():
                    self.flush(block=False)

                if self._should_checkpoint(state=state):
                    self._save_checkpoint(state=state)

        # Flush whatever is left
        if self._batch and self.auto_save:
            logger.debug(f"Flushing final batch of {len(self._batch)} states")
            self.flush(block=False)

        # Mark the stream as consumed
        self._consumed = True
        logger.debug(
            f"Stream exhausted: {self._yield_count} yielded, {self._saved_count} saved"
        )

    def __enter__(self) -> Self:
        """
        Context manager entry. Prepare for streaming.

        :return: Self for context manager usage
        """
        self._started = True
        logger.info(
            f"Started stream session to {self.store!s}"
            if self.store
            else "Started stream session (no persistence)"
        )
        return self

    def __exit__(
        self,
        exc_type: typing.Optional[typing.Type[BaseException]],
        exc_val: typing.Optional[BaseException],
        exc_tb: typing.Optional[typing.Any],
    ) -> None:
        """
        Context manager exit. Flushes any remaining states and ensure async I/O completion.

        Ensures all states saved even if iteration interrupted.
        For async I/O, waits for background thread to finish all pending writes.

        :param exc_type: Exception type if error occurred
        :param exc_val: Exception value if error occurred
        :param exc_tb: Exception traceback if error occurred
        """
        if self._batch and self.auto_save and self.store is not None:
            logger.warning(f"Flushing {len(self._batch)} unsaved states on exit")
            try:
                self.flush(block=False)  # Enqueue for I/O worker
            except Exception as exc:
                logger.error(f"Failed to flush states on exit: {exc}", exc_info=True)

        # Wait for background I/O to complete
        if self.background_io:
            try:
                logger.info("Waiting for background I/O to complete...")
                self._wait_for_queue()
                self._stop_io_thread()
            except Exception as exc:
                logger.error(f"Error during I/O worker shutdown: {exc}", exc_info=True)
                if exc_type is None:
                    raise

        if exc_type is None:
            logger.info(
                f"Stream complete: {self._saved_count} states saved, "
                f"{self._checkpoints_count} checkpoints created."
            )
        else:
            logger.error(
                f"Stream interrupted after {self._saved_count} states have been saved: {exc_val}"
            )

    def last(self) -> typing.Optional[ModelState[NDimension]]:
        """
        Get the last state from the stream.

        Iterates through the entire stream and returns the final state.
        Useful for quickly accessing only the end result of a simulation.

        :return: The last `ModelState` instance, or None if the stream is empty
        """
        logger.debug("Retrieving last state from stream")
        if self._consumed and self.store is not None:
            max_idx = self.store.max_index()
            if max_idx is None:
                return None
            results = list(self.store.load(ModelState, indices=[max_idx]))
            return results[0] if results else None

        last_state: typing.Optional[ModelState[NDimension]] = None
        for state in self:
            last_state = state

        if last_state is not None:
            logger.debug(f"Last state retrieved: step {last_state.step}")
        else:
            logger.debug("Stream is empty, no last state available")
        return last_state

    def consume(self) -> None:
        """
        Exhaust the entire stream without yielding states.

        This method iterates through all states, triggering any configured side effects
        (persistence, checkpointing, validation) without returning states. Useful when
        you only want the side effects (saving to store, creating checkpoints) without
        processing individual states.

        The stream's internal mechanisms (__iter__, batching, flushing, checkpointing)
        still occur normally, only the yielding to caller is skipped.

        Example:
        ```python
        # Just save all states to disk without processing them
        stream = StateStream(states=simulation(), store=store)
        stream.consume()  # States saved, nothing returned

        # Create checkpoints without holding states in memory
        stream = StateStream(
            states=simulation(),
            checkpoint_interval=100,
            checkpoint_store=HDF5Store("./checkpoints.h5")
        )
        stream.consume()  # Checkpoints created, stream exhausted
        ```

        Note: After calling `consume()`, the stream is exhausted. Calling it again has no effect.
        """
        if self._consumed:
            logger.debug("Stream already consumed")
            return

        logger.debug("Consuming stream (no yield to caller)")
        for _ in self:
            pass  # Iterate through, triggering side effects but not yielding
        logger.debug(f"Stream consumed: {self._yield_count} states processed")

    def replay(
        self,
        indices: typing.Optional[typing.Sequence[int]] = None,
        predicate: typing.Optional[typing.Callable[[EntryMeta], bool]] = None,
        steps: typing.Optional[
            typing.Union[typing.Sequence[int], typing.Callable[[int], bool]]
        ] = None,
        validator: typing.Optional[
            typing.Callable[[ModelState[NDimension]], ModelState[NDimension]]
        ] = None,
    ) -> typing.Iterator[ModelState[NDimension]]:
        """
        Load and iterate over previously saved states from the store.

        All filtering happens before any array data is deserialised, so skipped
        entries have no I/O cost.  `indices`, `steps`, and `predicate` can
        be combined: `indices` always takes priority and bypasses the other two;
        `steps` and `predicate` are composed with a logical AND when both are
        supplied.

        Note: each call to `replay(...)` continues to increment `yield_count`.
        Replaying 100 states after streaming 100 states gives `yield_count == 200`.

        :param indices: Load only the entries at these zero-based insertion-order
            positions.  When given, `steps` and `predicate` are ignored.
        :param steps: Filter by simulation step number.  Accepts either a sequence
            of exact step numbers (`steps=[0, 100, 200]`) or a callable that
            receives a step number and returns `bool`
            (`steps=lambda s: s % 50 == 0`).  Composed with `predicate` when
            both are provided.
        :param predicate: `(EntryMeta) -> bool` filter evaluated against stored
            entry metadata.  Use this for any metadata beyond step number, e.g.
            `predicate=lambda e: e.meta.get("converged")`.  Composed with
            `steps` when both are provided.
        :param validator: Optional post-load callable applied to each deserialised
            state before it is yielded.  Defaults to `validate_state` when the
            stream was constructed with `validate=True`.
        :return: Iterator over `ModelState` instances matching the filter criteria,
            in insertion order.
        :raises StreamError: If no store was provided at construction time.
            Or if any storage-related error occurs.
        """
        if self.store is None:
            raise StreamError("Cannot replay: no store provided")

        logger.debug(f"Replaying from {self.store}")

        pred = predicate
        if steps:
            if callable(steps):

                def _predicate(entry: EntryMeta) -> bool:
                    step = int(entry.meta.get("step", -1))
                    if predicate is not None:
                        return steps(step) and predicate(entry)  # type: ignore
                    return steps(step)  # type: ignore
            else:
                steps_set = set(steps)

                def _predicate(entry: EntryMeta) -> bool:
                    step = int(entry.meta.get("step", -1))
                    in_steps = step in steps_set
                    if predicate is not None:
                        return in_steps and predicate(entry)
                    return in_steps

            pred = _predicate

        try:
            for state in self.store.load(
                ModelState,
                indices=indices,
                predicate=pred,
                validator=validator or (validate_state if self.validate else None),
            ):
                self._yield_count += 1
                yield state
        except StorageError as exc:
            raise StreamError(
                f"An error occured while replaying stream: {exc}"
            ) from exc

        logger.debug(f"Replay complete: {self._yield_count} total yielded")

    def until(
        self, condition: typing.Callable[[ModelState[NDimension]], bool]
    ) -> typing.Iterator[ModelState[NDimension]]:
        """
        Iterate over states, optionally persisting as we go, until `condition` evaluates to True.

        This is a wrapper around `__iter__`, allowing you to stream states until a stop criteria is met

        :param condition: Iteration stop condition/criteria.
        :return: Iterator over `ModelState` instances
        :raises `StreamError`: If trying to iterate again after exhaustion (when auto_replay=False or no store)
        """
        for state in self:
            if condition(state):
                yield state
                break
            yield state

    def while_(
        self, condition: typing.Callable[[ModelState[NDimension]], bool]
    ) -> typing.Iterator[ModelState[NDimension]]:
        """
        Iterate over states, optionally persisting as we go, as long `condition` evaluates to True.
        Stop when `condition` evaluates to False.

        This is a wrapper around `__iter__`, allowing you to stream states
        based on a criteria until it can no longer be met

        :param condition: Iteration continuation condition/criteria.
        :return: Iterator over `ModelState` instances
        :raises `StreamError`: If trying to iterate again after exhaustion (when auto_replay=False or no store)
        """
        for state in self:
            if not condition(state):
                yield state
                break
            yield state

    def flush(self, block: bool = False) -> None:
        """
        Manually flush accumulated batch to store.

        Sync path:
            Each item in the batch is appended to the store in sequence, then
            the batch buffer is cleared.

        Async path
            The batch list is handed off to the I/O worker queue and the
            buffer is cleared immediately so the simulation can keep running.
            Pass `block=True` to wait until the queue has fully drained.

        :param block: If True, wait for I/O thread to complete all pending writes.
            If False (default), just enqueue and return immediately (async behavior).
            Only relevant when `background_io=True`.
        :raises StreamError: If no store provided or I/O error occurred
        """
        if self.store is None:
            raise StreamError("Cannot flush: no store provided")

        if not self._batch:
            if block and self.background_io:
                # Even with empty batch, block might want to wait for queue
                logger.debug("Flush called with empty batch, waiting for queue...")
                self._wait_for_queue()
            else:
                logger.debug("Flush called but batch is empty")
            return

        batch_size = len(self._batch)

        if self.background_io:
            logger.debug(
                f"Enqueuing batch of {batch_size} states to I/O thread (block={block})"
            )

            # Check for errors before enqueuing
            self._check_io_error()
            if (
                self._io_queue is None
                or self.store is None
                or self._shutdown_event is None
            ):
                raise StreamError("I/O infrastructure not properly initialized")

            try:
                # Put batch in queue. May block if queue is full (backpressure)
                self._io_queue.put(self._batch.copy(), timeout=3.0)
                logger.debug(f"Batch enqueued (queue size: ~{self._io_queue.qsize()})")

                # Clear batch and reassign to new list to free memory immediately
                self._batch = []
                # If blocking requested, wait for queue to drain
                if block:
                    self._wait_for_queue()

            except queue.Full:
                logger.error(
                    f"I/O queue full ({self.max_queue_size}) "
                    f"Consider increasing `max_queue_size`, ensure `max_queue_size` is a certain magnitude larger than "
                    f"`batch_size` ({self.batch_size}), or slow down the simulation."
                )
                raise StreamError("I/O queue full. Backpressure limit reached")
        else:
            logger.debug(f"Flushing batch of {batch_size} states to {self.store}")

            try:
                with self.store(mode="a") as store:
                    for state in self._batch:
                        store.append(
                            state,
                            validator=validate_state if self.validate else None,
                            meta=lambda s: {"step": s.step},
                        )

                self._saved_count += batch_size
                logger.debug(
                    f"Flushed {batch_size} states (total saved: {self._saved_count})"
                )
            except Exception as exc:
                logger.error(f"Failed to flush batch: {exc}")
                raise
            finally:
                # Reassign to new list to free memory immediately
                self._batch = []

    def get_pending_batch(self) -> typing.List[ModelState[NDimension]]:
        """
        Get a copy of states in the current batch (not yet flushed to store).

        Useful for:
        - Inspecting what will be saved on next flush
        - Recovering states if an error occurs before flush
        - Debugging batch accumulation behavior

        :return: Copy of the current batch buffer (safe to modify without affecting stream)
        """
        return self._batch.copy()

    def _estimate_state_size(self, state: ModelState[NDimension]) -> float:
        """
        Estimate memory footprint of a single state in MB.

        :param state: State to measure
        :return: Estimated size in MB
        """
        if self._state_size_mb is not None:
            return self._state_size_mb

        size_bytes = 0

        for attr_name in dir(state):
            if attr_name.startswith("_"):
                continue

            try:
                attr = getattr(state, attr_name)
                if isinstance(attr, np.ndarray):
                    size_bytes += attr.nbytes
                elif hasattr(attr, "__dict__"):
                    for nested_attr_name in dir(attr):
                        if nested_attr_name.startswith("_"):
                            continue
                        nested_attr = getattr(attr, nested_attr_name, None)
                        if isinstance(nested_attr, np.ndarray):
                            size_bytes += nested_attr.nbytes
            except Exception as exc:
                logger.warning(
                    f"Failed to estimate memory size for {attr_name!r}: {exc}",
                    exc_info=True,
                )
                continue

        self._state_size_mb = size_bytes / 1024 / 1024
        logger.debug(f"Estimated state size: {self._state_size_mb:.2f} MB")
        return self._state_size_mb

    def _should_save(self, state: ModelState[NDimension]) -> bool:
        """
        Determine if state should be saved based on `save`.

        :param state: State to evaluate
        :return: True if state should be saved, False otherwise
        """
        if not self._uses_save_func:
            return self.save  # type: ignore[return-value]
        return self.save(state)  # type: ignore[call-arg]

    def _should_flush(self) -> bool:
        """
        Determine if batch should be flushed based on batch size and memory limits.

        :return: True if batch should be flushed, False otherwise
        """
        if len(self._batch) >= self.batch_size:
            return True

        if self.max_batch_memory_usage is not None and self._batch:
            state_size = self._estimate_state_size(self._batch[0])
            batch_memory_usage = state_size * len(self._batch)

            if batch_memory_usage > self.max_batch_memory_usage:
                logger.warning(
                    f"Batch memory limit reached ({batch_memory_usage:.1f} MB > "
                    f"{self.max_batch_memory_usage:.1f} MB) with {len(self._batch)} states - "
                    f"flushing early"
                )
                return True

        return False

    def _should_checkpoint(self, state: ModelState[NDimension]) -> bool:
        """
        Determine if a checkpoint should be created for this state.

        :param state: Current state
        :return: True if checkpoint should be created, False otherwise
        """
        return (
            self.checkpoint_interval is not None
            and self.checkpoint_store is not None
            and state.step > 0
            and state.step % self.checkpoint_interval == 0
        )

    def _save_checkpoint(self, state: ModelState[NDimension]) -> None:
        """
        Save a checkpoint for crash recovery.

        Creates a separate checkpoint file that can be used to resume simulation
        from this point.

        :param state: State to checkpoint
        """
        if self.checkpoint_store is None:
            return
        try:
            self.checkpoint_store.append(
                state,
                validator=validate_state if self.validate else None,
                meta=lambda s: {"step": s.step},
            )
            self._checkpoints_count += 1
            logger.info(f"Checkpoint saved at step {state.step}")
        except Exception as exc:
            logger.error(
                f"Failed to save checkpoint at step {state.step}: {exc}", exc_info=True
            )

    def checkpoint(self, step: int) -> ModelState[NDimension]:
        """
        Load a specific checkpoint by step number.

        :param step: Step number of checkpoint to load
        :return: Loaded ModelState from checkpoint
        :raises `StreamError`: If checkpointing not configured
        :raises `FileNotFoundError`: If checkpoint doesn't exist
        """
        if self.checkpoint_store is None:
            raise StreamError(
                "Checkpointing not configured. No `checkpoint_store` found."
            )

        results = list(
            self.checkpoint_store.load(
                ModelState,
                predicate=lambda e: int(e.meta["step"]) == step,
            )
        )
        if not results:
            raise StreamError(f"No checkpoint found for step {step}.")
        return results[0]

    def checkpoints(self) -> typing.Generator[ModelState[NDimension], None, None]:
        """
        Yield all checkpointed states in insertion order.

        :return: Generator yielding state checkpoints
        :raises `StreamError`: If checkpointing not configured
        """
        if self.checkpoint_store is None:
            raise StreamError("No `checkpoint_store` configured.")
        yield from self.checkpoint_store.load(ModelState)

    def list_checkpoints(self) -> typing.List[int]:
        """
        List all available checkpoint step numbers.

        :return: Sorted list of checkpoint step numbers
        :raises `StreamError`: If checkpointing not configured
        """
        if self.checkpoint_store is None:
            raise StreamError(
                "Checkpointing not configured. No `checkpoint_store` found"
            )

        return sorted(
            int(e.meta["step"])
            for e in self.checkpoint_store.entries()
            if "step" in e.meta
        )

    def progress(self) -> StreamProgress:
        """
        Get streaming progress statistics.

        :return: Dictionary with progress metrics including:
            - yield_count: Total states yielded
            - saved_count: Total states saved to store
            - checkpoints_count: Total checkpoints created
            - batch_pending: States in current batch (not yet saved)
            - store_backend: Type of store being used (or None)
            - memory_usage: Estimated batch memory in MB
            - io_queue_size: Size of async I/O queue (if background_io enabled)
            - io_thread_alive: Whether I/O thread is running (if background_io enabled)
        """
        if self._batch and self._state_size_mb is not None:
            batch_memory_usage = self._state_size_mb * len(self._batch)
        else:
            batch_memory_usage = 0.0

        progress = StreamProgress(
            yield_count=self._yield_count,
            saved_count=self._saved_count,
            checkpoints_count=self._checkpoints_count,
            batch_pending=len(self._batch),
            store_backend=type(self.store).__name__ if self.store else None,
            memory_usage=batch_memory_usage,
        )
        if self.background_io and self._io_queue is not None:
            progress["io_queue_size"] = self._io_queue.qsize()  # type: ignore[typeddict-unknown-key]
            progress["io_thread_alive"] = (  # type: ignore[typeddict-unknown-key]
                self._io_thread.is_alive() if self._io_thread else False
            )
        return progress

    @property
    def yield_count(self) -> int:
        """
        Total number of states yielded (including replays).

        :return: Count of yielded states
        """
        return self._yield_count

    @property
    def is_consumed(self) -> bool:
        """
        Check if the underlying iterable has been exhausted.

        Once consumed, the stream cannot be iterated again unless:
        1. A store was provided and contains data (will auto-replay)
        2. A fresh stream is created with a new iterable

        :return: True if the iterable has been consumed, False otherwise
        """
        return self._consumed

    @property
    def saved_count(self) -> int:
        """
        Number of states saved to store so far.

        :return: Count of saved states
        """
        return self._saved_count

    @property
    def checkpoints_count(self) -> int:
        """
        Number of checkpoints created so far.

        :return: Count of checkpoints
        """
        return self._checkpoints_count

    def __repr__(self) -> str:
        store_info = f"store={self.store}" if self.store else "no store"
        io_mode = "async" if self.background_io else "sync"
        return (
            f"{self.__class__.__name__}({store_info}, {io_mode}, "
            f"batch_size={self.batch_size}, yielded={self._yield_count}, "
            f"saved={self._saved_count})"
        )
