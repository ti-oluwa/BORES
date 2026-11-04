"""
Stream data with optional persistence for memory-efficient iteration over any `Serializable` item iterable.
"""

import atexit
import logging
import queue
import threading
import time
import typing

import numpy as np
from typing_extensions import Self

from bores.errors import StorageError, StreamError
from bores.serde.base import SerializableT
from bores.serde.stores.base import DataStore, EntryMeta
from bores.utils import close_iterator

__all__ = ["DataStream", "StreamProgress"]

logger = logging.getLogger(__name__)


class StreamProgress(typing.TypedDict):
    """Progress statistics for a data stream."""

    yield_count: int
    saved_count: int
    checkpoints_count: int
    batch_pending: int
    store_backend: str | None
    memory_usage: float


STOP_IO = object()  # Signal for stopping I/O thread


class DataStream(typing.Generic[SerializableT]):
    """
    Memory-efficient stream for iterating any `Serializable` item
    with optional persistence in a `DataStore`.

    Wraps a generator/iterator of `SerializableT` items and optionally persists them to
    a `DataStore` as they're yielded, immediately freeing memory. Supports batching for
    I/O efficiency and async I/O for non-blocking disk writes. Generic over any
    `Serializable`/`StoreSerializable` type e.g, reservoir items, well items, or any
    other item your `DataStore` backends know how to (de)serialise.

    **Why stream items?**
    - Low memory overhead (items persisted immediately/eventually after yield)
    - Batch persistence for I/O efficiency
    - Optional async I/O (2-3x speedup when I/O slower than production of items)
    - Optional validation before save
    - Progress tracking and logging
    - Auto-save on context exit (no lost data)
    - Replay from store (load previously saved items)
    - Selective persistence (save only items matching predicate)
    - Checkpointing for crash recovery
    - Memory monitoring with automatic flushing

    **Async/Background I/O**
    When `background_io=True` disk writes happen in a background thread. The
    source fills a queue; the I/O worker drains it. `max_queue_size`
    applies back-pressure so memory stays bounded when the source produces
    items faster than disk can absorb them.

    **Persistence model**
    Items are accumulated in a local batch buffer. When the buffer reaches
    `batch_size` (or the memory limit), `flush(...)` is called:

    * **Synchronous path** - each item in the batch is appended to the store directly.
    * **Asynchronous/background path** - the batch list is enqueued; the I/O worker appends each
      item to the store and then discards the list.

    The store's `append(...)` method is used (not `dump`) so existing entries are
    never overwritten. The store must have `can_append=True`.

    **Replay**
    After the source is exhausted, `replay(...)` loads all saved items back
    from the store. `__iter__` does this automatically when `auto_replay=True` (the default).

    Example Usage:

    ```python
    store = HDF5Store("run01.h5")
    with DataStream(ReservoirState, source=run(), store=store, background_io=True) as stream:
        for item in stream.until(some_condition):
            analyse(item)  # background thread writes while we analyse

    # Replay the whole run later
    for item in stream.replay():
        plot(item)

    # Load only specific entries
    for item in stream.replay(indices=[0, 50, 99]):
        ...

    # Load entries matching a predicate on EntryMeta
    for item in stream.replay(predicate=lambda e: e.idx % 10 == 0):
        ...
    ```
    """

    def __init__(
        self,
        item_type: type[SerializableT],
        source: typing.Iterable[SerializableT] | None = None,
        store: DataStore[SerializableT, typing.Any] | None = None,
        batch_size: int = 50,
        validator: typing.Callable[[SerializableT], SerializableT] | None = None,
        meta: typing.Callable[[SerializableT], dict[str, typing.Any]] | None = None,
        auto_save: bool = True,
        auto_replay: bool = True,
        save: typing.Callable[[SerializableT], bool] | bool = True,
        checkpoint_store: DataStore[SerializableT, typing.Any] | None = None,
        checkpoint_interval: int | None = None,
        max_batch_memory_usage: float | None = None,
        background_io: bool = False,
        max_queue_size: int = 100,
        io_thread_name: str = "stream-io-worker",
        queue_timeout: float = 1.0,
    ) -> None:
        """
        Initializes data stream.

        :param item_type: The `Serializable` subclass being streamed. Required
            even when `source` is given because `store`-only (replay) construction has
            no live item to infer it from, and `DataStore.load(...)` always needs
            an explicit type to deserialise into, so this is never optional.
        :param source: Generator or iterator of `item_type` instances. May be
            omitted for store-only (replay) construction.
        :param store: Optional `DataStore` for persistence. If None, items are only
            yielded (no persistence). The data store must support appending new
            items, i.e. `store.can_append` must be True.
        :param batch_size: Number of items to accumulate before flushing to disk (default: 50)
        :param validator: Optional callable applied to each item before it is
            persisted (to `store`, `checkpoint_store`, and on `replay(...)`
            unless overridden there). Receives the item and must return a
            (possibly transformed) item of the same type; raise to abort the
            write. `None` (default) skips validation entirely.
        :param meta: Optional callable that receives an item and returns a plain
            `dict` of JSON-serialisable values to store alongside it (forwarded
            directly to `DataStore.append`'s own `meta` parameter).
        :param auto_save: Automatically flush remaining items on context exit (default: True)
        :param auto_replay: If True, automatically replay from store when iterating after
            consumption. If False, raises `StreamError` instead (default: True)
        :param save: Usually a function to filter which items to save. Can be a boolean flag
            that determines whether to save items or not.
            If provided, only items where save(item) returns True are saved.
            Example: ```lambda s: s.step % 10 == 0``` (save every 10th item)
        :param checkpoint_interval: Optional interval, in *yielded item count* (not any
            domain-specific counter), for checkpointing. If provided, creates a checkpoint
            every N yielded items for crash recovery. Example: 100
        :param checkpoint_store: Optional `DataStore` for checkpointing. This must be provided if
            `checkpoint_interval` is set. The data store must support appending new items, i.e.
            `store.can_append` must be True.
        :param max_batch_memory_usage: Maximum batch memory in MB before forcing flush.
            Estimated by sampling first item's memory footprint. Batch flushes when either
            `batch_size` or `max_batch_memory_usage` threshold is reached. Example: 50.0 MB
        :param background_io: Enable asynchronous I/O for non-blocking disk writes (default: False).
            When enabled, disk writes happen in a background thread, allowing the source to keep
            producing items. Provides 2-3x speedup when I/O is slower than item production.
        :param max_queue_size: Maximum items/batches in I/O queue before blocking (default: 100).
            Acts as backpressure to prevent unbounded memory growth when I/O can't keep up.
            Higher values allow more buffering but use more memory. Use a negative value for unbounded growth.
        :param io_thread_name: Name for I/O worker thread, useful for debugging (default: "stream-io-worker")
        :param queue_timeout: Timeout in seconds for queue operations (default: 1.0).
            Used for responsive shutdown and error checking.
        """
        self.item_type = item_type
        self.source = iter(source) if source is not None else None
        self.store = store
        self.batch_size = batch_size
        self.validator = validator
        self.meta = meta
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
            if self.validator:
                logger.warning(
                    "`validator` is set but no store provided. Items will be validated "
                    "but not persisted."
                )
            if self.auto_save:
                logger.debug("`auto_save=True` but no store provided. This setting has no effect.")
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
                logger.warning("`background_io=True` but no store provided. Async I/O disabled.")
                self.background_io = False

        if store is not None and not store.can_append:
            raise StreamError(
                f"Store {store!r} does not support appending. {self.__class__.__name__} requires `can_append=True`."
            )

        if checkpoint_interval is not None and checkpoint_store is None:
            raise StreamError(
                "`checkpoint_store` must be provided when `checkpoint_interval` is set."
            )

        if checkpoint_store is not None and not checkpoint_store.can_append:
            raise StreamError(
                f"`checkpoint_store` {checkpoint_store!r} does not support appending."
            )

        # Internal item
        self._batch: list[SerializableT] = []
        self._yield_count: int = 0
        self._saved_count: int = 0
        self._checkpoints_count: int = 0
        self._started: bool = False
        self._uses_save_func = callable(save)
        self._consumed: bool = False
        self._item_size_mb: float | None = None

        # Async I/O infrastructure
        self._io_queue: queue.Queue | None = None
        self._io_thread: threading.Thread | None = None
        self._io_error: Exception | None = None
        self._shutdown_event: threading.Event | None = None
        self._saved_count_lock = threading.Lock()  # Protects _saved_count in async mode

        # Store-only (replay) mode
        if self.source is None and self.store is None:
            raise StreamError("Either `source` or `store` must be provided.")

        if self.source is None and self.store is not None:
            # Store-only mode is intended for replay
            if not self.auto_replay:
                logger.warning(
                    "Creating stream with `store` but no `source`. forcing `auto_replay=True`."
                )
                self.auto_replay = True
            # Mark as already consumed since there's no items to iterate
            self._consumed = True

        if self.background_io:
            if max_queue_size <= 0:
                logger.warning(
                    "Unbounded queue configured (max_queue_size <= 0). "
                    "This can lead to memory exhaustion."
                )
            self._start_io_worker()

        atexit.register(self.close)  # Ensure resources cleaned up on exit

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
                        if item is STOP_IO:
                            logger.debug("I/O worker received shutdown signal")
                            self._io_queue.task_done()
                            break

                        # Process batch
                        batch: list[SerializableT] = item
                        logger.debug(f"I/O worker writing batch of {len(batch)} items")

                        try:
                            count = 0
                            for item in batch:
                                store.append(item, validator=self.validator, meta=self.meta)
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
        """Checks if I/O thread encountered an error and raises it."""
        with self._saved_count_lock:
            if self._io_error is not None:
                raise StreamError(
                    f"Background I/O thread failed: {self._io_error}"
                ) from self._io_error

    def _wait_for_queue(self) -> None:
        """Waits for all pending I/O operations to complete."""
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
        """Stops the I/O worker thread gracefully."""
        if not self.background_io or self._io_thread is None:
            return

        logger.debug("Stopping I/O worker thread...")

        if self._io_queue is None or self._shutdown_event is None:
            logger.error("I/O infrastructure not properly initialized")
            return

        # Signal shutdown
        self._shutdown_event.set()
        self._io_queue.put(STOP_IO)
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

    def __iter__(self) -> typing.Iterator[SerializableT]:
        """
        Iterates over items, optionally persisting as we go.

        Yields items one at a time, accumulating in a batch buffer.
        When batch is full, flush to store and clear buffer.

        Memory pattern:
            1. Yield item to user -> User processes it
            2. Add to batch buffer (small memory cost)
            3. When batch full -> Flush to disk, clear buffer
            4. Net effect: Only `batch_size` items stay in memory at once

        Note: If the underlying iterable is a generator, it can only be consumed once.
        After the first iteration:
            - If `auto_replay=True` and a store exists, automatically replays from store
            - If `auto_replay=False`, raises `StreamError` (use `replay()` explicitly)
            - If no store exists, raises `StreamError` (create fresh stream)

        :return: Iterator over `item_type` instances
        :raises `StreamError`: If trying to iterate again after exhaustion (when `auto_replay=False` or no store)
        """
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
                raise StreamError("Stream already consumed and no store available for replay.")

        # No items provided, this shouldn't happen as `_consumed` should already be set to false
        # but still handle it
        if self.source is None:
            raise StreamError("No items provided and stream not consumed.")

        if self.store is None:
            logger.info("No store provided, streaming without persistence")
            try:
                for item in self.source:
                    self._yield_count += 1
                    yield item
            finally:
                close_iterator(self.source)
            self._consumed = True
            return

        io_mode = "async" if self.background_io else "sync"
        logger.debug(f"Streaming -> {self.store} ({io_mode}, batch_size={self.batch_size})")

        try:
            for item in self.source:
                self._yield_count += 1

                # Surface any background I/O errors before continuing/yielding
                if self.background_io:
                    self._check_io_error()

                yield item

                if self.should_save(item=item):
                    self._batch.append(item)

                    if self.should_flush():
                        self.flush(block=False)

                    if self.should_checkpoint(item=item):
                        self.save_checkpoint(item=item)
        finally:
            close_iterator(self.source)

        # Flush whatever is left
        if self._batch and self.auto_save:
            logger.debug(f"Flushing final batch of {len(self._batch)} items")
            self.flush(block=False)

        # Mark the stream as consumed
        self._consumed = True
        logger.debug(f"Stream exhausted: {self._yield_count} yielded, {self._saved_count} saved")

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
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: typing.Any | None,
    ) -> None:
        """
        Context manager exit. Flushes any remaining items and ensure async I/O completion.

        Ensures all items saved even if iteration interrupted.
        For async I/O, waits for background thread to finish all pending writes.

        :param exc_type: Exception type if error occurred
        :param exc_val: Exception value if error occurred
        :param exc_tb: Exception traceback if error occurred
        """
        if self._batch and self.auto_save and self.store is not None:
            logger.warning(f"Flushing {len(self._batch)} unsaved items on exit")
            try:
                self.flush(block=False)  # Enqueue for I/O worker
            except Exception as exc:
                logger.error(f"Failed to flush items on exit: {exc}", exc_info=True)

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

        # Close the underlying items iterable if it has not already been closed
        if not self._consumed and self.source is not None:
            close_iterator(self.source)

        if exc_type is None:
            logger.info(
                f"Stream complete: {self._saved_count} items saved, "
                f"{self._checkpoints_count} checkpoints created."
            )
        else:
            logger.error(
                f"Stream interrupted after {self._saved_count} items have been saved: {exc_val}"
            )

    def close(self) -> None:
        """
        Manually close the stream, flushing any remaining items and stopping background I/O.

        This is useful if not using a context manager. After calling `close()`, the stream is exhausted.
        Calling `close()` again has no effect.
        """
        if self._consumed:
            logger.debug("Stream already consumed/closed")
            return

        logger.debug("Manually closing stream...")
        self.__exit__(None, None, None)

    def last(self) -> SerializableT | None:
        """
        Gets the last item from the stream.

        Iterates through the entire stream and returns the final item.
        Useful for quickly accessing only the end result of a simulation.

        :return: The last `item_type` instance, or None if the stream is empty
        """
        logger.debug("Retrieving last item from stream")
        if self._consumed and self.store is not None:
            max_idx = self.store.max_index()
            if max_idx is None:
                return None
            results = list(self.store.load(self.item_type, indices=[max_idx]))
            return results[0] if results else None

        last_item: SerializableT | None = None
        for item in self:
            last_item = item

        if last_item is not None:
            logger.debug(f"Last item retrieved (yield_count={self._yield_count})")
        else:
            logger.debug("Stream is empty, no last item available")
        return last_item

    def drain(self) -> None:
        """
        Returns drain/exhaust the entire stream without yielding items.

        This method iterates through all items, triggering any configured side effects
        (persistence, checkpointing, validation) without returning items. Useful when
        you only want the side effects (saving to store, creating checkpoints) without
        processing individual items.

        The stream's internal mechanisms (__iter__, batching, flushing, checkpointing)
        still occur normally, only the yielding to caller is skipped.

        Example:
        ```python
        # Just save all items to disk without processing them
        stream = DataStream(ItemType, source=produce(), store=store)
        stream.drain()  # Items saved, nothing returned

        # Create checkpoints without holding items in memory
        stream = DataStream(
            ItemType,
            source=produce(),
            checkpoint_interval=100,
            checkpoint_store=HDF5Store("./checkpoints.h5"),
        )
        stream.drain()  # Checkpoints created, stream exhausted
        ```

        Note: After calling `drain()`, the stream is exhausted. Calling it again has no effect.
        """
        if self._consumed:
            logger.debug("Stream already consumed")
            return

        logger.debug("Consuming stream (no yield to caller)")
        for _ in self:
            pass  # Iterate through, triggering side effects but not yielding
        logger.debug(f"Stream consumed: {self._yield_count} items processed")

    def replay(
        self,
        indices: typing.Sequence[int] | None = None,
        predicate: typing.Callable[[EntryMeta], bool] | None = None,
        validator: typing.Callable[[SerializableT], SerializableT] | None = None,
    ) -> typing.Iterator[SerializableT]:
        """
        Loads and iterate over previously saved items from the store.

        All filtering happens before any array data is deserialised, so skipped
        entries have no I/O cost. When both `indices` and `predicate` are
        supplied, `indices` takes priority and `predicate` is ignored.

        Note: each call to `replay(...)` continues to increment `yield_count`.
        Replaying 100 items after streaming 100 items gives `yield_count == 200`.

        :param indices: Load only the entries at these zero-based insertion-order
            positions. When given, `predicate` is ignored.
        :param predicate: `(EntryMeta) -> bool` filter evaluated against stored
            entry metadata - `entry.idx` (store-assigned insertion position) is
            always available regardless of what `meta` records; `entry.meta`
            holds whatever your `meta` callable recorded at append time, if
            anything (e.g. `predicate=lambda e: e.meta.get("step") == 500` for
            an item type that records a `"step"` key).
        :param validator: Optional post-load callable applied to each deserialised
            item before it is yielded. Defaults to `self.validator` when not given.
        :return: Iterator over `item_type` instances matching the filter criteria,
            in insertion order.
        :raises StreamError: If no store was provided at construction time.
            Or if any storage-related error occurs.
        """
        if self.store is None:
            raise StreamError("Cannot replay: no store provided")

        logger.debug(f"Replaying from {self.store}")

        items = None
        try:
            items = self.store.load(
                self.item_type,
                indices=indices,
                predicate=predicate,
                validator=validator or self.validator,
            )
            for item in items:
                self._yield_count += 1
                yield item
        except StorageError as exc:
            raise StreamError(f"An error occured while replaying stream: {exc}") from exc
        finally:
            if items is not None:
                close_iterator(items)

        logger.debug(f"Replay complete: {self._yield_count} total yielded")

    def until(
        self, condition: typing.Callable[[SerializableT], bool]
    ) -> typing.Iterator[SerializableT]:
        """
        Iterates over items, optionally persisting as we go, until `condition` evaluates to True.

        This is a wrapper around `__iter__`, allowing you to stream items until a stop criteria is met

        :param condition: Iteration stop condition/criteria.
        :return: Iterator over `item_type` instances
        :raises `StreamError`: If trying to iterate again after exhaustion (when auto_replay=False or no store)
        """
        for item in self:
            if condition(item):
                yield item
                break
            yield item

    def while_(
        self, condition: typing.Callable[[SerializableT], bool]
    ) -> typing.Iterator[SerializableT]:
        """
        Iterates over items, optionally persisting as we go, as long `condition` evaluates to True.
        Stop when `condition` evaluates to False.

        This is a wrapper around `__iter__`, allowing you to stream items
        based on a criteria until it can no longer be met

        :param condition: Iteration continuation condition/criteria.
        :return: Iterator over `item_type` instances
        :raises `StreamError`: If trying to iterate again after exhaustion (when auto_replay=False or no store)
        """
        for item in self:
            if not condition(item):
                yield item
                break
            yield item

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
            logger.debug(f"Enqueuing batch of {batch_size} items to I/O thread (block={block})")

            # Check for errors before enqueuing
            self._check_io_error()
            if self._io_queue is None or self.store is None or self._shutdown_event is None:
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

            except queue.Full as exc:
                logger.error(f"I/O queue full ({self.max_queue_size}). Cannot buffer more items.")
                raise StreamError(
                    "I/O queue exhausted. Simulation running faster than disk writes. "
                    "Increase `max_queue_size` or reduce `batch_size`."
                ) from exc
        else:
            logger.debug(f"Flushing batch of {batch_size} items to {self.store}")

            try:
                with self.store(mode="a") as store:
                    for item in self._batch:
                        store.append(
                            item,
                            validator=self.validator,
                            meta=self.meta,
                        )

                self._saved_count += batch_size
                logger.debug(f"Flushed {batch_size} items (total saved: {self._saved_count})")
            except Exception as exc:
                logger.error(f"Failed to flush batch: {exc}")
                raise
            finally:
                # Reassign to new list to free memory immediately
                self._batch = []

    def get_pending_batch(self) -> list[SerializableT]:
        """
        Gets a copy of items in the current batch (not yet flushed to store).

        Useful for:
        - Inspecting what will be saved on next flush
        - Recovering items if an error occurs before flush
        - Debugging batch accumulation behavior

        :return: Copy of the current batch buffer (safe to modify without affecting stream)
        """
        return self._batch.copy()

    def _estimate_item_size(self, item: SerializableT) -> float:
        """
        Estimates memory footprint of a single item in MB.

        :param item: Item to measure
        :return: Estimated size in MB
        """
        if self._item_size_mb is not None:
            return self._item_size_mb

        size_bytes = 0

        for attr_name in dir(item):
            if attr_name.startswith("_"):
                continue

            try:
                attr = getattr(item, attr_name)
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

        self._item_size_mb = size_bytes / 1024 / 1024
        logger.debug(f"Estimated item size: {self._item_size_mb:.2f} MB")
        return self._item_size_mb

    def should_save(self, item: SerializableT) -> bool:
        """
        Determines if item should be saved based on `save`.

        :param item: Item to evaluate
        :return: True if item should be saved, False otherwise
        """
        if not self._uses_save_func:
            return self.save  # type: ignore[return-value]  # ty:ignore[invalid-return-type]
        return self.save(item)  # type: ignore[call-arg]  # ty:ignore[call-non-callable]

    def should_flush(self) -> bool:
        """
        Determines if batch should be flushed based on batch size and memory limits.

        :return: True if batch should be flushed, False otherwise
        """
        if len(self._batch) >= self.batch_size:
            return True

        if self.max_batch_memory_usage is not None and self._batch:
            item_size = self._estimate_item_size(self._batch[0])
            batch_memory_usage = item_size * len(self._batch)
            if batch_memory_usage > self.max_batch_memory_usage:
                logger.warning(
                    f"Batch memory limit reached ({batch_memory_usage:.1f} MB > "
                    f"{self.max_batch_memory_usage:.1f} MB) with {len(self._batch)} items - "
                    f"flushing early"
                )
                return True

        return False

    def should_checkpoint(self, item: SerializableT) -> bool:
        """
        Determines if a checkpoint should be created after yielding `item`.

        Cadence is based on `self._yield_count` (how many items this stream
        has yielded so far), not any attribute of `item` itself - unlike
        `save`/`meta`, which do inspect the item, checkpoint *cadence* has no
        domain-specific equivalent to fall back on generically.

        :param item: Current item (unused directly; kept for a consistent
            call signature with `should_save`/`save_checkpoint`).
        :return: True if checkpoint should be created, False otherwise
        """
        return (
            self.checkpoint_interval is not None
            and self.checkpoint_store is not None
            and self._yield_count > 0
            and self._yield_count % self.checkpoint_interval == 0
        )

    def save_checkpoint(self, item: SerializableT) -> None:
        """
        Saves a checkpoint for crash recovery.

        Creates an entry in `checkpoint_store` that can be used to resume
        from this point later, via `checkpoint(index=...)`/`checkpoints()`.

        :param item: Item to checkpoint.
        """
        if self.checkpoint_store is None:
            return
        try:
            entry = self.checkpoint_store.append(item, validator=self.validator, meta=self.meta)
            self._checkpoints_count += 1
            logger.info(
                f"Checkpoint saved (checkpoint index={entry.idx}, yield_count={self._yield_count})"
            )
        except Exception as exc:
            logger.error(
                f"Failed to save checkpoint at yield_count={self._yield_count}: {exc}",
                exc_info=True,
            )

    def checkpoint(
        self,
        index: int | None = None,
        predicate: typing.Callable[[EntryMeta], bool] | None = None,
    ) -> SerializableT:
        """
        Loads a specific checkpoint.

        Exactly one of `index`/`predicate` must be given. `index` is
        `checkpoint_store`'s own insertion-order position among checkpoints
        (the *n*-th checkpoint ever saved, 0-based) - always available
        regardless of what `meta` records. `predicate` filters on whatever
        `meta` recorded, if anything (e.g.
        `predicate=lambda e: e.meta.get("step") == 500` for an item type
        that records a `"step"` key).

        :param index: Zero-based position among saved checkpoints.
        :param predicate: `(EntryMeta) -> bool` filter, evaluated against
            `checkpoint_store`'s stored metadata.
        :return: Loaded `item_type` instance.
        :raises `StreamError`: If checkpointing not configured, if neither or
            both of `index`/`predicate` are given, or if no matching
            checkpoint exists.
        """
        if self.checkpoint_store is None:
            raise StreamError("Checkpointing not configured. No `checkpoint_store` found.")
        if (index is None) == (predicate is None):
            raise StreamError("Exactly one of `index` or `predicate` must be given.")

        if index is not None:
            results = list(self.checkpoint_store.load(self.item_type, indices=[index]))
        else:
            results = list(self.checkpoint_store.load(self.item_type, predicate=predicate))
        if not results:
            raise StreamError(f"No checkpoint found for index={index!r} predicate={predicate!r}.")
        return results[0]

    def checkpoints(self) -> typing.Generator[SerializableT, None, None]:
        """
        Yields all checkpointed items in insertion order.

        :return: Generator yielding checkpointed items.
        :raises `StreamError`: If checkpointing not configured
        """
        if self.checkpoint_store is None:
            raise StreamError("No `checkpoint_store` configured.")
        yield from self.checkpoint_store.load(self.item_type)

    def list_checkpoints(self) -> list[EntryMeta]:
        """
        Lists metadata for all available checkpoints, in insertion order.

        Each `EntryMeta.idx` is usable directly as `checkpoint(index=...)`;
        `EntryMeta.meta` holds whatever `meta` recorded for that item, if
        anything.

        :return: List of `EntryMeta`, one per saved checkpoint.
        :raises `StreamError`: If checkpointing not configured
        """
        if self.checkpoint_store is None:
            raise StreamError("Checkpointing not configured. No `checkpoint_store` found")
        return list(self.checkpoint_store.entries())

    def progress(self) -> StreamProgress:
        """
        Gets streaming progress statistics.

        :return: Dictionary with progress metrics including:
            - yield_count: Total items yielded
            - saved_count: Total items saved to store
            - checkpoints_count: Total checkpoints created
            - batch_pending: States in current batch (not yet saved)
            - store_backend: Type of store being used (or None)
            - memory_usage: Estimated batch memory in MB
            - io_queue_size: Size of async I/O queue (if background_io enabled)
            - io_thread_alive: Whether I/O thread is running (if background_io enabled)
        """
        if self._batch and self._item_size_mb is not None:
            batch_memory_usage = self._item_size_mb * len(self._batch)
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
            progress["io_queue_size"] = self._io_queue.qsize()  # type: ignore[typeddict-unknown-key]  # ty:ignore[invalid-key]
            progress["io_thread_alive"] = (  # type: ignore[typeddict-unknown-key]  # ty:ignore[invalid-key]
                self._io_thread.is_alive() if self._io_thread else False
            )
        return progress

    @property
    def yield_count(self) -> int:
        """
        Returns the total number of items yielded (including replays).

        :return: Count of yielded items
        """
        return self._yield_count

    @property
    def is_consumed(self) -> bool:
        """
        Checks if the underlying iterable has been exhausted.

        Once consumed, the stream cannot be iterated again unless:
        1. A store was provided and contains data (will auto-replay)
        2. A fresh stream is created with a new iterable

        :return: True if the iterable has been consumed, False otherwise
        """
        return self._consumed

    @property
    def saved_count(self) -> int:
        """
        Returns the number of items saved to store so far.

        :return: Count of saved items
        """
        return self._saved_count

    @property
    def checkpoints_count(self) -> int:
        """
        Returns the number of checkpoints created so far.

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
