#!/usr/bin/env python

from typing import Optional, List
import traceback
from multiprocessing import Queue, Process, Pipe, cpu_count
from queue import Empty as EmptyQueueError
import logging
import os
from time import sleep

import h5py

from deeprank_gnn.models.query import Query


_log = logging.getLogger(__name__)


class _PreProcess(Process):
    def __init__(
        self,
        input_queue: Queue,
        output_path: str,
        error_queue: Queue,
        feature_modules: List,
    ):
        Process.__init__(self)
        self.daemon = True

        self._input_queue = input_queue
        self._error_queue = error_queue

        self._output_path = output_path

        self._receiver, self._sender = Pipe(duplex=False)

        self._feature_modules = feature_modules

    @property
    def output_path(self) -> str:
        return self._output_path

    def is_running(self) -> bool:
        with self._running_lock:
            return self._running

    def stop(self):
        self._sender.send(1)
        self.join()

    def _should_stop(self) -> bool:
        return self._receiver.poll()

    def run(self):
        while not self._should_stop():
            try:
                query = self._input_queue.get_nowait()

            except EmptyQueueError:
                continue

            graph = None
            try:
                graph = query.build_graph(self._feature_modules)
                if graph.has_nan():
                    self._error_queue.put(
                        f"skipping {query}, because of a generated NaN value in the graph"
                    )

                graph.write_to_hdf5(self._output_path)
            except BaseException:
                self._error_queue.put(traceback.format_exc())

                # Don't leave behind an unfinished hdf5 group.
                if graph is not None:
                    with h5py.File(self._output_path, "a") as f5:
                        if graph.id in f5:
                            del f5[graph.id]


class PreProcessor:
    "preprocesses a series of graph building operations (represented by queries)"

    def __init__(
        self,
        feature_modules: List,
        prefix: Optional[str] = None,
        process_count: Optional[int] = None,
    ):
        """
        Args:
            prefix: prefix for the output files, ./preprocessed-data- by default
            process_count: how many subprocesses will I run simultaneously, by default takes all available cpu cores.
            feature_modules: the feature modules used to generate features, each must implement the add_features function
        """

        self._input_queue = Queue()
        self._error_queue = Queue()

        if process_count is None:
            process_count = cpu_count()

        if prefix is None:
            prefix = "preprocessed-data"

        self._processes = [
            _PreProcess(
                self._input_queue,
                f"{prefix}-{index}.hdf5",
                self._error_queue,
                feature_modules,
            )
            for index in range(process_count)
        ]

    def start(self):
        "start the workers"

        _log.info("starting %d worker processes", len(self._processes))
        for process in self._processes:
            process.start()
            if not process.is_alive():
                raise RuntimeError(f"worker process {process.name} did not start")

    def _queue_empty(self):
        """
        Returns whether queue is empty.

        Note: This implementation is not reliable, especially not on Mac OSX. .qsize() seems to be more reliable than
        .empty(), however qsize relies on sem_getvalue() which is not implemented on Mac OSX.
        """
        try:
            return self._input_queue.qsize() == 0
        except NotImplementedError:
            return self._input_queue.empty()

    def wait(self):
        "wait for all graphs to be built"

        try:
            _log.info("waiting for the queue to be empty..")
            while not self._queue_empty():
                sleep(1.0)
        finally:
            self.shutdown()

        # log all errors in the subprocesses
        while not self._error_queue.empty():
            error_s = self._error_queue.get()
            _log.error(error_s)

    def add_query(self, query: Query):
        "add a single graph building query"

        self._input_queue.put(query)

    def add_queries(self, queries: List[Query]):
        "add multiple graph building queries"

        for query in queries:
            self._input_queue.put(query)

    @property
    def output_paths(self) -> List[str]:
        return [
            process.output_path
            for process in self._processes
            if os.path.isfile(process.output_path)
        ]

    def shutdown(self):
        "stop building graphs"

        _log.info("shutting down %d worker processes..", len(self._processes))

        for process in self._processes:
            process.stop()

    def __del__(self):
        # clean up all the created subprocesses
        self.shutdown()
