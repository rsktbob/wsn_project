"""Generic persistent-process worker pool shared by every SE market backend.

This module owns exactly one thing: process lifecycle and Pipe request/
response exchange for a fixed number of long-lived worker processes. It has
no idea what a worker computes, or whether it keeps any state across
rounds -- that is entirely up to the ``build_handler`` callable each caller
supplies to :meth:`PersistentWorkerPool.start`.

``ParallelSEMarket`` (SA_SETS and everything built on ``BaseSE``) and
``SI_SETS`` both use this same pool for the actual "open h processes, talk
to them over Pipes" mechanics; they differ only in what handler each worker
runs and how requests get routed to workers, which is exactly the part that
*should* differ between a region-owning worker and a stateless evaluator.
"""

from __future__ import annotations

from multiprocessing import Pipe, Process


def _worker_loop(build_handler, worker_id, seed, connection):
    """Run in the child process for its whole life.

    Builds this worker's handler once (``build_handler`` may seed RNGs,
    open problem-local caches, or close over persistent local state such as
    a region's goods pool), then answers requests until it receives the
    ``None`` stop sentinel.
    """
    handle = build_handler(worker_id, seed)
    while True:
        request = connection.recv()
        if request is None:
            break
        connection.send(handle(request))
    connection.close()


class PersistentWorkerPool:
    """Own ``count`` persistent worker processes, each with its own Pipe.

    ``build_handler(worker_id, seed)`` runs once inside the worker process
    and must return a ``handle(request) -> response`` callable. Because the
    worker process stays alive for the pool's whole lifetime, anything that
    callable closes over persists across calls -- that is what gives a
    worker "memory" between rounds (a region's goods pool, for example).
    A stateless handler (e.g. batch fitness evaluation) simply doesn't use
    that persistence and that is fine too.
    """

    def __init__(self):
        self.processes = []
        self.connections = []

    def start(self, count, build_handler, next_seed):
        for worker_id in range(count):
            parent_connection, child_connection = Pipe()
            process = Process(
                target=_worker_loop,
                args=(build_handler, worker_id, next_seed(), child_connection),
            )
            process.start()
            self.processes.append(process)
            self.connections.append(parent_connection)

    def send(self, worker_id, request):
        self.connections[worker_id].send(request)

    def recv(self, worker_id):
        return self.connections[worker_id].recv()

    def send_all(self, requests):
        """Broadcast one request per worker, in worker-id order."""
        for connection, request in zip(self.connections, requests):
            connection.send(request)

    def recv_all(self):
        """Collect one response per worker, in worker-id order."""
        return [connection.recv() for connection in self.connections]

    def close(self):
        for connection in self.connections:
            try:
                connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process, connection in zip(self.processes, self.connections):
            process.join()
            connection.close()
        self.processes.clear()
        self.connections.clear()


__all__ = ["PersistentWorkerPool"]
