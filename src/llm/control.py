"""Cancellation shared by model streams, application service and UI workers."""

from threading import Event, Lock, Thread

from src.llm.provider import ModelError


class GenerationCancelled(ModelError):
    def __init__(self):
        super().__init__("generation_cancelled", "生成已停止，未完成回答不会写入成功历史")


class CancellationToken:
    def __init__(self):
        self.event, self.lock, self.callbacks = Event(), Lock(), {}

    def check(self):
        if self.event.is_set():
            raise GenerationCancelled()

    def register(self, callback):
        key = object()
        with self.lock:
            self.callbacks[key] = callback
            already_cancelled = self.event.is_set()
        if already_cancelled:
            self._close_async(callback)
        return key

    def unregister(self, key):
        with self.lock:
            self.callbacks.pop(key, None)

    @staticmethod
    def _close_async(callback):
        def close():
            try:
                callback()
            except Exception:
                pass  # Do not surface transport internals/credentials during cancellation.
        Thread(target=close, daemon=True).start()

    def cancel(self):
        with self.lock:
            self.event.set()
            callbacks = list(self.callbacks.values())
        for callback in callbacks:
            self._close_async(callback)
