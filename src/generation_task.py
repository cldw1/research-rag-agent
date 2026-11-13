"""Bounded local UI workers. Not a durable queue or a multi-agent framework."""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from src.llm.control import CancellationToken, GenerationCancelled


class GenerationTask:
    def __init__(self, conversation_id, work):
        self.conversation_id, self.work = conversation_id, work
        self.token, self.lock = CancellationToken(), Lock()
        self.status, self.text, self.phase, self.error = "running", "", "准备生成", ""
        self.result = None
        self.future = None

    def update(self, event):
        with self.lock:
            if self.status != "running" or self.token.event.is_set():
                return
            if event["kind"] == "model_start":
                self.text, self.phase = "", "模型正在生成"
            elif event["kind"] == "text_delta":
                self.text += event["text"]
            elif event["kind"] == "tool_start":
                self.text, self.phase = "", "正在查询文献证据"

    def execute(self):
        try:
            result = self.work(self.token, self.update)
            with self.lock:
                self.status, self.result, self.text, self.phase = "completed", result, result["answer"], "已保存回答"
        except GenerationCancelled:
            with self.lock:
                self.status, self.text, self.phase = "cancelled", "", "已停止，未保存部分回答"
        except Exception as exc:
            from src.llm.provider import ModelError
            from src.storage.artifacts import StorageError
            with self.lock:
                self.status, self.text = "failed", ""
                self.error = str(exc) if isinstance(exc, (ModelError, StorageError)) else "生成未完成，请检查模型、网络、预算及索引状态；可使用新请求重试。"
        finally:
            self.work = None  # Release closures holding the per-request provider/key.

    def cancel(self):
        with self.lock:
            if self.status != "running":
                return False
            self.text, self.phase = "", "正在停止"
        self.token.cancel()
        return True

    def snapshot(self):
        with self.lock:
            return {"status": self.status, "text": self.text, "phase": self.phase, "error": self.error}


class GenerationPool:
    def __init__(self, max_workers=2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="research-generation")
        self.lock, self.tasks, self.max_workers = Lock(), [], max_workers

    def start(self, conversation_id, work):
        with self.lock:
            self.tasks = [task for task in self.tasks if not task.future.done()]
            if len(self.tasks) >= self.max_workers:
                from src.llm.provider import ModelError
                raise ModelError("generation_busy", "本地生成任务已达上限，请等待现有任务完成")
            task = GenerationTask(conversation_id, work)
            task.future = self.executor.submit(task.execute)
            self.tasks.append(task)
            return task

    def close(self):
        with self.lock:
            for task in self.tasks:
                task.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
