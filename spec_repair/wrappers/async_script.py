import threading
import subprocess
import queue


class AsyncScript:
    def __init__(self, args):
        self.args = args
        self.process = None
        self.running = False
        self.stdout_queue = queue.Queue()
        self.stderr_queue = queue.Queue()
        self.stdin_queue = queue.Queue()

    def _read_stdout(self):
        while self.running:
            line = self.process.stdout.readline()
            if line:
                self.stdout_queue.put(line)

    def _read_stderr(self):
        while self.running:
            line = self.process.stderr.readline()
            if line:
                self.stderr_queue.put(line)

    def _write_stdin(self):
        while self.running:
            input_str = self.stdin_queue.get()
            self.process.stdin.write(f"{input_str}\n")
            self.process.stdin.flush()

    def start(self):
        self.process = subprocess.Popen(
            self.args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            universal_newlines=True
        )
        self.running = True
        threading.Thread(target=self._read_stdout).start()
        threading.Thread(target=self._read_stderr).start()
        threading.Thread(target=self._write_stdin).start()

    def read_next_line(self):
        if not self.running:
            raise RuntimeError("Script not started yet")
        return self.stdout_queue.get()

    def write_line(self, input_str):
        if not self.running:
            raise RuntimeError("Script not started yet")
        self.stdin_queue.put(input_str)

    def end(self):
        if not self.running:
            raise RuntimeError("Script not started yet")
        self.running = False
        self.process.kill()
        self.process.wait()

    def is_running(self):
        return self.running
