import subprocess


class Script:
    def __init__(self, args: list[str]):
        self.process: subprocess.Popen
        self._args = args

    def start(self):
        self.process = subprocess.Popen(
            self._args,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text="True")

    def read_next_line(self):
        return self.process.stdout.readline()

    def read_next_paragraph(self):
        return self.process.stdout.read()

    def write_line(self, line):
        self.process.stdin.write(f"{line}\n")
        self.process.stdin.flush()

    def end(self):
        self.process.stdin.close()
        self.process.stdout.close()
        self.process.terminate()
