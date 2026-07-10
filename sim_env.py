import os
import socket
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
SIM_PROJECT = REPO / "simulation"

# Observation columns fed to the network (reward/done are training signals, not inputs).
STATE_DIM = 4


class SimEnv:
    def __init__(self, port=9999, host="127.0.0.1", launch=True, build=True,
                 godot=None, connect_timeout=30.0):
        self.port = port
        self.host = host
        self._proc = None
        self._sock = None
        self._buf = ""

        godot = godot or os.environ.get("GODOT", "godot-mono")

        if launch:
            if build:
                subprocess.run(
                    ["dotnet", "build", str(SIM_PROJECT / "simulation.sln"), "-c", "Debug"],
                    check=True, stdout=subprocess.DEVNULL)
            # headless + --port => blocking lockstep (deterministic, as fast as we step)
            # --fixed-fps disables real-time sync: without it Godot paces physics at
            # wall-clock 60Hz, so training crawls at real-time speed. With it each
            # frame advances one 60Hz physics tick as fast as the CPU can go.
            self._proc = subprocess.Popen(
                [godot, "--headless", "--fixed-fps", "60",
                 "--path", str(SIM_PROJECT), "--", f"--port={port}"])

        self._connect(connect_timeout)

    def _connect(self, timeout):
        deadline = time.time() + timeout
        while True:
            try:
                self._sock = socket.create_connection((self.host, self.port), timeout=5.0)
                self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.2)

    def _read_line(self):
        while "\n" not in self._buf:
            chunk = self._sock.recv(4096).decode("ascii")
            if not chunk:
                raise ConnectionError("sim closed the connection")
            self._buf += chunk
        line, self._buf = self._buf.split("\n", 1)
        return line

    @staticmethod
    def _parse(line):
        cart_v, pole_av, pole_a, cart_p, reward, done = line.split(",")
        state = [float(cart_v), float(pole_av), float(pole_a), float(cart_p)]
        return state, float(reward), int(done)

    def reset(self):
        """Read the initial observation of a fresh episode; return the state (len-4 list)."""
        state, _, _ = self._parse(self._read_line())
        return state

    def step(self, command):
        """Send a motor command in [-1,1], read the resulting obs.

        Returns (next_state, reward, done).
        """
        command = max(-1.0, min(1.0, float(command)))
        self._sock.sendall(f"{command}\n".encode("ascii"))
        return self._parse(self._read_line())

    def request_reset(self):
        """Force an episode reset mid-episode (sim's legacy 'R' command).

        Use when a step cap is hit: the sim reloads the scene and the next
        reset() call reads the fresh episode's first observation.
        """
        self._sock.sendall(b"R\n")

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
