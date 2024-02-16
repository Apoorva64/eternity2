import subprocess
from pathlib import Path

from eternitylib.board import Board


class Solver:
    def __init__(self):
        # exe is in the same dir as this file
        self.exe_path = Path(__file__).parent / "e2solver"

    def link_solver(self, exe_path: Path = Path(__file__).parent.parent.parent / "cmake-build-debug/solver/Solver"):
        subprocess.run(["ln", "-s", str(exe_path), str(self.exe_path)])

    def solve(self, board: Board, timeout: int = 60) -> str:
        # Write the board to a file
        board_path = Path(__file__).parent / f"board.{board.hash()}.txt"
        board_path.write_text(board.to_csv())

        # Run the solver
        result = subprocess.run([self.exe_path, str(board_path)], capture_output=True, timeout=timeout)
        return result.stdout.decode("utf-8")