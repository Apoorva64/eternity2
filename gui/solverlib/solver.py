import subprocess
import logging
from pathlib import Path

from eternitylib.board import Board
from solverlib.execution import Execution


class Solver:
    def __init__(self):
        # exe is in the same dir as this file
        self.exe_path = Path(__file__).parent / "e2solver"

    def link_solver(self, path: Path = Path(__file__).parent.parent.parent / "cmake-build-debug/solver/Solver"):
        # If file does not exist, raise an error
        if not path.exists():
            raise FileNotFoundError(f"Solver executable not found at {path}")

        if self.exe_path.exists():
            self.exe_path.unlink()

        self.exe_path.symlink_to(path)

    def solve(self, board: Board, timeout: int = 60) -> Execution:
        # Write the board to a file
        board_path = Path(__file__).parent.parent / f"tmp/board.{board.hash()}.txt"
        board_path.parent.mkdir(parents=True, exist_ok=True)
        board_path.write_text(board.to_csv())

        # print info to log
        logging.
        # Run the solver
        try:
            logging.info(f"Running solver with timeout of {timeout} seconds")
            result = subprocess.run([self.exe_path, str(board_path)], capture_output=True, timeout=timeout)
            logging.info("Solver finished.")
        except subprocess.TimeoutExpired:
            logging.warning(f"Solver timed out after {timeout} seconds")
            return Execution("")
        return Execution(result.stdout.decode("utf-8"))
