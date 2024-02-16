from pathlib import Path

import gradio as gr

from eternitylib.board import Board
from solverlib.solver import Solver


def board_to_image(text, file) -> (str, Path):
    if not text and not file:
        raise ValueError("No input provided")

    if text and file:
        raise ValueError("Both text and file provided, please provide only one")

    board = Board()

    if text:
        board.read_csv(text)

    if file:
        board.read_file(Path(file))

    return f"Board: {board.size}x{board.size}, {board.pattern_count} patterns", board.image


def generate_board(size: int, pattern_count: int) -> (str, Path, Path):
    board = Board()
    board.generate(size, pattern_count)
    unsorted_board = board.image
    board.shuffle()
    return board.to_csv(), unsorted_board, board.image


image_gen = gr.Interface(
    fn=board_to_image,
    inputs=["text", "file"],
    outputs=["text", "image"]
)

board_gen = gr.Interface(
    fn=generate_board,
    inputs=[
        gr.Slider(2, 32, 12, step=1, label="Puzzle Size"),
        gr.Slider(1, 128, 22, step=1, label="Pattern count"),
        # gr.Dataset(components=["slider"], samples=[["4x4 (2)", "8x8 (4)", "14x14 (22)", "15x15 (22)", "16x16 (22)", "Eternity II (16x16 22)"]])
    ],
    outputs=[gr.Textbox(label="Board Data", show_copy_button=True), "image", "image"]
)

interface = gr.TabbedInterface([board_gen, image_gen], ["Generate Board", "Generate Image"])

if __name__ == "__main__":
    solver = Solver()
    solver.link_solver()
    # interface.launch(share=True)