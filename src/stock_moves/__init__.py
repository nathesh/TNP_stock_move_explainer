"""Stock move explainer: detect major daily moves and explain them with news."""


def main() -> None:
    """Run the API server (entry point for `uv run stock-moves`)."""
    import uvicorn

    uvicorn.run("stock_moves.api.app:app", host="127.0.0.1", port=8000, reload=False)
