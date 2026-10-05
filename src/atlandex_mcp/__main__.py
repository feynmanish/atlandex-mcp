"""Entry point: `atlandex-mcp` or `python -m atlandex_mcp` serves over stdio."""

from .server import create_server


def main() -> None:
    create_server().run()


if __name__ == "__main__":
    main()
