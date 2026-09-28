from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.pydantic_models.source import Source

console = Console()


def format_sources(sources: list[Source]) -> None:
    if not sources:
        console.print(
            Panel(
                "No sources found.",
                title="Sources",
                border_style="yellow",
            )
        )
        return

    table = Table(
        title=f"Sources ({len(sources)})",
        show_lines=True,
    )

    table.add_column("#", justify="right", width=4)
    table.add_column("Title", ratio=3)
    table.add_column("Authors", ratio=2)
    table.add_column("arXiv ID", ratio=1)
    table.add_column("URL", ratio=2)

    for index, source in enumerate(sources, start=1):
        table.add_row(
            str(index),
            source.title,
            ", ".join(source.authors),
            source.arxiv_id,
            str(source.url),
        )

    console.print(table)


def format_metadata(message: Any) -> None:
    metadata = message.response_metadata

    if not metadata:
        return

    table = Table(show_header=False, box=None)

    table.add_column("Field", style="bold")
    table.add_column("Value")

    fields = {
        "Model": metadata.get("model_name"),
        "Provider": metadata.get("model_provider"),
        "Finish reason": metadata.get("finish_reason"),
        "Cost": metadata.get("cost"),
    }

    for key, value in fields.items():
        if value is not None:
            table.add_row(key, str(value))

    if table.row_count:
        console.print(
            Panel(
                table,
                title="Response Metadata",
                border_style="dim",
            )
        )


def format_tool_calls(message: Any) -> None:
    if not message.tool_calls:
        return

    table = Table(
        title="Tool Calls",
        show_lines=True,
    )

    table.add_column("Tool")
    table.add_column("Arguments")
    table.add_column("ID")

    for tool_call in message.tool_calls:
        table.add_row(
            tool_call["name"],
            str(tool_call["args"]),
            tool_call["id"],
        )

    console.print(table)


def format_message(index: int, message: Any) -> None:
    message_type = message.type

    if message_type == "human":
        title = f"HumanMessage #{index}"
        content = str(message.content)

    elif message_type == "ai":
        model = message.response_metadata.get("model_name", "unknown")
        finish_reason = message.response_metadata.get(
            "finish_reason",
            "unknown",
        )

        title = f"AIMessage #{index} | {model} | {finish_reason}"

        content = message.content or "No content"

    elif message_type == "tool":
        title = f"ToolMessage | {message.name or 'tool'}"
        content = str(message.content)

    else:
        title = f"{message_type.capitalize()} #{index}"
        content = str(message.content)

    console.print(
        Panel(
            Markdown(content) if message_type == "ai" else content,
            title=title,
            border_style="dim",
        )
    )

    if message_type == "ai":
        format_tool_calls(message)
        format_metadata(message)


def format_state(state: dict[str, Any]) -> None:
    console.rule("Research Copilot")

    question = state.get("question")

    if question:
        console.print(
            Panel(
                question,
                title="Question",
            )
        )

    messages = state.get("messages", [])

    for index, message in enumerate(messages, start=1):
        format_message(index, message)

    sources = state.get("sources", [])
    format_sources(sources)

    answer = state.get("answer")

    if answer:
        console.print(
            Panel(
                Markdown(answer),
                title="Final Answer",
                border_style="green",
            )
        )


def pretty_print_state(state: dict[str, Any]) -> None:
    format_state(state)
