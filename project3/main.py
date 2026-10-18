import os
import ast
import operator
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent

from ddgs import DDGS

load_dotenv()

# Initialize Gemini
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY missing. Add it to your .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
    temperature=0,
)

# ---------------- CALCULATOR TOOL ----------------

@tool
def calculator(expression: str) -> str:
    """Calculate basic mathematical expressions safely."""

    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)

            if op_type not in allowed_operators:
                raise ValueError("Unsupported operator")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if op_type is ast.Pow and abs(right) > 10:
                raise ValueError("Exponent too large")

            return allowed_operators[op_type](left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)

            if op_type not in allowed_operators:
                raise ValueError("Unsupported operator")

            return allowed_operators[op_type](evaluate(node.operand))

        raise ValueError("Invalid mathematical expression")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(evaluate(tree.body))
    except Exception as error:
        return f"Calculation error: {error}"


# ---------------- WEB SEARCH TOOL ----------------

@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo."""

    try:
        results = DDGS().text(query, max_results=5)

        if not results:
            return "No search results found."

        formatted_results = []

        for index, result in enumerate(results, start=1):
            title = result.get("title", "No title")
            body = result.get("body", "No description")
            url = result.get("href", "")

            formatted_results.append(
                f"{index}. {title}\n"
                f"Description: {body}\n"
                f"URL: {url}"
            )

        return "\n\n".join(formatted_results)

    except Exception as error:
        return f"Web search error: {error}"


# ---------------- LOCAL FILE READER TOOL ----------------

@tool
def read_local_file(filename: str) -> str:
    """Read a text file from the current project directory only."""

    try:
        base_directory = Path.cwd().resolve()
        file_path = (base_directory / filename).resolve()

        if not file_path.is_relative_to(base_directory):
            return "Access denied: files outside the project directory are not allowed."

        if not file_path.is_file():
            return f"File not found: {filename}"

        if file_path.suffix.lower() not in {
            ".txt", ".md", ".py", ".json", ".csv", ".log"
        }:
            return "Unsupported file type. Use a text, Markdown, Python, JSON, CSV, or log file."

        if file_path.stat().st_size > 1_000_000:
            return "File is too large to read (maximum 1 MB)."

        return file_path.read_text(encoding="utf-8", errors="replace")

    except Exception as error:
        return f"File reading error: {error}"


# ---------------- CREATE AI AGENT ----------------

tools = [calculator, web_search, read_local_file]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful multi-tool AI assistant. "
        "Use the calculator for mathematical calculations, "
        "web_search for web research, and read_local_file to read "
        "files in the current project directory. "
        "Use tools when necessary and provide clear, accurate answers. "
        "Never claim you used a tool if you did not."
    ),
)


# ---------------- RUN AGENT ----------------

def main():
    print("=" * 50)
    print("MULTI-TOOL AUTONOMOUS AI AGENT")
    print("=" * 50)
    print("Available tools: Calculator, Web Search, Local File Reader")
    print("Type 'exit' to quit.")
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in {"exit", "quit"}:
                print("Agent: Goodbye!")
                break

            if not user_input:
                continue

            print("\nAgent: Working...\n")

            response = agent.invoke({
                "messages": [
                    {"role": "user", "content": user_input}
                ]
            })

            messages = response.get("messages", [])
            final_answer = None

            for message in reversed(messages):
                if getattr(message, "type", "") == "ai":
                    if getattr(message, "tool_calls", None):
                        continue

                    content = getattr(message, "content", "")

                    if isinstance(content, str):
                        final_answer = content
                    elif isinstance(content, list):
                        final_answer = "\n".join(
                            item.get("text", "")
                            for item in content
                            if isinstance(item, dict)
                        )

                    if final_answer:
                        break

            print(f"Agent: {final_answer or 'No response received.'}\n")

        except KeyboardInterrupt:
            print("\nAgent stopped.")
            break

        except Exception as error:
            print(f"Error: {error}\n")


if __name__ == "__main__":
    main()