from w5.tools import caculation, format_date, format_text, read_csv

class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name: str, func):
        """Register a tool by name."""
        self._tools[name] = func

    def list_tools(self) -> list:
        """Return list of available tool names."""
        return list(self._tools.keys())

    def call(self, name: str, inputs: dict) -> dict:
        """Call a tool by name with inputs."""
        if name not in self._tools:
            return {"success": False, "result": None, "error": f"Unknown tool: {name}"}
        return self._tools[name](inputs)

# registry = ToolRegistry()
# registry.register("calculation", caculation)
# registry.register("date", format_date)
# registry.register("text", format_text)
# registry.register("csv", read_csv)


# input_data = input(f"Enter which tool to use {registry.list_tools()}: ")

# if input_data == "calculation":
#   input_expr = input("Enter a mathematical expression to evaluate: ")
#   print(registry.call("calculation", {"expression": input_expr}))
# elif input_data == "date":
#   input_format = input("Enter a date format (YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY, day, timestamp): ")
#   print(registry.call("date", {"format": input_format}))
# elif input_data == "text":
#   input_text = input("Enter some text: ")
#   input_format = input("enter a text format (uppercase, lowercase, titlecase, reversed): ")
#   print(registry.call("text", {"text": input_text, "case": input_format}))
# elif input_data == "csv":
#   input_file_path = input("Enter the path to the CSV file: ")
#   print(registry.call("csv", {"file_path": input_file_path}))
# else:
#   res = {
#     "success": False,
#     "result": None,
#     "error": "Invalid tool selection. Please choose from calculation, date, text, or csv."
#   }
    
    
# === REFLECTION ===
# What does Pydantic handle vs ast?
# - Pydantic: wrong types, empty strings, invalid Literal values
# - ast: dangerous code injection, invalid math syntax, division by zero
# - They complement each other — Pydantic is the door, ast is the lock

# What would break these tools?
# - calculator: valid syntax but semantic error e.g. "1/0" → handled by try/except
# - csv_loader: file exists but wrong encoding → caught by Exception
# - date: timestamp format "%s" not supported on Windows → worth noting

# Why ToolRegistry matters?
# - Agent can call registry.list_tools() to know what's available
# - Single call interface: registry.call(name, inputs) for any tool
# - Easy to add new tools without changing any other code