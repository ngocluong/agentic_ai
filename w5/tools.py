from datetime import datetime
import csv
from typing import Literal
from pydantic import BaseModel, Field, ValidationError
import ast
import operator as op

# Supported operators
operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod
}

date_formats = {
    "YYYY-MM-DD": "%Y-%m-%d",
    "MM/DD/YYYY": "%m/%d/%Y",
    "DD-MM-YYYY": "%d-%m-%Y",
    "day": "%A",
    "timestamp": "%s"
}

class CalculatorInput(BaseModel):
    """Input model for the calculator."""
    expression: str = Field(min_length=1, description="Math expression to evaluate")

def eval_node(node):
    """
    Recursively evaluate an AST node.
    
    Args:
        node (ast.AST): The AST node to evaluate.
        
    Returns:
        float: The result of the evaluated node.
        
    Raises:
        ValueError: If the node type is unsupported.
    """
    if isinstance(node, ast.Constant):  # <number>
        return node.value
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        left = eval_node(node.left)
        right = eval_node(node.right)
        operator_func = operators.get(type(node.op))
        if operator_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return operator_func(left, right)
    else:
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")
    
def caculation(expr: dict) -> dict:
    try:
        validated_input = CalculatorInput(**expr)
    except ValidationError as e:
        return {"success": False, "result": None, "error": str(e)}
    try:
        node = ast.parse(validated_input.expression, mode='eval').body
        result = eval_node(node)
        return {"success": True, "result": result, "error": None}
    except Exception as e:
        return {"success": False, "result": None, "error": str(e)}
    
class DateInput(BaseModel):
    """Input model for the date formatter."""
    format: Literal["YYYY-MM-DD", "MM/DD/YYYY", "DD-MM-YYYY", "day", "timestamp"] = Field(
        description="Desired date format"
    )

def format_date(date_format):
    """
    Format the current date based on the specified format.
    Args:
        date_format (str): The desired date format.
    Returns:
        str: The current date formatted according to the specified format.
    """
    try:
        validated_input = DateInput(**date_format)
        return {"success": True, "result": datetime.now().strftime(date_formats.get(validated_input.format, validated_input.format)), "error": None}
    except ValidationError as e:
        return {"success": False, "result": None, "error": str(e)}

class TextFormatInput(BaseModel):
    """Input model for the text formatter."""
    text: str = Field(min_length=1, description="Text to format")
    case: Literal["uppercase", "lowercase", "titlecase", "reversed"] = Field(description="Desired case format")
    
def format_text(text_format):
    """
    Format the input text based on the specified case.
    
    Args:
        text_format (dict): A dictionary containing 'text' and 'case'.
        
    Returns:
        str: The formatted text.
        
    Raises:
        ValueError: If the input is invalid or the case is unsupported.
    """
    try:
        validated_input = TextFormatInput(**text_format)
    except ValidationError as e:
        return {"success": False, "result": None, "error": str(e)}

    if validated_input.case == "uppercase":
        return {"success": True, "result": validated_input.text.upper(), "error": None}
    elif validated_input.case == "lowercase":
        return {"success": True, "result": validated_input.text.lower(), "error": None}
    elif validated_input.case == "titlecase":
        return {"success": True, "result": validated_input.text.title(), "error": None}
    elif validated_input.case == "reversed":
        return {"success": True, "result": validated_input.text[::-1], "error": None}
    else:
        return {"success": False, "result": None, "error": f"Unsupported case format: {validated_input.case}"}
    
class CSVInput(BaseModel):
    """Input model for the CSV formatter."""
    file_path: str = Field(min_length=1, description="Path to the CSV file")
    
def read_csv(file_path):
    """
    Read a CSV file and return its content as a list of dictionaries.
    
    Args:
        file_path (str): The path to the CSV file.
    Returns:
        list: A list of dictionaries representing the CSV rows.
    """
    try:
        validated_input = CSVInput(**file_path)
    except ValidationError as e:
        return {"success": False, "result": None, "error": str(e)}

    try:
        with open(validated_input.file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = [row for row in reader]
            return {
                "success": True,
                "result": {
                    "row_count": len(rows),
                    "columns": list(rows[0].keys()) if rows else [],
                    "preview": rows[:3]
                },
                "error": None
            }
    except Exception as e:
        return {"success": False, "result": None, "error": f"Error reading CSV file at {validated_input.file_path}: {e}"}