from canar.app.bot_tools.models import ToolDefinition
from canar.app.bot_tools.openai_adapter import (
    tool_definition_to_openai,
    tools_to_openai_schemas,
)


def test_tool_definition_to_openai_nominal():
    tool = ToolDefinition(
        provider_id="echo",
        name="echo__echo",
        description="Renvoie le texte reçu.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    schema = tool_definition_to_openai(tool)

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "echo__echo"
    assert schema["function"]["description"] == "Renvoie le texte reçu."
    assert schema["function"]["parameters"]["type"] == "object"
    assert "text" in schema["function"]["parameters"]["properties"]


def test_tool_definition_to_openai_empty_description_and_schema():
    tool = ToolDefinition(
        provider_id="local",
        name="local__custom",
        description="",
        input_schema={},
    )

    schema = tool_definition_to_openai(tool)

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "local__custom"
    assert schema["function"]["description"] == ""
    assert schema["function"]["parameters"] == {"type": "object", "properties": {}}


def test_tools_to_openai_schemas_list():
    tools = [
        ToolDefinition(
            provider_id="p1",
            name="p1__t1",
            description="Tool 1",
            input_schema={"type": "object"},
        ),
        ToolDefinition(
            provider_id="p2",
            name="p2__t2",
            description="Tool 2",
            input_schema={"type": "object"},
        ),
    ]

    schemas = tools_to_openai_schemas(tools)

    assert len(schemas) == 2
    assert schemas[0]["function"]["name"] == "p1__t1"
    assert schemas[1]["function"]["name"] == "p2__t2"


def test_tools_to_openai_schemas_empty_list():
    schemas = tools_to_openai_schemas([])
    assert schemas == []
