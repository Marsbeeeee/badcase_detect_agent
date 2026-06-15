from prompt_optimizer_agent.json_utils import function_call_wrappers_to_tool_calls, parse_conversation_json


def test_parse_valid_conversation() -> None:
    raw = """
    {
      "system_prompt": "Follow policy.",
      "interactions": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"}
      ]
    }
    """
    result = parse_conversation_json(raw)
    assert result.error is None
    assert result.data is not None
    assert result.data.system_prompt == "Follow policy."


def test_repairs_trailing_commas() -> None:
    raw = """
    {
      "system_prompt": "Follow policy.",
      "interactions": [
        {"role": "user", "content": "Hi"},
      ],
    }
    """
    result = parse_conversation_json(raw)
    assert result.error is None
    assert result.data is not None
    assert "Trailing commas were removed." in result.warnings


def test_validation_error_is_friendly() -> None:
    result = parse_conversation_json('{"system_prompt": "", "interactions": []}')
    assert result.error is not None
    assert "system_prompt" in result.error


def test_normalizes_messages_shape() -> None:
    raw = """
    {
      "messages": [
        {"role": "system", "content": "Follow policy."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"}
      ]
    }
    """
    result = parse_conversation_json(raw)
    assert result.error is None
    assert result.data is not None
    assert result.data.system_prompt == "Follow policy."
    assert len(result.data.interactions) == 2


def test_preserves_assistant_tool_call_messages() -> None:
    raw = """
    {
      "messages": [
        {"role": "system", "content": "Call tools."},
        {"role": "user", "content": "Search promo"},
        {
          "role": "assistant",
          "tool_calls": [
            {
              "type": "function",
              "id": "call_1",
              "function": {
                "name": "search_promotion",
                "arguments": "{\\"query\\": \\"promo\\"}"
              }
            }
          ]
        },
        {"role": "tool", "content": "{\\"result\\": \\"ok\\"}", "tool_call_id": "call_1"}
      ]
    }
    """
    result = parse_conversation_json(raw)
    assert result.error is None
    assert result.data is not None
    assert len(result.data.interactions) == 3
    assert result.data.interactions[1].role == "assistant"
    assert result.data.interactions[1].content == (
        '<function-call>search_promotion:{"query": "promo"}</function-call>'
    )
    assert result.data.interactions[1].tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "search_promotion",
                "arguments": '{"query": "promo"}',
            },
        }
    ]
    assert result.data.interactions[2].tool_call_id == "call_1"


def test_function_call_wrappers_to_tool_calls() -> None:
    tool_calls = function_call_wrappers_to_tool_calls(
        '<function-call>search_promotion:{"query":"promo"}</function-call>',
        id_prefix="call_rerun_8",
    )

    assert tool_calls == [
        {
            "id": "call_rerun_8_1",
            "type": "function",
            "function": {
                "name": "search_promotion",
                "arguments": '{"query":"promo"}',
            },
        }
    ]


def test_normalizes_dialog_shape_with_text() -> None:
    raw = """
    {
      "prompt": "Follow policy.",
      "dialog": [
        {"from": "human", "text": "Hi"},
        {"from": "gpt", "text": "Hello"}
      ]
    }
    """
    result = parse_conversation_json(raw)
    assert result.error is None
    assert result.data is not None
    assert result.data.system_prompt == "Follow policy."
    assert result.data.interactions[0].role == "user"


def test_normalizes_top_level_message_list() -> None:
    raw = """
    [
      {"role": "system", "content": "Follow policy."},
      {"role": "user", "content": "Hi"},
      {"role": "assistant", "content": "Hello"}
    ]
    """
    result = parse_conversation_json(raw)
    assert result.error is None
    assert result.data is not None
    assert result.data.system_prompt == "Follow policy."


def test_normalizes_first_conversation_from_top_level_record_list() -> None:
    raw = """
    [
      {
        "system_prompt": "Follow first policy.",
        "interactions": [
          {"role": "user", "content": "First question"},
          {"role": "assistant", "content": "First answer"}
        ]
      },
      {
        "system_prompt": "Follow second policy.",
        "interactions": [
          {"role": "user", "content": "Second question"},
          {"role": "assistant", "content": "Second answer"}
        ]
      }
    ]
    """

    result = parse_conversation_json(raw)

    assert result.error is None
    assert result.data is not None
    assert result.data.system_prompt == "Follow first policy."
    assert result.data.interactions[0].content == "First question"
    assert "Detected a list of records. Loaded the first record." in result.warnings


def test_normalizes_company_compress_dialog_shape() -> None:
    raw = """
    {
      "type": "compress",
      "dialog": [
        {"turn_index": 0, "role": "system", "content": "Follow company flow."},
        {"turn_index": 1, "role": "user", "content": "Hi"},
        {
          "turn_index": 2,
          "role": "assistant",
          "content": "Hello",
          "evaluate": {
            "H200_01_fc_9010@C0": {
              "meta": {
                "model_info": {"provider": "openai_api_like", "model": "H200_01_fc_9010"},
                "kwargs": {
                  "url": "http://192.168.101.15:9898",
                  "models": ["openai_api_like:H200_01_fc_9010"]
                }
              }
            }
          }
        }
      ],
      "tools": {
        "search_user_account": {
          "type": "function",
          "function": {"name": "search_user_account"}
        }
      }
    }
    """
    result = parse_conversation_json(raw)
    assert result.error is None
    assert result.data is not None
    assert result.data.system_prompt == "Follow company flow."
    assert len(result.data.interactions) == 2
    assert result.data.tools is not None
    assert result.data.source_meta is not None
    assert result.data.source_meta["model_info"]["model"] == "H200_01_fc_9010"


def test_normalizes_list_tools_by_function_name() -> None:
    raw = """
    {
      "messages": [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Any promo?"},
        {"role": "assistant", "content": "Let me check."}
      ],
      "tools": [
        {
          "type": "function",
          "function": {
            "name": "MandiriCX_Call_Center_search_promotion",
            "description": "Search current promotions.",
            "parameters": {"type": "object", "properties": {}}
          }
        }
      ]
    }
    """

    result = parse_conversation_json(raw)

    assert result.error is None
    assert result.data is not None
    assert result.data.tools is not None
    assert list(result.data.tools) == ["MandiriCX_Call_Center_search_promotion"]
    assert (
        result.data.tools["MandiriCX_Call_Center_search_promotion"]["function"]["name"]
        == "MandiriCX_Call_Center_search_promotion"
    )
