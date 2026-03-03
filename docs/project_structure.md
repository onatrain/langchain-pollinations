# langchain-pollinations provider library

## Project structure

```
langchain-pollinations/
├── .env.example
├── .github/
│   └── workflows/
│       └── publish.yml
├── .gitignore
├── academy/
│   ├── Chinook.db
│   ├── groovy.jpg
│   ├── L1_fast_agent.py
│   ├── L2_agent_messages.py
│   ├── L2_model_messages.py
│   ├── L3_custom_streaming.py
│   ├── L3_messages_streaming.py
│   ├── L3_values_streaming.py
│   ├── L4_doc_tools.py
│   ├── L4_tools.py
│   ├── L5_MCP.py
│   ├── L6_Memory.py
│   ├── L7_basic_structured_output.py
│   ├── L7_complex_structured_output.py
│   ├── L8_dynamic_prompt.py
│   ├── L9_human_in_the_loop_allow.py
│   ├── L9_human_in_the_loop_reject.py
│   ├── marycorto.mp3
│   ├── pollinations_audio.py
│   └── pollinations_image.py
├── assets/
│   └── doki.png
├── CHANGELOG.md
├── debugging/
│   ├── debug_httpx_raw.py
│   ├── debug_raw_chat.py
│   ├── debug_raw_stream.py
│   └── debugging.md
├── docs/
│   ├── api.json
│   ├── api_reference/
│   │   ├── 01_Overview.md
│   │   ├── 02_ChatPollinations.md
│   │   ├── 03_ChatPollinationsConfig.md
│   │   ├── 04_Types.md
│   │   ├── 05_ImagePollinations.md
│   │   ├── 06_AccountInformation.md
│   │   ├── 07_ModelInformation.md
│   │   └── 08_PollinationsAPIError.md
│   ├── code_structure_and_composition.md
│   ├── design_decissions.md
│   ├── project_structure.md
│   └── tooling.md
├── examples/
│   ├── chat_completions/
│   │   ├── __init__.py
│   │   ├── audio.mp3
│   │   ├── audio_generation.py
│   │   ├── errors.py
│   │   ├── messages_streaming_chat.py
│   │   ├── models_list.py
│   │   ├── multiimage_question.py
│   │   ├── simple_chat.py
│   │   ├── transcribe_audio.py
│   │   ├── url_audio.py
│   │   ├── url_image.py
│   │   ├── url_ocr.py
│   │   ├── utils.py
│   │   └── video_question.py
│   └── image/
│       ├── basic_image_generation.py
│       ├── custom_image_generation.py
│       ├── framed_video.py
│       ├── image_to_image.py
│       ├── image_to_video.py
│       ├── interpolated_video.py
│       ├── referential_image_generation.py
│       ├── simple_image_generation.py
│       ├── text_to_video.py
│       ├── transparent_image_generation.py
│       ├── utils.py
│       └── video_generation.py
├── LICENSE.md
├── pyproject.toml
├── README.md
├── src/
│   └── langchain_pollinations/
│       ├── __init__.py
│       ├── _auth.py
│       ├── _client.py
│       ├── _errors.py
│       ├── _openai_compat.py
│       ├── _sse.py
│       ├── account.py
│       ├── chat.py
│       ├── image.py
│       ├── models.py
│       └── py.typed
├── tests/
│   ├── integration/
│   │   ├── conftest.py
│   │   ├── test_chat_completions.py
│   │   ├── test_chat_multimodal.py
│   │   ├── test_chat_streaming.py
│   │   ├── test_image_endpoint.py
│   │   ├── test_models_endpoints.py
│   │   └── with/
│   │       ├── conftest.py
│   │       ├── test_with_config.py
│   │       ├── test_with_fallbacks.py
│   │       ├── test_with_listeners.py
│   │       ├── test_with_retry.py
│   │       ├── test_with_structured_output_pydantic.py
│   │       └── test_with_structured_output_typed_dict.py
│   └── unit/
│       ├── test_account.py
│       ├── test_auth.py
│       ├── test_chat.py
│       ├── test_client.py
│       ├── test_content_block_handling.py
│       ├── test_image.py
│       ├── test_models.py
│       ├── test_openai_compat.py
│       ├── test_sse.py
│       └── test_structured_errors.py
└── uv.lock
```
