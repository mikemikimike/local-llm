import ast
import asyncio
import copy
import json
import logging
import mimetypes
import os
import random
import re
import sys
import textwrap
import time
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse
from starlette.responses import JSONResponse, StreamingResponse

from local_llm.config import (
    CODE_INTERPRETER_BLOCKED_MODULES,
    CODE_INTERPRETER_JUPYTER_PROMPT,
    CODE_INTERPRETER_PYODIDE_PROMPT,
    DEFAULT_CODE_INTERPRETER_PROMPT,
    DEFAULT_TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE,
    DEFAULT_VOICE_MODE_PROMPT_TEMPLATE,
)
from local_llm.constants import TASKS
from local_llm.env import (
    CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS,
    CHAT_RESPONSE_STREAM_DELTA_CHUNK_SIZE,
    ENABLE_API_OUTLET_FILTERS,
    ENABLE_CHAT_RESPONSE_BASE64_IMAGE_URL_CONVERSION,
    ENABLE_PLUGINS,
    ENABLE_QUERIES_CACHE,
    ENABLE_RESPONSES_API_STATEFUL,
    GLOBAL_LOG_LEVEL,
    RAG_SYSTEM_CONTEXT,
)
from local_llm.events import EVENTS, publish_event
from local_llm.models.access_grants import AccessGrants
from local_llm.models.chats import Chats
from local_llm.models.config import Config
from local_llm.models.folders import Folders
from local_llm.models.notes import Notes
from local_llm.models.users import UserModel
from local_llm.retrieval.utils import get_sources_from_items
from local_llm.routers.images import (
    CreateImageForm,
    EditImageForm,
    image_edits,
    image_generations,
)
from local_llm.routers.pipelines import (
    get_sorted_filters,
    process_pipeline_inlet_filter,
    process_pipeline_outlet_filter,
)
from local_llm.routers.retrieval import (
    SearchForm,
    process_web_search,
)
from local_llm.routers.tasks import (
    generate_chat_tags,
    generate_follow_ups,
    generate_image_prompt,
    generate_queries,
    generate_title,
)
from local_llm.socket.main import (
    get_event_call,
    get_event_emitter,
)
from local_llm.tasks import clear_response_stream, save_response_stream
from local_llm.utils.access_control import has_connection_access, has_permission
from local_llm.utils.access_control.files import get_owner_accessible_folder_files
from local_llm.utils.access_control.folders import has_folder_access
from local_llm.utils.ask_user import stage_ask_user_tool_calls
from local_llm.utils.chat import generate_chat_completion
from local_llm.utils.chat_id import is_saved_chat_id
from local_llm.utils.code_interpreter import chat_files, execute_code_jupyter
from local_llm.utils.context_compaction import compact_messages_for_request
from local_llm.utils.files import (
    convert_markdown_base64_images,
    get_file_url_from_base64,
    get_image_base64_from_url,
    get_image_url_from_base64,
)
from local_llm.utils.filter import (
    FilterContext,
    get_filter_context,
    get_filter_functions,
    process_filter_functions,
)
from local_llm.utils.json_codec import JSONCodec
from local_llm.utils.mcp.client import MCPClient
from local_llm.utils.memory import add_memory_context, review_memory_after_turn
from local_llm.utils.misc import (
    add_or_update_system_message,
    add_or_update_user_message,
    convert_output_to_messages,
    get_content_from_message,
    get_last_assistant_message,
    get_last_user_message,
    get_last_user_message_item,
    get_message_list,
    get_output_text,
    get_reasoning_details,
    get_response_error_detail,
    get_system_message,
    get_text_from_message,
    is_string_allowed,
    merge_system_messages,
    replace_system_message_content,
    set_last_user_message_content,
    strip_empty_content_blocks,
)
from local_llm.utils.payload import apply_params_to_form_data, apply_system_prompt_to_body, resolve_system_prompt
from local_llm.utils.response import merge_usage, normalize_usage
from local_llm.utils.sanitize import sanitize_code
from local_llm.utils.task import (
    get_task_model_id,
    rag_template,
    tools_function_calling_generation_template,
)
from local_llm.utils.tools import (
    build_tool_server_headers,
    get_attached_knowledge,
    get_builtin_tools,
    get_terminal_tools,
    get_tools,
    get_updated_tool_function,
)

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)


def _is_tool_result_error(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if (
            text.startswith('error:')
            or text.startswith('exception:')
            or text.startswith('traceback')
            or text.startswith('http error!')
        ):
            return True

    parsed = value
    while isinstance(parsed, str):
        try:
            parsed = JSONCodec.loads(parsed)
        except (JSONCodec.JSONDecodeError, TypeError, ValueError):
            break

    if not isinstance(parsed, dict):
        return False

    error = parsed.get('error')
    if isinstance(error, str):
        has_error = bool(error.strip())
    else:
        has_error = isinstance(error, (dict, list)) and bool(error)
    if has_error:
        return True

    status = parsed.get('status')
    if isinstance(status, str) and status.strip().lower() in {'error', 'failed'}:
        return True

    if parsed.get('success') is False or parsed.get('ok') is False:
        message = parsed.get('message')
        return has_error or (
            bool(message.strip()) if isinstance(message, str) else isinstance(message, (dict, list)) and bool(message)
        )

    return False


def normalize_messages_for_model(form_data: dict) -> dict:
    form_data['messages'] = strip_empty_content_blocks(form_data.get('messages', []))
    form_data['messages'] = merge_system_messages(form_data.get('messages', []))
    return form_data


async def publish_chat_finished_event(
    request: Request, user: UserModel, metadata: dict, title: str, content: str, output: list | None = None
):
    chat_id = metadata.get('chat_id')
    if getattr(request.state, 'internal', False) is True or not is_saved_chat_id(chat_id):
        return

    content = content or get_output_text(output)
    webui_url = await Config.get('webui.url')
    await publish_event(
        request,
        EVENTS.CHAT_FINISHED,
        actor=user,
        subject_id=chat_id,
        subject_type='chat',
        data={
            'user_id': user.id,
            'chat_id': chat_id,
            'message_id': metadata.get('message_id'),
            'model_id': metadata.get('model_id'),
            'title': title,
            'url': f'{webui_url}/c/{chat_id}' if webui_url else f'/c/{chat_id}',
            'message': content,
        },
        message=title or 'Chat finished',
    )
    event_emitter = await get_event_emitter(metadata, update_db=False)
    if event_emitter:
        folder_id = metadata.get('folder_id') or await Chats.get_chat_folder_id(chat_id, metadata.get('user_id'))
        await event_emitter({'type': 'chat:list', 'data': {'chat_id': chat_id, 'folder_id': folder_id}})


# We believe in one maker of all models, seen and unseen,
# and in the reasoning which proceeds from the architect.
# We look for the resurrection of dead processes and the
# inference of the world to come.
DEFAULT_REASONING_TAGS = [
    ('<think>', '</think>'),
    ('<thinking>', '</thinking>'),
    ('<reason>', '</reason>'),
    ('<reasoning>', '</reasoning>'),
    ('<thought>', '</thought>'),
    ('<Thought>', '</Thought>'),
    ('<|begin_of_thought|>', '<|end_of_thought|>'),
    ('◁think▷', '◁/think▷'),
]

DEFAULT_SOLUTION_TAGS = [('<|begin_of_solution|>', '<|end_of_solution|>')]
DEFAULT_CODE_INTERPRETER_TAGS = [('<code_interpreter>', '</code_interpreter>')]


def _start_tag_pattern(start_tag: str) -> str:
    if start_tag.startswith('<') and start_tag.endswith('>'):
        return rf'<{re.escape(start_tag[1:-1])}(\s.*?)?>'
    return re.escape(start_tag)


def output_id(prefix: str) -> str:
    """Generate OR-style ID: prefix + 24-char hex UUID."""
    return f'{prefix}_{uuid4().hex[:24]}'


def build_terminal_file_tool_result(
    tool_function_name: str,
    tool_function_params: dict,
    tool_result: Any,
    tool: dict | None,
    metadata: dict | None,
) -> dict | None:
    if isinstance(tool_result, (list, tuple)) and tool_result and isinstance(tool_result[0], dict):
        tool_result = tool_result[0]

    if tool_function_name != 'display_file' or not isinstance(tool_result, dict) or tool_result.get('exists') is False:
        return None

    tool_id = (tool or {}).get('tool_id', '')
    terminal_id = metadata.get('terminal_id') if metadata else None
    if isinstance(tool_id, str) and tool_id.startswith('terminal:'):
        terminal_id = tool_id.split(':', 1)[1]

    server_url = ((tool or {}).get('server') or {}).get('url')
    terminal_selector = terminal_id or server_url
    path = tool_result.get('path') or tool_function_params.get('path')
    if not terminal_selector or not path:
        return None
    mime_type, _ = mimetypes.guess_type(path)
    mime_type = mime_type or 'application/octet-stream'
    page = tool_result.get('page') or tool_function_params.get('page')

    return {
        **tool_result,
        'type': 'file',
        'source': 'open_terminal',
        **({'displayed': True} if tool_function_params.get('inline') is True else {}),
        'terminal_selector': terminal_selector,
        **({'terminal_id': terminal_id} if terminal_id else {}),
        **({'terminal_url': server_url} if server_url and not terminal_id else {}),
        'session_id': metadata.get('chat_id') if metadata else None,
        'path': path,
        'full_path': tool_result.get('full_path') or path,
        'name': tool_result.get('name') or os.path.basename(path),
        'mime_type': tool_result.get('mime_type') or tool_result.get('content_type') or mime_type,
        'content_type': tool_result.get('content_type') or tool_result.get('mime_type') or mime_type,
        **({'page': page} if page else {}),
    }


def tool_result_content(tool_result: Any) -> str:
    if not tool_result:
        return ''
    if isinstance(tool_result, (dict, list)):
        return JSONCodec.dumps(tool_result, ensure_ascii=False)
    return str(tool_result)


def merge_streamed_reasoning_details(target: list, details) -> None:
    items = details if isinstance(details, list) else [details]
    for item in items:
        if not isinstance(item, dict):
            continue

        index = item.get('index')
        existing = (
            next((detail for detail in target if detail.get('index') == index), None)
            if isinstance(index, int)
            else None
        )
        if existing is None:
            target.append(dict(item))
            continue

        for key, value in item.items():
            if key in ('text', 'summary') and isinstance(value, str) and isinstance(existing.get(key), str):
                existing[key] += value
            else:
                existing[key] = value


def _split_tool_calls(
    tool_calls: list[dict],
) -> list[dict]:
    """Expand tool calls whose arguments contain multiple back-to-back JSON objects.

    Some models (e.g. GPT-5.4) send multiple complete JSON argument objects
    under the same tool call index, producing concatenated invalid JSON like:
        '{"query":"A","count":5}{"query":"B","count":5}'

    Each such tool call is split into separate entries so each gets executed
    independently. Single-object arguments pass through unchanged.
    """

    def split_json_objects(raw: str) -> list[str]:
        if not isinstance(raw, str):
            raw = '' if raw is None else JSONCodec.dumps(raw)

        decoder = json.JSONDecoder()
        results = []
        position = 0

        while position < len(raw):
            while position < len(raw) and raw[position].isspace():
                position += 1
            if position >= len(raw):
                break
            try:
                _, end = decoder.raw_decode(raw, position)
                results.append(raw[position:end].strip())
                position = end
            except JSONCodec.JSONDecodeError:
                return [raw]

        return results or [raw]

    expanded = []
    for tool_call in tool_calls:
        function = tool_call.setdefault('function', {})
        arguments = function.get('arguments')
        if not isinstance(arguments, str):
            arguments = '' if arguments is None else JSONCodec.dumps(arguments)
            function['arguments'] = arguments
        split_arguments = split_json_objects(arguments)

        if len(split_arguments) <= 1:
            expanded.append(tool_call)
        else:
            for argument in split_arguments:
                cloned = copy.deepcopy(tool_call)
                cloned['id'] = f'call_{uuid4().hex[:24]}'
                cloned['function']['arguments'] = argument
                expanded.append(cloned)

    return expanded


def get_citation_source_from_tool_result(
    tool_name: str, tool_params: dict, tool_result: str, tool_id: str = ''
) -> list[dict]:
    """
    Parse a tool's result and convert it to source dicts for citation display.

    Follows the source format conventions from get_sources_from_items:
    - source: file/item info object with id, name, type
    - document: list of document contents
    - metadata: list of metadata objects with source, file_id, name fields

    Returns a list of sources (usually one, but query_knowledge_files/query_chat_files may return multiple).
    """
    _EXPECTS_LIST = {'search_web', 'query_knowledge_files', 'query_chat_files'}
    _EXPECTS_DICT = {'view_knowledge_file', 'view_file'}

    try:
        try:
            tool_result = JSONCodec.loads(tool_result)
        except (JSONCodec.JSONDecodeError, TypeError):
            pass  # keep tool_result as-is (e.g. fetch_url returns plain text)
        if isinstance(tool_result, dict) and 'error' in tool_result:
            return []

        # Validate tool_result type based on what the branch expects
        if tool_name in _EXPECTS_LIST and not isinstance(tool_result, list):
            return []
        elif tool_name in _EXPECTS_DICT and not isinstance(tool_result, dict):
            return []

        if tool_name == 'search_web':
            # Parse JSON array: [{"title": "...", "link": "...", "snippet": "..."}]
            results = tool_result
            documents = []
            metadata = []

            for result in results:
                title = result.get('title', '')
                link = result.get('link', '')
                snippet = result.get('snippet', '')

                documents.append(f'{title}\n{snippet}')
                metadata.append(
                    {
                        'source': link,
                        'name': title,
                        'url': link,
                    }
                )

            return [
                {
                    'source': {'name': 'search_web', 'id': 'search_web'},
                    'document': documents,
                    'metadata': metadata,
                }
            ]

        elif tool_name in ('view_knowledge_file', 'view_file'):
            file_data = tool_result
            filename = file_data.get('filename', 'Unknown File')
            file_id = file_data.get('id', '')
            knowledge_name = file_data.get('knowledge_name', '')

            return [
                {
                    'source': {
                        'id': file_id,
                        'name': filename,
                        'type': 'file',
                    },
                    'document': [file_data.get('content', '')],
                    'metadata': [
                        {
                            'file_id': file_id,
                            'name': filename,
                            'source': filename,
                            **({'knowledge_name': knowledge_name} if knowledge_name else {}),
                        }
                    ],
                }
            ]

        elif tool_name == 'fetch_url':
            url = tool_params.get('url', '')
            content = tool_result if isinstance(tool_result, str) else str(tool_result)
            snippet = content[:500] + ('...' if len(content) > 500 else '')

            return [
                {
                    'source': {'name': url or 'fetch_url', 'id': url or 'fetch_url'},
                    'document': [snippet],
                    'metadata': [
                        {
                            'source': url,
                            'name': url,
                            'url': url,
                        }
                    ],
                }
            ]

        elif tool_name in ('query_knowledge_files', 'query_chat_files'):
            chunks = tool_result

            # Group chunks by source for better citation display
            # Each unique source becomes a separate source entry
            sources_by_file = {}

            for chunk in chunks:
                source_name = chunk.get('source', 'Unknown')
                file_id = chunk.get('file_id', '')
                note_id = chunk.get('note_id', '')
                chunk_type = chunk.get('type', 'file')
                content = chunk.get('content', '')

                # Use file_id or note_id as the key
                key = file_id or note_id or source_name

                if key not in sources_by_file:
                    sources_by_file[key] = {
                        'source': {
                            'id': file_id or note_id,
                            'name': source_name,
                            'type': chunk_type,
                        },
                        'document': [],
                        'metadata': [],
                    }

                sources_by_file[key]['document'].append(content)
                sources_by_file[key]['metadata'].append(
                    {
                        'file_id': file_id,
                        'name': source_name,
                        'source': source_name,
                        **({'note_id': note_id} if note_id else {}),
                    }
                )

            # Return all grouped sources as a list
            if sources_by_file:
                return list(sources_by_file.values())

            # Empty result fallback
            return []

        else:
            # Fallback for other tools
            return [
                {
                    'source': {
                        'name': tool_name,
                        'type': 'tool',
                        'id': tool_id or tool_name,
                    },
                    'document': [str(tool_result)],
                    'metadata': [{'source': tool_name, 'name': tool_name}],
                }
            ]
    except Exception as e:
        log.exception(f'Error parsing tool result for {tool_name}: {e}')
        return [
            {
                'source': {'name': tool_name, 'type': 'tool'},
                'document': [str(tool_result)],
                'metadata': [{'source': tool_name}],
            }
        ]


def deep_merge(target, source):
    """
    Merge source into target recursively (returning new structure).
    - Dicts: Recursive merge.
    - Strings: Concatenation.
    - Others: Overwrite.
    """
    if isinstance(target, dict) and isinstance(source, dict):
        new_target = target.copy()
        for k, v in source.items():
            if k in new_target:
                new_target[k] = deep_merge(new_target[k], v)
            else:
                new_target[k] = v
        return new_target
    elif isinstance(target, str) and isinstance(source, str):
        return target + source
    else:
        return source


RESPONSE_COMPLETION_RESPONSE_FIELDS = ('error', 'id', 'output', 'usage')


def get_response_completion_event_data(event: dict) -> dict:
    """Build the data payload for response:completion events."""
    response = event.get('response')
    if not isinstance(response, dict):
        return event

    response_data = {key: response[key] for key in RESPONSE_COMPLETION_RESPONSE_FIELDS if key in response}

    return {
        **event,
        'response': response_data,
    }


def _output_index_in_range(current_output: list, output_index) -> bool:
    return bool(current_output) and 0 <= output_index < len(current_output)


def _copy_output_item(current_output: list, output_index: int) -> tuple[list, dict]:
    """A shallow copy of the output with the item at output_index copied too, ready to change."""
    new_output = list(current_output)
    item = new_output[output_index].copy()
    new_output[output_index] = item
    return new_output, item


def _copied_parts(item: dict, key: str) -> list:
    """item[key] as a fresh list (an empty one when it is missing), so the given item's list is untouched."""
    item[key] = list(item[key]) if key in item else []
    return item[key]


def _copied_part(parts: list, index: int, empty_part: dict) -> dict:
    """parts[index] as a fresh copy, padding parts with empty ones up to index."""
    while len(parts) <= index:
        parts.append(dict(empty_part))
    part = parts[index].copy()
    parts[index] = part
    return part


def _item_added(data: dict, current_output: list) -> tuple[list, dict | None]:
    item = data.get('item', {})
    if not item:
        return current_output, None
    new_output = list(current_output)
    output_index = data.get('output_index', len(new_output))
    existing_index = next(
        (
            idx
            for idx, existing in enumerate(new_output)
            if (item.get('id') and existing.get('id') == item.get('id'))
            or (item.get('call_id') and existing.get('call_id') == item.get('call_id'))
        ),
        None,
    )
    if existing_index is not None:
        new_output[existing_index] = item
    elif 0 <= output_index < len(new_output):
        new_output.insert(output_index, item)
    else:
        new_output.append(item)
    return new_output, None


def _part_added(parts_key: str):
    """A handler adding the event's part to the item's content (or summary) list."""

    def handle(data: dict, current_output: list) -> tuple[list, dict | None]:
        part = data.get('part', {})
        output_index = data.get('output_index', len(current_output) - 1)
        if not _output_index_in_range(current_output, output_index):
            return current_output, None
        new_output, item = _copy_output_item(current_output, output_index)
        parts = _copied_parts(item, parts_key)
        # Reasoning items take summary parts, never content parts
        if not (parts_key == 'content' and item.get('type') == 'reasoning'):
            parts.append(part)
        return new_output, None

    return handle


def _empty_value(delta):
    return {} if isinstance(delta, dict) else ''


def _apply_message_delta(item: dict, delta_type: str, delta, data: dict) -> None:
    # "text"/"output_text" go to the content part's "text"; reasoning deltas belong to reasoning items.
    if delta_type in ['reasoning_text', 'reasoning_summary_text']:
        return
    key = 'text' if delta_type in ['text', 'output_text'] else delta_type
    part = _copied_part(_copied_parts(item, 'content'), data.get('content_index', 0), {'type': 'text', 'text': ''})
    current_val = part.get(key)
    if current_val is None:
        current_val = _empty_value(delta)
    part[key] = deep_merge(current_val, delta)


def _apply_reasoning_delta(item: dict, delta_type: str, delta, data: dict) -> None:
    # Summary text goes to item['summary'], reasoning text to item['content']; plain text belongs to messages.
    if delta_type == 'reasoning_summary_text':
        parts, index, empty_part = _copied_parts(item, 'summary'), data.get('summary_index', 0), 'summary_text'
    elif delta_type == 'reasoning_text':
        parts, index, empty_part = _copied_parts(item, 'content'), data.get('content_index', 0), 'text'
    else:
        return
    part = _copied_part(parts, index, {'type': empty_part, 'text': ''})
    part['text'] = deep_merge(part.get('text', ''), delta)


def _apply_other_delta(item: dict, delta_type: str, delta) -> None:
    key = 'text' if delta_type in ['text', 'output_text'] else delta_type
    current_val = item.get(key)
    if current_val is None:
        current_val = _empty_value(delta)
    item[key] = deep_merge(current_val, delta)


def _delta(data: dict, current_output: list) -> tuple[list, dict | None]:
    """response.<type>.delta: append the delta to the field it targets on the item at output_index."""
    parts = data.get('type', '').split('.')
    if len(parts) < 3:
        return current_output, None
    delta_type = parts[1]
    delta = data.get('delta', '')
    output_index = data.get('output_index', len(current_output) - 1)
    if not _output_index_in_range(current_output, output_index):
        return current_output, None

    new_output, item = _copy_output_item(current_output, output_index)
    item_type = item.get('type', '')
    if delta_type == 'function_call_arguments':
        # Function call args are usually strings
        if item_type == 'function_call':
            item['arguments'] = item.get('arguments', '') + str(delta)
    elif item_type == 'message':
        _apply_message_delta(item, delta_type, delta, data)
    elif item_type == 'reasoning':
        _apply_reasoning_delta(item, delta_type, delta, data)
    else:
        _apply_other_delta(item, delta_type, delta)
    return new_output, None


def _item_done(data: dict, current_output: list) -> tuple[list, dict | None]:
    item = data.get('item')
    output_index = data.get('output_index', len(current_output) - 1)
    new_output = list(current_output)
    if item and 0 <= output_index < len(current_output):
        new_output[output_index] = item
    elif item:
        new_output.append(item)
    return new_output, {}


def _part_done(data: dict, current_output: list, parts_key: str, index_key: str) -> tuple[list, dict | None]:
    """A finished content (or summary) part replaces the one at its index, when the event carries it."""
    part = data.get('part')
    output_index = data.get('output_index', len(current_output) - 1)
    if part and _output_index_in_range(current_output, output_index):
        new_output, item = _copy_output_item(current_output, output_index)
        if parts_key in item:
            item[parts_key] = list(item[parts_key])
            index = data.get(index_key, len(item[parts_key]) - 1)
            if 0 <= index < len(item[parts_key]):
                item[parts_key][index] = part
                return new_output, {}
    return current_output, None


def _field_done(data: dict, current_output: list, type_name: str) -> tuple[list, dict | None]:
    """A finished field (text.done, audio.done, ...) sets the item's final value when the event carries it."""
    output_index = data.get('output_index', len(current_output) - 1)
    if type_name in ['completed', 'failed'] or not _output_index_in_range(current_output, output_index):
        return current_output, None

    if type_name == 'function_call_arguments':
        key = 'arguments'
    elif type_name in ['text', 'output_text', 'reasoning_text', 'reasoning_summary_text']:
        key = 'text'
    else:
        key = type_name
    if key not in data:
        return current_output, None

    final_value = data[key]
    new_output, item = _copy_output_item(current_output, output_index)
    item_type = item.get('type', '')
    if type_name == 'function_call_arguments':
        if item_type == 'function_call':
            item['arguments'] = final_value
    elif item_type == 'message':
        content_index = data.get('content_index', 0)
        if 'content' in item:
            item['content'] = list(item['content'])
            if len(item['content']) > content_index:
                part = item['content'][content_index].copy()
                item['content'][content_index] = part
                part[key] = final_value
    elif item_type == 'reasoning':
        item['status'] = 'completed'
    else:
        item[key] = final_value
    return new_output, {}


def _done(data: dict, current_output: list) -> tuple[list, dict | None]:
    """response.<type>.done: a finished part or field."""
    parts = data.get('type', '').split('.')
    if len(parts) < 3:
        return current_output, None
    type_name = parts[1]
    if type_name == 'content_part':
        return _part_done(data, current_output, 'content', 'content_index')
    if type_name == 'reasoning_summary_part':
        return _part_done(data, current_output, 'summary', 'summary_index')
    return _field_done(data, current_output, type_name)


def _completed(data: dict, current_output: list) -> tuple[list, dict | None]:
    response_data = data.get('response', {})
    # Some providers send an empty output on response.completed despite having streamed items
    new_output = response_data.get('output') or current_output

    # Ensure reasoning items are marked as completed in the final output
    for item in new_output:
        if item.get('type') == 'reasoning' and item.get('status') != 'completed':
            item['status'] = 'completed'

    return new_output, {
        'usage': response_data.get('usage'),
        'done': True,
        'response_id': response_data.get('id'),
    }


def _failed(data: dict, current_output: list) -> tuple[list, dict | None]:
    return current_output, {'error': data.get('response', {}).get('error', {})}


_RESPONSES_EVENT_HANDLERS = {
    'response.output_item.added': _item_added,
    'response.content_part.added': _part_added('content'),
    'response.reasoning_summary_part.added': _part_added('summary'),
    'response.output_item.done': _item_done,
    'response.completed': _completed,
    'response.failed': _failed,
}


def handle_responses_streaming_event(
    data: dict,
    current_output: list,
) -> tuple[list, dict | None]:
    """
    Handle Responses API streaming events in a pure functional way.

    Args:
        data: The event data
        current_output: List of output items (treated as immutable)

    Returns:
        tuple[list, dict | None]: (new_output, metadata)
        - new_output: The updated output list.
        - metadata: Metadata to emit (e.g. usage), {} if update occurred, None if skip.
    """
    # Note: treating current_output as immutable, but avoiding full deepcopy for perf.
    # Handlers shallow copy only the list and the items they change.
    event_type = data.get('type', '')
    handler = _RESPONSES_EVENT_HANDLERS.get(event_type)
    if handler is None and event_type.startswith('response.'):
        if event_type.endswith('.delta'):
            handler = _delta
        elif event_type.endswith('.done'):
            handler = _done
    # Anything else (response.in_progress, response.created, ...) leaves the output as it is.
    return handler(data, current_output) if handler else (current_output, None)


def get_source_context(sources: list, source_ids: dict = None, include_content: bool = True) -> str:
    """
    Build <source> tag context string from citation sources.
    """
    context_string = ''
    if source_ids is None:
        source_ids = {}
    for source in sources:
        for doc, meta in zip(source.get('document', []), source.get('metadata', []), strict=False):
            source_id = meta.get('source') or source.get('source', {}).get('id') or 'N/A'
            if source_id not in source_ids:
                source_ids[source_id] = len(source_ids) + 1
            src_name = source.get('source', {}).get('name')
            src_type = source.get('source', {}).get('type')
            src_rid = source.get('source', {}).get('id')
            body = doc if include_content else ''
            context_string += (
                f'<source id="{source_ids[source_id]}"'
                + (f' name="{src_name}"' if src_name else '')
                + (f' resource-type="{src_type}"' if src_type else '')
                + (f' resource-id="{src_rid}"' if src_rid else '')
                + f'>{body}</source>\n'
            )
    return context_string


async def apply_source_context_to_messages(
    request: Request,
    messages: list,
    sources: list,
    user_message: str,
    include_content: bool = True,
) -> list:
    """
    Build source context from citation sources and apply to messages.
    Uses RAG template to format context for model consumption.

    When include_content is False, emit <source> tags with id/name but no
    document body — useful when the content is already present elsewhere
    (e.g. in a tool result message) and only citation markers are needed.
    """
    if not sources or not user_message:
        return messages

    context = get_source_context(sources, include_content=include_content)

    context = context.strip()
    if not context:
        return messages

    if RAG_SYSTEM_CONTEXT:
        return add_or_update_system_message(
            await rag_template(await Config.get('rag.template'), context, user_message),
            messages,
            append=True,
        )
    else:
        return add_or_update_user_message(
            await rag_template(await Config.get('rag.template'), context, user_message),
            messages,
            append=False,
        )


def _ui_component_result(tool_function_name: str, status: str, detail: str) -> dict:
    return {'status': status, 'code': 'ui_component', 'message': f'{tool_function_name}: {detail}'}


def _embedded_ui_result(tool_function_name: str, result_context):
    """What the model is told about an embed the user can see: the tool's own context, or a generic note."""
    if result_context is not None and isinstance(result_context, (str, dict, list)):
        return result_context
    return _ui_component_result(tool_function_name, 'success', 'Embedded UI result is active and visible to the user.')


def _html_response_result(tool_function_name: str, response: HTMLResponse, result_context, embeds: list):
    """An HTMLResponse: shown inline when its Content-Disposition says so, otherwise its body as text."""
    if 'inline' not in response.headers.get('Content-Disposition', ''):
        return response.body.decode('utf-8', 'replace')

    embeds.append(response.body.decode('utf-8', 'replace'))
    status_code = response.status_code
    if 200 <= status_code < 300:
        return _embedded_ui_result(tool_function_name, result_context)
    if 400 <= status_code < 500:
        return _ui_component_result(tool_function_name, 'error', f'Client error {status_code} from embedded UI result.')
    if 500 <= status_code < 600:
        return _ui_component_result(tool_function_name, 'error', f'Server error {status_code} from embedded UI result.')
    return _ui_component_result(
        tool_function_name, 'error', f'Unexpected status code {status_code} from embedded UI result.'
    )


def _header(headers: dict, name: str) -> str:
    return headers.get(name, headers.get(name.lower(), ''))


def _headed_result(tool_function_name: str, tool_result, tool_response_headers, embeds: list):
    """A (data, headers) result: inline HTML or a Location becomes an embed, anything else stays data."""
    try:
        if not isinstance(tool_response_headers, dict):
            tool_response_headers = dict(tool_response_headers)
    except Exception as e:
        tool_response_headers = {}
        log.debug(e)

    if not (tool_response_headers and isinstance(tool_response_headers, dict)):
        return tool_result
    if 'inline' not in _header(tool_response_headers, 'Content-Disposition'):
        return tool_result

    location = _header(tool_response_headers, 'Location')
    is_html = 'text/html' in _header(tool_response_headers, 'Content-Type')
    if not (is_html or location):
        return tool_result

    # Either may come as (html_content, result_context)
    result_context = None
    html_content = tool_result
    if isinstance(tool_result, (tuple, list)) and len(tool_result) == 2:
        html_content, result_context = tool_result
    # Display as iframe embed
    embeds.append(html_content if is_html else location)
    return _embedded_ui_result(tool_function_name, result_context)


def _json_or_text(text):
    if isinstance(text, str):
        try:
            return JSONCodec.loads(text)
        except JSONCodec.JSONDecodeError:
            pass
    return text


def _mcp_resource_result(resource: dict, tool_response: list, tool_result_files: list) -> None:
    text = resource.get('text', '')
    if isinstance(text, str) and text:
        tool_response.append(_json_or_text(text))
    elif resource.get('blob'):
        resource_mime_type = resource.get('mimeType') or 'application/octet-stream'
        resource_blob = resource.get('blob', '')
        if resource_mime_type.startswith('image/'):
            tool_result_files.append({'type': 'image', 'url': f'data:{resource_mime_type};base64,{resource_blob}'})
        else:
            resource_uri = resource.get('uri', 'resource')
            tool_response.append(f'[Resource: {resource_uri}] (binary data, mimeType: {resource_mime_type})')
    elif resource.get('uri'):
        tool_response.append(resource.get('uri'))


async def _mcp_list_result(request, tool_result: list, metadata, user, tool_result_files: list):
    """MCP content items: text and resources become the result, images and audio become files."""
    tool_response = []
    for item in tool_result:
        if not isinstance(item, dict):
            continue
        item_type = item.get('type')
        if item_type == 'text':
            tool_response.append(_json_or_text(item.get('text', '')))
        elif item_type in ['image', 'audio']:
            file_url = await get_file_url_from_base64(
                request,
                f'data:{item.get("mimeType")};base64,{item.get("data", item.get("blob", ""))}',
                {
                    'chat_id': metadata.get('chat_id', None),
                    'message_id': metadata.get('message_id', None),
                    'session_id': metadata.get('session_id', None),
                    'result': item,
                },
                user,
            )
            tool_result_files.append({'type': item.get('type', 'data'), 'url': file_url})
        elif item_type == 'resource':
            _mcp_resource_result(item.get('resource', {}), tool_response, tool_result_files)
    return tool_response[0] if len(tool_response) == 1 else tool_response


def _split_openapi_data_items(tool_result: list, tool_result_files: list) -> list:
    """The result without its data: items, which become files; the tool's own list is left as it was."""
    kept = []
    for item in tool_result:
        if isinstance(item, str) and item.startswith('data:'):
            tool_result_files.append(
                {
                    'type': 'data',
                    'content': item,
                }
            )
        else:
            kept.append(item)
    return kept


def _tool_result_text(tool_result):
    """The result as the text the model reads: JSON for structures, and always a string (or None)."""
    if isinstance(tool_result, list):
        tool_result = {'results': tool_result}

    if isinstance(tool_result, dict) or isinstance(tool_result, list):
        tool_result = json.dumps(tool_result, indent=2, ensure_ascii=False)

    # Safety: ensure tool_result is always a string (or None) to prevent
    # downstream TypeError when concatenating (e.g. if an upstream callable
    # returned a tuple that was not unpacked by the branches above).
    if tool_result is not None and not isinstance(tool_result, str):
        if isinstance(tool_result, tuple):
            # execute_tool_server returns (data, headers); unpack the data part
            tool_result = json.dumps(tool_result[0], indent=2, ensure_ascii=False) if len(tool_result) > 0 else ''
        else:
            tool_result = str(tool_result)
    return tool_result


async def process_tool_result(
    request,
    tool_function_name,
    tool_result,
    tool_type,
    direct_tool=False,
    metadata=None,
    user=None,
):
    tool_result_embeds = []
    EXTERNAL_TOOL_TYPES = ('external', 'action', 'terminal')

    # Support (HTMLResponse, result_context) tuples: the optional second
    # element lets tool authors provide the LLM with actionable context
    # about the generated embed instead of the generic fallback message.
    result_context = None
    if isinstance(tool_result, tuple) and len(tool_result) == 2 and isinstance(tool_result[0], HTMLResponse):
        tool_result, result_context = tool_result

    if isinstance(tool_result, HTMLResponse):
        tool_result = _html_response_result(tool_function_name, tool_result, result_context, tool_result_embeds)
    elif (tool_type in EXTERNAL_TOOL_TYPES and isinstance(tool_result, tuple)) or (
        direct_tool and isinstance(tool_result, list) and len(tool_result) == 2
    ):
        tool_result, tool_response_headers = tool_result
        tool_result = _headed_result(tool_function_name, tool_result, tool_response_headers, tool_result_embeds)

    tool_result_files = []

    # Detect base64 image data URIs from tool results (e.g. binary image
    # responses from execute_tool_server).  Move the data URI to
    # tool_result_files and replace tool_result with a text summary.
    if isinstance(tool_result, str) and tool_result.startswith('data:image/'):
        tool_result_files.append({'type': 'image', 'url': tool_result})
        tool_result = f'{tool_function_name}: Image file read successfully.'

    if isinstance(tool_result, list):
        if tool_type == 'mcp':
            tool_result = await _mcp_list_result(request, tool_result, metadata, user, tool_result_files)
        else:  # OpenAPI
            tool_result = _split_openapi_data_items(tool_result, tool_result_files)

    return _tool_result_text(tool_result), tool_result_files, tool_result_embeds


async def terminal_event_handler(
    tool_function_name: str,
    tool_function_params: dict,
    tool_result,
    event_emitter,
):
    """Emit terminal:* events for Open Terminal tools.

    - display_file  → emits 'terminal:display_file' to open the file preview.
    - write_file / replace_file_content → emits 'terminal:write_file' to refresh.
    - run_command → emits 'terminal:run_command' with cwd to refresh if relevant.
    """
    if not event_emitter:
        return

    if tool_function_name == 'display_file':
        if tool_function_params.get('inline') is True:
            return
        path = tool_function_params.get('path', '')
        if not path:
            return
        # Only emit if the file actually exists
        parsed = tool_result
        if isinstance(parsed, str):
            try:
                parsed = JSONCodec.loads(parsed)
            except (JSONCodec.JSONDecodeError, TypeError):
                pass
        if isinstance(parsed, dict) and parsed.get('exists') is False:
            return
        page = tool_function_params.get('page')

        await event_emitter(
            {
                'type': f'terminal:{tool_function_name}',
                'data': {
                    'path': path,
                    **({'page': page} if page else {}),
                },
            }
        )
    elif tool_function_name in ('write_file', 'replace_file_content'):
        path = tool_function_params.get('path', '')
        if not path:
            return
        await event_emitter(
            {
                'type': f'terminal:{tool_function_name}',
                'data': {'path': path},
            }
        )
    elif tool_function_name == 'run_command':
        await event_emitter(
            {
                'type': 'terminal:run_command',
                'data': {},
            }
        )


async def chat_completion_tools_handler(
    request: Request, body: dict, extra_params: dict, user: UserModel, models, tools
) -> tuple[dict, dict]:
    async def get_content_from_response(response) -> str | None:
        content = None
        if hasattr(response, 'body_iterator'):
            async for chunk in response.body_iterator:
                data = JSONCodec.loads(chunk.decode('utf-8', 'replace'))
                content = data['choices'][0]['message']['content']

            # Cleanup any remaining background tasks if necessary
            if response.background is not None:
                await response.background()
        else:
            content = response['choices'][0]['message']['content']
        return content

    def get_tools_function_calling_payload(messages, task_model_id, content):
        user_message = get_last_user_message(messages)

        if user_message and messages and messages[-1]['role'] == 'user':
            # Remove the last user message to avoid duplication
            messages = messages[:-1]

        recent_messages = messages[-4:] if len(messages) > 4 else messages
        chat_history = '\n'.join(
            f'{message["role"].upper()}: """{get_content_from_message(message)}"""' for message in recent_messages
        )

        prompt = f'History:\n{chat_history}\nQuery: {user_message}' if chat_history else f'Query: {user_message}'

        return {
            'model': task_model_id,
            'messages': [
                {'role': 'system', 'content': content},
                {'role': 'user', 'content': prompt},
            ],
            'stream': False,
            'metadata': {'task': str(TASKS.FUNCTION_CALLING)},
        }

    event_caller = extra_params['__event_call__']
    event_emitter = extra_params['__event_emitter__']
    metadata = extra_params['__metadata__']

    # One batched SELECT instead of four sequential round trips.
    task_config = await Config.get_many(
        'task.model.default',
        'task.model.external',
        'task.tools.prompt_template',
    )
    task_model_id = get_task_model_id(
        body['model'],
        task_config.get('task.model.default'),
        task_config.get('task.model.external'),
        models,
    )

    skip_files = False
    sources = []

    specs = [tool['spec'] for tool in tools.values()]
    tools_specs = JSONCodec.dumps(specs, ensure_ascii=False)

    tools_prompt_template = task_config.get('task.tools.prompt_template')
    if tools_prompt_template != '':
        template = tools_prompt_template
    else:
        template = DEFAULT_TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE

    tools_function_calling_prompt = tools_function_calling_generation_template(template, tools_specs)
    payload = get_tools_function_calling_payload(body['messages'], task_model_id, tools_function_calling_prompt)

    try:
        response = await generate_chat_completion(request, form_data=payload, user=user)
        log.debug('response=%r', response)
        content = await get_content_from_response(response)
        log.debug('content=%r', content)

        if not content:
            return body, {}

        try:
            content = content[content.find('{') : content.rfind('}') + 1]
            if not content:
                raise Exception('No JSON object found in the response')

            result = JSONCodec.loads(content)

            async def tool_call_handler(tool_call):
                nonlocal skip_files

                log.debug('tool_call=%r', tool_call)

                tool_function_name = tool_call.get('name', None)
                if tool_function_name not in tools:
                    log.warning(f'Tool "{tool_function_name}" not found')
                    return

                tool_function_params = tool_call.get('parameters', {})

                tool = None
                tool_type = ''
                direct_tool = False

                try:
                    tool = tools[tool_function_name]
                    tool_type = tool.get('type', '')
                    direct_tool = tool.get('direct', False)

                    spec = tool.get('spec', {})
                    allowed_params = spec.get('parameters', {}).get('properties', {}).keys()
                    tool_function_params = {k: v for k, v in tool_function_params.items() if k in allowed_params}

                    if tool.get('direct', False):
                        tool_result = await event_caller(
                            {
                                'type': 'execute:tool',
                                'data': {
                                    'id': str(uuid4()),
                                    'name': tool_function_name,
                                    'params': tool_function_params,
                                    'server': tool.get('server', {}),
                                    'session_id': metadata.get('session_id', None),
                                },
                            }
                        )
                    else:
                        tool_function = tool['callable']
                        tool_result = await tool_function(**tool_function_params)

                except Exception as e:
                    tool_result = {'error': str(e)}

                tool_result, tool_result_files, tool_result_embeds = await process_tool_result(
                    request,
                    tool_function_name,
                    tool_result,
                    tool_type,
                    direct_tool,
                    metadata,
                    user,
                )

                if event_emitter:
                    await terminal_event_handler(
                        tool_function_name,
                        tool_function_params,
                        tool_result,
                        event_emitter,
                    )

                    if tool_result_files:
                        await event_emitter(
                            {
                                'type': 'files',
                                'data': {
                                    'files': tool_result_files,
                                },
                            }
                        )

                    if tool_result_embeds:
                        await event_emitter(
                            {
                                'type': 'embeds',
                                'data': {
                                    'embeds': tool_result_embeds,
                                },
                            }
                        )

                if tool_result:
                    tool = tools[tool_function_name]
                    tool_id = tool.get('tool_id', '')

                    tool_name = f'{tool_id}/{tool_function_name}' if tool_id else f'{tool_function_name}'

                    # Citation is enabled for this tool
                    sources.append(
                        {
                            'source': {
                                'name': (f'{tool_name}'),
                            },
                            'document': [str(tool_result)],
                            'metadata': [
                                {
                                    'source': (f'{tool_name}'),
                                    'parameters': tool_function_params,
                                }
                            ],
                            'tool_result': True,
                        }
                    )

                    if tools[tool_function_name].get('metadata', {}).get('file_handler', False):
                        skip_files = True

            # check if "tool_calls" in result
            if result.get('tool_calls'):
                for tool_call in result.get('tool_calls'):
                    await tool_call_handler(tool_call)
            else:
                await tool_call_handler(result)

        except Exception as e:
            log.debug('Error: %s', e)
            content = None
    except Exception as e:
        log.debug('Error: %s', e)
        content = None

    log.debug('tool_contexts: %s', sources)

    if skip_files and 'files' in body.get('metadata', {}):
        del body['metadata']['files']

    return body, {'sources': sources}


async def chat_web_search_handler(request: Request, form_data: dict, extra_params: dict, user):
    event_emitter = extra_params['__event_emitter__']
    await event_emitter(
        {
            'type': 'status',
            'data': {
                'action': 'web_search',
                'description': 'Searching the web',
                'done': False,
            },
        }
    )

    messages = form_data['messages']
    user_message = get_last_user_message(messages)

    queries = []
    try:
        res = await generate_queries(
            request,
            {
                'model': form_data['model'],
                'messages': messages,
                'prompt': user_message,
                'type': 'web_search',
                'chat_id': extra_params.get('__chat_id__'),
            },
            user,
        )

        # generate_queries returns a JSONResponse on error (e.g. model not
        # found, chat completion failure).  Extract the error detail and
        # re-raise so the outer except block falls back to using the raw
        # user message as the search query.
        if isinstance(res, JSONResponse):
            try:
                error_body = JSONCodec.loads(res.body)
                detail = error_body.get('detail', 'Query generation failed')
            except Exception:
                detail = 'Query generation failed'
            raise Exception(detail)

        response = res['choices'][0]['message']['content']

        try:
            bracket_start = response.rfind('{')
            bracket_end = response.rfind('}') + 1

            if bracket_start == -1 or bracket_end == -1:
                raise Exception('No JSON object found in the response')

            response = response[bracket_start:bracket_end]
            queries = JSONCodec.loads(response)
            queries = queries.get('queries', [])
        except Exception:
            queries = [response]

        if ENABLE_QUERIES_CACHE:
            request.state.cached_queries = queries

    except Exception as e:
        log.exception(e)
        queries = [user_message or '']

    # Check if generated queries are empty
    if len(queries) == 1 and queries[0].strip() == '':
        queries = [user_message or '']

    # Check if queries are not found
    if len(queries) == 0:
        await event_emitter(
            {
                'type': 'status',
                'data': {
                    'action': 'web_search',
                    'description': 'No search query generated',
                    'done': True,
                },
            }
        )
        return form_data

    await event_emitter(
        {
            'type': 'status',
            'data': {
                'action': 'web_search_queries_generated',
                'queries': queries,
                'done': False,
            },
        }
    )

    try:
        results = await process_web_search(
            request,
            SearchForm(queries=queries),
            user=user,
        )

        if results:
            files = form_data.get('files', [])

            if collection_names := results.get('collection_names'):
                for collection_name in collection_names:
                    files.append(
                        {
                            'collection_name': collection_name,
                            'name': ', '.join(queries),
                            'type': 'web_search',
                            'urls': results['filenames'],
                            'queries': queries,
                        }
                    )
            elif results.get('docs'):
                # Invoked when bypass embedding and retrieval is set to True
                docs = results['docs']
                files.append(
                    {
                        'docs': docs,
                        'name': ', '.join(queries),
                        'type': 'web_search',
                        'urls': results['filenames'],
                        'queries': queries,
                    }
                )

            form_data['files'] = files

            await event_emitter(
                {
                    'type': 'status',
                    'data': {
                        'action': 'web_search',
                        'description': 'Searched {{count}} sites',
                        'urls': results['filenames'],
                        'items': results.get('items', []),
                        'done': True,
                    },
                }
            )
        else:
            await event_emitter(
                {
                    'type': 'status',
                    'data': {
                        'action': 'web_search',
                        'description': 'No search results found',
                        'done': True,
                        'error': True,
                    },
                }
            )

    except Exception as e:
        log.exception(e)
        detail = e.detail if isinstance(e, HTTPException) else None
        await event_emitter(
            {
                'type': 'status',
                'data': {
                    'action': 'web_search',
                    'description': (str(detail) if detail else 'An error occurred while searching the web'),
                    'queries': queries,
                    'done': True,
                    'error': True,
                },
            }
        )

    return form_data


def get_images_from_messages(message_list):
    images = []

    for message in reversed(message_list):
        message_images = []
        for file in message.get('files', []):
            if file.get('type') == 'image':
                message_images.append(file.get('url'))
            elif file.get('content_type', '').startswith('image/'):
                message_images.append(file.get('url'))

        if message_images:
            images.append(message_images)

    return images


async def get_image_urls(delta_images, request, metadata, user) -> list[str]:
    if not isinstance(delta_images, list):
        return []

    image_urls = []
    for img in delta_images:
        if not isinstance(img, dict) or img.get('type') != 'image_url':
            continue

        url = img.get('image_url', {}).get('url')
        if not url:
            continue

        if url.startswith('data:image/png;base64'):
            url = await get_image_url_from_base64(request, url, metadata, user)

        image_urls.append(url)

    return image_urls


async def add_file_context(messages: list, chat_id: str, user) -> list:
    """
    Add file URLs to messages for native function calling.
    """
    if not is_saved_chat_id(chat_id):
        return messages

    chat = await Chats.get_chat_by_id_and_user_id(chat_id, user.id)
    if not chat:
        return messages

    history = chat.chat.get('history', {})
    stored_messages = get_message_list(history.get('messages', {}), history.get('currentId'))

    def format_file_tag(file):
        # Every file reaching here has a url or a chat id, so id is always set.
        attrs = f'type="{file.get("type", "file")}" id="{file.get("id") or file.get("url")}"'
        if file.get('url'):
            attrs += f' url="{file["url"]}"'
        if file.get('content_type'):
            attrs += f' content_type="{file["content_type"]}"'
        if file.get('name'):
            attrs += f' name="{file["name"]}"'
        return f'<file {attrs}/>'

    # Pair only user-role messages from both lists to avoid misalignment.
    # After process_messages_with_output(), assistant messages with tool calls
    # are expanded into multiple messages (assistant + tool results), making
    # the payload message list longer than the stored message list. A naive
    # positional zip() would pair user messages with wrong stored messages,
    # causing later images to lose their file context (see #21878).
    user_messages = [m for m in messages if m.get('role') == 'user']
    stored_user_messages = [m for m in stored_messages if m.get('role') == 'user']

    for message, stored_message in zip(user_messages, stored_user_messages, strict=False):
        # Chat references carry no url - they are addressed by id via view_chat.
        attached_files = [
            file
            for file in stored_message.get('files', [])
            if (file.get('url') and not file.get('url').startswith('data:'))
            or (file.get('type') == 'chat' and file.get('id'))
        ]
        if not attached_files:
            continue

        file_tags = [format_file_tag(file) for file in attached_files]
        file_context = '<attached_files>\n' + '\n'.join(file_tags) + '\n</attached_files>\n\n'

        content = message.get('content', '')
        if isinstance(content, list):
            message['content'] = [{'type': 'text', 'text': file_context}] + content
        else:
            message['content'] = file_context + content

    return messages


async def chat_image_generation_handler(request: Request, form_data: dict, extra_params: dict, user):
    metadata = extra_params.get('__metadata__', {})
    chat_id = metadata.get('chat_id', None)
    __event_emitter__ = extra_params.get('__event_emitter__', None)

    if not chat_id or not isinstance(chat_id, str) or not __event_emitter__:
        return form_data

    is_channel_chat = chat_id.startswith('channel:')
    image_metadata = {
        'message_id': metadata.get('message_id', None),
        **({'channel_id': chat_id.removeprefix('channel:')} if is_channel_chat else {'chat_id': chat_id}),
    }

    if not is_saved_chat_id(chat_id):
        message_list = form_data.get('messages', [])
    else:
        chat = await Chats.get_chat_by_id_and_user_id(chat_id, user.id)

        messages_map = chat.chat.get('history', {}).get('messages', {})
        message_id = chat.chat.get('history', {}).get('currentId')
        message_list = get_message_list(messages_map, message_id)

    user_message = get_last_user_message(message_list)

    prompt = user_message
    message_images = get_images_from_messages(message_list)

    # Limit to first 2 sets of images
    # We may want to change this in the future to allow more images
    input_images = []
    for idx, images in enumerate(message_images):
        if idx >= 2:
            break
        for image in images:
            input_images.append(image)

    # Called directly, bypassing the /images routes that enforce these switches.
    editing = len(input_images) > 0 and await Config.get('images.edit.enable')
    if not editing and not await Config.get('image_generation.enable'):
        return form_data

    if is_saved_chat_id(chat_id):
        await __event_emitter__(
            {
                'type': 'status',
                'data': {'description': 'Creating image', 'done': False},
            }
        )

    system_message_content = ''

    if editing:
        # Edit image(s)
        try:
            images = await image_edits(
                request=request,
                form_data=EditImageForm(**{'prompt': prompt, 'image': input_images}),
                metadata=image_metadata,
                user=user,
            )

            await __event_emitter__(
                {
                    'type': 'status',
                    'data': {'description': 'Image created', 'done': True},
                }
            )

            await __event_emitter__(
                {
                    'type': 'files',
                    'data': {
                        'files': [
                            {
                                'type': 'image',
                                **image,
                            }
                            for image in images
                        ]
                    },
                }
            )

            system_message_content = (
                '<context>The requested image has been edited and created and is now being shown to the user. Let '
                'them know that it has been generated.</context>'
            )
        except Exception as e:
            log.debug(e)

            error_message = ''
            if isinstance(e, HTTPException):
                if e.detail and isinstance(e.detail, dict):
                    error_message = e.detail.get('message', str(e.detail))
                else:
                    error_message = str(e.detail)

            await __event_emitter__(
                {
                    'type': 'status',
                    'data': {
                        'description': 'An error occurred while generating an image',
                        'done': True,
                    },
                }
            )

            system_message_content = (
                '<context>Image generation was attempted but failed. The system is currently unable to generate the '
                f'image. Tell the user that the following error occurred: {error_message}</context>'
            )

    elif not await Config.get('image_generation.enable'):
        await __event_emitter__(
            {
                'type': 'status',
                'data': {
                    'description': 'Image generation is disabled',
                    'done': True,
                },
            }
        )

        system_message_content = (
            '<context>Image generation was requested but the feature is currently disabled by the administrator, so '
            'no image was created. Let the user know that image generation is currently unavailable.</context>'
        )

    else:
        # Create image(s)
        if await Config.get('image_generation.prompt.enable'):
            try:
                res = await generate_image_prompt(
                    request,
                    {
                        'model': form_data['model'],
                        'messages': form_data['messages'],
                        'chat_id': metadata.get('chat_id'),
                    },
                    user,
                )

                # Handle JSONResponse from error paths
                if isinstance(res, JSONResponse):
                    try:
                        error_body = JSONCodec.loads(res.body)
                        detail = error_body.get('detail', 'Image prompt generation failed')
                    except Exception:
                        detail = 'Image prompt generation failed'
                    raise Exception(detail)

                response = res['choices'][0]['message']['content']

                try:
                    bracket_start = response.rfind('{')
                    bracket_end = response.rfind('}') + 1

                    if bracket_start == -1 or bracket_end == -1:
                        raise Exception('No JSON object found in the response')

                    response = response[bracket_start:bracket_end]
                    response = JSONCodec.loads(response)
                    prompt = response.get('prompt', [])
                except Exception:
                    prompt = user_message

            except Exception as e:
                log.exception(e)
                prompt = user_message

        try:
            images = await image_generations(
                request=request,
                form_data=CreateImageForm(**{'prompt': prompt}),
                metadata=image_metadata,
                user=user,
            )

            await __event_emitter__(
                {
                    'type': 'status',
                    'data': {'description': 'Image created', 'done': True},
                }
            )

            await __event_emitter__(
                {
                    'type': 'files',
                    'data': {
                        'files': [
                            {
                                'type': 'image',
                                **image,
                            }
                            for image in images
                        ]
                    },
                }
            )

            system_message_content = (
                '<context>The requested image has been created by the system successfully and is now being shown to '
                'the user. Let the user know that the image they requested has been generated and is now shown in '
                'the chat.</context>'
            )
        except Exception as e:
            log.debug(e)

            error_message = ''
            if isinstance(e, HTTPException):
                if e.detail and isinstance(e.detail, dict):
                    error_message = e.detail.get('message', str(e.detail))
                else:
                    error_message = str(e.detail)

            await __event_emitter__(
                {
                    'type': 'status',
                    'data': {
                        'description': 'An error occurred while generating an image',
                        'done': True,
                    },
                }
            )

            system_message_content = (
                '<context>Image generation was attempted but failed because of an error. The system is currently '
                'unable to generate the image. Tell the user that the following error occurred: '
                f'{error_message}</context>'
            )

    if system_message_content:
        form_data['messages'] = add_or_update_system_message(system_message_content, form_data['messages'])

    return form_data


async def chat_completion_files_handler(
    request: Request, body: dict, extra_params: dict, user: UserModel
) -> tuple[dict, dict[str, list]]:
    __event_emitter__ = extra_params['__event_emitter__']
    sources = []

    files = [item for item in (body.get('metadata', {}).get('files', None) or []) if item.get('type') != 'filesystem']
    if files:
        # Check if all files are in full context mode
        all_full_context = all(item.get('context') == 'full' for item in files)

        queries = []
        if not all_full_context:
            try:
                queries_response = await generate_queries(
                    request,
                    {
                        'model': body['model'],
                        'messages': body['messages'],
                        'type': 'retrieval',
                        'chat_id': body.get('metadata', {}).get('chat_id'),
                    },
                    user,
                )
                queries_response = queries_response['choices'][0]['message']['content']

                try:
                    bracket_start = queries_response.rfind('{')
                    bracket_end = queries_response.rfind('}') + 1

                    if bracket_start == -1 or bracket_end == -1:
                        raise Exception('No JSON object found in the response')

                    queries_response = queries_response[bracket_start:bracket_end]
                    queries_response = JSONCodec.loads(queries_response)
                except Exception:
                    queries_response = {'queries': [queries_response]}

                queries = queries_response.get('queries', [])
            except Exception:
                pass

            await __event_emitter__(
                {
                    'type': 'status',
                    'data': {
                        'action': 'queries_generated',
                        'queries': queries,
                        'done': False,
                    },
                }
            )

        if len(queries) == 0:
            queries = [get_last_user_message(body['messages']) or '']

        try:
            # One batched SELECT instead of six sequential round trips.
            rag_config = await Config.get_many(
                'rag.top_k',
                'rag.top_k_reranker',
                'rag.relevance_threshold',
                'rag.hybrid_bm25_weight',
                'rag.enable_hybrid_search',
                'rag.full_context',
            )
            # Directly await async get_sources_from_items (no thread needed - fully async now)
            sources = await get_sources_from_items(
                request=request,
                items=files,
                queries=queries,
                embedding_function=lambda query, prefix: request.app.state.EMBEDDING_FUNCTION(
                    query, prefix=prefix, user=user
                ),
                k=rag_config.get('rag.top_k'),
                reranking_function=(
                    (lambda query, documents: request.app.state.RERANKING_FUNCTION(query, documents, user=user))
                    if request.app.state.RERANKING_FUNCTION
                    else None
                ),
                k_reranker=rag_config.get('rag.top_k_reranker'),
                r=rag_config.get('rag.relevance_threshold'),
                hybrid_bm25_weight=rag_config.get('rag.hybrid_bm25_weight'),
                hybrid_search=rag_config.get('rag.enable_hybrid_search'),
                full_context=all_full_context or rag_config.get('rag.full_context'),
                user=user,
            )
        except Exception as e:
            log.exception(e)

        log.debug('rag_contexts:sources: %s', sources)

        unique_ids = set()
        for source in sources or []:
            if not source or len(source.keys()) == 0:
                continue

            documents = source.get('document') or []
            metadatas = source.get('metadata') or []
            src_info = source.get('source') or {}

            for index, _ in enumerate(documents):
                metadata = metadatas[index] if index < len(metadatas) else None
                _id = (metadata or {}).get('source') or (src_info or {}).get('id') or 'N/A'
                unique_ids.add(_id)

        sources_count = len(unique_ids)
        await __event_emitter__(
            {
                'type': 'status',
                'data': {
                    'action': 'sources_retrieved',
                    'count': sources_count,
                    'done': True,
                },
            }
        )

    return body, {'sources': sources}


async def convert_url_images_to_base64(form_data, user=None):
    messages = form_data.get('messages', [])

    for message in messages:
        content = message.get('content')
        if not isinstance(content, list):
            continue

        new_content = []

        for item in content:
            if not isinstance(item, dict) or item.get('type') != 'image_url':
                new_content.append(item)
                continue

            image_url_data = item.get('image_url', {})
            if isinstance(image_url_data, dict):
                image_url = image_url_data.get('url') or ''
            elif isinstance(image_url_data, str):
                image_url = image_url_data
            else:
                image_url = ''
            if image_url.startswith('data:image/'):
                new_content.append(item)
                continue

            try:
                base64_data = await get_image_base64_from_url(image_url, user=user)
                if base64_data:
                    image_url_payload = {'url': base64_data}
                    if isinstance(image_url_data, dict) and image_url_data.get('detail'):
                        image_url_payload['detail'] = image_url_data['detail']
                    new_content.append(
                        {
                            'type': 'image_url',
                            'image_url': image_url_payload,
                        }
                    )
                else:
                    new_content.append(item)
            except Exception as e:
                log.debug('Error converting image URL to base64: %s', e)
                new_content.append(item)

        message['content'] = new_content

    return form_data


MESSAGE_REPLAY_KEYS = ('id', 'role', 'content', 'output', 'files', 'contextSummary', 'usage', 'model')


async def load_messages_from_db(chat_id: str, message_id: str) -> list[dict] | None:
    """
    Load the message chain from DB up to message_id,
    keeping only fields needed to rebuild the LLM payload.
    """
    messages_map = await Chats.get_messages_map_by_chat_id(chat_id)
    if not messages_map:
        return None

    db_messages = get_message_list(messages_map, message_id)
    if not db_messages:
        return None

    return [{k: v for k, v in msg.items() if k in MESSAGE_REPLAY_KEYS} for msg in db_messages]


def get_reasoning_format(model: dict) -> str | None:
    """
    Determine how reasoning should be included in reconstructed messages.

    Returns:
        'thinking': Ollama expects reasoning in the native thinking field.
        'think_tags': wrap reasoning in <think> tags inside content.
        'reasoning_content': llama.cpp supports reasoning_content as a top-level field.
        None: skip reasoning (safe default for strict providers).
    """
    provider = model.get('provider', '')
    if model.get('owned_by') == 'ollama':
        return 'thinking'
    if provider == 'llama.cpp':
        return 'reasoning_content'
    return None


def strip_reasoning_details(output: list) -> list:
    return [
        {key: value for key, value in item.items() if key != 'reasoning_details'} if isinstance(item, dict) else item
        for item in output
    ]


def process_messages_with_output(
    messages: list[dict],
    reasoning_format: str | None = None,
) -> list[dict]:
    """
    Process messages with OR-aligned output items for LLM consumption.

    For assistant messages with 'output' field, produces properly formatted
    OpenAI-style messages (tool_calls + tool results). Strips 'output' before LLM.
    """
    processed = []

    for message in messages:
        if message.get('role') == 'assistant' and message.get('output'):
            # Use output items for clean OpenAI-format messages
            output_messages = convert_output_to_messages(
                message['output'],
                raw=True,
                reasoning_format=reasoning_format,
                flatten_tool_images=True,
            )
            if output_messages:
                processed.extend(output_messages)
                continue

        clean_message = dict(message)
        for key in ('id', 'files', 'output', 'model', 'contextSummary', 'context_summary', 'usage'):
            clean_message.pop(key, None)
        processed.append(clean_message)

    return processed


def sanitize_tool_pairs(messages: list[dict]) -> list[dict]:
    tool_result_ids = {
        message.get('tool_call_id')
        for message in messages
        if message.get('role') == 'tool' and message.get('tool_call_id')
    }

    tool_call_ids = {
        tool_call.get('id')
        for message in messages
        for tool_call in (message.get('tool_calls') or [])
        if message.get('role') == 'assistant' and tool_call.get('id')
    }

    sanitized = []
    for message in messages:
        if message.get('role') == 'assistant' and message.get('tool_calls'):
            kept = [
                tool_call for tool_call in message.get('tool_calls') or [] if tool_call.get('id') in tool_result_ids
            ]
            if kept:
                sanitized.append({**message, 'tool_calls': kept})
            else:
                clean = dict(message)
                clean.pop('tool_calls', None)
                clean.pop('reasoning_items', None)
                if clean.get('content'):
                    sanitized.append(clean)
        elif message.get('role') != 'tool' or message.get('tool_call_id') in tool_call_ids:
            sanitized.append(message)

    return sanitized


# Ids are validated as [a-z0-9_-]+ on create; matching that keeps ordinary "<$..." text intact.
# Labels exclude `<` as well as `>`, so a scan stops at the next tag: with
# `[^>]*`, a message of many unclosed `<$a|` took quadratic time.
SKILL_MENTION_RE = re.compile(r'<(?:\$([a-z0-9_-]+)(?:\|[^<>]*)?|/([a-z0-9_-]+)\|[^<>]*)>')


def _get_text_parts(message: dict) -> list[str]:
    """Return all text segments from a message's content."""
    content = message.get('content')
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [p.get('text', '') for p in content if isinstance(p, dict) and p.get('type') == 'text']
    return []


def extract_skill_ids_from_messages(messages: list[dict]) -> set[str]:
    """Extract skill IDs from <$skillId|label> and </skillId|label> mention tags."""
    ids: set[str] = set()
    for message in messages:
        for text in _get_text_parts(message):
            ids.update(m.group(1) or m.group(2) for m in SKILL_MENTION_RE.finditer(text))
    return ids


SKILL_MENTION_STRIP_RE = re.compile(r'<(?:\$[a-z0-9_-]+(?:\|([^<>]*))?|/[a-z0-9_-]+\|([^<>]*))>')


def strip_skill_mentions(messages: list[dict]) -> None:
    """Replace <$skillId|label> and </skillId|label> mention tags with the label in-place."""

    def label(match):
        return match.group(1) or match.group(2) or ''

    for message in messages:
        content = message.get('content')
        if isinstance(content, str) and SKILL_MENTION_STRIP_RE.search(content):
            message['content'] = SKILL_MENTION_STRIP_RE.sub(label, content).strip()
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'text':
                    text = part.get('text', '')
                    if SKILL_MENTION_STRIP_RE.search(text):
                        part['text'] = SKILL_MENTION_STRIP_RE.sub(label, text).strip()


async def connect_mcp_server(
    request,
    server_id: str,
    user,
    metadata: dict,
    extra_params: dict,
) -> tuple[MCPClient, list[dict]] | None:
    """Resolve an MCP server connection, authenticate, and return (client, tool_specs).

    Returns None if the server is not found or access is denied.
    """
    mcp_server_connection = None
    for server_connection in await Config.get('tool_server.connections', []):
        if server_connection.get('type', '') == 'mcp' and (server_connection.get('info') or {}).get('id') == server_id:
            mcp_server_connection = server_connection
            break

    if not mcp_server_connection:
        log.error(f'MCP server with id {server_id} not found')
        return None

    if not await has_connection_access(user, mcp_server_connection):
        log.warning(f'Access denied to MCP server {server_id} for user {user.id}')
        return None

    headers, _ = await build_tool_server_headers(
        mcp_server_connection,
        request,
        user,
        server_id=server_id,
        metadata=metadata,
        extra_params=extra_params,
    )

    client = MCPClient()
    await client.connect(
        url=mcp_server_connection.get('url', ''),
        headers=headers if headers else None,
    )

    # The caller registers the client for cleanup only once this returns, so close it here on failure.
    try:
        function_name_filter_list = mcp_server_connection.get('config', {}).get('function_name_filter_list', '')
        if isinstance(function_name_filter_list, str):
            function_name_filter_list = function_name_filter_list.split(',')

        tool_specs = await client.list_tool_specs()
        if function_name_filter_list:
            tool_specs = [spec for spec in tool_specs if is_string_allowed(spec['name'], function_name_filter_list)]
    except BaseException:
        # Cancellation too (a chat stopped while tools/list is pending): disconnect()
        # runs in this same task and re-raises the task's own cancellation.
        await client.disconnect()
        raise

    return client, tool_specs


def _resolve_arena_model(request, form_data: dict, metadata: dict, model: dict) -> dict:
    """An arena model's pick of a sub-model, so downstream processing uses its settings, not the wrapper's."""
    arena_model_ids = model.get('info', {}).get('meta', {}).get('model_ids')
    arena_filter_mode = model.get('info', {}).get('meta', {}).get('filter_mode')
    if arena_model_ids and arena_filter_mode == 'exclude':
        arena_model_ids = [
            available_model['id']
            for available_model in request.app.state.MODELS.values()
            if available_model.get('owned_by') != 'arena' and available_model['id'] not in arena_model_ids
        ]

    if not (isinstance(arena_model_ids, list) and arena_model_ids):
        arena_model_ids = [
            available_model['id']
            for available_model in request.app.state.MODELS.values()
            if available_model.get('owned_by') != 'arena'
        ]
    selected_model_id = random.choice(arena_model_ids)

    selected_model = request.app.state.MODELS.get(selected_model_id)
    if not selected_model:
        return model
    form_data['model'] = selected_model_id
    metadata['selected_model_id'] = selected_model_id
    return selected_model


def _inline_message_images(messages: list[dict]) -> None:
    """Put a user message's image files into its content as image_url parts (mirrors frontend logic)."""
    for message in messages:
        image_files = [
            f
            for f in message.get('files', [])
            if f.get('type') == 'image' or (f.get('content_type') or '').startswith('image/')
        ]
        if message.get('role') == 'user' and image_files:
            text_content = message.get('content', '')
            if isinstance(text_content, str):
                message['content'] = [
                    {'type': 'text', 'text': text_content},
                    *[
                        {
                            'type': 'image_url',
                            'image_url': {'url': f['url']},
                        }
                        for f in image_files
                        if f.get('url')
                    ],
                ]
        # Strip files field — it's been incorporated into content
        message.pop('files', None)


async def _replay_history_from_db(form_data: dict, metadata: dict, chat_id: str, user_message_id: str) -> None:
    """Use the stored history when there is one: it keeps the structured 'output' items the frontend strips."""
    db_messages = await load_messages_from_db(chat_id, user_message_id)
    if not db_messages:
        return

    # Continue: frontend sends assistant_message_id when continuing
    # an existing response. Load its content so the LLM sees prior output.
    assistant_message_id = metadata.get('assistant_message_id')
    if assistant_message_id:
        assistant_message = await Chats.get_message_by_id_and_message_id(chat_id, assistant_message_id)
        if assistant_message and (assistant_message.get('content') or assistant_message.get('output')):
            db_messages.append({k: v for k, v in assistant_message.items() if k in MESSAGE_REPLAY_KEYS})

    system_message = get_system_message(form_data.get('messages', []))
    form_data['messages'] = [system_message, *db_messages] if system_message else db_messages
    _inline_message_images(form_data['messages'])


async def _compact_history(request, user, form_data: dict, metadata: dict) -> None:
    if getattr(request.state, 'direct', False) and hasattr(request.state, 'model'):
        compaction_models = {
            **dict(request.app.state.MODELS.items()),
            request.state.model['id']: request.state.model,
        }
    else:
        compaction_models = request.app.state.MODELS

    system_message = get_system_message(form_data.get('messages', []))
    system_prompt = get_content_from_message(system_message) if system_message else ''

    try:
        form_data['messages'], context_summary, _ = await compact_messages_for_request(
            request,
            user,
            form_data.get('messages', []),
            metadata,
            form_data.get('model'),
            compaction_models,
            system_prompt,
        )
        if context_summary:
            form_data['messages'] = add_or_update_system_message(
                f'[CONVERSATION SUMMARY]\n{context_summary}',
                form_data['messages'],
                append=True,
            )
    except Exception:
        log.exception('Context compaction failed; continuing with full chat history')


def _messages_for_model(messages: list[dict], model: dict) -> list[dict]:
    """Replayable messages: OR-aligned output items, reasoning kept only for this model's own output."""
    for message in messages:
        output = message.get('output')
        # reasoning_details can be model/provider-bound, so only replay them
        # for output produced by the same model.
        if message.get('role') == 'assistant' and message.get('model') != model['id'] and isinstance(output, list):
            message['output'] = strip_reasoning_details(output)

    messages = process_messages_with_output(messages, reasoning_format=get_reasoning_format(model))
    return sanitize_tool_pairs(messages)


async def _payload_extra_params(request, user, metadata: dict, model: dict, event_emitter, event_caller) -> dict:
    return {
        '__event_emitter__': event_emitter,
        '__event_call__': event_caller,
        '__user__': user.model_dump() if isinstance(user, UserModel) else {},
        '__metadata__': metadata,
        '__oauth_token__': await get_system_oauth_token(request, user),
        '__request__': request,
        '__model__': model,
        '__chat_id__': metadata.get('chat_id'),
        '__message_id__': metadata.get('message_id'),
    }


def _payload_models(request) -> dict:
    # A direct connection only knows its own model.
    if getattr(request.state, 'direct', False) and hasattr(request.state, 'model'):
        return {
            request.state.model['id']: request.state.model,
        }
    return request.app.state.MODELS


async def _apply_folder(form_data: dict, metadata: dict, user) -> dict:
    """A chat's folder ("Project") adds its system prompt and its files or folder knowledge."""
    # Uses lightweight column query — only fetches folder_id, not the full chat JSON blob
    chat_id = metadata.get('chat_id', None)
    folder_id = None
    if user and is_saved_chat_id(chat_id):
        folder_id = await Chats.get_chat_folder_id(chat_id, user.id)

    # Fallback: use folder_id from metadata (temporary chats have no DB record)
    if not folder_id:
        folder_id = metadata.get('folder_id', None)

    if not (folder_id and user):
        return form_data

    folder = await Folders.get_folder_by_id(folder_id)
    if folder and user.role != 'admin' and not await has_folder_access(user.id, folder, 'read', db=None):
        folder = None

    if folder and folder.data:
        if 'system_prompt' in folder.data:
            form_data = await apply_system_prompt_to_body(folder.data['system_prompt'], form_data, metadata, user)
        if 'files' in folder.data:
            if metadata.get('params', {}).get('function_calling') == 'legacy':
                form_data['files'] = [
                    {'type': 'folder', 'id': folder.id},
                    *form_data.get('files', []),
                ]
            else:
                # Native FC: skip RAG injection, builtin tools
                # will read folder knowledge from metadata.
                metadata['folder_knowledge'] = await get_owner_accessible_folder_files(folder)
    return form_data


def _legacy_knowledge_file(item: dict) -> dict:
    if item.get('collection_name'):
        return {
            'id': item.get('collection_name'),
            'name': item.get('name'),
            'legacy': True,
        }
    if item.get('collection_names'):
        return {
            'name': item.get('name'),
            'type': 'collection',
            'collection_names': item.get('collection_names'),
            'legacy': True,
        }
    return item


async def _search_model_knowledge(form_data: dict, model_knowledge: list, event_emitter, user_message) -> None:
    """Legacy function calling: the model's knowledge joins the files RAG searches."""
    await event_emitter(
        {
            'type': 'status',
            'data': {
                'action': 'knowledge_search',
                'query': user_message,
                'done': False,
            },
        }
    )

    files = form_data.get('files', [])
    files.extend(_legacy_knowledge_file(item) for item in model_knowledge)
    form_data['files'] = files


async def _run_inlet_filters(request, form_data: dict, user, model: dict, metadata: dict, models, extra_params):
    """The pipeline inlet, then the inlet filter functions; returns the form data and the filters to reuse."""
    form_data = await process_pipeline_inlet_filter(request, form_data, user, models)

    filter_functions = []
    filter_context = get_filter_context(request) if ENABLE_PLUGINS else None
    if ENABLE_PLUGINS:
        try:
            filter_functions = await get_filter_functions(request, model, metadata.get('filter_ids', []))

            form_data, flags = await process_filter_functions(
                request=request,
                filter_context=filter_context,
                filter_functions=filter_functions,
                filter_type='inlet',
                form_data=form_data,
                extra_params=extra_params,
            )
        except Exception as e:
            raise Exception(f'{e}')
    return form_data, filter_functions, filter_context


async def _may_use_feature(user, feature_key: str) -> bool:
    # features is client-supplied; re-check the permission the native FC path enforces.
    return getattr(user, 'role', None) == 'admin' or await has_permission(
        getattr(user, 'id', ''),
        f'features.{feature_key}',
        await Config.get('user.permissions'),
    )


async def _apply_code_interpreter(form_data: dict, metadata: dict) -> None:
    engine = await Config.get('code_interpreter.engine', 'pyodide')

    # Skip XML-tag prompt injection when native FC is enabled —
    # execute_code will be injected as a builtin tool instead
    if metadata.get('params', {}).get('function_calling') == 'legacy':
        ci_prompt_template = await Config.get('code_interpreter.prompt_template')
        prompt = ci_prompt_template if ci_prompt_template != '' else DEFAULT_CODE_INTERPRETER_PROMPT

        # Append the engine's own environment and filesystem notes
        prompt += CODE_INTERPRETER_JUPYTER_PROMPT if engine == 'jupyter' else CODE_INTERPRETER_PYODIDE_PROMPT

        form_data['messages'] = add_or_update_user_message(
            prompt,
            form_data['messages'],
        )
    else:
        # Native FC: tool docstring can't be dynamic, so inject the
        # engine's environment and filesystem notes into the system
        # message.  Appending to the system prompt (instead of the
        # user message) keeps it in the stable cached prefix so
        # providers with prefix caching don't re-bill the full
        # conversation on every turn.
        form_data['messages'] = add_or_update_system_message(
            CODE_INTERPRETER_JUPYTER_PROMPT if engine == 'jupyter' else CODE_INTERPRETER_PYODIDE_PROMPT,
            form_data['messages'],
            append=True,
        )


async def _apply_features(request, form_data: dict, user, model: dict, metadata: dict, extra_params, features: dict):
    """The chat's feature toggles: voice, memory, and (legacy FC only) web search, image generation, code."""
    is_legacy = metadata.get('params', {}).get('function_calling') == 'legacy'

    if features.get('voice') and await Config.get('task.voice.prompt.enable'):
        template = await Config.get('task.voice.prompt_template')
        if not template:
            template = DEFAULT_VOICE_MODE_PROMPT_TEMPLATE
        form_data['messages'] = add_or_update_system_message(template, form_data['messages'])

    if (
        features.get('memory')
        and await Config.get('memories.system_context.enable')
        and await _may_use_feature(user, 'memories')
    ):
        form_data = await add_memory_context(request, form_data, user, model)

    if (
        features.get('web_search')
        and await Config.get('web.search.enable')
        and await _may_use_feature(user, 'web_search')
    ):
        # Skip forced RAG web search when native FC is enabled - model can use web_search tool
        if is_legacy:
            form_data = await chat_web_search_handler(request, form_data, extra_params, user)

    # features is client-supplied; re-check the permission the direct /images routes enforce.
    if features.get('image_generation') and await _may_use_feature(user, 'image_generation'):
        # Skip forced image generation when native FC is enabled - model can use generate_image tool
        if is_legacy:
            form_data = await chat_image_generation_handler(request, form_data, extra_params, user)

    if features.get('code_interpreter'):
        await _apply_code_interpreter(form_data, metadata)
    return form_data


async def _note_chat_files(chat, user, files):
    """A note's own chat also gets the note's non-image files, when the user can read the note."""
    note_id = (chat.meta or {}).get('note_id')
    note = await Notes.get_note_by_id(note_id) if note_id else None
    if not note or not (
        user.role == 'admin'
        or note.user_id == user.id
        or await AccessGrants.has_access(
            user_id=user.id,
            resource_type='note',
            resource_id=note.id,
            permission='read',
        )
    ):
        return files
    note_files = [
        file
        for file in ((note.data or {}).get('files') or [])
        if isinstance(file, dict)
        and file.get('type') != 'image'
        and not (file.get('content_type') or '').startswith('image/')
    ]
    return [*(files or []), *note_files] if note_files else files


async def _apply_skills(form_data: dict, user, skill_ids: list, mentioned_skill_ids: set, use_builtin_tools: bool):
    """Mentioned skills (or all, without builtin tools) go in whole; the rest are listed for view_skill.

    Returns the available skills and the ids view_skill may load."""
    from local_llm.models.skills import Skills as SkillsModel

    accessible_skills = {s.id: s for s in await SkillsModel.get_skills(user_id=user.id, ids=skill_ids)}
    available_skills = [
        accessible_skills[sid] for sid in skill_ids if sid in accessible_skills and accessible_skills[sid].is_active
    ]

    view_skill_ids = []
    skill_manifest = ''
    for skill in available_skills:
        if skill.id in mentioned_skill_ids or not use_builtin_tools:
            form_data['messages'] = add_or_update_system_message(
                f'<skill name="{skill.name}">\n{skill.content}\n</skill>',
                form_data['messages'],
                append=True,
            )
        else:
            view_skill_ids.append(skill.id)
            skill_manifest += (
                f'<skill>\n<id>{skill.id}</id>\n<name>{skill.name}</name>\n'
                f'<description>{skill.description or ""}</description>\n</skill>\n'
            )

    if skill_manifest:
        form_data['messages'] = add_or_update_system_message(
            f'<available_skills>\n{skill_manifest}</available_skills>',
            form_data['messages'],
            append=True,
        )
    return available_skills, view_skill_ids


def _mcp_tool_function(client, function_name: str):
    async def tool_function(**kwargs):
        return await client.call_tool(
            function_name,
            function_args=kwargs,
        )

    return tool_function


async def _add_mcp_server_tools(
    request, server_id: str, user, metadata: dict, extra_params: dict, mcp_clients: dict, mcp_tools_dict: dict
) -> None:
    """Connect one MCP server into mcp_clients and add its tools, keyed <server>_<tool>, to mcp_tools_dict."""
    result = await connect_mcp_server(
        request,
        server_id,
        user,
        metadata,
        extra_params,
    )
    if result is None:
        return

    client, tool_specs = result
    mcp_clients[server_id] = client
    for tool_spec in tool_specs:
        mcp_tools_dict[f'{server_id}_{tool_spec["name"]}'] = {
            'spec': {
                **tool_spec,
                'name': f'{server_id}_{tool_spec["name"]}',
            },
            'callable': _mcp_tool_function(client, tool_spec['name']),
            'type': 'mcp',
            'client': client,
            'direct': False,
        }


async def _selected_tools(
    request,
    form_data: dict,
    user,
    metadata: dict,
    extra_params: dict,
    models,
    task_model_id,
    event_emitter,
    mcp_clients,
) -> dict:
    """The tools picked by id: MCP servers (server:mcp:<id>) and, with plugins on, the database's."""
    tool_ids = metadata.get('tool_ids', None)
    if not tool_ids:
        return {}

    tools_dict = {}
    mcp_tools_dict = {}
    db_tool_ids = []
    for tool_id in tool_ids:
        if tool_id.startswith('server:mcp:'):
            server_id = tool_id[len('server:mcp:') :]
            try:
                await _add_mcp_server_tools(
                    request, server_id, user, metadata, extra_params, mcp_clients, mcp_tools_dict
                )
            except Exception as e:
                log.debug(e)
                if event_emitter:
                    await event_emitter(
                        {
                            'type': 'chat:message:error',
                            'data': {'error': {'content': f"Failed to connect to MCP server '{server_id}'"}},
                        }
                    )
        elif ENABLE_PLUGINS:
            db_tool_ids.append(tool_id)

    if db_tool_ids:
        tools_dict = await get_tools(
            request,
            db_tool_ids,
            user,
            {
                **extra_params,
                '__model__': models[task_model_id],
                '__messages__': form_data['messages'],
                '__files__': metadata.get('files', []),
            },
        )

    if mcp_tools_dict:
        tools_dict = {**tools_dict, **mcp_tools_dict}
    return tools_dict


async def _add_terminal_tools(request, form_data: dict, user, model: dict, terminal_id, extra_params, tools_dict):
    # Outside the tool_ids check so system terminals work even when no other tools are selected
    terminal_capability = (model.get('info', {}).get('meta', {}).get('capabilities') or {}).get('terminal', True)
    if not (terminal_id and terminal_capability):
        return tools_dict
    try:
        terminal_result = await get_terminal_tools(
            request,
            terminal_id,
            user,
            extra_params,
        )
        if isinstance(terminal_result, tuple):
            terminal_tools, system_prompt = terminal_result
        else:
            terminal_tools = terminal_result
            system_prompt = None
        if terminal_tools:
            tools_dict = {**tools_dict, **terminal_tools}
        if system_prompt:
            form_data['messages'] = add_or_update_system_message(
                system_prompt,
                form_data['messages'],
                append=True,
            )
    except Exception as e:
        log.exception(e)
        raise HTTPException(status_code=503, detail=f'Terminal unavailable: {e}') from e
    return tools_dict


def _add_direct_server_tools(form_data: dict, direct_tool_servers, tools_dict: dict) -> None:
    # Read the servers, never pop from them: metadata['tool_servers'] is shared by every model of a
    # multi-model chat, and is saved for a tool-approval resume and handed to sub-agents.
    for tool_server in direct_tool_servers or []:
        system_prompt = tool_server.get('system_prompt')
        if system_prompt:
            form_data['messages'] = add_or_update_system_message(
                system_prompt,
                form_data['messages'],
                append=True,
            )

        server = {key: value for key, value in tool_server.items() if key not in ('system_prompt', 'specs')}
        for tool in tool_server.get('specs', []):
            tools_dict[tool['name']] = {
                'spec': tool,
                'direct': True,
                'server': server,
            }


def _attached_knowledge_tags(model: dict, metadata: dict) -> list[str]:
    from html import escape

    knowledge_tags = []
    for item in get_attached_knowledge(model, metadata):
        if not item.get('id') or not item.get('type'):
            continue
        attrs = f'type="{escape(str(item["type"]), quote=True)}" id="{escape(str(item["id"]), quote=True)}"'
        if item.get('name'):
            attrs += f' name="{escape(str(item["name"]), quote=True)}"'
        if item.get('source'):
            attrs += f' source="{escape(str(item["source"]), quote=True)}"'
        knowledge_tags.append(f'<knowledge {attrs}/>')
    return knowledge_tags


async def _add_builtin_tools(
    request,
    form_data: dict,
    user,
    model,
    metadata,
    extra_params,
    event_emitter,
    view_skill_ids,
    features,
    is_note_chat,
    tools_dict,
):
    """Native function calling from the UI: file context, the attached knowledge list and the builtin tools."""
    # Add file context to user messages
    form_data['messages'] = await add_file_context(form_data.get('messages', []), metadata.get('chat_id'), user)

    if (model.get('info', {}).get('meta', {}).get('builtinTools') or {}).get('knowledge', True):
        knowledge_tags = _attached_knowledge_tags(model, metadata)
        if knowledge_tags:
            form_data['messages'] = add_or_update_system_message(
                '<attached_knowledge>\n' + '\n'.join(knowledge_tags) + '\n</attached_knowledge>',
                form_data['messages'],
                append=True,
            )

    builtin_tools = await get_builtin_tools(
        request,
        {
            **extra_params,
            '__event_emitter__': event_emitter,
            '__skill_ids__': view_skill_ids,
        },
        features,
        model,
        is_note_chat=is_note_chat,
    )
    for name, tool_dict in builtin_tools.items():
        if name not in tools_dict:
            tools_dict[name] = tool_dict


async def _resolve_payload_tools(
    request,
    form_data: dict,
    user,
    model: dict,
    metadata: dict,
    extra_params: dict,
    models,
    task_model_id,
    event_emitter,
    terminal_id,
    use_builtin_tools: bool,
    view_skill_ids: list,
    features: dict,
    is_note_chat: bool,
    inlet_filter_tools,
    sources: list,
):
    """Resolve the chat's tools server-side and hand them to the model (native) or the tools handler (legacy)."""
    mcp_clients = {}
    try:
        tools_dict = await _selected_tools(
            request, form_data, user, metadata, extra_params, models, task_model_id, event_emitter, mcp_clients
        )
        tools_dict = await _add_terminal_tools(request, form_data, user, model, terminal_id, extra_params, tools_dict)
        # Client side tools
        _add_direct_server_tools(form_data, metadata.get('tool_servers', None), tools_dict)
    finally:
        # Even when a later step raises, so the chat's teardown disconnects every client that connected.
        if mcp_clients:
            metadata['mcp_clients'] = mcp_clients

    # Inject builtin tools for native function calling based on enabled features and model capability.
    # Only inject when the request originates from the UI (identified by session_id).
    # API callers don't expect hidden tools; they can explicitly request tools via tool_ids.
    if use_builtin_tools:
        await _add_builtin_tools(
            request,
            form_data,
            user,
            model,
            metadata,
            extra_params,
            event_emitter,
            view_skill_ids,
            features,
            is_note_chat,
            tools_dict,
        )

    if not tools_dict:
        return form_data

    # Always store resolved tools in metadata so downstream consumers
    # (e.g. pipe functions) can access all tools including MCP and builtins.
    metadata['tools'] = tools_dict

    if metadata.get('params', {}).get('function_calling') != 'legacy':
        form_data['tools'] = [{'type': 'function', 'function': tool.get('spec', {})} for tool in tools_dict.values()]
        if inlet_filter_tools:
            form_data['tools'].extend(inlet_filter_tools)
        return form_data

    # Legacy function calling: a task model picks the tool calls up front
    try:
        form_data, flags = await chat_completion_tools_handler(
            request, form_data, extra_params, user, models, tools_dict
        )
        sources.extend(flags.get('sources', []))
    except Exception as e:
        log.exception(e)
    return form_data


async def _record_system_prompt(form_data: dict, metadata: dict, user, model_system_prompt) -> None:
    """Save the pre-RAG system prompt, so the native tool call loop can restore the true original.

    A snapshot taken after file-source injection would already have the RAG template baked in."""
    system_message = get_system_message(form_data['messages'])
    # All of it: the tool loop restores this, and a list-content system
    # message once came back as its first part only (#187).
    system_content = get_text_from_message(system_message) if system_message else ''
    resolved_model_system_prompt = await resolve_system_prompt(
        model_system_prompt,
        metadata,
        user,
    )
    if resolved_model_system_prompt:
        system_content = (
            f'{resolved_model_system_prompt}\n{system_content}' if system_content else resolved_model_system_prompt
        )
    metadata['system_prompt'] = system_content or None


async def process_chat_payload(request, form_data, user, metadata, model):
    # Ensure chat_id is always a string — external API clients may omit it.
    if not isinstance(metadata.get('chat_id'), str):
        metadata['chat_id'] = ''

    # Pipeline Inlet -> Filter Inlet -> Chat Memory -> Chat Web Search -> Chat Image Generation
    # -> Chat Code Interpreter (Form Data Update) -> (Default) Chat Tools Function Calling
    # -> Chat Files

    # Arena model resolution — pick the sub-model now so all downstream
    # processing (knowledge, capabilities, tools, params) uses its settings
    # instead of the empty arena wrapper.
    if model.get('owned_by') == 'arena':
        model = _resolve_arena_model(request, form_data, metadata, model)

    # Captured before apply_params_to_form_data pops 'params'; populates metadata['system_prompt'] below
    model_system_prompt = (form_data.get('params') or {}).get('system')

    form_data = apply_params_to_form_data(form_data, model)
    log.debug('form_data: %s', form_data)

    # Guided regeneration: extract before it reaches the LLM provider
    regeneration_prompt = form_data.pop('regeneration_prompt', None)

    chat_id = metadata.get('chat_id')
    user_message_id = metadata.get('user_message_id')
    has_stored_history = is_saved_chat_id(chat_id) and user_message_id
    if has_stored_history:
        await _replay_history_from_db(form_data, metadata, chat_id, user_message_id)

    if regeneration_prompt:
        form_data['messages'].append({'role': 'user', 'content': regeneration_prompt})

    if has_stored_history:
        await _compact_history(request, user, form_data, metadata)

    form_data['messages'] = _messages_for_model(form_data.get('messages', []), model)

    system_message = get_system_message(form_data.get('messages', []))
    if system_message:  # Chat Controls/User Settings
        try:
            form_data = await apply_system_prompt_to_body(
                system_message.get('content'), form_data, metadata, user, replace=True
            )  # Required to handle system prompt variables
        except Exception:
            pass

    form_data = await convert_url_images_to_base64(form_data, user=user)

    event_emitter = await get_event_emitter(metadata)
    event_caller = await get_event_call(metadata)
    extra_params = await _payload_extra_params(request, user, metadata, model, event_emitter, event_caller)

    models = _payload_models(request)
    task_model_id = get_task_model_id(
        form_data['model'],
        await Config.get('task.model.default'),
        await Config.get('task.model.external'),
        models,
    )

    # Additional events to send to the client, and the sources cited
    events = []
    sources = []

    form_data = await _apply_folder(form_data, metadata, user)

    # Model "Knowledge" handling
    user_message = get_last_user_message(form_data['messages'])
    model_knowledge = model.get('info', {}).get('meta', {}).get('knowledge', False)
    if model_knowledge and metadata.get('params', {}).get('function_calling') == 'legacy':
        await _search_model_knowledge(form_data, model_knowledge, event_emitter, user_message)

    form_data.pop('variables', None)
    payload_tools = form_data.get('tools', None)  # snapshot before filters

    form_data, filter_functions, filter_context = await _run_inlet_filters(
        request, form_data, user, model, metadata, models, extra_params
    )

    features = form_data.pop('features', None) or {}
    extra_params['__features__'] = features
    form_data = await _apply_features(request, form_data, user, model, metadata, extra_params, features)

    tool_ids = form_data.pop('tool_ids', None)
    terminal_id = form_data.pop('terminal_id', None)
    files = form_data.pop('files', None)
    form_data.pop('folder_id', None)

    # If the original caller provided tools, use them as-is (skip resolution).
    # Otherwise, save any tools that filter inlets added for merging later.
    inlet_filter_tools = None if payload_tools is not None else form_data.get('tools', None)

    # Mentioned skills get full content; selected/default skills can be loaded through view_skill.
    mentioned_skill_ids = extract_skill_ids_from_messages(form_data.get('messages', []))
    skill_ids = sorted(
        set(form_data.pop('skill_ids', None) or [])
        | set(model.get('info', {}).get('meta', {}).get('skillIds', []))
        | mentioned_skill_ids
    )
    chat = await Chats.get_chat_by_id(metadata['chat_id']) if is_saved_chat_id(metadata.get('chat_id')) else None
    is_note_chat = bool(chat and (chat.meta or {}).get('internal') is True and (chat.meta or {}).get('type') == 'note')
    if is_note_chat:
        files = await _note_chat_files(chat, user, files)

    use_builtin_tools = is_note_chat or (
        bool(metadata.get('session_id'))
        and metadata.get('params', {}).get('function_calling') != 'legacy'
        and (model.get('info', {}).get('meta', {}).get('capabilities') or {}).get('builtin_tools', True)
    )

    available_skills, view_skill_ids = [], []
    if skill_ids:
        available_skills, view_skill_ids = await _apply_skills(
            form_data, user, skill_ids, mentioned_skill_ids, use_builtin_tools
        )

    # Strip <$skillId|label> mention tags so the model doesn't see raw markup.
    strip_skill_mentions(form_data.get('messages', []))

    prompt = get_last_user_message(form_data['messages'])

    # Guard against empty user message after skill mention stripping.
    # When a user selects a skill ($skill-name) without typing additional text,
    # the stripped result is an empty string which causes 400 errors on providers
    # that reject empty content blocks (e.g. AWS Bedrock ConverseStream).
    if not prompt or not prompt.strip():
        fallback = ', '.join(s.name for s in available_skills)
        if fallback:
            set_last_user_message_content(fallback, form_data['messages'])
            prompt = fallback
    # TODO: re-enable URL extraction from prompt
    # urls = []
    # if prompt and len(prompt or "") < 500 and (not files or len(files) == 0):
    #     urls = extract_urls(prompt)

    if files:
        # files = [*files, *[{"type": "url", "url": url, "name": url} for url in urls]]
        # Remove duplicate files based on their content
        files = list({json.dumps(f, sort_keys=True): f for f in files}.values())

    metadata.update(
        {
            'model_id': form_data.get('model'),
            'tool_ids': tool_ids,
            'skill_ids': skill_ids,
            'terminal_id': terminal_id,
            'files': files,
            'features': features,
        }
    )
    form_data['metadata'] = metadata

    # When the caller provides an explicit `tools` key in the request body,
    # skip all server-side tool resolution and pass the caller's tools through
    # unchanged.  Sending `tools: []` explicitly opts out of builtin injection.
    if payload_tools is None:
        form_data = await _resolve_payload_tools(
            request,
            form_data,
            user,
            model,
            metadata,
            extra_params,
            models,
            task_model_id,
            event_emitter,
            terminal_id,
            use_builtin_tools,
            view_skill_ids,
            features,
            is_note_chat,
            inlet_filter_tools,
            sources,
        )

    # Check if file context extraction is enabled for this model (default True)
    if (model.get('info', {}).get('meta', {}).get('capabilities') or {}).get('file_context', True):
        try:
            form_data, flags = await chat_completion_files_handler(request, form_data, extra_params, user)
            sources.extend(flags.get('sources', []))
        except Exception as e:
            log.exception(e)

    await _record_system_prompt(form_data, metadata, user, model_system_prompt)
    metadata['user_prompt'] = get_last_user_message(form_data['messages'])
    metadata['sources'] = sources[:] if sources else []

    # If context is not empty, insert it into the messages
    if sources and prompt:
        form_data['messages'] = await apply_source_context_to_messages(request, form_data['messages'], sources, prompt)

    # Only sources with a name or id are shown as citations
    sources = [
        source
        for source in sources
        if source.get('source', {}).get('name', '') or source.get('source', {}).get('id', '')
    ]
    if len(sources) > 0:
        events.append({'sources': sources})

    if model_knowledge:
        await event_emitter(
            {
                'type': 'status',
                'data': {
                    'action': 'knowledge_search',
                    'query': user_message,
                    'done': True,
                    'hidden': True,
                },
            }
        )

    if ENABLE_PLUGINS:
        try:
            form_data, _ = await process_filter_functions(
                request=request,
                filter_context=filter_context,
                filter_functions=filter_functions,
                filter_type='request',
                form_data=form_data,
                extra_params=extra_params,
            )
        except Exception as e:
            raise Exception(f'{e}')

    form_data = normalize_messages_for_model(form_data)

    return form_data, metadata, events


async def get_event_emitter_and_caller(metadata):
    event_emitter = None
    event_caller = None

    # event_emitter only needs user_id + chat_id + message_id.
    # It broadcasts to user:{user_id} room AND persists to DB,
    # so it works for backend-initiated calls (automations, API).
    if metadata.get('chat_id') and metadata.get('message_id'):
        event_emitter = await get_event_emitter(metadata)

    # event_caller needs session_id — it calls back to a specific
    # websocket session (used by direct tools, pyodide code interpreter).
    if metadata.get('session_id') and metadata.get('chat_id') and metadata.get('message_id'):
        event_caller = await get_event_call(metadata)

    return event_emitter, event_caller


async def build_chat_response_context(request, form_data, user, model, metadata, tasks, events):
    event_emitter, event_caller = await get_event_emitter_and_caller(metadata)
    return {
        'request': request,
        'form_data': form_data,
        'user': user,
        'model': model,
        'metadata': metadata,
        'tasks': tasks,
        'events': events,
        'event_emitter': event_emitter,
        'event_caller': event_caller,
    }


async def execute_tool_call_for_output(request, form_data, user, metadata, event_caller, event_emitter, tool_call):
    tools = metadata.get('tools', {})
    name = tool_call.get('function', {}).get('name', '')
    tool_args = tool_call.get('function', {}).get('arguments', '{}')
    params = {}
    if tool_args and tool_args.strip():
        try:
            params = JSONCodec.loads(tool_args)
        except Exception:
            try:
                params = ast.literal_eval(tool_args)
            except Exception as e:
                log.debug(e)
                return {
                    'tool_call_id': tool_call.get('id', ''),
                    'content': (
                        'Error: Tool call arguments could not be parsed. '
                        'The model generated malformed or incomplete JSON.'
                    ),
                }
    tool_call.setdefault('function', {})['arguments'] = JSONCodec.dumps(params)

    tool = tools.get(name)
    if not tool:
        return {'tool_call_id': tool_call.get('id', ''), 'content': f'Error: Tool "{name}" not found.'}

    spec = tool.get('spec', {})
    tool_type = tool.get('type', '')
    direct_tool = tool.get('direct', False)
    allowed_params = spec.get('parameters', {}).get('properties', {}).keys()
    params = {key: value for key, value in params.items() if key in allowed_params}

    try:
        if direct_tool:
            if not event_caller:
                result = 'Error: Browser session is not connected for this direct tool.'
            else:
                result = await event_caller(
                    {
                        'type': 'execute:tool',
                        'data': {
                            'id': str(uuid4()),
                            'name': name,
                            'params': params,
                            'server': tool.get('server', {}),
                            'session_id': metadata.get('session_id'),
                        },
                    }
                )
        else:
            function = await get_updated_tool_function(
                function=tool['callable'],
                extra_params={
                    '__messages__': form_data.get('messages', []),
                    '__files__': metadata.get('files', []),
                },
            )
            result = await function(**params)
    except Exception as e:
        result = {'error': str(e)}

    terminal_file_result = build_terminal_file_tool_result(name, params, result, tool, metadata)
    if terminal_file_result:
        result = terminal_file_result

    result, files, embeds = await process_tool_result(
        request,
        name,
        result,
        tool_type,
        direct_tool,
        metadata,
        user,
    )

    await terminal_event_handler(name, params, result, event_emitter)

    return {
        'tool_call_id': tool_call.get('id', ''),
        'content': tool_result_content(result),
        **({'files': files} if files else {}),
        **({'embeds': embeds} if embeds else {}),
    }


async def drain_approved_tool_calls(request, form_data, user, model, metadata) -> bool:
    chat_id = metadata.get('chat_id')
    assistant_message_id = metadata.get('assistant_message_id')
    # Only a resume/continue payload re-enters an existing message; other paths mint a fresh id with nothing to drain.
    if not is_saved_chat_id(chat_id) or not assistant_message_id:
        return False

    message_id = metadata.get('message_id') or assistant_message_id
    message = await Chats.get_message_by_id_and_message_id(chat_id, message_id)
    output = message.get('output') if message else None
    if not isinstance(output, list):
        return False

    result_call_ids = {
        item.get('call_id') for item in output if item.get('type') == 'function_call_output' and item.get('call_id')
    }
    approved_calls = [
        item
        for item in output
        if item.get('type') == 'function_call'
        and item.get('call_id')
        and item.get('status') == 'queued'
        and item.get('approved') is True
        and item.get('call_id') not in result_call_ids
    ]
    if not approved_calls:
        if metadata.get('params', {}).get('tool_approval_mode', 'full') == 'ask' and any(
            item.get('type') == 'function_call'
            and item.get('name') != 'ask_user'
            and (item.get('call_id') or item.get('id'))
            and item.get('status') == 'queued'
            and item.get('approved') is not True
            and (item.get('call_id') or item.get('id')) not in result_call_ids
            for item in output
        ):
            event_emitter, _ = await get_event_emitter_and_caller(metadata)
            await pause_for_tool_approval(chat_id, message_id, output, form_data, metadata)
            if event_emitter:
                await event_emitter({'type': 'chat:completion', 'data': {'done': False, 'output': output}})
            return True
        return False

    event_emitter, event_caller = await get_event_emitter_and_caller(metadata)
    changed = False
    for item in approved_calls:
        if item.get('name') == 'ask_user':
            item['status'] = 'pending'
            item.pop('approved', None)
            changed = True
            continue

        tool_call = {
            'id': item.get('call_id', ''),
            'type': 'function',
            'function': {
                'name': item.get('name', ''),
                'arguments': item.get('arguments', '{}'),
            },
        }
        result = await execute_tool_call_for_output(
            request,
            form_data,
            user,
            metadata,
            event_caller,
            event_emitter,
            tool_call,
        )
        item['arguments'] = tool_call.get('function', {}).get('arguments', '{}')
        output_parts = [{'type': 'input_text', 'text': result.get('content', '')}]
        item['status'] = 'failed' if _is_tool_result_error(result.get('content', '')) else 'completed'
        display_files = []
        for file_item in result.get('files', []):
            if file_item.get('type') == 'image' and file_item.get('url', '').startswith('data:'):
                output_parts.append({'type': 'input_image', 'image_url': file_item['url']})
            else:
                display_files.append(file_item)

        output.append(
            {
                'type': 'function_call_output',
                'id': output_id('fco'),
                'call_id': result.get('tool_call_id', ''),
                'output': output_parts,
                'status': item['status'],
                **({'files': display_files} if display_files else {}),
                **({'embeds': result.get('embeds')} if result.get('embeds') else {}),
            }
        )
        changed = True

    if changed:
        result_call_ids = {
            item.get('call_id') for item in output if item.get('type') == 'function_call_output' and item.get('call_id')
        }
        if metadata.get('params', {}).get('tool_approval_mode', 'full') == 'ask' and any(
            item.get('type') == 'function_call'
            and item.get('name') != 'ask_user'
            and (item.get('call_id') or item.get('id'))
            and item.get('status') == 'queued'
            and item.get('approved') is not True
            and (item.get('call_id') or item.get('id')) not in result_call_ids
            for item in output
        ):
            await pause_for_tool_approval(chat_id, message_id, output, form_data, metadata)
            result_call_ids = {
                item.get('call_id')
                for item in output
                if item.get('type') == 'function_call_output' and item.get('call_id')
            }
        paused = any(
            item.get('type') == 'function_call'
            and item.get('call_id')
            and item.get('status') in {'pending', 'queued', 'requires_approval'}
            and item.get('call_id') not in result_call_ids
            for item in output
        )
        if not paused:
            output.append(
                {
                    'type': 'message',
                    'id': output_id('msg'),
                    'status': 'in_progress',
                    'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': ''}],
                }
            )

        await Chats.upsert_message_to_chat_by_id_and_message_id(
            chat_id,
            message_id,
            {'done': False, 'output': output},
            touch=False,
        )
        if event_emitter:
            await event_emitter(
                {
                    'type': 'chat:completion',
                    'data': {
                        'done': False,
                        'output': output,
                    },
                }
            )

        db_messages = await load_messages_from_db(chat_id, metadata.get('user_message_id'))
        if db_messages:
            assistant_message = await Chats.get_message_by_id_and_message_id(chat_id, message_id)
            if assistant_message:
                db_messages.append({k: v for k, v in assistant_message.items() if k in MESSAGE_REPLAY_KEYS})
            for message in db_messages:
                output = message.get('output')
                # reasoning_details can be model/provider-bound, so only replay them
                # for output produced by the same model.
                if (
                    message.get('role') == 'assistant'
                    and message.get('model') != model['id']
                    and isinstance(output, list)
                ):
                    message['output'] = strip_reasoning_details(output)

            form_data['messages'] = process_messages_with_output(
                db_messages,
                reasoning_format=get_reasoning_format(model),
            )
            form_data['messages'] = sanitize_tool_pairs(form_data['messages'])

        if not paused and ENABLE_PLUGINS:
            filter_functions = await get_filter_functions(request, model, metadata.get('filter_ids', []))
            if filter_functions:
                filtered_form_data, _ = await process_filter_functions(
                    request=request,
                    filter_context=get_filter_context(request),
                    filter_functions=filter_functions,
                    filter_type='request',
                    form_data=form_data,
                    extra_params={
                        '__event_emitter__': event_emitter,
                        '__event_call__': event_caller,
                        '__user__': user.model_dump() if isinstance(user, UserModel) else {},
                        '__metadata__': metadata,
                        '__oauth_token__': await get_system_oauth_token(request, user),
                        '__request__': request,
                        '__model__': model,
                        '__chat_id__': metadata.get('chat_id'),
                        '__message_id__': metadata.get('message_id'),
                    },
                )
                if filtered_form_data is not form_data:
                    form_data.clear()
                    form_data.update(filtered_form_data)

        if not paused:
            normalize_messages_for_model(form_data)

        return paused

    return False


async def pause_for_tool_approval(chat_id: str, message_id: str, output: list[dict], form_data: dict, metadata: dict):
    result_call_ids = {
        item.get('call_id') for item in output if item.get('type') == 'function_call_output' and item.get('call_id')
    }
    has_pending_approval = False
    for item in output:
        if item.get('type') == 'function_call' and not item.get('call_id') and item.get('id'):
            item['call_id'] = item['id']

        if (
            item.get('type') == 'function_call'
            and item.get('call_id')
            and item.get('call_id') not in result_call_ids
            and item.get('status') != 'rejected'
        ):
            if not has_pending_approval:
                item['status'] = 'pending'
                has_pending_approval = True
            elif item.get('status') == 'in_progress':
                item['status'] = 'queued'

    await Chats.upsert_message_to_chat_by_id_and_message_id(
        chat_id,
        message_id,
        {
            'done': False,
            'output': output,
            'meta': {
                **(metadata.get('tool_approval') or {}),
                'session_id': metadata.get('session_id'),
                'tool_ids': metadata.get('tool_ids') or [],
                'skill_ids': metadata.get('skill_ids') or [],
                'terminal_id': metadata.get('terminal_id'),
                'tool_servers': metadata.get('tool_servers'),
                'filter_ids': metadata.get('filter_ids') or [],
                'features': metadata.get('features') or {},
                'variables': metadata.get('variables') or {},
                'files': metadata.get('files') or [],
                'params': metadata.get('params') or {},
            },
        },
        touch=False,
    )


def get_response_data(response):
    if isinstance(response, list) and len(response) == 1:
        # If the response is a single-item list, unwrap it #17213
        response = response[0]

    if isinstance(response, JSONResponse):
        if isinstance(response.body, bytes):
            try:
                response_data = JSONCodec.loads(response.body.decode('utf-8', 'replace'))
            except JSONCodec.JSONDecodeError:
                response_data = {'error': {'detail': 'Invalid JSON response'}}
        else:
            response_data = response
    elif isinstance(response, dict):
        response_data = response
    else:
        response_data = None

    return response, response_data


def merge_events_into_response(response_data, events):
    if events and isinstance(events, list):
        extra_response = {}
        for event in events:
            if isinstance(event, dict):
                extra_response.update(event)
            else:
                extra_response[event] = True

        return {
            **extra_response,
            **response_data,
        }
    return response_data


def build_response_object(response, response_data):
    if isinstance(response, dict):
        return response_data
    if isinstance(response, JSONResponse):
        return JSONResponse(
            content=response_data,
            headers=response.headers,
            status_code=response.status_code,
        )
    return response


def update_assistant_message_from_stream(assistant_message, raw):
    line = raw.decode('utf-8', 'replace') if isinstance(raw, bytes) else raw
    if not isinstance(line, str):
        return

    def append_output_text(item, text):
        parts = item.setdefault('content', [])
        if parts and parts[-1].get('type') == 'output_text':
            parts[-1]['text'] += text
        else:
            parts.append({'type': 'output_text', 'text': text})

    for raw_part in line.splitlines():
        part = raw_part.removeprefix('data:').strip()
        if not part or part == '[DONE]':
            continue

        try:
            data = JSONCodec.loads(part)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        if data.get('type', '').startswith('response.'):
            output, meta = handle_responses_streaming_event(data, assistant_message.get('output', []))
            if output:
                assistant_message['output'] = output
            if meta and meta.get('usage'):
                assistant_message['usage'] = merge_usage(assistant_message.get('usage'), meta['usage'])
            continue

        raw_usage = data.get('usage', {}) or {}
        raw_usage.update(data.get('timings', {}))
        if raw_usage:
            assistant_message['usage'] = merge_usage(assistant_message.get('usage'), raw_usage)

        for choice in data.get('choices', []):
            delta = choice.get('delta', {}) or {}
            content = delta.get('content')
            reasoning_content = delta.get('reasoning_content') or delta.get('reasoning') or delta.get('thinking')

            if reasoning_content:
                output = assistant_message.setdefault('output', [])
                if not output or output[-1].get('type') != 'reasoning':
                    output.append(
                        {
                            'type': 'reasoning',
                            'id': output_id('r'),
                            'status': 'in_progress',
                            'start_tag': '<think>',
                            'end_tag': '</think>',
                            'attributes': {'type': 'reasoning_content'},
                            'content': [],
                            'summary': None,
                            'started_at': time.time(),
                        }
                    )

                append_output_text(output[-1], reasoning_content)

            if content:
                output = assistant_message.get('output')
                if output:
                    if output[-1].get('type') == 'reasoning':
                        output[-1]['status'] = 'completed'
                        output[-1]['ended_at'] = time.time()
                        output[-1]['duration'] = int(output[-1]['ended_at'] - output[-1]['started_at'])

                    if not output or output[-1].get('type') != 'message':
                        output.append(
                            {
                                'type': 'message',
                                'id': output_id('msg'),
                                'status': 'in_progress',
                                'role': 'assistant',
                                'content': [],
                            }
                        )

                    append_output_text(output[-1], content)

                assistant_message['content'] = assistant_message.get('content', '') + content


async def get_system_oauth_token(request, user):
    """Get the system OAuth token for a user.

    Primary path: use the oauth_session_id cookie (browser requests).
    Fallback: look up the user's most recent OAuth session from the DB
    (covers automations, API calls, and other cookie-less contexts).
    """
    oauth_token = None
    try:
        oauth_session_id = request.cookies.get('oauth_session_id', None)
        if oauth_session_id:
            oauth_token = await request.app.state.oauth_manager.get_oauth_token(
                user.id,
                oauth_session_id,
            )

        # Fallback: no cookie (automation, API key, etc.) — use most recent session
        if oauth_token is None:
            from local_llm.models.oauth_sessions import OAuthSessions

            sessions = await OAuthSessions.get_sessions_by_user_id(user.id)
            # Filter out MCP-provider sessions — their token refresh is handled
            # separately by oauth_client_manager.  Passing them to the SSO
            # oauth_manager causes a failed refresh and session deletion (#24618).
            sessions = [s for s in sessions if not (s.provider or '').startswith('mcp:')]
            if sessions:
                best = max(sessions, key=lambda s: s.updated_at)
                oauth_token = await request.app.state.oauth_manager.get_oauth_token(
                    user.id,
                    best.id,
                )
    except Exception as e:
        log.error(f'Error getting OAuth token: {e}')
    return oauth_token


def _messages_for_tasks(message_list: list[dict]) -> list[dict]:
    """The conversation as the task models see it: text only, without details blocks or images."""
    # get_message_list creates a new list, so this does not affect
    # the original messages outside of this handler
    messages = []
    for message in message_list:
        content = message.get('content', '')
        if isinstance(content, list):
            for item in content:
                if item.get('type') == 'text':
                    content = item['text']
                    break

        if isinstance(content, str):
            content = re.sub(
                r'<details\b[^>]*>.*?<\/details>|!\[.*?\]\(.*?\)',
                '',
                content,
                flags=re.S | re.I,
            ).strip()

        messages.append(
            {
                **message,
                'role': message.get('role', 'assistant'),  # Safe fallback for missing role
                'content': content,
            }
        )
    return messages


def _task_reply_json(res: dict, text_from_message) -> str:
    """The {...} span of a task model's reply, from its single choice's message text."""
    if len(res.get('choices', [])) == 1:
        # A reply may carry content and reasoning_content as null; read that as no text.
        text = text_from_message(res.get('choices', [])[0].get('message', {})) or ''
    else:
        text = ''
    return text[text.find('{') : text.rfind('}') + 1]


def _content_or_reasoning(response_message: dict):
    return response_message.get('content') or response_message.get('reasoning_content', '')


async def _generate_follow_ups(request, user, metadata: dict, event_emitter, model_id: str, messages: list[dict]):
    res = await generate_follow_ups(
        request,
        {
            'model': model_id,
            'messages': messages,
            'message_id': metadata['message_id'],
            'chat_id': metadata['chat_id'],
        },
        user,
    )
    if not (res and isinstance(res, dict)):
        return

    follow_ups_string = _task_reply_json(res, _content_or_reasoning)
    try:
        follow_ups = JSONCodec.loads(follow_ups_string).get('follow_ups', [])
        await event_emitter(
            {
                'type': 'chat:message:follow_ups',
                'data': {
                    'follow_ups': follow_ups,
                },
            }
        )

        if is_saved_chat_id(metadata.get('chat_id')):
            await Chats.upsert_message_to_chat_by_id_and_message_id(
                metadata['chat_id'],
                metadata['message_id'],
                {
                    'followUps': follow_ups,
                },
                touch=False,
            )
    except Exception:
        pass


async def _generate_chat_title(
    request, user, metadata: dict, event_emitter, message: dict, messages: list[dict], messages_map, enabled
) -> None:
    user_message = get_last_user_message(messages)
    if user_message and len(user_message) > 100:
        user_message = user_message[:100] + '...'

    title = None
    if enabled:
        res = await generate_title(
            request,
            {
                'model': message['model'],
                'messages': messages,
                'chat_id': metadata['chat_id'],
            },
            user,
        )

        if res and isinstance(res, dict):
            title_string = _task_reply_json(
                res,
                lambda response_message: (
                    response_message.get('content')
                    or response_message.get('reasoning_content')
                    or message.get('content', user_message)
                ),
            )

            try:
                title = JSONCodec.loads(title_string).get('title', user_message)
            except Exception:
                title = ''

            if not title:
                title = messages[0].get('content', user_message)

            await Chats.update_chat_title_by_id(metadata['chat_id'], title)
            await event_emitter({'type': 'chat:title', 'data': title})

    # Without a generated title, a first exchange is titled after its opening message.
    if title is None and len(messages) == 2 and (not messages_map or len(messages_map) <= 2):
        title = messages[0].get('content', user_message)

        await Chats.update_chat_title_by_id(metadata['chat_id'], title)
        await event_emitter({'type': 'chat:title', 'data': title})


async def _generate_chat_tags(request, user, metadata: dict, event_emitter, model_id: str, messages: list[dict]):
    res = await generate_chat_tags(
        request,
        {
            'model': model_id,
            'messages': messages,
            'chat_id': metadata['chat_id'],
        },
        user,
    )
    if not (res and isinstance(res, dict)):
        return

    tags_string = _task_reply_json(res, _content_or_reasoning)
    try:
        tags = JSONCodec.loads(tags_string).get('tags', [])
        await Chats.update_chat_tags_by_id(metadata['chat_id'], tags, user)
        await event_emitter({'type': 'chat:tags', 'data': tags})
    except Exception:
        pass


async def background_tasks_handler(ctx):
    request = ctx['request']
    form_data = ctx['form_data']
    user = ctx['user']
    metadata = ctx['metadata']
    tasks = ctx['tasks']
    event_emitter = ctx['event_emitter']

    messages_map = None
    if is_saved_chat_id(metadata.get('chat_id')):
        messages_map = await Chats.get_messages_map_by_chat_id(metadata['chat_id'])
        if not messages_map:
            # Chat was deleted while the response was streaming — skip background tasks
            return
        message = messages_map.get(metadata['message_id'])
        messages = _messages_for_tasks(get_message_list(messages_map, metadata['message_id']))
    else:
        # Local temp chat, get the model and message from the form_data
        message = get_last_user_message_item(form_data.get('messages', []))
        messages = form_data.get('messages', [])
        if message:
            message['model'] = form_data.get('model')

    if not (message and 'model' in message):
        return

    if tasks and messages:
        if TASKS.FOLLOW_UP_GENERATION in tasks and tasks[TASKS.FOLLOW_UP_GENERATION]:
            await _generate_follow_ups(request, user, metadata, event_emitter, message['model'], messages)

        # Only saved chats get titles and tags
        if is_saved_chat_id(metadata.get('chat_id')):
            if TASKS.TITLE_GENERATION in tasks:
                await _generate_chat_title(
                    request,
                    user,
                    metadata,
                    event_emitter,
                    message,
                    messages,
                    messages_map,
                    tasks[TASKS.TITLE_GENERATION],
                )

            if TASKS.TAGS_GENERATION in tasks and tasks[TASKS.TAGS_GENERATION]:
                await _generate_chat_tags(request, user, metadata, event_emitter, message['model'], messages)

    if messages:
        await review_memory_after_turn(
            request=request,
            user=user,
            model=ctx['model'],
            metadata=metadata,
            form_data=form_data,
            assistant_message=ctx.get('assistant_message') or {},
            messages=messages,
        )


async def outlet_filter_handler(ctx):
    """Run outlet filters inline after chat completion.

    Replaces the separate POST /api/chat/completed round-trip.
    Persists outlet-modified content to DB and emits a chat:outlet event
    so the frontend can sync its in-memory state. Returns immediately when
    the model has no filters.

    For temp/API chats, messages are built from form_data plus ctx['assistant_message'].
    """
    request = ctx['request']
    user = ctx['user']
    model = ctx['model']
    metadata = ctx['metadata']
    event_emitter = ctx.get('event_emitter')
    event_caller = ctx.get('event_caller')

    chat_id = metadata.get('chat_id', '')
    message_id = metadata.get('message_id')

    if not chat_id and not ctx.get('assistant_message'):
        return

    if not message_id:
        message_id = output_id('msg')

    is_unsaved_chat = not is_saved_chat_id(chat_id)
    try:
        filter_functions = (
            await get_filter_functions(request, model, metadata.get('filter_ids', [])) if ENABLE_PLUGINS else []
        )
        model_id = model.get('id') if isinstance(model, dict) else model
        models = request.app.state.MODELS
        has_pipeline_outlet_filters = bool(
            (isinstance(model, dict) and 'pipeline' in model) or get_sorted_filters(model_id, models)
        )
        if not filter_functions and not has_pipeline_outlet_filters:
            return

        messages_map = None

        if is_unsaved_chat:
            form_messages = ctx.get('form_data', {}).get('messages', [])
            assistant_message = ctx.get('assistant_message', {})

            message_list = [
                {
                    'role': m.get('role'),
                    'content': m.get('content') or get_output_text(m.get('output')),
                }
                for m in form_messages
            ]

            if assistant_message:
                message_list.append(
                    {
                        'id': message_id,
                        'role': 'assistant',
                        **assistant_message,
                    }
                )

            if not message_list:
                return
        else:
            messages_map = await Chats.get_messages_map_by_chat_id(chat_id)
            if not messages_map:
                return

            message_list = get_message_list(messages_map, message_id)
            if not message_list:
                return

        outlet_data = {
            'model': model_id,
            'messages': [
                {
                    'id': m.get('id'),
                    'role': m.get('role'),
                    'content': m.get('content') or get_output_text(m.get('output')),
                    'info': m.get('info'),
                    'timestamp': m.get('timestamp'),
                    # Deepcopy so in-place filter mutations do not alias messages_map's baseline
                    **({'output': copy.deepcopy(m['output'])} if m.get('output') else {}),
                    **({'usage': m['usage']} if m.get('usage') else {}),
                    **({'sources': m['sources']} if m.get('sources') else {}),
                }
                for m in message_list
            ],
            'filter_ids': metadata.get('filter_ids', []),
            'chat_id': chat_id,
            'session_id': metadata.get('session_id'),
            'id': message_id,
        }

        # Pipeline outlet filters
        try:
            outlet_data = await process_pipeline_outlet_filter(request, outlet_data, user, models)
        except Exception as e:
            log.debug('Pipeline outlet filter error: %s', e)

        # Function outlet filters
        extra_params = {
            '__event_emitter__': event_emitter,
            '__event_call__': event_caller,
            '__user__': user.model_dump() if isinstance(user, UserModel) else {},
            '__metadata__': metadata,
            '__request__': request,
            '__model__': model,
        }

        if filter_functions:
            outlet_result, _ = await process_filter_functions(
                request=request,
                filter_context=None,
                filter_functions=filter_functions,
                filter_type='outlet',
                form_data=outlet_data,
                extra_params=extra_params,
            )
        else:
            outlet_result = outlet_data

        if outlet_result and outlet_result.get('messages'):
            if not is_unsaved_chat and messages_map:
                for message in outlet_result['messages']:
                    outlet_message_id = message.get('id')
                    if outlet_message_id and outlet_message_id in messages_map:
                        original_message = messages_map[outlet_message_id]
                        original_content = original_message.get('content') or get_output_text(
                            original_message.get('output')
                        )
                        message_content = message.get('content') or get_output_text(message.get('output'))
                        content_changed = original_content != message_content
                        output_changed = message.get('output') and message.get('output') != original_message.get(
                            'output'
                        )
                        if content_changed or output_changed:
                            message_update = {
                                'originalContent': original_content,
                                **({'output': message['output']} if output_changed else {}),
                            }
                            if content_changed:
                                message_update['content'] = message_content or ''
                            await Chats.upsert_message_to_chat_by_id_and_message_id(
                                chat_id,
                                outlet_message_id,
                                message_update,
                            )

            if event_emitter:
                await event_emitter(
                    {
                        'type': 'chat:outlet',
                        'data': {'messages': outlet_result['messages']},
                    }
                )
    except Exception as e:
        log.debug('Error running outlet filters: %s', e)


async def non_streaming_chat_response_handler(response, ctx):
    request = ctx['request']

    user = ctx['user']
    metadata = ctx['metadata']
    events = ctx['events']

    event_emitter = ctx['event_emitter']

    response, response_data = get_response_data(response)
    if response_data is None:
        return response

    chat_id = metadata.get('chat_id') or ''
    save_to_chat = is_saved_chat_id(chat_id)

    if event_emitter:
        try:
            if 'error' in response_data:
                error = response_data.get('error')

                if isinstance(error, dict):
                    error = error.get('detail', error)
                else:
                    error = str(error)

                log.error('Provider returned error (non-streaming): %s', error)

                if save_to_chat:
                    await Chats.upsert_message_to_chat_by_id_and_message_id(
                        metadata['chat_id'],
                        metadata['message_id'],
                        {
                            'error': {'content': error},
                        },
                    )
                if isinstance(error, str) or isinstance(error, dict):
                    await event_emitter(
                        {
                            'type': 'chat:message:error',
                            'data': {'error': {'content': error}},
                        }
                    )

            if 'selected_model_id' in response_data and save_to_chat:
                await Chats.upsert_message_to_chat_by_id_and_message_id(
                    metadata['chat_id'],
                    metadata['message_id'],
                    {
                        'selectedModelId': response_data['selected_model_id'],
                    },
                    touch=False,
                )

            choices = response_data.get('choices', [])
            response_output = response_data.get('output')
            content = choices[0].get('message', {}).get('content') if choices else ''

            if choices and (content or response_output):
                if content or response_output:
                    await event_emitter(
                        {
                            'type': 'chat:completion',
                            'data': response_data,
                        }
                    )

                    title = await Chats.get_chat_title_by_id(metadata['chat_id']) if save_to_chat else ''

                    # Use output from backend if provided (OR-compliant backends),
                    # otherwise generate from response content
                    if not response_output:
                        choice_message = choices[0].get('message', {})
                        reasoning_content = choice_message.get('reasoning_content') or choice_message.get('reasoning')
                        reasoning_details = get_reasoning_details(choice_message)
                        response_output = []
                        if reasoning_content or reasoning_details:
                            reasoning_item = {
                                'type': 'reasoning',
                                'id': output_id('r'),
                                'status': 'completed',
                                'start_tag': '<think>',
                                'end_tag': '</think>',
                                'attributes': {'type': 'reasoning_content'},
                                'content': (
                                    [{'type': 'output_text', 'text': reasoning_content}] if reasoning_content else []
                                ),
                                'summary': None,
                            }
                            if reasoning_details:
                                reasoning_item['reasoning_details'] = (
                                    reasoning_details if isinstance(reasoning_details, list) else [reasoning_details]
                                )
                            response_output.append(reasoning_item)
                        response_output.append(
                            {
                                'type': 'message',
                                'id': output_id('msg'),
                                'status': 'completed',
                                'role': 'assistant',
                                'content': [{'type': 'output_text', 'text': content}],
                            }
                        )

                    await event_emitter(
                        {
                            'type': 'chat:completion',
                            'data': {
                                'done': True,
                                'output': response_output,
                                'title': title,
                            },
                        }
                    )

                    # Save message in the database
                    usage = normalize_usage(response_data.get('usage', {}) or {})

                    if save_to_chat:
                        await Chats.upsert_message_to_chat_by_id_and_message_id(
                            metadata['chat_id'],
                            metadata['message_id'],
                            {
                                'done': True,
                                'role': 'assistant',
                                'output': response_output,
                                **({'usage': usage} if usage else {}),
                            },
                        )

                    await publish_chat_finished_event(request, user, metadata, title, content, response_output)

                    ctx['assistant_message'] = {
                        'content': content,
                        'output': response_output,
                        **({'usage': usage} if usage else {}),
                    }
                    await outlet_filter_handler(ctx)
                    await background_tasks_handler(ctx)

            response = build_response_object(response, merge_events_into_response(response_data, events))
        except Exception as e:
            log.debug('Error occurred while processing request: %s', e)
            chat_id = metadata.get('chat_id')
            if getattr(request.state, 'internal', False) is not True and chat_id and is_saved_chat_id(chat_id):
                webui_url = await Config.get('webui.url')
                await publish_event(
                    request,
                    EVENTS.CHAT_FAILED,
                    actor=user,
                    subject_id=chat_id,
                    subject_type='chat',
                    data={
                        'user_id': user.id,
                        'chat_id': chat_id,
                        'message_id': metadata.get('message_id'),
                        'model_id': metadata.get('model_id'),
                        'url': f'{webui_url}/c/{chat_id}' if webui_url else f'/c/{chat_id}',
                        'message': str(e),
                    },
                    message='Chat failed',
                )
            pass

        return response

    choices = response_data.get('choices', [])
    output = response_data.get('output')
    content = choices[0].get('message', {}).get('content') if choices else ''
    if ENABLE_API_OUTLET_FILTERS and (content or output):
        usage = normalize_usage(response_data.get('usage', {}) or {})
        ctx['assistant_message'] = {
            **({'content': content} if content else {}),
            **({'output': output} if output else {}),
            **({'usage': usage} if usage else {}),
        }
        await outlet_filter_handler(ctx)

    if isinstance(response, dict):
        response = merge_events_into_response(response_data, events)

    return response


def _message_output_item(text: str = '', **extra) -> dict:
    return {
        'type': 'message',
        'id': output_id('msg'),
        'status': 'in_progress',
        'role': 'assistant',
        'content': [{'type': 'output_text', 'text': text}],
        **extra,
    }


def _tag_attributes(tag_content: str) -> dict:
    """Extract attributes from a tag if they exist."""
    attributes = {}
    if not tag_content:
        return attributes
    matches = re.findall(r'(\w+)\s*=\s*"([^"]+)"', tag_content)
    for key, value in matches:
        attributes[key] = value
    return attributes


def _last_output_text(output: list) -> str:
    """Get text from last message item, or empty string."""
    if output and output[-1].get('type') == 'message':
        parts = output[-1].get('content', [])
        if parts and parts[-1].get('type') == 'output_text':
            return parts[-1].get('text', '')
    return ''


def _set_last_output_text(output: list, text: str) -> None:
    """Set text on last message item's output_text."""
    if output and output[-1].get('type') == 'message':
        parts = output[-1].get('content', [])
        if parts and parts[-1].get('type') == 'output_text':
            parts[-1]['text'] = text


# Map content_type to output item type
_TAG_OUTPUT_TYPES = {
    'reasoning': 'reasoning',
    'solution': 'message',  # solution tags just produce text
    'code_interpreter': 'open_webui:code_interpreter',
}


def _tag_block_item(output_item_type: str, content_type: str, start_tag: str, end_tag: str, attributes: dict) -> dict:
    """The in-progress output item a start tag opens."""
    if output_item_type == 'reasoning':
        return {
            'type': 'reasoning',
            'id': output_id('r'),
            'status': 'in_progress',
            'start_tag': start_tag,
            'end_tag': end_tag,
            'attributes': attributes,
            'content': [],
            'summary': None,
            'started_at': time.time(),
        }
    if output_item_type == 'open_webui:code_interpreter':
        return {
            'type': 'open_webui:code_interpreter',
            'id': output_id('ci'),
            'status': 'in_progress',
            'start_tag': start_tag,
            'end_tag': end_tag,
            'attributes': attributes,
            'lang': attributes.get('lang', 'python'),
            'code': '',
            'output': None,
            'started_at': time.time(),
        }
    # solution or other text-producing tag
    return {
        'type': 'message',
        'id': output_id('msg'),
        'status': 'in_progress',
        'role': 'assistant',
        'content': [{'type': 'output_text', 'text': ''}],
        '_tag_type': content_type,
        'start_tag': start_tag,
        'end_tag': end_tag,
        'attributes': attributes,
        'started_at': time.time(),
    }


class _TagScanner:
    """
    Detect special tags (reasoning, solution, code_interpreter) in streaming
    content and create corresponding OR-aligned output items directly.
    Operates on output items instead of content_blocks.

    Uses the text from the output items themselves for tag detection,
    eliminating state divergence between accumulated content and items.
    Remembers, per item and tag kind, how much text it has scanned, so each
    chunk only searches what is new.
    """

    def __init__(self):
        self.scan_positions = {}
        self.boundary_positions = {}

    def _scanned_length(self, item: dict, content_type: str, text: str) -> int:
        item_id = item.get('id')
        if not item_id:
            return 0

        scanned_length = self.scan_positions.get((item_id, content_type), 0)
        return scanned_length if scanned_length <= len(text) else 0

    def _save_scanned_length(self, item: dict, content_type: str, text: str) -> None:
        item_id = item.get('id')
        if item_id:
            self.scan_positions[(item_id, content_type)] = len(text)

    def _clear_scanned_length(self, item: dict, content_type: str) -> None:
        item_id = item.get('id')
        if item_id:
            self.scan_positions.pop((item_id, content_type), None)
            self.boundary_positions.pop((item_id, content_type), None)

    def _tag_boundaries(self, item: dict, content_type: str, text: str, scanned_length: int) -> tuple[int, int]:
        """Index of the last '<', and of the last '>' or newline, before scanned_length."""
        key = (item.get('id'), content_type)
        scanned, last_open, last_boundary = self.boundary_positions.get(key, (0, -1, -1))
        if scanned > scanned_length:  # the item was rewritten, so the cached positions are stale
            scanned, last_open, last_boundary = 0, -1, -1

        if scanned < scanned_length:
            # only text added since the last call can move either position
            open_tag = text.rfind('<', scanned, scanned_length)
            if open_tag != -1:
                last_open = open_tag
            boundary = max(
                text.rfind('>', scanned, scanned_length),
                text.rfind('\n', scanned, scanned_length),
            )
            if boundary != -1:
                last_boundary = boundary
            self.boundary_positions[key] = (scanned_length, last_open, last_boundary)

        return last_open, last_boundary

    def scan(self, content_type: str, tags: list, output: list) -> tuple[list, bool]:
        """Open a block at a start tag in the last message, or close the open block at its end tag.

        Returns the output and whether a block was closed."""
        last_type = output[-1].get('type', '') if output else ''
        # An open block of this kind is checked for its end tag first: a tagged message
        # (a solution block) is a message too, and would otherwise only be scanned for start tags.
        if (
            (last_type == 'reasoning' and content_type == 'reasoning')
            or (last_type == 'open_webui:code_interpreter' and content_type == 'code_interpreter')
            or (last_type == 'message' and output[-1].get('_tag_type') == content_type)
        ):
            return output, self._close_block(content_type, output, last_type)
        if last_type == 'message':
            return self._open_block(content_type, tags, output)
        return output, False

    def _open_block(self, content_type: str, tags: list, output: list) -> tuple[list, bool]:
        # Use the output item's own text for tag detection
        item = output[-1]
        item_text = _last_output_text(output)
        scanned_length = self._scanned_length(item, content_type, item_text)
        max_start_tag_length = max((len(start_tag) for start_tag, _ in tags), default=1)
        search_start = max(0, scanned_length - max_start_tag_length + 1)

        if scanned_length and any(start_tag.startswith('<') and start_tag.endswith('>') for start_tag, _ in tags):
            open_tag_start, last_tag_boundary = self._tag_boundaries(item, content_type, item_text, scanned_length)
            if open_tag_start > last_tag_boundary:
                search_start = min(search_start, open_tag_start)

        for start_tag, end_tag in tags:
            match = re.compile(_start_tag_pattern(start_tag)).search(item_text, search_start)
            if match:
                return output, self._split_at_start_tag(
                    content_type, tags, output, item, item_text, match, start_tag, end_tag
                )

        self._save_scanned_length(item, content_type, item_text)
        return output, False

    def _split_at_start_tag(self, content_type, tags, output, item, item_text, match, start_tag, end_tag) -> bool:
        """Keep the text before the tag in the message and open the tag's block with the text after it."""
        self._clear_scanned_length(item, content_type)
        try:
            attr_content = match.group(1) if match.group(1) else ''
        except Exception:
            attr_content = ''

        attributes = _tag_attributes(attr_content)

        before_tag = item_text[: match.start()]
        after_tag = item_text[match.end() :]

        # Keep only text before the tag in the message
        _set_last_output_text(output, before_tag)

        if not before_tag.strip():
            # Remove empty message item
            if output and output[-1].get('type') == 'message':
                output.pop()

        output_item_type = _TAG_OUTPUT_TYPES.get(content_type, content_type)
        output.append(_tag_block_item(output_item_type, content_type, start_tag, end_tag, attributes))

        if not after_tag:
            return False
        # Set the after_tag content on the new item
        if output_item_type == 'reasoning':
            output[-1]['content'] = [{'type': 'output_text', 'text': after_tag}]
        elif output_item_type == 'open_webui:code_interpreter':
            output[-1]['code'] = after_tag
        else:
            _set_last_output_text(output, after_tag)

        _, recursive_end = self.scan(content_type, tags, output)
        return bool(recursive_end)

    def _close_block(self, content_type: str, output: list, last_type: str) -> bool:
        item = output[-1]
        start_tag = item.get('start_tag', '')
        end_tag = item.get('end_tag', '')

        # Get the block content from the item itself
        if last_type == 'reasoning':
            parts = item.get('content', [])
            block_content = ''
            if parts and parts[-1].get('type') == 'output_text':
                block_content = parts[-1].get('text', '')
        elif last_type == 'open_webui:code_interpreter':
            block_content = item.get('code', '')
        else:
            block_content = _last_output_text(output)

        scanned_length = self._scanned_length(item, content_type, block_content)
        end_tag_search_start = max(0, scanned_length - max(len(end_tag), 1) + 1)

        if block_content.find(end_tag, end_tag_search_start) == -1:
            self._save_scanned_length(item, content_type, block_content)
            return False

        self._clear_scanned_length(item, content_type)

        # Strip start and end tags from content
        block_content = re.sub(_start_tag_pattern(start_tag), '', block_content).strip()

        end_tag_regex = re.compile(rf'{re.escape(end_tag)}', re.DOTALL)
        split_content = end_tag_regex.split(block_content, maxsplit=1)

        block_content = split_content[0].strip() if split_content else ''
        leftover_content = split_content[1].strip() if len(split_content) > 1 else ''

        if block_content:
            # Update the item with final content
            if last_type == 'reasoning':
                item['content'] = [{'type': 'output_text', 'text': block_content}]
                item['ended_at'] = time.time()
                item['duration'] = int(item['ended_at'] - item['started_at'])
                item['status'] = 'completed'
            elif last_type == 'open_webui:code_interpreter':
                item['code'] = block_content
                item['ended_at'] = time.time()
                item['duration'] = int(item['ended_at'] - item['started_at'])
            else:
                _set_last_output_text(output, block_content)
                item['ended_at'] = time.time()
        else:
            # Remove the block if content is empty
            output.pop()

        # Reset by appending a new message item for leftover
        output.append(_message_output_item(leftover_content))
        return True


def _message_error_content(error):
    if isinstance(error, HTTPException):
        error = error.detail
    elif isinstance(error, dict):
        error = error.get('detail', error)
    else:
        error = str(error)

    return error if isinstance(error, (str, dict)) else str(error)


def _response_delta_key(delta_data: dict):
    event_type = delta_data.get('type', '')
    if not event_type.startswith('response.') or not event_type.endswith('.delta'):
        return None
    return (
        event_type,
        delta_data.get('item_id'),
        delta_data.get('output_index'),
        delta_data.get('content_index'),
        delta_data.get('summary_index'),
    )


class _DeltaBuffer:
    """Coalesces consecutive deltas for the same field into one event, flushing every chunk_size of them."""

    def __init__(self, event_emitter, save_stream, chunk_size: int):
        self.event_emitter = event_emitter
        self.save_stream = save_stream
        self.chunk_size = chunk_size
        self.count = 0
        self.data = None
        self.type = None
        self.key = None

    async def flush(self, threshold: int = 0) -> None:
        if self.count >= threshold and self.data:
            await self.event_emitter(
                {
                    'type': 'response:completion',
                    'data': self.data,
                }
            )
            await self.save_stream()
            self.count = 0
            self.data = None
            self.type = None
            self.key = None

    async def queue(self, delta_data: dict, delta_type: str) -> None:
        delta_key = _response_delta_key(delta_data)
        if (
            self.data
            and self.key == delta_key
            and isinstance(self.data.get('delta'), str)
            and isinstance(delta_data.get('delta'), str)
        ):
            self.data['delta'] += delta_data['delta']
            self.count += 1
        else:
            if self.data and (self.type != delta_type or self.key != delta_key):
                await self.flush()

            self.count += 1
            self.data = delta_data
            self.type = delta_type
            self.key = delta_key

        if self.count >= self.chunk_size:
            await self.flush(self.chunk_size)


def _url_citation_source(title: str, url: str) -> dict:
    return {
        'type': 'source',
        'data': {
            'source': {
                'name': title,
                'url': url,
            },
            'document': [title],
            'metadata': [
                {
                    'source': url,
                    'name': title,
                }
            ],
        },
    }


def _parse_tool_params(tool_call: dict) -> dict | None:
    """The call's arguments as a dict (JSON, or a Python literal), written back onto it as JSON.

    None when they can't be read."""
    tool_args = tool_call.get('function', {}).get('arguments', '{}')
    params = {}
    if tool_args and tool_args.strip():
        try:
            params = JSONCodec.loads(tool_args)
        except Exception:
            try:
                params = ast.literal_eval(tool_args)
            except Exception as e:
                log.debug(e)
                return None
    tool_call.setdefault('function', {})['arguments'] = JSONCodec.dumps(params)
    return params


_CITATION_TOOLS = [
    'search_web',
    'fetch_url',
    'view_file',
    'view_knowledge_file',
    'query_knowledge_files',
    'query_chat_files',
]


def _merge_tool_call_delta(current_response_tool_call: dict, delta_tool_call: dict) -> None:
    """Fold a later delta of a streamed tool call into the call: a new name, more arguments."""
    delta_name = delta_tool_call.get('function', {}).get('name')
    delta_arguments = delta_tool_call.get('function', {}).get('arguments')

    if delta_name:
        current_response_tool_call['function']['name'] = delta_name

    if delta_arguments is not None:
        if not isinstance(delta_arguments, str):
            delta_arguments = JSONCodec.dumps(delta_arguments)
        current_response_tool_call.setdefault('function', {})
        if not isinstance(current_response_tool_call['function'].get('arguments'), str):
            current_response_tool_call['function']['arguments'] = ''
        current_response_tool_call['function']['arguments'] += delta_arguments


class _StreamingReply:
    """One streamed reply: its output items, the tool-call and code-interpreter rounds after it, and the save."""

    def __init__(self, ctx: dict, extra_params: dict, filter_functions: list):
        self.ctx = ctx
        self.request = ctx['request']
        self.form_data = ctx['form_data']
        self.user = ctx['user']
        self.model = ctx['model']
        self.metadata = ctx['metadata']
        self.event_emitter = ctx['event_emitter']
        self.event_caller = ctx['event_caller']
        self.chat_id = self.metadata.get('chat_id') or ''
        self.save_to_chat = is_saved_chat_id(self.chat_id)
        self.extra_params = extra_params
        self.filter_functions = filter_functions
        self.model_id = self.form_data.get('model', '')

        self.filter_context = FilterContext()
        self.tags = _TagScanner()
        self.response_stream_task_id = self.metadata.get('task_id') or self.metadata.get('message_id')

        self.tool_calls = []
        self.content_parts = []
        self.output = []
        self.prior_output = []
        self.usage = None
        self.last_response_id = None

        # Joined content_parts, cached until content_parts grows
        self.joined_content = ''
        self.joined_part_count = 0

    def full_output(self) -> list:
        return self.prior_output + self.output if self.prior_output else self.output

    async def emit_message_error(self, error_content) -> None:
        if self.save_to_chat:
            await Chats.upsert_message_to_chat_by_id_and_message_id(
                self.metadata['chat_id'],
                self.metadata['message_id'],
                {'error': {'content': error_content}},
            )
        await self.event_emitter(
            {
                'type': 'chat:message:error',
                'data': {'error': {'content': error_content}},
            }
        )

    async def _load_starting_output(self) -> None:
        """Start from the stored message when continuing one, else from a trailing assistant message, else empty."""
        message = (
            await Chats.get_message_by_id_and_message_id(self.metadata['chat_id'], self.metadata['message_id'])
            if self.save_to_chat
            else None
        )

        last_assistant_message = None
        try:
            if self.form_data['messages'][-1]['role'] == 'assistant':
                last_assistant_message = get_last_assistant_message(self.form_data['messages'])
        except Exception:
            pass

        initial_content = (
            message.get('content', '') if message else last_assistant_message if last_assistant_message else ''
        )
        self.content_parts = [initial_content] if initial_content else []

        # Initialize output: use existing from message if continuing, else create new
        existing_output = message.get('output') if message else None
        if existing_output and self.metadata.get('assistant_message_id'):
            self.prior_output = list(existing_output)
            self._trim_empty_trailing_message(self.prior_output)
            self.output = []
            self.content_parts = []
        elif existing_output:
            self.output = existing_output
        elif initial_content:
            # Only create an initial message item if there is content to initialize with
            self.output = [_message_output_item(initial_content)]
        else:
            self.output = []

    @staticmethod
    def _trim_empty_trailing_message(output: list) -> None:
        # A trailing in-progress message with no text would persist as a ghost item.
        if output and output[-1].get('type') == 'message' and output[-1].get('status') == 'in_progress':
            msg_parts = output[-1].get('content', [])
            if not msg_parts or (len(msg_parts) == 1 and not msg_parts[0].get('text', '').strip()):
                output.pop()

    async def _detect_code_interpreter(self) -> bool:
        # Legacy tool-calling only: native FC gets execute_code as a builtin tool.
        # Same five authz gates as utils/tools.py get_builtin_tools.
        features = self.metadata.get('features', {}) or {}
        model_capabilities = self.model.get('info', {}).get('meta', {}).get('capabilities') or {}
        builtin_tools_meta = self.model.get('info', {}).get('meta', {}).get('builtinTools', {})
        return (
            self.metadata.get('params', {}).get('function_calling') == 'legacy'
            and bool(features.get('code_interpreter'))
            and builtin_tools_meta.get('code_interpreter', True)
            and await Config.get('code_interpreter.enable')
            and model_capabilities.get('code_interpreter', True)
            and (
                getattr(self.user, 'role', None) == 'admin'
                or await has_permission(
                    getattr(self.user, 'id', ''),
                    'features.code_interpreter',
                    await Config.get('user.permissions'),
                )
            )
        )

    def _reasoning_tags(self) -> list:
        if not self.detect_reasoning_tags:
            return []
        if isinstance(self.reasoning_tags_param, list) and len(self.reasoning_tags_param) == 2:
            return [(self.reasoning_tags_param[0], self.reasoning_tags_param[1])]
        return DEFAULT_REASONING_TAGS

    async def save_stream(self, stream_output: list | None = None) -> None:
        """Snapshot the reply so far for clients that reconnect mid-stream."""
        if not self.chat_id or not self.metadata.get('message_id'):
            return

        # content_parts is append-only, so its length tells us when the join is stale
        if self.joined_part_count != len(self.content_parts):
            self.joined_content = ''.join(self.content_parts)
            self.joined_part_count = len(self.content_parts)

        current_stream_output = stream_output if stream_output is not None else self.full_output()
        await save_response_stream(
            self.request.app.state.redis,
            self.response_stream_task_id,
            self.chat_id,
            self.metadata.get('message_id'),
            self.joined_content or get_output_text(current_stream_output),
            current_stream_output,
        )

    def _with_full_output_index(self, response_data: dict) -> dict:
        if self.prior_output and isinstance(response_data.get('output_index'), int):
            return {
                **response_data,
                'output_index': response_data['output_index'] + len(self.prior_output),
            }
        return response_data

    async def _queue_delta(self, stream: '_StreamPass', delta_data: dict, delta_type: str) -> None:
        await stream.deltas.queue(self._with_full_output_index(delta_data), delta_type)

    async def emit_response_event(self, stream: '_StreamPass', response_data: dict, stream_output: list | None = None):
        if response_data.get('type', '').endswith('.delta'):
            await self._queue_delta(stream, response_data, response_data.get('type', 'response.delta'))
            return

        response_data = self._with_full_output_index(response_data)
        await stream.deltas.flush()
        await self.event_emitter(
            {
                'type': 'response:completion',
                'data': get_response_completion_event_data(response_data),
            }
        )
        await self.save_stream(stream_output)

    # --- consuming one stream ---------------------------------------------------

    async def consume(self, response, form_data: dict) -> None:
        """Read one streamed response into the output, then queue the tool calls it asked for."""
        stream = _StreamPass(self, form_data)

        async for line in response.body_iterator:
            line = line.decode('utf-8', 'replace') if isinstance(line, bytes) else line
            # Skip empty lines
            if not line or line.isspace():
                continue
            if not line.startswith('data:'):
                await self._handle_plain_line(line, stream)
                continue
            try:
                if await self._handle_data_line(stream, line[5:].strip()):
                    break
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as e:
                if 'data: [DONE]' not in line:
                    log.debug('Error: %s', e)
        await stream.deltas.flush()

        self._finish_stream_output()
        await self._queue_tool_calls(stream)

    async def _handle_plain_line(self, data: str, stream: '_StreamPass') -> None:
        # Some upstreams return plain JSON error lines in a streaming response
        # (without SSE `data:` prefix). Try to normalize these into standard
        # error events so frontend and DB paths still receive them.
        try:
            raw_obj = JSONCodec.loads(data)
            raw_error = raw_obj.get('error') if isinstance(raw_obj, dict) else None
            if raw_error:
                stream.provider_error = True
                await self._save_error(raw_error)
                await self.event_emitter({'type': 'chat:completion', 'data': {'error': raw_error}})
        except Exception:
            pass

    async def _save_error(self, error) -> None:
        if self.save_to_chat:
            try:
                await Chats.upsert_message_to_chat_by_id_and_message_id(
                    self.metadata['chat_id'],
                    self.metadata['message_id'],
                    {
                        'error': {'content': error},
                    },
                )
            except Exception:
                pass

    async def _handle_data_line(self, stream: '_StreamPass', payload: str) -> bool:
        """Handle one SSE data payload; True when the stream must stop (a code block closed)."""
        data = JSONCodec.loads(payload)

        if self.filter_functions:
            data, _ = await process_filter_functions(
                request=self.request,
                filter_context=self.filter_context,
                filter_functions=self.filter_functions,
                filter_type='stream',
                form_data=data,
                extra_params=stream.filter_extra_params,
            )

        if not data:
            return False

        if 'event' in data and not getattr(self.request.state, 'direct', False):
            await self.event_emitter(data.get('event', {}))

        if 'selected_model_id' in data:
            await self._handle_selected_model(data)
            return False
        # Check for Responses API events (type field starts with "response.")
        if data.get('type', '').startswith('response.'):
            await self._handle_responses_event(stream, data)
            return False
        return await self._handle_chat_chunk(stream, data)

    async def _handle_selected_model(self, data: dict) -> None:
        if self.save_to_chat:
            await Chats.upsert_message_to_chat_by_id_and_message_id(
                self.metadata['chat_id'],
                self.metadata['message_id'],
                {
                    'selectedModelId': data['selected_model_id'],
                },
                touch=False,
            )
        await self.event_emitter(
            {
                'type': 'chat:completion',
                'data': data,
            }
        )

    async def _handle_responses_event(self, stream: '_StreamPass', data: dict) -> None:
        response_data_type = data.get('type', '')
        self.output, response_metadata = handle_responses_streaming_event(data, self.output)

        if not response_data_type.endswith('.delta'):
            await stream.deltas.flush()

        # Emit citation sources from finalized output items
        # (mirrors Chat Completions annotation handling at delta level)
        if response_data_type == 'response.output_item.done':
            await self._emit_item_citations(data.get('item', {}))

        # Merge any metadata (usage, etc.)
        # Strip 'done' — response.completed emits
        # it but we may still need to execute tool
        # calls. The outer middleware manages the
        # actual completion signal.
        if response_metadata:
            if ENABLE_RESPONSES_API_STATEFUL:
                response_id = response_metadata.pop('response_id', None)
                if response_id:
                    self.last_response_id = response_id

            # Normalize and capture usage for DB persistence
            if response_metadata.get('usage'):
                self.usage = merge_usage(self.usage, response_metadata['usage'])
                response_metadata['usage'] = self.usage

            if response_metadata.get('error'):
                stream.provider_error = True
                await self.event_emitter(
                    {
                        'type': 'chat:completion',
                        'data': {'error': response_metadata['error']},
                    }
                )

        await self.emit_response_event(stream, data)

        if response_metadata and response_metadata.get('usage'):
            await self.event_emitter(
                {
                    'type': 'chat:completion',
                    'data': {'usage': self.usage},
                }
            )

    async def _emit_item_citations(self, item: dict) -> None:
        if item.get('type') != 'message':
            return
        for part in item.get('content', []):
            for annotation in part.get('annotations', []):
                if annotation.get('type') == 'url_citation':
                    # Handle both flat (Responses API) and nested (Chat Completions) formats
                    url_citation = annotation.get('url_citation', annotation)

                    url = url_citation.get('url', '')
                    title = url_citation.get('title', url)

                    if url:
                        await self.event_emitter(_url_citation_source(title, url))

    async def _handle_chat_chunk(self, stream: '_StreamPass', data: dict) -> bool:
        """A Chat Completions chunk: usage, errors, then its delta. True when the stream must stop."""
        choices = data.get('choices', [])

        # Normalize usage data to standard format
        raw_usage = data.get('usage', {}) or {}
        raw_usage.update(data.get('timings', {}))  # llama.cpp
        if raw_usage:
            self.usage = merge_usage(self.usage, raw_usage)
            await self.event_emitter(
                {
                    'type': 'chat:completion',
                    'data': {
                        'usage': self.usage,
                    },
                }
            )

        # A provider error can come beside choices, not only instead of them:
        # OpenRouter ends a failed reply with a top-level `error` and a choice
        # whose finish_reason is "error", with HTTP 200 (#162).
        error = data.get('error') or (choices[0].get('error') if choices else None)
        if not error and choices and choices[0].get('finish_reason') == 'error':
            error = {'message': 'The provider ended the reply with an error.'}
        if error:
            stream.provider_error = True
            log.error('Provider returned error (streaming): %s', error)
            # Text still buffered goes out first, so the reply is not marked
            # failed (and done) before its last words arrive.
            await stream.deltas.flush()
            await self._save_error(error)
            await self.event_emitter(
                {
                    'type': 'chat:completion',
                    'data': {
                        'error': error,
                    },
                }
            )
            if not (choices and (choices[0].get('delta') or {}).get('content')):
                return False

        if not choices:
            return False

        delta = choices[0].get('delta', {})
        await self._emit_delta_citations(delta.get('annotations'))

        event_data, delta_type = data, 'content'
        delta_tool_calls = delta.get('tool_calls', None)
        if delta_tool_calls:
            if await self._track_tool_call_deltas(stream, delta_tool_calls):
                event_data, delta_type = None, 'tool_call'

        await self._handle_delta_images(delta.get('images'))

        event_data, delta_type = await self._handle_reasoning_delta(delta, event_data, delta_type)

        # content deltas are raw JSON: a stream filter can make them any type
        value = delta.get('content')
        if value and not isinstance(value, str):
            value = f'{value}'
        if value:
            content_event = await self._handle_content(value)
            if content_event is None:
                return True
            event_data, delta_type = content_event

        if delta and event_data:
            await self._queue_delta(stream, event_data, delta_type)
        elif event_data:
            await self.event_emitter(
                {
                    'type': 'chat:completion',
                    'data': event_data,
                }
            )
        return False

    async def _emit_delta_citations(self, annotations) -> None:
        for annotation in annotations or []:
            if annotation.get('type') == 'url_citation' and 'url_citation' in annotation:
                url_citation = annotation['url_citation']

                url = url_citation.get('url', '')
                title = url_citation.get('title', url)

                await self.event_emitter(_url_citation_source(title, url))

    async def _track_tool_call_deltas(self, stream: '_StreamPass', delta_tool_calls: list) -> bool:
        """Accumulate streamed tool calls and show them as function_call items. True once any are pending."""
        for delta_tool_call in delta_tool_calls:
            tool_call_index = delta_tool_call.get('index')
            if tool_call_index is None:
                continue

            # Check if the tool call already exists
            current_response_tool_call = next(
                (tc for tc in stream.response_tool_calls if tc.get('index') == tool_call_index), None
            )
            if current_response_tool_call is None:
                # Add the new tool call
                delta_tool_call.setdefault('function', {})
                delta_tool_call['function'].setdefault('name', '')
                delta_tool_call['id'] = delta_tool_call.get('id') or output_id('fc')
                delta_arguments = delta_tool_call['function'].get('arguments')
                if not isinstance(delta_arguments, str):
                    delta_tool_call['function']['arguments'] = (
                        '' if delta_arguments is None else JSONCodec.dumps(delta_arguments)
                    )
                stream.response_tool_calls.append(delta_tool_call)
            else:
                _merge_tool_call_delta(current_response_tool_call, delta_tool_call)

        if not stream.response_tool_calls:
            return False

        # Emit pending tool calls in real-time as Responses events.
        output_by_call_id = await self._show_pending_tool_calls(stream)
        for delta_tool_call in delta_tool_calls:
            tool_call_index = delta_tool_call.get('index')
            current_response_tool_call = next(
                (tc for tc in stream.response_tool_calls if tc.get('index') == tool_call_index),
                None,
            )
            if not current_response_tool_call:
                continue
            call_id = current_response_tool_call.get('id')
            output_index, _ = output_by_call_id.get(call_id, (len(self.output) - 1, {}))
            delta_arguments = delta_tool_call.get('function', {}).get('arguments')
            if delta_arguments is not None:
                if not isinstance(delta_arguments, str):
                    delta_arguments = JSONCodec.dumps(delta_arguments)
                await self.emit_response_event(
                    stream,
                    {
                        'type': 'response.function_call_arguments.delta',
                        'item_id': call_id,
                        'output_index': output_index,
                        'delta': delta_arguments,
                    },
                )

        await self.save_stream()
        return True

    async def _show_pending_tool_calls(self, stream: '_StreamPass') -> dict:
        """A function_call item per pending call (added on first sight); returns them by call id."""
        output_by_call_id = {
            item.get('call_id'): (idx, item)
            for idx, item in enumerate(self.output)
            if item.get('type') == 'function_call'
        }

        for tc in stream.response_tool_calls:
            call_id = tc.get('id') or output_id('fc')
            tc['id'] = call_id
            func = tc.get('function', {})
            if call_id in output_by_call_id:
                output_index, item = output_by_call_id[call_id]
                item['name'] = func.get('name', item.get('name', ''))
                item['arguments'] = func.get('arguments', item.get('arguments', ''))
                item['status'] = 'in_progress'
            else:
                output_index = len(self.output)
                item = {
                    'type': 'function_call',
                    'id': call_id,
                    'call_id': call_id,
                    'name': func.get('name', ''),
                    'arguments': '',
                    'status': 'in_progress',
                }
                self.output.append(item)
                output_by_call_id[call_id] = (output_index, item)
                await self.emit_response_event(
                    stream,
                    {
                        'type': 'response.output_item.added',
                        'output_index': output_index,
                        'item': item.copy(),
                    },
                )
                item['arguments'] = func.get('arguments', '')
        return output_by_call_id

    async def _handle_delta_images(self, delta_images) -> None:
        image_urls = await get_image_urls(delta_images, self.request, self.metadata, self.user) if delta_images else []
        if not image_urls:
            return
        image_file_list = [{'type': 'image', 'url': url} for url in image_urls]
        message_files = image_file_list
        if self.save_to_chat:
            message_files = await Chats.add_message_files_by_id_and_message_id(
                self.metadata['chat_id'],
                self.metadata['message_id'],
                image_file_list,
            )
            if message_files is None:
                message_files = image_file_list

        await self.event_emitter(
            {
                'type': 'files',
                'data': {'files': message_files},
            }
        )

    async def _handle_reasoning_delta(self, delta: dict, event_data, delta_type):
        """Reasoning text and details go to a reasoning item; returns the event to send for this chunk."""
        # reasoning deltas are raw JSON: a stream filter can make them any type
        reasoning_content = delta.get('reasoning_content') or delta.get('reasoning') or delta.get('thinking')
        if reasoning_content and not isinstance(reasoning_content, str):
            reasoning_content = f'{reasoning_content}'
        reasoning_details = get_reasoning_details(delta)
        reasoning_detail_items = (
            [item for item in reasoning_details if isinstance(item, dict)]
            if isinstance(reasoning_details, list)
            else [reasoning_details]
            if isinstance(reasoning_details, dict)
            else []
        )
        existing_reasoning_item = next(
            (item for item in reversed(self.output) if item.get('type') == 'reasoning'),
            None,
        )
        message_index = next(
            (i for i, item in enumerate(self.output) if item.get('type') == 'message'),
            None,
        )
        if not (
            reasoning_content
            or (
                reasoning_detail_items
                and (
                    existing_reasoning_item
                    or any(
                        item.get('text') or item.get('summary') or item.get('data') for item in reasoning_detail_items
                    )
                )
            )
        ):
            return event_data, delta_type

        reasoning_item = (
            existing_reasoning_item
            if (reasoning_detail_items and not reasoning_content) or message_index is not None
            else None
        )
        if reasoning_item is None:
            reasoning_item = self._open_reasoning_item(message_index)

        if reasoning_content:
            # Append to reasoning content
            parts = reasoning_item.get('content', [])
            if parts and parts[-1].get('type') == 'output_text':
                parts[-1]['text'] += reasoning_content
            else:
                reasoning_item['content'] = [
                    {
                        'type': 'output_text',
                        'text': reasoning_content,
                    }
                ]

            event_data = {
                'type': 'response.reasoning_text.delta',
                'item_id': reasoning_item.get('id'),
                'output_index': self.output.index(reasoning_item),
                'content_index': max(
                    len(reasoning_item.get('content', [])) - 1,
                    0,
                ),
                'delta': reasoning_content,
            }
            delta_type = 'response.reasoning_text.delta'

        if reasoning_detail_items:
            merge_streamed_reasoning_details(
                reasoning_item.setdefault('reasoning_details', []),
                reasoning_detail_items,
            )
            await self.save_stream()
            # Providers such as OpenRouter send reasoning_details
            # alongside the reasoning text: only drop the event when
            # the details were all there was to report, otherwise the
            # reasoning delta never reaches the client.
            if not reasoning_content:
                event_data = None
        return event_data, delta_type

    def _open_reasoning_item(self, message_index):
        """The reasoning item to stream into: the open last one, or a new one (completed if a message began)."""
        if self.output and self.output[-1].get('type') == 'reasoning':
            return self.output[-1]
        reasoning_item = {
            'type': 'reasoning',
            'id': output_id('r'),
            'status': 'in_progress',
            'start_tag': '<think>',
            'end_tag': '</think>',
            'attributes': {'type': 'reasoning_content'},
            'content': [],
            'summary': None,
            'started_at': time.time(),
        }
        if message_index is not None:
            reasoning_item['ended_at'] = time.time()
            reasoning_item['duration'] = 0
            reasoning_item['status'] = 'completed'
            self.output.insert(message_index, reasoning_item)
        else:
            self.output.append(reasoning_item)
        return reasoning_item

    async def _handle_content(self, value: str):
        """Append streamed text to the output and scan it for tags.

        Returns the delta event and its type, or None when a code block closed and the stream must stop."""
        self._close_streamed_reasoning()

        if ENABLE_CHAT_RESPONSE_BASE64_IMAGE_URL_CONVERSION:
            value = await convert_markdown_base64_images(
                self.request,
                value,
                {
                    'chat_id': self.metadata.get('chat_id', None),
                    'message_id': self.metadata.get('message_id', None),
                },
                self.user,
            )

        # closure-cell str += recopies per chunk; append + join once at read is O(n)
        self.content_parts.append(value)
        self._append_text(value)

        if self.detect_reasoning_tags:
            self.output, _ = self.tags.scan('reasoning', self.reasoning_tags, self.output)
            self.output, _ = self.tags.scan('solution', DEFAULT_SOLUTION_TAGS, self.output)

        if self.detect_code_interpreter:
            self.output, end = self.tags.scan('code_interpreter', DEFAULT_CODE_INTERPRETER_TAGS, self.output)
            if end:
                return None

        target_index = len(self.output) - 1
        target_item = self.output[target_index] if target_index >= 0 else {}
        target_content = target_item.get('content', [])
        delta_event_type = (
            'response.reasoning_text.delta' if target_item.get('type') == 'reasoning' else 'response.output_text.delta'
        )
        return (
            {
                'type': delta_event_type,
                'item_id': target_item.get('id'),
                'output_index': target_index,
                'content_index': max(len(target_content) - 1, 0),
                'delta': value,
            },
            delta_event_type,
        )

    def _close_streamed_reasoning(self) -> None:
        # Text after reasoning_content ends the reasoning item and starts the answer.
        if (
            self.output
            and self.output[-1].get('type') == 'reasoning'
            and self.output[-1].get('attributes', {}).get('type') == 'reasoning_content'
        ):
            reasoning_item = self.output[-1]
            reasoning_item['ended_at'] = time.time()
            reasoning_item['duration'] = int(reasoning_item['ended_at'] - reasoning_item['started_at'])
            reasoning_item['status'] = 'completed'
            self.output.append(_message_output_item())

    def _append_text(self, value: str) -> None:
        # Inside a tag-based block (reasoning, code_interpreter, or solution), append
        # to the in-progress item instead of starting a message — otherwise the
        # tag scanner re-detects the start tag on every chunk and fragments the output.
        last_item = self.output[-1] if self.output else None
        last_item_type = last_item.get('type', '') if last_item else ''
        inside_tag_block = (
            last_item is not None
            and last_item.get('status') == 'in_progress'
            and last_item.get('attributes', {}).get('type') != 'reasoning_content'
            and (
                last_item_type == 'reasoning'
                or last_item_type == 'open_webui:code_interpreter'
                or (last_item_type == 'message' and last_item.get('_tag_type') is not None)
            )
        )

        if inside_tag_block and last_item_type == 'open_webui:code_interpreter':
            last_item['code'] = last_item.get('code', '') + value
            return
        if not inside_tag_block and (not self.output or self.output[-1].get('type') != 'message'):
            self.output.append(_message_output_item())

        # The reasoning item, the tagged message, or the last message gets the text.
        item = last_item if inside_tag_block else self.output[-1]
        parts = item.get('content', [])
        if parts and parts[-1].get('type') == 'output_text':
            parts[-1]['text'] += value
        else:
            item['content'] = [{'type': 'output_text', 'text': value}]

    def _finish_stream_output(self) -> None:
        if not self.output:
            return
        # Clean up the last message item
        if self.output[-1].get('type') == 'message':
            parts = self.output[-1].get('content', [])
            if parts and parts[-1].get('type') == 'output_text':
                parts[-1]['text'] = parts[-1]['text'].strip()

                if not parts[-1]['text']:
                    self.output.pop()

                    if not self.output:
                        self.output.append(_message_output_item())

        if self.output[-1].get('type') == 'reasoning':
            reasoning_item = self.output[-1]
            if reasoning_item.get('ended_at') is None:
                reasoning_item['ended_at'] = time.time()
                if reasoning_item.get('started_at') is not None:
                    reasoning_item['duration'] = int(reasoning_item['ended_at'] - reasoning_item['started_at'])
                reasoning_item['status'] = 'completed'

    async def _queue_tool_calls(self, stream: '_StreamPass') -> None:
        """Finish the streamed function_call items and queue their calls; or, for the Responses API, queue
        the function_call items that have no output yet."""
        if stream.provider_error:
            return
        if stream.response_tool_calls:
            for tc in stream.response_tool_calls:
                call_id = tc.get('id', '')
                arguments = tc.get('function', {}).get('arguments', '{}')
                for output_index, item in enumerate(self.output):
                    if item.get('type') == 'function_call' and item.get('call_id') == call_id:
                        item['arguments'] = arguments
                        item['status'] = 'completed'
                        await self.emit_response_event(
                            stream,
                            {
                                'type': 'response.function_call_arguments.done',
                                'item_id': item.get('id'),
                                'output_index': output_index,
                                'arguments': arguments,
                            },
                        )
                        await self.emit_response_event(
                            stream,
                            {
                                'type': 'response.output_item.done',
                                'output_index': output_index,
                                'item': item.copy(),
                            },
                        )
                        break
            self.tool_calls.append(_split_tool_calls(stream.response_tool_calls))
            return

        if not self.output:
            return
        # Collect call_ids that already have results, including those from
        # prior_output so we don't re-process tool calls from a previous turn.
        handled_call_ids = {
            item.get('call_id')
            for item in (self.prior_output + self.output)
            if item.get('type') == 'function_call_output'
        }
        responses_api_tool_calls = []
        for item in self.output:
            call_id = item.get('call_id') or item.get('id') or output_id('fc')
            if item.get('type') == 'function_call' and call_id not in handled_call_ids:
                arguments = item.get('arguments', '{}')
                responses_api_tool_calls.append(
                    {
                        'id': call_id,
                        'index': len(responses_api_tool_calls),
                        'function': {
                            'name': item.get('name', ''),
                            'arguments': (arguments if isinstance(arguments, str) else JSONCodec.dumps(arguments)),
                        },
                    }
                )
        if responses_api_tool_calls:
            self.tool_calls.append(_split_tool_calls(responses_api_tool_calls))

    # --- the reply ------------------------------------------------------------

    async def run(self, response, events: list) -> None:
        await self._load_starting_output()

        self.reasoning_tags_param = self.metadata.get('params', {}).get('reasoning_tags')
        self.detect_reasoning_tags = self.reasoning_tags_param is not False
        self.detect_code_interpreter = await self._detect_code_interpreter()
        self.reasoning_tags = self._reasoning_tags()

        try:
            await self._emit_events(events)

            try:
                await self.consume(response, self.form_data)
            finally:
                if response.background:
                    await response.background()

            if not await self._run_tool_calls():
                return
            if self.detect_code_interpreter:
                await self._run_code_interpreter()
            await self._finish()
        except asyncio.CancelledError:
            log.warning('Task was cancelled!')
            await self._save_cancelled_state(response)
            raise  # re-raise CancelledError for proper propagation

        if response.background is not None:
            await response.background()

    async def _emit_events(self, events: list) -> None:
        for event in events:
            await self.event_emitter(
                {
                    'type': 'chat:completion',
                    'data': event,
                }
            )

            # Save message in the database
            if self.save_to_chat:
                await Chats.upsert_message_to_chat_by_id_and_message_id(
                    self.metadata['chat_id'],
                    self.metadata['message_id'],
                    {
                        **event,
                    },
                )

    async def _finish(self) -> None:
        # Mark all in-progress items as completed
        for item in self.output:
            if item.get('status') == 'in_progress':
                item['status'] = 'completed'

        current_output = self.full_output()
        title = await Chats.get_chat_title_by_id(self.metadata['chat_id']) if self.save_to_chat else ''
        data = {
            'done': True,
            'output': current_output,
            'title': title,
            **({'usage': self.usage} if self.usage else {}),
        }

        if self.save_to_chat:
            # Save final output once. The delta path keeps in-progress
            # state in response_streams instead of writing tokens to DB.
            await Chats.upsert_message_to_chat_by_id_and_message_id(
                self.metadata['chat_id'],
                self.metadata['message_id'],
                {
                    'done': True,
                    'output': current_output,
                    **({'usage': self.usage} if self.usage else {}),
                },
            )

        await clear_response_stream(self.request.app.state.redis, self.response_stream_task_id)
        await publish_chat_finished_event(
            self.request, self.user, self.metadata, title, ''.join(self.content_parts), current_output
        )

        await self.event_emitter(
            {
                'type': 'chat:completion',
                'data': data,
            }
        )

        self.ctx['assistant_message'] = {
            'content': ''.join(self.content_parts) or get_output_text(current_output),
            'output': current_output,
            **({'usage': self.usage} if self.usage else {}),
        }
        await outlet_filter_handler(self.ctx)
        await background_tasks_handler(self.ctx)

    async def _save_cancelled_state(self, response) -> None:
        # Close the response body iterator to trigger cleanup
        # in stream_wrapper's finally block and release the
        # upstream connection.  Without this, the async
        # generator is orphaned and may spin in anyio internals.
        if hasattr(response, 'body_iterator') and hasattr(response.body_iterator, 'aclose'):
            try:
                await asyncio.shield(response.body_iterator.aclose())
            except (asyncio.CancelledError, Exception):
                pass

        async def save_cancelled_state():
            await self.event_emitter({'type': 'chat:tasks:cancel'})
            if self.save_to_chat:
                await Chats.upsert_message_to_chat_by_id_and_message_id(
                    self.metadata['chat_id'],
                    self.metadata['message_id'],
                    {
                        'done': True,
                        'output': self.full_output(),
                    },
                )
            await clear_response_stream(self.request.app.state.redis, self.response_stream_task_id)

        try:
            await asyncio.shield(save_cancelled_state())
        except (asyncio.CancelledError, Exception):
            pass

    # --- tool calls -------------------------------------------------------------

    async def _run_tool_calls(self) -> bool:
        """Run each round of tool calls and continue the reply with their results.

        Returns False when the reply paused for the user (approval or ask_user)."""
        tool_call_iterations = 0
        max_tool_call_iterations = getattr(
            self.request.state,
            'max_tool_call_iterations',
            CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS,
        )
        rounds = _ToolCallRounds(self)

        while self.tool_calls and (max_tool_call_iterations is None or tool_call_iterations < max_tool_call_iterations):
            tool_call_iterations += 1

            response_tool_calls = self.tool_calls.pop(0)
            response_tool_calls, paused = await rounds.stage(response_tool_calls)
            if paused:
                return False

            await self.event_emitter(
                {
                    'type': 'chat:completion',
                    'data': {
                        'output': self.full_output(),
                    },
                }
            )

            results = await rounds.execute(response_tool_calls)
            rounds.record_results(results, response_tool_calls)
            await rounds.apply_sources()
            await self._emit_frontend_output()

            if not await self._continue_after_tools():
                break

        if (
            max_tool_call_iterations is not None
            and self.tool_calls
            and tool_call_iterations >= max_tool_call_iterations
        ):
            log.warning('Tool-call iteration limit reached (%s)', max_tool_call_iterations)
            error_content = f'Tool-call limit reached ({max_tool_call_iterations} iterations).'
            await self.emit_message_error(error_content)
        return True

    async def _emit_frontend_output(self) -> None:
        # Strip input_image parts (large base64 data URIs) from the
        # output sent to the frontend — they're only for LLM consumption
        # via convert_output_to_messages.
        frontend_output = []
        for item in self.full_output():
            if item.get('type') == 'function_call_output':
                parts = item.get('output', [])
                if any(p.get('type') == 'input_image' for p in parts):
                    item = {**item, 'output': [p for p in parts if p.get('type') != 'input_image']}
            frontend_output.append(item)

        await self.event_emitter(
            {
                'type': 'chat:completion',
                'data': {
                    'output': frontend_output,
                },
            }
        )

    def _continuation_form_data(self) -> dict:
        new_form_data = {
            **self.form_data,
            'model': self.model_id,
            'stream': True,
            'metadata': self.metadata,
        }

        if ENABLE_RESPONSES_API_STATEFUL and self.last_response_id:
            system_message = get_system_message(self.form_data['messages'])
            new_form_data['messages'] = ([system_message] if system_message else []) + convert_output_to_messages(
                self.output, raw=True, reasoning_format=get_reasoning_format(self.model)
            )
            new_form_data['previous_response_id'] = self.last_response_id
        else:
            # Tool images are flattened into a user message after the tool
            # messages, since Chat Completions tool messages are text-only.
            new_form_data['messages'] = [
                *self.form_data['messages'],
                *convert_output_to_messages(
                    self.output,
                    raw=True,
                    reasoning_format=get_reasoning_format(self.model),
                    flatten_tool_images=True,
                ),
            ]
        return new_form_data

    async def _request_continuation(self, new_form_data: dict):
        if self.filter_functions:
            new_form_data, _ = await process_filter_functions(
                request=self.request,
                filter_context=self.filter_context,
                filter_functions=self.filter_functions,
                filter_type='request',
                form_data=new_form_data,
                extra_params=self.extra_params,
            )

        new_form_data = normalize_messages_for_model(new_form_data)

        res = await generate_chat_completion(
            self.request,
            new_form_data,
            self.user,
            bypass_system_prompt=True,
        )
        return res, new_form_data

    async def _continue_after_tools(self) -> bool:
        """Ask the model to carry on from the tool results. False when the reply ends here."""
        try:
            res, new_form_data = await self._request_continuation(self._continuation_form_data())

            if isinstance(res, StreamingResponse):
                # Save accumulated output and start fresh.
                # Responses API output_index values are relative
                # to the current response — a clean output list
                # keeps indices aligned. The display prefix
                # ensures the UI shows tool history during
                # streaming.
                self.prior_output = list(self.full_output())
                # Trim the trailing empty placeholder message
                # so it doesn't persist as a ghost item once
                # the new stream produces real content.
                self._trim_empty_trailing_message(self.prior_output)
                self.output = []
                await self.consume(res, new_form_data)
                self.output[:0] = self.prior_output
                self.prior_output = []
                return True
            if getattr(res, 'status_code', 200) >= 400:
                await self.emit_message_error(_message_error_content(get_response_error_detail(res)))
            return False
        except Exception as e:
            error_content = _message_error_content(e)
            log.exception('Tool-call continuation failed: %s', error_content)
            await self.emit_message_error(error_content)
            return False

    # --- code interpreter -------------------------------------------------------

    async def _run_code_interpreter(self) -> None:
        MAX_RETRIES = 5
        retries = 0
        while self.output and self.output[-1].get('type') == 'open_webui:code_interpreter' and retries < MAX_RETRIES:
            await self.event_emitter(
                {
                    'type': 'chat:completion',
                    'data': {
                        'output': self.full_output(),
                    },
                }
            )

            retries += 1
            log.debug('Attempt count: %s', retries)

            ci_item = self.output[-1]
            try:
                ci_output = (
                    await self._execute_code(ci_item) if ci_item.get('attributes', {}).get('type') == 'code' else ''
                )
            except Exception as e:
                ci_output = str(e)

            ci_item['output'] = ci_output
            ci_item['status'] = 'completed'

            self.output.append(_message_output_item())

            await self.event_emitter(
                {
                    'type': 'chat:completion',
                    'data': {
                        'output': self.full_output(),
                    },
                }
            )

            if not await self._continue_after_code():
                break

    async def _execute_code(self, ci_item: dict):
        # Sanitize code (strips ANSI codes and markdown fences)
        code = sanitize_code(ci_item.get('code', ''))

        if CODE_INTERPRETER_BLOCKED_MODULES:
            blocking_code = textwrap.dedent(f"""
                import builtins

                BLOCKED_MODULES = {CODE_INTERPRETER_BLOCKED_MODULES}

                _real_import = builtins.__import__
                def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
                    if name.split('.')[0] in BLOCKED_MODULES:
                        importer_name = globals.get('__name__') if globals else None
                        if importer_name == '__main__':
                            raise ImportError(
                                f"Direct import of module {{name}} is restricted."
                            )
                    return _real_import(name, globals, locals, fromlist, level)

                builtins.__import__ = restricted_import
            """)
            code = blocking_code + '\n' + code

        ci_engine = await Config.get('code_interpreter.engine')
        if ci_engine == 'pyodide':
            ci_output = await self.event_caller(
                {
                    'type': 'execute:python',
                    'data': {
                        'id': str(uuid4()),
                        'code': code,
                        'session_id': self.metadata.get('session_id', None),
                        'files': self.metadata.get('files', []),
                    },
                }
            )
        elif ci_engine == 'jupyter':
            files, notes = await chat_files(self.metadata.get('files'), self.user)
            ci_output = await execute_code_jupyter(
                await Config.get('code_interpreter.jupyter.url'),
                code,
                (
                    await Config.get('code_interpreter.jupyter.auth_token')
                    if await Config.get('code_interpreter.jupyter.auth') == 'token'
                    else None
                ),
                (
                    await Config.get('code_interpreter.jupyter.auth_password')
                    if await Config.get('code_interpreter.jupyter.auth') == 'password'
                    else None
                ),
                await Config.get('code_interpreter.jupyter.timeout'),
                files=files,
                notes=notes,
            )
        else:
            ci_output = {'stdout': 'Code interpreter engine not configured.'}

        log.debug('Code interpreter output: %s', ci_output)

        # Handle error responses from event_caller
        # (e.g. session disconnected, timeout)
        if isinstance(ci_output, dict) and ci_output.get('error'):
            ci_output = {'stderr': ci_output['error']}

        if isinstance(ci_output, dict):
            await self._inline_output_images(ci_output)
        return ci_output

    async def _inline_output_images(self, ci_output: dict) -> None:
        """Replace base64 image lines in stdout and the result with uploaded images."""
        stdout = ci_output.get('stdout', '')
        if isinstance(stdout, str):
            stdout_lines = stdout.split('\n')
            for idx, line in enumerate(stdout_lines):
                if re.match(r'data:image/\w+;base64', line):
                    image_url = await get_image_url_from_base64(self.request, line, self.metadata, self.user)
                    if image_url:
                        stdout_lines[idx] = f'![Output Image]({image_url})'
            ci_output['stdout'] = '\n'.join(stdout_lines)

        result = ci_output.get('result', '')
        if isinstance(result, str):
            result_lines = result.split('\n')
            for idx, line in enumerate(result_lines):
                if re.match(r'data:image/\w+;base64', line):
                    image_url = await get_image_url_from_base64(self.request, line, self.metadata, self.user)
                    result_lines[idx] = f'![Output Image]({image_url})'
            ci_output['result'] = '\n'.join(result_lines)

    async def _continue_after_code(self) -> bool:
        """Ask the model to carry on from the code's output. False when the reply ends here."""
        try:
            res, new_form_data = await self._request_continuation(
                {
                    **self.form_data,
                    'model': self.model_id,
                    'stream': True,
                    'metadata': self.metadata,
                    'messages': [
                        *self.form_data['messages'],
                        *convert_output_to_messages(
                            self.output,
                            raw=True,
                            reasoning_format=get_reasoning_format(self.model),
                            flatten_tool_images=True,
                        ),
                    ],
                }
            )

            if isinstance(res, StreamingResponse):
                await self.consume(res, new_form_data)
                return True
            if getattr(res, 'status_code', 200) >= 400:
                await self.emit_message_error(_message_error_content(get_response_error_detail(res)))
            return False
        except Exception as e:
            error_content = _message_error_content(e)
            log.exception('Code interpreter continuation failed: %s', error_content)
            await self.emit_message_error(error_content)
            return False


class _StreamPass:
    """State for reading one streamed response: its tool calls and its pending deltas."""

    def __init__(self, reply: _StreamingReply, form_data: dict):
        self.response_tool_calls = []
        self.provider_error = False
        self.deltas = _DeltaBuffer(
            reply.event_emitter,
            reply.save_stream,
            max(
                CHAT_RESPONSE_STREAM_DELTA_CHUNK_SIZE,
                int(reply.metadata.get('params', {}).get('stream_delta_chunk_size') or 1),
            ),
        )
        self.filter_extra_params = {'__body__': form_data, **reply.extra_params} if reply.filter_functions else None


class _ToolCallRounds:
    """Running the reply's tool calls, round by round, and feeding their results and citations back."""

    def __init__(self, reply: _StreamingReply):
        self.reply = reply
        self.tool_call_sources = []  # Track citation sources from tool results
        self.all_tool_call_sources = []  # Accumulated sources across all iterations
        self.user_message = get_last_user_message(reply.form_data['messages'])

        # Check if citations are enabled for this model
        self.citations_enabled = (reply.model.get('info', {}).get('meta', {}).get('capabilities') or {}).get(
            'citations', True
        )

        # Use the pre-RAG system content captured before the
        # initial file-source injection in process_chat_payload.
        # This ensures restore truly undoes the RAG template.
        self.original_system_content = reply.metadata.get('system_prompt')
        if self.original_system_content is None:
            original_system_message = get_system_message(reply.form_data['messages'])
            self.original_system_content = (
                get_text_from_message(original_system_message) if original_system_message else None
            )

    async def _pause(self) -> None:
        reply = self.reply
        if is_saved_chat_id(reply.metadata.get('chat_id')) and reply.metadata.get('message_id'):
            await pause_for_tool_approval(
                reply.metadata['chat_id'],
                reply.metadata['message_id'],
                reply.full_output(),
                reply.form_data,
                reply.metadata,
            )
        await reply.event_emitter({'type': 'chat:completion', 'data': {'output': reply.full_output()}})

    async def stage(self, response_tool_calls: list) -> tuple[list, bool]:
        """Add the round's function_call items; pause for ask_user or tool approval. Returns the calls to run
        and whether the reply paused."""
        reply = self.reply
        ask_user_staged, ask_user_error = stage_ask_user_tool_calls(response_tool_calls, reply.output, output_id)
        if ask_user_error:
            response_tool_calls = [
                tool_call
                for tool_call in response_tool_calls
                if tool_call.get('function', {}).get('name') != 'ask_user'
            ]
        elif ask_user_staged:
            await self._pause()
            return response_tool_calls, True

        # Append function_call items for each tool call
        # (Responses API already has them from streaming, so skip duplicates)
        existing_call_ids = {item.get('call_id') for item in reply.output if item.get('type') == 'function_call'}
        for tc in response_tool_calls:
            call_id = tc.get('id', '')
            if call_id not in existing_call_ids:
                func = tc.get('function', {})
                reply.output.append(
                    {
                        'type': 'function_call',
                        'id': call_id or output_id('fc'),
                        'call_id': call_id,
                        'name': func.get('name', ''),
                        'arguments': func.get('arguments', '{}'),
                        'status': 'in_progress',
                    }
                )

        tool_approval_mode = reply.metadata.get('params', {}).get('tool_approval_mode', 'full')
        if (
            response_tool_calls
            and tool_approval_mode == 'ask'
            and is_saved_chat_id(reply.metadata.get('chat_id'))
            and reply.metadata.get('message_id')
        ):
            await self._pause()
            return response_tool_calls, True
        return response_tool_calls, False

    async def _execute_tool_call(self, tools: dict, tool_call: dict):
        reply = self.reply
        name = tool_call.get('function', {}).get('name', '')
        params = _parse_tool_params(tool_call)
        if params is None:
            return {}, None, None, None, False
        tool = tools.get(name)
        if not tool:
            return params, f'Error: Tool "{name}" not found.', None, None, False
        spec = tool.get('spec', {})
        tool_type = tool.get('type', '')
        direct_tool = tool.get('direct', False)
        allowed_params = spec.get('parameters', {}).get('properties', {}).keys()
        params = {key: value for key, value in params.items() if key in allowed_params}
        try:
            if direct_tool:
                result = await reply.event_caller(
                    {
                        'type': 'execute:tool',
                        'data': {
                            'id': str(uuid4()),
                            'name': name,
                            'params': params,
                            'server': tool.get('server', {}),
                            'session_id': reply.metadata.get('session_id'),
                        },
                    }
                )
            else:
                function = await get_updated_tool_function(
                    function=tool['callable'],
                    extra_params={
                        '__messages__': reply.form_data.get('messages', []),
                        '__files__': reply.metadata.get('files', []),
                    },
                )
                result = await function(**params)
        except Exception as e:
            result = {'error': str(e)}
        return params, result, tool, tool_type, direct_tool

    async def execute(self, response_tool_calls: list) -> list[dict]:
        """Run the round's calls (delegate_task calls together, the rest in order) and return their results."""
        tools = self.reply.metadata.get('tools', {})
        delegate_calls = [
            tool_call
            for tool_call in response_tool_calls
            if tool_call.get('function', {}).get('name') == 'delegate_task'
        ]
        tool_results = {}
        for tool_call in response_tool_calls:
            if tool_call.get('function', {}).get('name') != 'delegate_task':
                tool_results[id(tool_call)] = await self._execute_tool_call(tools, tool_call)
        tool_results.update(
            zip(
                [id(tool_call) for tool_call in delegate_calls],
                await asyncio.gather(*(self._execute_tool_call(tools, tool_call) for tool_call in delegate_calls)),
                strict=True,
            )
        )

        results = []
        for tool_call in response_tool_calls:
            results.append(await self._tool_call_result(tool_call, tool_results[id(tool_call)]))
        return results

    async def _tool_call_result(self, tool_call: dict, tool_execution) -> dict:
        reply = self.reply
        tool_call_id = tool_call.get('id', '')
        tool_function_name = tool_call.get('function', {}).get('name', '')
        tool_function_params, tool_result, tool, tool_type, direct_tool = tool_execution
        if tool_result is None:
            return {
                'tool_call_id': tool_call_id,
                'content': (
                    'Error: Tool call arguments could not be parsed. The model generated '
                    f'malformed or incomplete JSON for `{tool_function_name}`. Please try again.'
                ),
            }

        terminal_file_result = build_terminal_file_tool_result(
            tool_function_name,
            tool_function_params,
            tool_result,
            tool,
            reply.metadata,
        )
        if terminal_file_result:
            tool_result = terminal_file_result

        tool_result, tool_result_files, tool_result_embeds = await process_tool_result(
            reply.request,
            tool_function_name,
            tool_result,
            tool_type,
            direct_tool,
            reply.metadata,
            reply.user,
        )

        await terminal_event_handler(
            tool_function_name,
            tool_function_params,
            tool_result,
            reply.event_emitter,
        )

        # Extract citation sources from tool results
        if self.citations_enabled and tool_function_name in _CITATION_TOOLS and tool_result:
            try:
                citation_sources = get_citation_source_from_tool_result(
                    tool_name=tool_function_name,
                    tool_params=tool_function_params,
                    tool_result=tool_result,
                    tool_id=tool.get('tool_id', '') if tool else '',
                )
                self.tool_call_sources.extend(citation_sources)
            except Exception as e:
                log.exception(f'Error extracting citation source: {e}')

        return {
            'tool_call_id': tool_call_id,
            'content': tool_result_content(tool_result),
            **({'files': tool_result_files} if tool_result_files else {}),
            **({'embeds': tool_result_embeds} if tool_result_embeds else {}),
        }

    def record_results(self, results: list[dict], response_tool_calls: list) -> None:
        """A function_call_output per result, and the round's function_call items marked done or failed."""
        output = self.reply.output
        result_status_by_call_id = {}
        for result in results:
            output_parts = [{'type': 'input_text', 'text': result.get('content', '')}]
            local_output_status = 'failed' if _is_tool_result_error(result.get('content', '')) else 'completed'
            result_status_by_call_id[result.get('tool_call_id', '')] = local_output_status

            # Separate image data URIs (for LLM via input_image) from
            # other files (for frontend display via files attribute).
            display_files = []
            for file_item in result.get('files', []):
                if file_item.get('type') == 'image' and file_item.get('url', '').startswith('data:'):
                    # LLM-only: add as input_image part, not frontend display output.
                    output_parts.append({'type': 'input_image', 'image_url': file_item['url']})
                else:
                    # Frontend display (MCP images, audio, etc.)
                    display_files.append(file_item)

            output.append(
                {
                    'type': 'function_call_output',
                    'id': output_id('fco'),
                    'call_id': result.get('tool_call_id', ''),
                    'output': output_parts,
                    'status': local_output_status,
                    **({'files': display_files} if display_files else {}),
                    **({'embeds': result.get('embeds')} if result.get('embeds') else {}),
                }
            )

        # Update function_call statuses and parsed/sanitized arguments.
        for tc in response_tool_calls:
            call_id = tc.get('id', '')
            for item in output:
                if item.get('type') == 'function_call' and item.get('call_id') == call_id:
                    item['status'] = result_status_by_call_id.get(call_id, 'completed')
                    item['arguments'] = tc.get('function', {}).get('arguments', '{}')
                    break

    async def apply_sources(self) -> None:
        """Show the round's citations and rebuild the RAG context in the messages for the next request."""
        if not self.citations_enabled:
            return
        reply = self.reply
        for source in self.tool_call_sources:
            await reply.event_emitter({'type': 'source', 'data': source})

        # Apply tool source context to messages for the model.
        # Restoring to pre-RAG original prevents duplicating
        # the RAG template across file and tool sources.
        self.all_tool_call_sources.extend(self.tool_call_sources)
        if self.all_tool_call_sources and self.user_message:
            self._restore_original_prompts()
            await self._add_source_context()
        self.tool_call_sources.clear()

    def _restore_original_prompts(self) -> None:
        # Restore pre-RAG message state before re-applying to prevent RAG template duplication.
        form_data = self.reply.form_data
        original_user_message = self.reply.metadata.get('user_prompt') or self.user_message
        set_last_user_message_content(original_user_message, form_data['messages'])
        if self.original_system_content is None:
            replace_system_message_content('', form_data['messages'])
        elif get_system_message(form_data['messages']):
            replace_system_message_content(self.original_system_content, form_data['messages'])
        else:
            form_data['messages'] = add_or_update_system_message(self.original_system_content, form_data['messages'])

    async def _add_source_context(self) -> None:
        # Build context: file sources with content, tool sources as citation markers only.
        form_data = self.reply.form_data
        source_ids = {}
        source_context = get_source_context(self.reply.metadata.get('sources', []), source_ids) + get_source_context(
            self.all_tool_call_sources,
            source_ids,
            include_content=False,
        )
        source_context = source_context.strip()
        if not source_context:
            return
        rag_content = await rag_template(
            await Config.get('rag.template'),
            source_context,
            self.user_message,
        )
        if RAG_SYSTEM_CONTEXT:
            form_data['messages'] = add_or_update_system_message(rag_content, form_data['messages'], append=True)
        else:
            form_data['messages'] = add_or_update_user_message(rag_content, form_data['messages'], append=False)


async def _passthrough_stream(response, ctx: dict, events: list, filter_functions: list, extra_params: dict):
    """Without an event emitter (direct API callers), relay the stream as is, through any stream filters,
    and run the outlet filters on the assembled reply."""
    request = ctx['request']
    model = ctx['model']

    def wrap_item(item):
        return f'data: {item}\n\n'

    assistant_message = {}
    filter_context = FilterContext()
    has_api_outlet_filters = ENABLE_API_OUTLET_FILTERS and bool(filter_functions)
    if ENABLE_API_OUTLET_FILTERS and not has_api_outlet_filters:
        try:
            model_id = model.get('id') if isinstance(model, dict) else model
            has_api_outlet_filters = bool(
                (isinstance(model, dict) and 'pipeline' in model)
                or get_sorted_filters(model_id, request.app.state.MODELS)
            )
        except Exception:
            has_api_outlet_filters = True

    for event in events:
        event, _ = await process_filter_functions(
            request=request,
            filter_context=filter_context,
            filter_functions=filter_functions,
            filter_type='stream',
            form_data=event,
            extra_params=extra_params,
        )

        if event:
            yield wrap_item(JSONCodec.dumps(event))

    async for data in response.body_iterator:
        if filter_functions:
            data = await _filter_stream_line(request, filter_context, filter_functions, extra_params, data, wrap_item)

        if data:
            if has_api_outlet_filters:
                update_assistant_message_from_stream(assistant_message, data)
            yield data

    if has_api_outlet_filters and assistant_message:
        ctx['assistant_message'] = assistant_message
        await outlet_filter_handler(ctx)


async def _filter_stream_line(request, filter_context, filter_functions, extra_params, data, wrap_item):
    """A data line after the stream filters (None when they drop it); other lines pass unchanged."""
    line = data.decode('utf-8', 'replace') if isinstance(data, bytes) else data
    if not (isinstance(line, str) and line.startswith('data:')):
        return data
    payload = line.removeprefix('data:').strip()
    if not payload or payload == '[DONE]':
        return data
    try:
        event = JSONCodec.loads(payload)
    except JSONCodec.JSONDecodeError:
        event = None

    if not isinstance(event, dict):
        return data
    event, _ = await process_filter_functions(
        request=request,
        filter_context=filter_context,
        filter_functions=filter_functions,
        filter_type='stream',
        form_data=event,
        extra_params=extra_params,
    )
    return wrap_item(JSONCodec.dumps(event)) if event else None


async def streaming_chat_response_handler(response, ctx):
    request = ctx['request']
    user = ctx['user']
    model = ctx['model']
    metadata = ctx['metadata']

    extra_params = {
        '__event_emitter__': ctx['event_emitter'],
        '__event_call__': ctx['event_caller'],
        '__user__': user.model_dump() if isinstance(user, UserModel) else {},
        '__metadata__': metadata,
        '__oauth_token__': await get_system_oauth_token(request, user),
        '__request__': request,
        '__model__': model,
        '__chat_id__': metadata.get('chat_id'),
        '__message_id__': metadata.get('message_id'),
    }

    filter_functions = (
        await get_filter_functions(request, model, metadata.get('filter_ids', [])) if ENABLE_PLUGINS else []
    )

    # Standard streaming response handler
    # event_caller is optional — only needed for direct (client-side) tools
    # and pyodide code interpreter. Server-side tools work without it.
    if ctx['event_emitter']:
        return await _StreamingReply(ctx, extra_params, filter_functions).run(response, ctx['events'])

    # Fallback to the original response
    return StreamingResponse(
        _passthrough_stream(response, ctx, ctx['events'], filter_functions, extra_params),
        headers=dict(response.headers),
        background=response.background,
    )


async def process_chat_response(response, ctx):
    # Non-streaming response
    if not isinstance(response, StreamingResponse):
        return await non_streaming_chat_response_handler(response, ctx)

    # Non standard response
    if not any(
        content_type in response.headers['Content-Type']
        for content_type in ['text/event-stream', 'application/x-ndjson']
    ):
        return response

    # Streaming response
    return await streaming_chat_response_handler(response, ctx)
