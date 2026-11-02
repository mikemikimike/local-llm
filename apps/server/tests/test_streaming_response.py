"""Characterization tests for streaming_chat_response_handler in utils/middleware.py.

streaming_chat_response_handler, which turns a model's stream into the
reply the user sees (with reasoning, tags, tool calls, code interpreter
runs and every continuation), was refactored for ruff's complexity limit
with no tests of its own, so these pin what it did before the refactor:
every event emitted, chat write, stream snapshot, tool call, follow-up
request and the final assistant message, in order. Streams are real
StreamingResponses fed fixed SSE lines; the event emitter and caller, the
chats, config, permissions, filters, uploads, the continuation requests
and the after-reply handlers are fakes that record their calls. Its pure
helpers (output shaping, tool result processing, message conversion) run
for real. uuid4 and the clock are fixed. Each case was recorded from the
untouched function into fixtures/streaming_response.json and must not
change.

The golden file is only ever written by hand, never by this test. To record
it again (only for an intended behaviour change, reviewed as such), run
`STREAMING_RESPONSE_GOLDEN_RECORD=1 pytest tests/test_streaming_response.py`.
"""

import asyncio
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

from local_llm.models.users import UserModel
from local_llm.utils import middleware

GOLDEN = Path(__file__).parent / 'fixtures' / 'streaming_response.json'
RECORD = os.environ.get('STREAMING_RESPONSE_GOLDEN_RECORD') == '1'

SAVED_CHAT = '7f1c0d9e-3b5a-4c1e-9a2f-0b8d6e4c2a10'


def sse(obj) -> str:
    return f'data: {json.dumps(obj)}'


def chunk(content=None, reasoning=None, tool_calls=None, **delta_extra):
    delta = {**delta_extra}
    if content is not None:
        delta['content'] = content
    if reasoning is not None:
        delta['reasoning_content'] = reasoning
    if tool_calls is not None:
        delta['tool_calls'] = tool_calls
    return sse({'choices': [{'delta': delta}]})


def call_delta(index, id=None, name=None, arguments=None):
    function = {}
    if name is not None:
        function['name'] = name
    if arguments is not None:
        function['arguments'] = arguments
    return {'index': index, **({'id': id} if id else {}), 'function': function}


DONE = 'data: [DONE]'
TEXT = [chunk('Hel'), chunk('lo'), DONE]


def tool_call_stream(name, arguments, call_id='call_1'):
    return [
        chunk(tool_calls=[call_delta(0, call_id, name, '')]),
        chunk(tool_calls=[call_delta(0, arguments=arguments)]),
        DONE,
    ]


def tool(name, properties=('q',), type='local', result=None, raises=None, direct=False, tool_id=None):
    return {
        'name': name,
        'properties': list(properties),
        'type': type,
        'result': result,
        'raises': raises,
        'direct': direct,
        'tool_id': tool_id or f'tool-{name}',
    }


RESP_MSG = {'type': 'message', 'id': 'msg_up', 'role': 'assistant', 'content': []}
RESP_TEXT = [
    sse({'type': 'response.output_item.added', 'output_index': 0, 'item': RESP_MSG}),
    sse({'type': 'response.content_part.added', 'output_index': 0, 'part': {'type': 'output_text', 'text': ''}}),
    sse(
        {
            'type': 'response.output_text.delta',
            'output_index': 0,
            'item_id': 'msg_up',
            'content_index': 0,
            'delta': 'Hi',
        }
    ),
    sse(
        {
            'type': 'response.output_text.delta',
            'output_index': 0,
            'item_id': 'msg_up',
            'content_index': 0,
            'delta': ' there',
        }
    ),
    sse(
        {
            'type': 'response.output_item.done',
            'output_index': 0,
            'item': {
                **RESP_MSG,
                'content': [
                    {
                        'type': 'output_text',
                        'text': 'Hi there',
                        'annotations': [
                            {'type': 'url_citation', 'url': 'https://ex.test/a', 'title': 'A'},
                            {'type': 'url_citation', 'url_citation': {'url': 'https://ex.test/b'}},
                            {'type': 'url_citation', 'url': ''},
                            {'type': 'file_citation'},
                        ],
                    }
                ],
            },
        }
    ),
    sse(
        {
            'type': 'response.completed',
            'response': {'id': 'resp_1', 'usage': {'input_tokens': 3, 'output_tokens': 2}, 'output': []},
        }
    ),
]
RESP_CALL = [
    sse(
        {
            'type': 'response.output_item.added',
            'output_index': 0,
            'item': {'type': 'function_call', 'id': 'fc_up', 'call_id': 'rc1', 'name': 'lookup', 'arguments': ''},
        }
    ),
    sse({'type': 'response.function_call_arguments.delta', 'output_index': 0, 'item_id': 'fc_up', 'delta': '{"q": '}),
    sse({'type': 'response.function_call_arguments.delta', 'output_index': 0, 'item_id': 'fc_up', 'delta': '"x"}'}),
    sse(
        {
            'type': 'response.function_call_arguments.done',
            'output_index': 0,
            'item_id': 'fc_up',
            'arguments': '{"q": "x"}',
        }
    ),
    sse(
        {
            'type': 'response.output_item.done',
            'output_index': 0,
            'item': {
                'type': 'function_call',
                'id': 'fc_up',
                'call_id': 'rc1',
                'name': 'lookup',
                'arguments': '{"q": "x"}',
            },
        }
    ),
    sse({'type': 'response.completed', 'response': {'id': 'resp_2', 'output': []}}),
]

ASK_USER_ARGS = json.dumps(
    {
        'questions': [
            {
                'id': 'which',
                'question': 'Which?',
                'options': [{'label': 'A', 'description': 'first'}, {'label': 'B', 'description': 'second'}],
            }
        ]
    }
)

CI_CODE = '<code_interpreter type="code" lang="python">print(1)</code_interpreter>'

LEGACY_CI = {'function_calling': 'legacy'}
CI_FEATURES = {'code_interpreter': True}


def _case(name, streams, metadata=None, world=None, form=None, model=None):
    return (name, streams, metadata or {}, world or {}, form or {}, model)


CASES_SPEC = [
    # --- chat completions text
    _case('local_text', [TEXT]),
    _case(
        'saved_text_with_usage_and_events',
        [
            [
                chunk('Answer'),
                sse(
                    {
                        'choices': [],
                        'usage': {'prompt_tokens': 3, 'completion_tokens': 1},
                        'timings': {'predicted_ms': 5},
                    }
                ),
                DONE,
            ]
        ],
        {'chat_id': SAVED_CHAT},
        {'events': [{'sources': [{'source': {'name': 'doc'}}]}], 'title': 'Saved title'},
    ),
    _case(
        'odd_lines',
        [
            [
                '',
                '   ',
                '{"error": {"message": "upstream exploded"}}',
                'not json at all',
                '{"ok": true}',
                'data: {bad json',
                chunk('fine'),
                DONE,
            ]
        ],
        {'chat_id': SAVED_CHAT},
    ),
    _case(
        'odd_lines_error_save_fails',
        [['{"error": "boom"}', chunk('x'), DONE]],
        {'chat_id': SAVED_CHAT},
        {'upsert_raise_on': 'error'},
    ),
    _case(
        'provider_error_chunk',
        [[sse({'choices': [], 'error': {'message': 'rate limited'}}), DONE]],
        {'chat_id': SAVED_CHAT},
    ),
    _case(
        'provider_error_chunk_save_fails',
        [[sse({'error': {'message': 'x'}}), DONE]],
        {'chat_id': SAVED_CHAT},
        {'upsert_raise_on': 'error'},
    ),
    _case('empty_choices_no_error', [[sse({'choices': []}), chunk('ok'), DONE]]),
    _case(
        'selected_model_and_custom_event',
        [
            [
                sse({'selected_model_id': 'm2'}),
                sse({'event': {'type': 'status', 'data': {'d': 1}}, 'choices': [{'delta': {'content': 'x'}}]}),
                DONE,
            ]
        ],
        {'chat_id': SAVED_CHAT},
    ),
    _case(
        'custom_event_on_direct_request',
        [[sse({'event': {'type': 'status'}, 'choices': [{'delta': {'content': 'x'}}]}), DONE]],
        world={'direct': True},
    ),
    _case(
        'selected_model_after_content', [[chunk('a'), sse({'selected_model_id': 'm2'}), DONE]], {'chat_id': SAVED_CHAT}
    ),
    _case(
        'selected_model_after_finish_chunk',
        [[chunk('a'), sse({'choices': [{'delta': {}}]}), sse({'selected_model_id': 'm2'}), DONE]],
    ),
    _case(
        'delta_annotations',
        [
            [
                chunk(
                    'Cited',
                    annotations=[
                        {'type': 'url_citation', 'url_citation': {'url': 'https://ex.test', 'title': 'Ex'}},
                        {'type': 'url_citation', 'url_citation': {'url': 'https://no-title.test'}},
                        {'type': 'url_citation'},
                    ],
                ),
                DONE,
            ]
        ],
    ),
    _case('non_string_content', [[chunk(42), chunk(reasoning=7), DONE]]),
    _case(
        'delta_chunk_size_groups',
        [[chunk('a'), chunk('b'), chunk('c'), chunk('d'), DONE]],
        {'params': {'stream_delta_chunk_size': 3}},
    ),
    _case('whitespace_only_reply', [[chunk('   '), DONE]]),
    _case('no_reply_at_all', [[DONE]]),
    _case(
        'base64_image_conversion',
        [[chunk('![x](data:image/png;base64,AAA)'), DONE]],
        world={'env': {'ENABLE_CHAT_RESPONSE_BASE64_IMAGE_URL_CONVERSION': True}},
    ),
    _case(
        'delta_images',
        [[chunk('', images=[{'image_url': {'url': 'data:image/png;base64,AA'}}]), chunk('see image'), DONE]],
        {'chat_id': SAVED_CHAT},
        {'message_files': None},
    ),
    _case(
        'delta_images_local', [[chunk(images=[{'url': 'u'}]), chunk(images=[]), DONE]], world={'no_image_urls': True}
    ),
    _case('stream_filters', [[chunk('filtered'), chunk('drop me'), DONE]], world={'filters': True}),
    # --- reasoning
    _case(
        'reasoning_content_then_answer', [[chunk(reasoning='Let me'), chunk(reasoning=' think'), chunk('Answer'), DONE]]
    ),
    _case('reasoning_after_answer_started', [[chunk('Ans'), chunk(reasoning='late thought'), chunk('wer'), DONE]]),
    _case(
        'reasoning_details_only',
        [
            [
                chunk(reasoning_details=[{'type': 'reasoning.encrypted', 'data': 'enc', 'index': 0}]),
                chunk(reasoning_details=[{'type': 'reasoning.text', 'text': 'detail', 'index': 1}, 'junk']),
                chunk('done'),
                DONE,
            ]
        ],
    ),
    _case('reasoning_details_empty', [[chunk(reasoning_details={'type': 'reasoning.text'}), chunk('ok'), DONE]]),
    _case(
        'reasoning_text_with_details',
        [[chunk(reasoning='why', reasoning_details=[{'type': 'reasoning.text', 'text': 'why'}]), chunk('x'), DONE]],
    ),
    _case(
        'reasoning_alias_fields',
        [[chunk(reasoning=None, thinking='hmm'), chunk(reasoning=None, reasoning_alt=None, reasoning2=None), DONE]],
    ),
    _case('reasoning_still_open_at_end', [[chunk(reasoning='unfinished'), DONE]]),
    _case(
        'think_tags_across_chunks',
        [[chunk('Intro <thi'), chunk('nk>plan'), chunk(' more</think>'), chunk(' Final'), DONE]],
    ),
    _case('think_tag_whole', [[chunk('<think>all at once</think>Then answer'), DONE]]),
    _case('think_tag_empty_block', [[chunk('<think></think>answer'), DONE]]),
    _case('think_tag_unclosed', [[chunk('<think>never closed'), DONE]]),
    _case('think_tag_with_attributes', [[chunk('<think type="deep">x</think>y'), DONE]]),
    _case('reasoning_tags_disabled', [[chunk('<think>kept</think>'), DONE]], {'params': {'reasoning_tags': False}}),
    _case(
        'reasoning_tags_custom',
        [[chunk('[[r]]why[[/r]]so'), DONE]],
        {'params': {'reasoning_tags': ['[[r]]', '[[/r]]']}},
    ),
    _case('solution_tags', [[chunk('<|begin_of_solution|>The solution<|end_of_solution|> tail'), DONE]]),
    _case('solution_tag_open_then_more', [[chunk('<|begin_of_solution|>part'), chunk(' two'), DONE]]),
    _case('tag_like_text_not_a_tag', [[chunk('a < b and c > d'), chunk(' ok <'), chunk('br> fine'), DONE]]),
    # --- tool calls (chat completions)
    _case(
        'tool_call_then_answer',
        [tool_call_stream('lookup', '{"q": "x", "extra": 1}'), [chunk('Found it'), DONE]],
        {'chat_id': SAVED_CHAT, 'tools': [tool('lookup', result={'hits': 1})]},
    ),
    _case(
        'tool_call_arguments_as_objects',
        [
            [
                chunk(tool_calls=[call_delta(0, None, 'lookup', {'q': 1})]),
                chunk(tool_calls=[call_delta(0, name='lookup', arguments={'more': 2})]),
                chunk(tool_calls=[{'function': {'name': 'no_index'}}]),
                DONE,
            ],
            [DONE],
        ],
        {'tools': [tool('lookup', result='ok')]},
    ),
    _case(
        'tool_call_existing_non_string_args',
        [
            [
                chunk(tool_calls=[{'index': 0, 'id': 'c1', 'function': {'name': 'lookup', 'arguments': None}}]),
                chunk(tool_calls=[call_delta(0, arguments='{"q": 1}')]),
                DONE,
            ],
            [DONE],
        ],
        {'tools': [tool('lookup', result='ok')]},
    ),
    _case(
        'tool_calls_bad_and_missing',
        [
            [
                chunk(
                    tool_calls=[call_delta(0, 'c1', 'missing_tool', '{}'), call_delta(1, 'c2', 'lookup', '{not json')]
                ),
                chunk(tool_calls=[call_delta(2, 'c3', 'lookup', "{'q': 'single'}"), call_delta(3, 'c4', 'lookup', '')]),
                DONE,
            ],
            [chunk('after'), DONE],
        ],
        {'tools': [tool('lookup', result={'error': 'bad query'})]},
    ),
    _case(
        'tool_raises_and_direct_tool',
        [tool_call_stream('boom', '{"q": 1}'), [chunk('ok'), DONE]],
        {'tools': [tool('boom', raises='kaput')]},
    ),
    _case(
        'direct_tool',
        [tool_call_stream('remote', '{"q": 1}'), [DONE]],
        {'tools': [tool('remote', direct=True)], 'session_id': 'sess'},
        {'direct_result': {'remote': 'answer'}},
    ),
    _case(
        'tool_approval_pauses',
        [tool_call_stream('lookup', '{"q": 1}')],
        {'chat_id': SAVED_CHAT, 'params': {'tool_approval_mode': 'ask'}, 'tools': [tool('lookup')]},
    ),
    _case(
        'tool_approval_local_chat_runs',
        [tool_call_stream('lookup', '{"q": 1}'), [DONE]],
        {'params': {'tool_approval_mode': 'ask'}, 'tools': [tool('lookup', result='r')]},
    ),
    _case(
        'ask_user_pauses',
        [tool_call_stream('ask_user', ASK_USER_ARGS)],
        {'chat_id': SAVED_CHAT, 'tools': [tool('ask_user', properties=['questions'])]},
    ),
    _case(
        'ask_user_local_chat',
        [tool_call_stream('ask_user', ASK_USER_ARGS)],
        {'tools': [tool('ask_user', properties=['questions'])]},
    ),
    _case(
        'ask_user_malformed_answers_itself',
        [tool_call_stream('ask_user', '{"questions": [{"question": "Q?"}]}'), [DONE]],
        {'tools': [tool('ask_user', properties=['questions'])]},
    ),
    _case(
        'ask_user_invalid_dropped',
        [
            [
                chunk(
                    tool_calls=[
                        call_delta(0, 'c1', 'ask_user', '{"questions": "nope"}'),
                        call_delta(1, 'c2', 'lookup', '{"q": 1}'),
                    ]
                ),
                DONE,
            ],
            [DONE],
        ],
        {'tools': [tool('ask_user', properties=['questions']), tool('lookup', result='r')]},
    ),
    _case(
        'search_citations_and_rag',
        [tool_call_stream('search_web', '{"query": "q"}'), [chunk('cited answer'), DONE]],
        {
            'tools': [
                tool(
                    'search_web',
                    properties=['query'],
                    result=json.dumps([{'link': 'https://r.test', 'title': 'R', 'snippet': 'S'}]),
                )
            ],
            'system_prompt': 'Original system',
            'user_prompt': 'What is new?',
            'sources': [
                {'source': {'name': 'file.pdf'}, 'document': ['file text'], 'metadata': [{'source': 'file.pdf'}]}
            ],
        },
        {'config': {'rag.template': '[RAG {{CONTEXT}}]'}},
        {
            'messages': [
                {'role': 'system', 'content': 'Sys + old rag'},
                {'role': 'user', 'content': 'What is new? + rag'},
            ]
        },
    ),
    _case(
        'search_citations_no_system_message',
        [tool_call_stream('search_web', '{"query": "q"}'), [DONE]],
        {
            'tools': [
                tool('search_web', properties=['query'], result=json.dumps([{'link': 'https://r.test', 'title': 'R'}]))
            ],
            'system_prompt': 'Original system',
        },
        {'env': {'RAG_SYSTEM_CONTEXT': True}},
        {'messages': [{'role': 'user', 'content': 'q?'}]},
    ),
    _case(
        'search_citations_no_original_system',
        [tool_call_stream('search_web', '{"query": "q"}'), [DONE]],
        {'tools': [tool('search_web', properties=['query'], result=json.dumps([{'link': 'https://r.test'}]))]},
        {},
        {'messages': [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'q?'}]},
    ),
    _case(
        'search_citations_disabled',
        [tool_call_stream('search_web', '{"query": "q"}'), [DONE]],
        {'tools': [tool('search_web', properties=['query'], result=json.dumps([{'link': 'https://r.test'}]))]},
        model={'id': 'm1', 'info': {'meta': {'capabilities': {'citations': False}}}},
    ),
    _case(
        'tool_result_images',
        [tool_call_stream('camera', '{"q": 1}'), [DONE]],
        {'tools': [tool('camera', result='data:image/png;base64,QUJD')]},
    ),
    _case(
        'tool_result_mcp_files',
        [tool_call_stream('mcp_tool', '{"q": 1}'), [DONE]],
        {
            'tools': [
                tool(
                    'mcp_tool',
                    type='mcp',
                    result=[
                        {'type': 'image', 'mimeType': 'image/png', 'data': 'QQ=='},
                        {'type': 'text', 'text': 'cap'},
                    ],
                )
            ]
        },
    ),
    _case(
        'delegate_calls_gathered',
        [
            [
                chunk(
                    tool_calls=[
                        call_delta(0, 'd1', 'delegate_task', '{"task": "a"}'),
                        call_delta(1, 'd2', 'lookup', '{"q": 1}'),
                        call_delta(2, 'd3', 'delegate_task', '{"task": "b"}'),
                    ]
                ),
                DONE,
            ],
            [DONE],
        ],
        {'tools': [tool('delegate_task', properties=['task'], result='delegated'), tool('lookup', result='r')]},
    ),
    _case(
        'continuation_error_response',
        [tool_call_stream('lookup', '{"q": 1}'), 'error_json'],
        {'chat_id': SAVED_CHAT, 'tools': [tool('lookup', result='r')]},
    ),
    _case(
        'continuation_raises',
        [tool_call_stream('lookup', '{"q": 1}'), 'raise'],
        {'tools': [tool('lookup', result='r')]},
    ),
    _case(
        'continuation_not_streaming',
        [tool_call_stream('lookup', '{"q": 1}'), 'plain'],
        {'tools': [tool('lookup', result='r')]},
    ),
    _case(
        'tool_call_iteration_limit',
        [tool_call_stream('lookup', '{"q": 1}', 'c1'), tool_call_stream('lookup', '{"q": 2}', 'c2')],
        {'tools': [tool('lookup', result='r')]},
        {'max_tool_call_iterations': 1},
    ),
    _case(
        'continuation_filters',
        [tool_call_stream('lookup', '{"q": 1}'), [chunk('z'), DONE]],
        {'tools': [tool('lookup', result='r')]},
        {'filters': True},
    ),
    _case(
        'continuation_after_empty_placeholder',
        [[chunk('  '), chunk(tool_calls=[call_delta(0, 'c1', 'lookup', '{}')]), DONE], [chunk('next'), DONE]],
        {'tools': [tool('lookup', result='r')]},
    ),
    # --- responses API
    _case('responses_text', [RESP_TEXT], {'chat_id': SAVED_CHAT}),
    _case('responses_text_stateful', [RESP_TEXT], world={'env': {'ENABLE_RESPONSES_API_STATEFUL': True}}),
    _case('responses_function_call', [RESP_CALL, RESP_TEXT], {'tools': [tool('lookup', result={'v': 1})]}),
    _case(
        'responses_function_call_stateful',
        [RESP_CALL, [DONE]],
        {'tools': [tool('lookup', result='r')]},
        {'env': {'ENABLE_RESPONSES_API_STATEFUL': True}},
        {'messages': [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'q'}]},
    ),
    _case(
        'responses_function_call_stateful_no_system',
        [RESP_CALL, [DONE]],
        {'tools': [tool('lookup', result='r')]},
        {'env': {'ENABLE_RESPONSES_API_STATEFUL': True}},
    ),
    _case(
        'responses_failed', [[sse({'type': 'response.failed', 'response': {'error': {'code': 'server_error'}}}), DONE]]
    ),
    _case(
        'responses_arguments_not_string',
        [
            [
                sse(
                    {
                        'type': 'response.output_item.added',
                        'item': {'type': 'function_call', 'call_id': 'rc9', 'name': 'lookup', 'arguments': {'q': 1}},
                    }
                ),
                DONE,
            ],
            [DONE],
        ],
        {'tools': [tool('lookup', result='r')]},
    ),
    # --- continuing an existing message
    _case(
        'continue_existing_message',
        [[chunk(' more'), DONE]],
        {'chat_id': SAVED_CHAT, 'assistant_message_id': 'a1'},
        {
            'stored_message': {
                'content': 'partial',
                'output': [
                    {
                        'type': 'message',
                        'id': 'msg_old',
                        'status': 'completed',
                        'content': [{'type': 'output_text', 'text': 'partial'}],
                    },
                    {
                        'type': 'message',
                        'id': 'msg_empty',
                        'status': 'in_progress',
                        'content': [{'type': 'output_text', 'text': ' '}],
                    },
                ],
            }
        },
    ),
    _case(
        'continue_responses_offsets_index',
        [RESP_TEXT],
        {'chat_id': SAVED_CHAT, 'assistant_message_id': 'a1'},
        {
            'stored_message': {
                'output': [
                    {
                        'type': 'message',
                        'id': 'msg_old',
                        'status': 'completed',
                        'content': [{'type': 'output_text', 'text': 'old'}],
                    }
                ]
            }
        },
    ),
    _case(
        'stored_output_without_continue',
        [[chunk('new'), DONE]],
        {'chat_id': SAVED_CHAT},
        {
            'stored_message': {
                'content': 'stored',
                'output': [
                    {
                        'type': 'message',
                        'id': 'msg_s',
                        'status': 'in_progress',
                        'content': [{'type': 'output_text', 'text': 'stored'}],
                    }
                ],
            }
        },
    ),
    _case(
        'stored_message_content_only',
        [[chunk(' tail'), DONE]],
        {'chat_id': SAVED_CHAT},
        {'stored_message': {'content': 'head'}},
    ),
    _case(
        'last_assistant_message_in_form',
        [[chunk(' continued'), DONE]],
        form={'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'prefix'}]},
    ),
    _case('form_without_messages', [[chunk('x'), DONE]], form={'messages': []}),
    # --- code interpreter (legacy)
    _case(
        'code_interpreter_pyodide',
        [[chunk('Running: '), chunk(CI_CODE), DONE], [chunk('It printed 1'), DONE]],
        {'params': LEGACY_CI, 'features': CI_FEATURES, 'session_id': 'sess'},
        {
            'admin': True,
            'config': {'code_interpreter.enable': True, 'code_interpreter.engine': 'pyodide'},
            'ci_outputs': [{'stdout': 'line\ndata:image/png;base64,AA', 'result': 'data:image/png;base64,BB\nplain'}],
        },
    ),
    _case(
        'code_interpreter_jupyter_token',
        [[chunk(CI_CODE), DONE], [DONE]],
        {'params': LEGACY_CI, 'features': CI_FEATURES},
        {
            'permitted': True,
            'config': {
                'code_interpreter.enable': True,
                'code_interpreter.engine': 'jupyter',
                'code_interpreter.jupyter.url': 'http://jup.test',
                'code_interpreter.jupyter.auth': 'token',
                'code_interpreter.jupyter.auth_token': 'tok',
                'code_interpreter.jupyter.timeout': 30,
            },
            'env': {'CODE_INTERPRETER_BLOCKED_MODULES': ['os']},
        },
    ),
    _case(
        'code_interpreter_jupyter_password',
        [[chunk(CI_CODE), DONE], 'error_json'],
        {'params': LEGACY_CI, 'features': CI_FEATURES},
        {
            'admin': True,
            'config': {
                'code_interpreter.enable': True,
                'code_interpreter.engine': 'jupyter',
                'code_interpreter.jupyter.auth': 'password',
                'code_interpreter.jupyter.auth_password': 'pw',
            },
        },
    ),
    _case(
        'code_interpreter_unconfigured_engine',
        [[chunk(CI_CODE), DONE], 'plain'],
        {'params': LEGACY_CI, 'features': CI_FEATURES},
        {'admin': True, 'config': {'code_interpreter.enable': True, 'code_interpreter.engine': 'cloud'}},
    ),
    _case(
        'code_interpreter_error_output',
        [[chunk(CI_CODE), DONE], 'raise'],
        {'params': LEGACY_CI, 'features': CI_FEATURES},
        {
            'admin': True,
            'config': {'code_interpreter.enable': True, 'code_interpreter.engine': 'pyodide'},
            'ci_outputs': [{'error': 'session disconnected'}],
        },
    ),
    _case(
        'code_interpreter_caller_raises',
        [[chunk(CI_CODE), DONE], [DONE]],
        {'params': LEGACY_CI, 'features': CI_FEATURES},
        {
            'admin': True,
            'config': {'code_interpreter.enable': True, 'code_interpreter.engine': 'pyodide'},
            'ci_outputs': ['raise'],
        },
    ),
    _case(
        'code_interpreter_not_code_type',
        [[chunk('<code_interpreter type="other">x</code_interpreter>'), DONE], [DONE]],
        {'params': LEGACY_CI, 'features': CI_FEATURES},
        {'admin': True, 'config': {'code_interpreter.enable': True}},
    ),
    _case(
        'code_interpreter_not_permitted',
        [[chunk(CI_CODE), DONE]],
        {'params': LEGACY_CI, 'features': CI_FEATURES},
        {'config': {'code_interpreter.enable': True}},
    ),
    _case(
        'code_interpreter_capability_off',
        [[chunk(CI_CODE), DONE]],
        {'params': LEGACY_CI, 'features': CI_FEATURES},
        {'admin': True, 'config': {'code_interpreter.enable': True}},
        model={'id': 'm1', 'info': {'meta': {'capabilities': {'code_interpreter': False}}}},
    ),
    _case(
        'code_interpreter_filters_and_retries',
        [[chunk(CI_CODE), DONE]] + [[chunk(CI_CODE), DONE]] * 6,
        {'params': LEGACY_CI, 'features': CI_FEATURES},
        {
            'admin': True,
            'filters': True,
            'config': {'code_interpreter.enable': True, 'code_interpreter.engine': 'pyodide'},
            'ci_outputs': [{'stdout': 5}] * 6,
        },
    ),
    # --- cancellation and the fallback (no event emitter)
    _case('cancelled_mid_stream', [[chunk('partial'), 'CANCEL']], {'chat_id': SAVED_CHAT}),
    _case('cancelled_save_fails', [[chunk('partial'), 'CANCEL']], {'chat_id': SAVED_CHAT}, {'upsert_raise_on': 'done'}),
    _case('fallback_passthrough', [TEXT], world={'no_emitter': True, 'events': [{'sources': []}]}),
    _case(
        'fallback_with_filters',
        [[chunk('a'), 'data: [DONE]', 'data: {bad', b'data: ' + json.dumps({'x': 1}).encode(), 'plain line']],
        world={'no_emitter': True, 'filters': True, 'events': [{'e': 1}]},
    ),
    _case(
        'fallback_outlet_filters',
        [TEXT],
        world={'no_emitter': True, 'env': {'ENABLE_API_OUTLET_FILTERS': True}, 'sorted_filters': ['f']},
    ),
    _case(
        'fallback_outlet_filters_lookup_fails',
        [TEXT],
        world={'no_emitter': True, 'env': {'ENABLE_API_OUTLET_FILTERS': True}, 'sorted_filters': 'raise'},
    ),
    _case(
        'fallback_outlet_filters_none', [TEXT], world={'no_emitter': True, 'env': {'ENABLE_API_OUTLET_FILTERS': True}}
    ),
    _case(
        'fallback_pipeline_model',
        [TEXT],
        world={'no_emitter': True, 'env': {'ENABLE_API_OUTLET_FILTERS': True}},
        model={'id': 'm1', 'pipeline': {}, 'info': {}},
    ),
    _case('response_background_runs', [TEXT], world={'background': True}),
    # --- edges for branch coverage
    _case(
        'local_odd_lines_and_events',
        [
            [
                '{"error": "local boom"}',
                sse({'selected_model_id': 'm2'}),
                sse({'choices': [], 'error': {'m': 1}}),
                chunk('x'),
                DONE,
            ]
        ],
        world={'events': [{'sources': []}]},
    ),
    _case(
        'finish_chunk_without_delta', [[chunk('x'), sse({'choices': [{'delta': {}, 'finish_reason': 'stop'}]}), DONE]]
    ),
    _case(
        'mixed_deltas_grouped',
        [
            [
                chunk(reasoning='r1'),
                chunk(reasoning='r2'),
                chunk('a'),
                chunk('b'),
                chunk(reasoning='late'),
                chunk('c'),
                DONE,
            ]
        ],
        {'params': {'stream_delta_chunk_size': 3}},
    ),
    _case('tag_at_chunk_end', [[chunk('<think>'), chunk('inside'), chunk('</think>after'), DONE]]),
    _case(
        'code_block_streams_into_item',
        [[chunk('<code_interpreter type="code">'), chunk('x = 1'), chunk('</code_interpreter>'), DONE], [DONE]],
        {'params': LEGACY_CI, 'features': CI_FEATURES},
        {
            'admin': True,
            'config': {'code_interpreter.enable': True, 'code_interpreter.engine': 'pyodide'},
            'ci_outputs': ['plain string output'],
        },
    ),
    _case(
        'code_interpreter_outputs_odd',
        [[chunk(CI_CODE), DONE], [DONE]],
        {'params': LEGACY_CI, 'features': CI_FEATURES},
        {
            'admin': True,
            'ci_image_none': True,
            'config': {'code_interpreter.enable': True, 'code_interpreter.engine': 'pyodide'},
            'ci_outputs': [{'stdout': 'data:image/png;base64,ZZ', 'result': {'not': 'text'}}],
        },
    ),
    _case(
        'tool_call_name_only_update',
        [
            [
                chunk(tool_calls=[{'function': {'name': 'ignored'}}]),
                chunk(tool_calls=[call_delta(0, 'c1', 'look', '{"q"')]),
                chunk(tool_calls=[{'index': 0, 'function': {'name': 'lookup'}}]),
                chunk(tool_calls=[call_delta(0, arguments=': 1}')]),
                DONE,
            ],
            [DONE],
        ],
        {'tools': [tool('lookup', result='r')]},
    ),
    _case(
        'delta_images_saved_and_local',
        [[chunk(images=[{'url': 'u'}]), chunk('pic'), DONE]],
        {'chat_id': SAVED_CHAT},
    ),
    _case('delta_images_local_chat', [[chunk(images=[{'url': 'u'}]), DONE]]),
    _case(
        'stored_output_without_ids',
        [[chunk('<think>t</think>after'), DONE]],
        {'chat_id': SAVED_CHAT},
        {'stored_message': {'output': [{'type': 'message', 'status': 'in_progress', 'content': []}]}},
    ),
    _case(
        'stored_output_refusal_only',
        [[DONE]],
        {'chat_id': SAVED_CHAT},
        {
            'stored_message': {
                'output': [
                    {
                        'type': 'message',
                        'id': 'm',
                        'status': 'completed',
                        'content': [{'type': 'refusal', 'refusal': 'no'}],
                    }
                ]
            }
        },
    ),
    _case(
        'stored_output_open_reasoning',
        [[DONE]],
        {'chat_id': SAVED_CHAT},
        {'stored_message': {'output': [{'type': 'reasoning', 'id': 'r', 'status': 'in_progress', 'content': []}]}},
    ),
    _case(
        'stored_output_closed_reasoning',
        [[DONE]],
        {'chat_id': SAVED_CHAT},
        {
            'stored_message': {
                'output': [{'type': 'reasoning', 'id': 'r', 'status': 'completed', 'started_at': 1, 'ended_at': 2}]
            }
        },
    ),
    _case(
        'continue_keeps_nonempty_in_progress',
        [[chunk(' more'), DONE]],
        {'chat_id': SAVED_CHAT, 'assistant_message_id': 'a1'},
        {
            'stored_message': {
                'output': [
                    {
                        'type': 'message',
                        'id': 'msg_p',
                        'status': 'in_progress',
                        'content': [{'type': 'output_text', 'text': 'kept'}],
                    }
                ]
            }
        },
    ),
    _case(
        'continuation_raises_http',
        [tool_call_stream('lookup', '{"q": 1}'), 'raise_http'],
        {'tools': [tool('lookup', result='r')]},
    ),
    _case(
        'continuation_error_detail_dict',
        [tool_call_stream('lookup', '{"q": 1}'), 'error_json_dict'],
        {'tools': [tool('lookup', result='r')]},
    ),
    _case(
        'responses_completed_stateful_without_id',
        [[sse({'type': 'response.completed', 'response': {'output': [], 'usage': {'input_tokens': 1}}}), DONE]],
        world={'env': {'ENABLE_RESPONSES_API_STATEFUL': True}},
    ),
    _case(
        'responses_call_then_empty_message',
        [
            [
                sse(
                    {
                        'type': 'response.output_item.added',
                        'item': {
                            'type': 'function_call',
                            'id': 'fc9',
                            'call_id': 'rc9',
                            'name': 'lookup',
                            'arguments': '{}',
                        },
                    }
                ),
                sse(
                    {
                        'type': 'response.output_item.added',
                        'item': {'type': 'message', 'id': 'm9', 'status': 'in_progress', 'content': []},
                    }
                ),
                DONE,
            ],
            [chunk('after tool'), DONE],
        ],
        {'tools': [tool('lookup', result='r')]},
    ),
    _case(
        'terminal_display_file',
        [tool_call_stream('display_file', '{"path": "notes.md"}'), [DONE]],
        {
            'tools': [
                tool(
                    'display_file',
                    properties=['path'],
                    type='terminal',
                    tool_id='terminal:t1',
                    result={'path': 'notes.md', 'exists': True},
                )
            ]
        },
    ),
    _case(
        'citation_extraction_fails',
        [tool_call_stream('view_file', '{"file_id": "f"}'), [DONE]],
        {
            'tools': [
                tool(
                    'view_file',
                    properties=['file_id'],
                    result=json.dumps({'id': 'f', 'name': None, 'content': object.__name__}),
                )
            ]
        },
        {'citation_raise': True},
    ),
    _case(
        'stored_output_without_ids_plain',
        [[chunk(' plain'), DONE]],
        {'chat_id': SAVED_CHAT},
        {
            'stored_message': {
                'output': [
                    {'type': 'message', 'status': 'in_progress', 'content': [{'type': 'output_text', 'text': 'x'}]}
                ]
            }
        },
    ),
    _case(
        'citations_without_any_system_prompt',
        [tool_call_stream('search_web', '{"query": "q"}'), [DONE]],
        {
            'tools': [
                tool('search_web', properties=['query'], result=json.dumps([{'link': 'https://r.test', 'title': 'R'}]))
            ]
        },
        {},
        {'messages': [{'role': 'user', 'content': 'q?'}]},
    ),
    _case(
        'citations_without_documents',
        [tool_call_stream('search_web', '{"query": "q"}'), [DONE]],
        {'tools': [tool('search_web', properties=['query'], result='[]')]},
        {'citation_sources': [{'source': {'name': 'empty'}, 'document': [], 'metadata': []}]},
    ),
    _case('cancelled_in_filter', [[chunk('a'), chunk('CANCELME'), DONE]], world={'filters': True}),
    _case('cancelled_local_chat', [[chunk('partial'), 'CANCEL']]),
    _case(
        'cancelled_iterator_without_aclose',
        [[chunk('partial'), 'CANCEL']],
        {'chat_id': SAVED_CHAT},
        {'aclose': 'missing'},
    ),
    _case('cancelled_aclose_fails', [[chunk('partial'), 'CANCEL']], {'chat_id': SAVED_CHAT}, {'aclose': 'raise'}),
    _case(
        'fallback_filters_drop',
        [[chunk('keep'), chunk('drop me'), DONE]],
        world={'no_emitter': True, 'filters': True, 'events': [{'drop me': True}, {'keep': True}]},
    ),
]

CASES = [(f'stream:{c[0]}', *c[1:]) for c in CASES_SPEC]

ENV = {
    'ENABLE_PLUGINS': True,
    'ENABLE_RESPONSES_API_STATEFUL': False,
    'ENABLE_CHAT_RESPONSE_BASE64_IMAGE_URL_CONVERSION': False,
    'CHAT_RESPONSE_STREAM_DELTA_CHUNK_SIZE': 1,
    'CHAT_RESPONSE_MAX_TOOL_CALL_ITERATIONS': 10,
    'CODE_INTERPRETER_BLOCKED_MODULES': [],
    'RAG_SYSTEM_CONTEXT': False,
    'ENABLE_API_OUTLET_FILTERS': False,
}


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, UserModel):
        return f'<user {value.id}>'
    if isinstance(value, JSONResponse):
        return {'json_response': value.status_code}
    if callable(value):
        return f'<callable {getattr(value, "__name__", "?")}>'
    return repr(value)


class _Background:
    def __init__(self, log, name):
        self.log, self.name = log, name

    async def __call__(self):
        self.log.append(['response.background', self.name])


class _LineIterator:
    """A body iterator without the async generator's aclose, or with one that fails."""

    def __init__(self, lines, log, aclose):
        self.lines, self.log = list(lines), log
        if aclose == 'raise':
            self.aclose = self._failing_aclose

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.lines:
            raise StopAsyncIteration
        line = self.lines.pop(0)
        if line == 'CANCEL':
            raise asyncio.CancelledError()
        return line

    async def _failing_aclose(self):
        self.log.append(['body_iterator.aclose', 'raise'])
        raise RuntimeError('already closed')


def _stream_response(lines, log, name, background=False, aclose=None):
    async def body():
        for line in lines:
            if line == 'CANCEL':
                raise asyncio.CancelledError()
            yield line

    content = _LineIterator(lines, log, aclose) if aclose else body()
    return StreamingResponse(content, background=_Background(log, name) if background else None)


def _tools(specs, log, world):
    tools = {}
    for spec in specs:

        def make(spec):
            async def tool_function(**kwargs):
                log.append(['tool', spec['name'], kwargs])
                if spec['raises']:
                    raise RuntimeError(spec['raises'])
                return copy.deepcopy(spec['result'])

            return tool_function

        tools[spec['name']] = {
            'tool_id': spec['tool_id'],
            'spec': {
                'name': spec['name'],
                'parameters': {'type': 'object', 'properties': {p: {} for p in spec['properties']}},
            },
            'callable': make(spec),
            'type': spec['type'],
            **({'direct': True, 'server': {'url': 'https://direct.test'}} if spec['direct'] else {}),
        }
    return tools


def _install(m, world, log, continuations):
    config = {'code_interpreter.enable': False, **world.get('config', {})}
    clock = iter(range(1000, 100000, 3))
    ids = iter(range(1, 100000))

    def fake(name, result=None):
        async def call(*args, **kwargs):
            log.append([name, _plain(args), _plain(kwargs)])
            return result(*args, **kwargs) if callable(result) else copy.deepcopy(result)

        return call

    async def config_get(key, default=None):
        log.append(['config_get', key])
        return copy.deepcopy(config.get(key, default))

    async def upsert(chat_id, message_id, data, touch=True):
        log.append(['chats.upsert_message', message_id, _plain(data), touch])
        raise_on = world.get('upsert_raise_on')
        if raise_on and raise_on in data:
            raise RuntimeError('database is locked')

    async def process_filter_functions(
        request=None, filter_context=None, filter_functions=None, filter_type=None, form_data=None, extra_params=None
    ):
        log.append(['filter', filter_type, _plain(form_data)])
        if filter_type == 'stream' and 'CANCELME' in json.dumps(form_data):
            raise asyncio.CancelledError()
        if filter_type == 'stream' and isinstance(form_data, dict) and 'drop me' in json.dumps(form_data):
            return None, {}
        if isinstance(form_data, dict):
            return {**form_data, 'filtered': filter_type}, {}
        return form_data, {}

    async def generate_chat_completion(request, form_data, user, bypass_system_prompt=False):
        log.append(['generate_chat_completion', _plain(form_data), bypass_system_prompt])
        if not continuations:
            return {'nothing': 'left'}
        nxt = continuations.pop(0)
        if nxt == 'error_json':
            return JSONResponse(status_code=429, content={'error': {'message': 'too many requests'}})
        if nxt == 'raise':
            raise RuntimeError('provider unreachable')
        if nxt == 'raise_http':
            raise HTTPException(status_code=502, detail='bad gateway')
        if nxt == 'error_json_dict':
            return JSONResponse(status_code=500, content={'detail': {'code': 'upstream', 'message': 'broke'}})
        if nxt == 'plain':
            return {'choices': []}
        return _stream_response(nxt, log, f'continuation-{len(continuations)}')

    def get_sorted_filters(model_id, models):
        if world.get('sorted_filters') == 'raise':
            raise RuntimeError('filter lookup failed')
        return world.get('sorted_filters', [])

    patches = {
        'get_system_oauth_token': fake('get_system_oauth_token', None),
        'get_filter_functions': fake('get_filter_functions', ['<filter>'] if world.get('filters') else []),
        'process_filter_functions': process_filter_functions,
        'save_response_stream': fake('save_response_stream'),
        'clear_response_stream': fake('clear_response_stream'),
        'has_permission': fake('has_permission', lambda *a, **k: bool(world.get('permitted'))),
        'get_image_urls': fake('get_image_urls', [] if world.get('no_image_urls') else ['/files/img1']),
        'convert_markdown_base64_images': fake(
            'convert_markdown_base64_images',
            lambda request, value, meta, user: value.replace('data:image/png;base64,AAA', '/files/converted'),
        ),
        'get_image_url_from_base64': fake(
            'get_image_url_from_base64', None if world.get('ci_image_none') else '/files/ci-image'
        ),
        'get_file_url_from_base64': fake('get_file_url_from_base64', '/files/mcp-image'),
        'pause_for_tool_approval': fake('pause_for_tool_approval'),
        'rag_template': fake(
            'rag_template', lambda template, context, query: f'[RAG {len(context)} chars for {query}]'
        ),
        'generate_chat_completion': generate_chat_completion,
        'chat_files': fake('chat_files', ([('sales.csv', '/data/uploads/f1_sales.csv')], [])),
        'execute_code_jupyter': fake('execute_code_jupyter', {'stdout': 'jupyter ran', 'result': ''}),
        'publish_chat_finished_event': fake('publish_chat_finished_event'),
        'outlet_filter_handler': fake('outlet_filter_handler', lambda ctx: None),
        'background_tasks_handler': fake('background_tasks_handler', lambda ctx: None),
        'get_sorted_filters': get_sorted_filters,
        'uuid4': lambda: SimpleNamespace(hex=f'{next(ids):024x}'),
        'time': SimpleNamespace(time=lambda: float(next(clock))),
    }
    if 'citation_sources' in world:
        patches['get_citation_source_from_tool_result'] = lambda **kwargs: copy.deepcopy(world['citation_sources'])
    if world.get('citation_raise'):
        patches['get_citation_source_from_tool_result'] = lambda **kwargs: (_ for _ in ()).throw(
            ValueError('bad source')
        )
    for name, value in {**patches, **ENV, **world.get('env', {})}.items():
        m.setattr(middleware, name, value)
    m.setattr(middleware.Config, 'get', config_get)
    m.setattr(middleware.Chats, 'upsert_message_to_chat_by_id_and_message_id', upsert)
    m.setattr(
        middleware.Chats, 'get_message_by_id_and_message_id', fake('chats.get_message', world.get('stored_message'))
    )
    m.setattr(
        middleware.Chats,
        'add_message_files_by_id_and_message_id',
        fake(
            'chats.add_message_files',
            lambda *a: world.get('message_files', [{'type': 'image', 'url': '/files/img1', 'saved': True}]),
        ),
    )
    m.setattr(middleware.Chats, 'get_chat_title_by_id', fake('chats.get_chat_title', world.get('title', 'Chat title')))


def _ctx(world, metadata_spec, form, model, log):
    user = UserModel(
        id='u1',
        email='a@x.test',
        name='Ada',
        role='admin' if world.get('admin') else 'user',
        last_active_at=1,
        updated_at=1,
        created_at=1,
    )
    ci_outputs = list(world.get('ci_outputs', []))

    async def event_emitter(event):
        log.append(['emit', _plain(event)])

    async def event_caller(event):
        log.append(['call', _plain(event)])
        if event['type'] == 'execute:tool':
            return world.get('direct_result', {}).get(event['data']['name'])
        output = ci_outputs.pop(0) if ci_outputs else {'stdout': ''}
        if output == 'raise':
            raise RuntimeError('pyodide crashed')
        return copy.deepcopy(output)

    metadata = {
        'message_id': 'a1',
        'chat_id': '',
        **{k: v for k, v in copy.deepcopy(metadata_spec).items() if k != 'tools'},
    }
    metadata['tools'] = _tools(metadata_spec.get('tools', []), log, world)
    state = SimpleNamespace(
        **({'direct': True} if world.get('direct') else {}),
        **(
            {'max_tool_call_iterations': world['max_tool_call_iterations']}
            if 'max_tool_call_iterations' in world
            else {}
        ),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(redis='<redis>', MODELS={'m1': {'id': 'm1'}})), state=state
    )
    return {
        'request': request,
        'form_data': {'model': 'm1', 'messages': [{'role': 'user', 'content': 'q?'}], **copy.deepcopy(form)},
        'user': user,
        'model': copy.deepcopy(model or {'id': 'm1', 'info': {'meta': {}}}),
        'metadata': metadata,
        'events': copy.deepcopy(world.get('events', [])),
        'event_emitter': None if world.get('no_emitter') else event_emitter,
        'event_caller': event_caller,
        'tasks': {},
    }


async def _drive(streams, world, ctx, log):
    response = _stream_response(
        streams[0], log, 'first', background=world.get('background', False), aclose=world.get('aclose')
    )
    try:
        returned = await middleware.streaming_chat_response_handler(response, ctx)
    except asyncio.CancelledError:
        return {'cancelled': True}
    if isinstance(returned, StreamingResponse):
        body = [_plain(item) async for item in returned.body_iterator]
        return {'returned_stream': body}
    return {'returned': _plain(returned)}


def _run(streams, metadata, world, form, model, monkeypatch):
    log = []
    continuations = list(streams[1:])
    _install(monkeypatch, world, log, continuations)
    ctx = _ctx(world, metadata, form, model, log)
    outcome = asyncio.run(_drive(streams, world, ctx, log))
    return json.loads(json.dumps({**outcome, 'assistant_message': _plain(ctx.get('assistant_message')), 'calls': log}))


def _golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding='utf-8'))


def test_case_ids_are_unique():
    ids = [case[0] for case in CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(('case_id', 'streams', 'metadata', 'world', 'form', 'model'), CASES, ids=[c[0] for c in CASES])
def test_streaming_response_matches_golden(monkeypatch, case_id, streams, metadata, world, form, model):
    observed = _run(streams, metadata, world, form, model, monkeypatch)
    if RECORD:
        pytest.skip('recording')
    assert observed == _golden()[case_id]


def test_record_golden(monkeypatch):
    """Writes the golden file, only when STREAMING_RESPONSE_GOLDEN_RECORD=1."""
    if not RECORD:
        pytest.skip('set STREAMING_RESPONSE_GOLDEN_RECORD=1 to record the golden file')
    golden = {}
    for case_id, streams, metadata, world, form, model in CASES:
        with monkeypatch.context() as m:
            golden[case_id] = _run(streams, metadata, world, form, model, m)
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(golden, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


# --- behaviour fixed since the golden file was recorded ---------------------
# These run through the same fakes as the cases above but are asserted here,
# not in the golden file, which pins the refactor and changes only by hand.

OPENROUTER_ERROR_CHUNK = sse(
    {
        'id': 'gen-abc123',
        'object': 'chat.completion.chunk',
        'error': {'code': 502, 'message': 'Provider disconnected unexpectedly'},
        'choices': [{'index': 0, 'delta': {'content': ''}, 'finish_reason': 'error'}],
    }
)


def _emitted(observed, kind='chat:completion'):
    return [c[1]['data'] for c in observed['calls'] if c[0] == 'emit' and c[1].get('type') == kind]


def _saved(observed):
    return [c[2] for c in observed['calls'] if c[0] == 'chats.upsert_message']


def test_openrouter_mid_stream_error_chunk_is_saved_and_shown(monkeypatch):
    """#162: OpenRouter ends a failed reply with a top-level error beside its choices.

    That chunk once took the ordinary delta path, so the cut-off text was
    saved and shown as a complete answer with no error anywhere.
    """
    error = {'code': 502, 'message': 'Provider disconnected unexpectedly'}
    observed = _run(
        [[chunk('The answer is'), OPENROUTER_ERROR_CHUNK, DONE]], {'chat_id': SAVED_CHAT}, {}, {}, None, monkeypatch
    )

    assert {'error': {'content': error}} in _saved(observed)
    assert {'error': error} in _emitted(observed)
    # The error chunk itself is not forwarded as one more delta.
    assert not [d for d in _emitted(observed, 'response:completion') if 'error' in d]


def test_provider_error_after_tool_call_does_not_execute_tool(monkeypatch):
    """A failed streamed completion must not run tool calls accumulated before the error."""
    error = {'code': 502, 'message': 'Provider disconnected unexpectedly'}
    failed_tool_stream = tool_call_stream('lookup', '{"q": 1}')[:-1] + [OPENROUTER_ERROR_CHUNK, DONE]
    observed = _run(
        [failed_tool_stream],
        {'chat_id': SAVED_CHAT, 'tools': [tool('lookup', result={'should_not': 'run'})]},
        {},
        {},
        None,
        monkeypatch,
    )

    assert not [call for call in observed['calls'] if call[0] == 'tool']
    assert {'error': {'content': error}} in _saved(observed)
    assert {'error': error} in _emitted(observed)


def test_finish_reason_error_without_an_error_object_is_still_an_error(monkeypatch):
    chunk_without_detail = sse({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'error'}]})
    observed = _run([[chunk('Half'), chunk_without_detail, DONE]], {'chat_id': SAVED_CHAT}, {}, {}, None, monkeypatch)

    errors = [d['error'] for d in _emitted(observed) if 'error' in d]
    assert errors and errors[0]['message']
    assert {'error': {'content': errors[0]}} in _saved(observed)


def test_a_null_error_key_beside_choices_is_not_an_error(monkeypatch):
    observed = _run(
        [[sse({'error': None, 'choices': [{'index': 0, 'delta': {'content': 'fine'}}]}), DONE]],
        {'chat_id': SAVED_CHAT},
        {},
        {},
        None,
        monkeypatch,
    )

    assert not [d for d in _emitted(observed) if 'error' in d]
    assert not [s for s in _saved(observed) if 'error' in s]


def test_text_buffered_before_a_provider_error_is_sent_before_it(monkeypatch):
    # Deltas are coalesced three at a time, so 'The answer is' is still
    # buffered when the error chunk arrives.
    metadata = {'chat_id': SAVED_CHAT, 'params': {'stream_delta_chunk_size': 3}}
    observed = _run([[chunk('The answer is'), OPENROUTER_ERROR_CHUNK, DONE]], metadata, {}, {}, None, monkeypatch)

    emits = [c[1] for c in observed['calls'] if c[0] == 'emit']
    text_at = max(
        i for i, e in enumerate(emits) if e.get('type') == 'response:completion' and 'The answer is' in json.dumps(e)
    )
    error_at = next(i for i, e in enumerate(emits) if e.get('type') == 'chat:completion' and 'error' in e['data'])
    assert text_at < error_at


def test_an_error_chunk_that_carries_text_keeps_the_text(monkeypatch):
    last_words = sse(
        {
            'error': {'message': 'cut off'},
            'choices': [{'index': 0, 'delta': {'content': ' forty-two'}, 'finish_reason': 'error'}],
        }
    )
    observed = _run([[chunk('The answer is'), last_words, DONE]], {'chat_id': SAVED_CHAT}, {}, {}, None, monkeypatch)

    assert {'error': {'message': 'cut off'}} in _emitted(observed)
    assert 'The answer is forty-two' in json.dumps(observed['assistant_message'])


def test_an_error_detail_on_the_choice_is_kept(monkeypatch):
    on_choice = sse(
        {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'error', 'error': {'message': 'upstream 503'}}]}
    )
    observed = _run([[chunk('Half'), on_choice, DONE]], {'chat_id': SAVED_CHAT}, {}, {}, None, monkeypatch)

    assert {'error': {'message': 'upstream 503'}} in _emitted(observed)
