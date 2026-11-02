"""Characterization tests for main.chat_completion, the chat completions endpoint.

chat_completion was refactored for ruff's complexity limit with no tests of
its own, so these pin what it did before the refactor: what it returns (or
raises), the form data and metadata it hands to the chat pipeline, and every
model lookup, access check, chat and message write, event, background task
and cleanup step along the way, in order. Everything it calls is a fake that
records its calls: models, config, chats, channels, messages, permissions,
events, the task registry and the chat pipeline itself. Background tasks
(one per model, and the first title generation) are collected and run after
the call returns, in the order they were started, so they run the same way
every time. Each case was recorded from the untouched function into
fixtures/chat_completion.json and must not change.

The golden file is only ever written by hand, never by this test. To record
it again (only for an intended behaviour change, reviewed as such), run
`CHAT_COMPLETION_GOLDEN_RECORD=1 pytest tests/test_chat_completion.py`.
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

from local_llm import main
from local_llm.constants import TASKS
from local_llm.utils import subagents, timers

GOLDEN = Path(__file__).parent / 'fixtures' / 'chat_completion.json'
RECORD = os.environ.get('CHAT_COMPLETION_GOLDEN_RECORD') == '1'

SAVED_CHAT = '7f1c0d9e-3b5a-4c1e-9a2f-0b8d6e4c2a10'
USAGE_MODEL = {'id': 'm1', 'info': {'meta': {'capabilities': {'usage': True}}}}
MODELS = {
    'm1': USAGE_MODEL,
    'm2': {'id': 'm2', 'info': None},
    'custom': {'id': 'custom', 'info': {'meta': {}}},
    'fallback': {'id': 'fallback'},
}


class _Params:
    def __init__(self, params):
        self.params = params

    def model_dump(self):
        return copy.deepcopy(self.params)


class _ModelInfo:
    def __init__(self, id, base_model_id=None, params=None):
        self.id, self.base_model_id = id, base_model_id
        self.params = _Params(params) if params is not None else None

    def model_copy(self, update):
        clone = _ModelInfo(self.id, self.base_model_id, self.params.params if self.params else None)
        for key, value in update.items():
            setattr(clone, key, value)
        return clone


MODEL_INFOS = {
    'm1': {'params': {'temperature': 0.2, 'stream_delta_chunk_size': 4, 'function_calling': 'default'}},
    'custom': {'base_model_id': 'gone-base', 'params': {'reasoning_tags': ['<t>', '</t>']}},
}

USER_MESSAGE = {'id': 'um1', 'role': 'user', 'content': 'hello there', 'parentId': 'a0', 'files': []}
NEW_CHAT = {'parent_id': None, 'session_id': 'sess-1', 'user_message': {'id': 'um1', 'role': 'user', 'content': 'hi'}}
FOLLOW_UP = {'chat_id': SAVED_CHAT, 'session_id': 'sess-1', 'user_message': USER_MESSAGE, 'parent_id': 'a0'}

CASES_SPEC = [
    # --- model resolution and access
    ('model_not_found', {'model': 'nope'}, {}),
    ('models_loaded_on_demand', {'model': 'm2', 'messages': []}, {'models_unloaded': True}),
    ('access_denied', {'model': 'm2'}, {'denied': ['m2']}),
    ('access_bypassed_globally', {'model': 'm2'}, {'denied': ['m2'], 'env': {'BYPASS_MODEL_ACCESS_CONTROL': True}}),
    ('admin_bypass', {'model': 'm2'}, {'denied': ['m2'], 'admin': True, 'env': {'BYPASS_ADMIN_ACCESS_CONTROL': True}}),
    ('admin_without_bypass_checked', {'model': 'm2'}, {'denied': ['m2'], 'admin': True}),
    ('direct_model', {'model': 'ext', 'model_item': {'id': 'ext', 'direct': True, 'info': {'meta': {}}}}, {}),
    ('missing_base_model_no_fallback', {'model': 'custom'}, {}),
    (
        'missing_base_model_fallback',
        {'model': 'custom', 'stream': True},
        {'env': {'ENABLE_CUSTOM_MODEL_FALLBACK': True}, 'config': {'ui.default_models': ' , fallback,m1'}},
    ),
    (
        'missing_base_model_fallback_unknown',
        {'model': 'custom'},
        {'env': {'ENABLE_CUSTOM_MODEL_FALLBACK': True}, 'config': {'ui.default_models': 'ghost'}},
    ),
    (
        'missing_base_model_fallback_none_configured',
        {'model': 'custom'},
        {'env': {'ENABLE_CUSTOM_MODEL_FALLBACK': True}},
    ),
    (
        'fallback_access_denied',
        {'model': 'custom'},
        {
            'env': {'ENABLE_CUSTOM_MODEL_FALLBACK': True},
            'config': {'ui.default_models': 'fallback'},
            'denied': ['fallback'],
        },
    ),
    # --- params
    (
        'params_merge_and_overrides',
        {'model': 'm1', 'stream': True, 'stream_options': {'x': 1}, 'params': {'temperature': 0.9, 'top_p': None}},
        {
            'config': {'models.default_params': {'top_k': 5, 'stream_response': False}},
            'model_infos': {
                'm1': {
                    'params': {
                        'stream_response': True,
                        'stream_delta_chunk_size': 8,
                        'reasoning_tags': False,
                        'compact_token_threshold': 1000,
                    }
                }
            },
        },
    ),
    ('stream_usage_requested', {'model': 'm1', 'stream': True}, {}),
    ('no_params_anywhere', {'model': 'm2', 'params': {'a': None}}, {}),
    (
        'request_params_only',
        {'model': 'm2', 'params': {'stream_delta_chunk_size': 2, 'reasoning_tags': True, 'compact_token_threshold': 9}},
        {},
    ),
    # --- message ids and chat variables
    ('message_ids_list', {'model': 'm1', 'id': 'x', 'message_ids': [{'model_id': 'm1', 'message_id': 'a1'}]}, {}),
    ('message_ids_legacy_dict', {'model': 'm1', 'id': 'x', 'message_ids': {'m1': 'a1', 'm2': 'a2'}}, {}),
    ('chat_variables_given', {'model': 'm1', 'chat_variables': {'name': 'Ada'}, 'parent_message': {'id': 'p1'}}, {}),
    ('chat_variables_from_saved_chat', {'model': 'm1', 'chat_id': SAVED_CHAT}, {'saved_variables': {'tone': 'dry'}}),
    ('chat_variables_saved_chat_missing', {'model': 'm1', 'chat_id': SAVED_CHAT}, {'chat_missing': True}),
    # --- tool servers and approval mode
    ('tool_servers_dropped_without_permission', {'model': 'm1', 'tool_servers': [{'url': 'x'}]}, {}),
    ('tool_servers_kept_with_permission', {'model': 'm1', 'tool_servers': [{'url': 'x'}]}, {'permitted': True}),
    ('tool_servers_kept_for_admin', {'model': 'm1', 'tool_servers': [{'url': 'x'}]}, {'admin': True}),
    (
        'tool_approval_from_params',
        {'model': 'm1', 'params': {'tool_approval_mode': 'ask'}},
        {'config': {'chat.tool_permissions.enable': True}},
    ),
    ('tool_approval_disabled', {'model': 'm1', 'params': {'tool_approval_mode': 'ask'}}, {}),
    (
        'tool_approval_automation',
        {'model': 'm1', 'automation_id': 'auto-1', 'params': {'tool_approval_mode': 'ask'}},
        {'config': {'chat.tool_permissions.enable': True}},
    ),
    ('tool_approval_unset', {'model': 'm1'}, {'config': {'chat.tool_permissions.enable': True}}),
    # --- channels
    ('channel_not_found', {'model': 'm1', 'chat_id': 'channel:ch-missing'}, {}),
    ('channel_group_member', {'model': 'm1', 'chat_id': 'channel:ch-group', 'id': 'cm1'}, {'channel_members': ['u1']}),
    ('channel_group_not_member', {'model': 'm1', 'chat_id': 'channel:ch-group'}, {}),
    ('channel_standard_writer', {'model': 'm1', 'chat_id': 'channel:ch-std'}, {'channel_grants': True}),
    ('channel_standard_no_write', {'model': 'm1', 'chat_id': 'channel:ch-std'}, {}),
    ('channel_admin_any_message', {'model': 'm1', 'chat_id': 'channel:ch-std', 'id': 'cm-other'}, {'admin': True}),
    (
        'channel_message_from_other_channel',
        {'model': 'm1', 'chat_id': 'channel:ch-std', 'id': 'cm-elsewhere'},
        {'admin': True},
    ),
    (
        'channel_message_not_own',
        {'model': 'm1', 'chat_id': 'channel:ch-group', 'id': 'cm-other'},
        {'channel_members': ['u1']},
    ),
    (
        'channel_message_unknown',
        {'model': 'm1', 'chat_id': 'channel:ch-group', 'id': 'cm-ghost'},
        {'channel_members': ['u1']},
    ),
    # --- new chats
    (
        'new_chat_multi_model_with_title',
        {
            **NEW_CHAT,
            'model': 'm1',
            'user_message': {
                'id': 'um1',
                'role': 'user',
                'content': 'hi',
                'files': [{'type': 'file', 'id': 'f1'}, {'type': 'web', 'id': 'w'}],
            },
            'message_ids': [
                {'model_id': 'm1', 'message_id': 'a1'},
                {'model_id': 'm2', 'message_id': 'a2', 'modelIdx': 1},
                {'model_id': 'm2', 'message_id': None},
            ],
            'background_tasks': {
                TASKS.TITLE_GENERATION: True,
                TASKS.TAGS_GENERATION: True,
                TASKS.FOLLOW_UP_GENERATION: True,
            },
            'files': [{'id': 'f1'}],
            'folder_id': 'fold-1',
        },
        {'folder_write': True},
    ),
    ('new_chat_folder_denied', {**NEW_CHAT, 'model': 'm1', 'folder_id': 'fold-x'}, {}),
    (
        'new_chat_without_user_message',
        {**NEW_CHAT, 'model': 'm1', 'user_message': None, 'message_ids': [{'model_id': 'm1', 'message_id': 'a1'}]},
        {},
    ),
    (
        'new_chat_chat_files_fail',
        {
            **NEW_CHAT,
            'model': 'm1',
            'user_message': {'id': 'um1', 'content': 'hi', 'files': [{'type': 'file', 'id': 'f1'}]},
            'id': 'a1',
        },
        {'chat_files_raise': True},
    ),
    (
        'new_chat_title_task_fails',
        {**NEW_CHAT, 'model': 'm1', 'id': 'a1', 'background_tasks': {TASKS.TITLE_GENERATION: True}},
        {'title_raise': True},
    ),
    (
        'new_chat_title_without_assistant',
        {**NEW_CHAT, 'model': 'm1', 'background_tasks': {TASKS.TITLE_GENERATION: True}},
        {},
    ),
    ('new_chat_with_chat_id_is_not_new', {**NEW_CHAT, 'model': 'm1', 'chat_id': SAVED_CHAT, 'id': 'a1'}, {}),
    # --- existing chats
    ('existing_chat_not_owner', {**FOLLOW_UP, 'model': 'm1', 'id': 'a1'}, {'not_owner': True}),
    (
        'existing_chat_admin_not_owner',
        {**FOLLOW_UP, 'model': 'm1', 'id': 'a1', 'files': [{'id': 'f9'}]},
        {'not_owner': True, 'admin': True},
    ),
    (
        'existing_chat_follow_up',
        {
            **FOLLOW_UP,
            'model': 'm1',
            'user_message': {**USER_MESSAGE, 'models': ['m1', 'm2'], 'files': [{'type': 'file', 'id': 'f2'}]},
            'message_ids': [
                {'model_id': 'm1', 'message_id': 'a1'},
                {'model_id': 'm2', 'message_id': 'a2', 'modelIdx': 1},
            ],
            'assistant_message_id': 'a1',
        },
        {
            'grandparent': {'id': 'a0', 'childrenIds': ['old']},
            'existing_user_message': {'id': 'um1', 'childrenIds': ['a1']},
        },
    ),
    (
        'existing_chat_grandparent_already_linked',
        {**FOLLOW_UP, 'model': 'm1', 'id': 'a1'},
        {'grandparent': {'id': 'a0', 'childrenIds': ['um1']}},
    ),
    ('existing_chat_grandparent_missing', {**FOLLOW_UP, 'model': 'm1', 'id': 'a1'}, {}),
    (
        'existing_chat_no_parent_no_models',
        {
            **FOLLOW_UP,
            'model': 'm1',
            'user_message': {'id': 'um1', 'content': 'x', 'meta': {'internal': True}},
            'id': None,
        },
        {},
    ),
    ('existing_chat_internal_request', {**FOLLOW_UP, 'model': 'm1', 'id': 'a1'}, {'internal': True}),
    ('existing_chat_timer_cancel_fails', {**FOLLOW_UP, 'model': 'm1', 'id': 'a1'}, {'timers_raise': True}),
    (
        'existing_chat_files_fail',
        {
            **FOLLOW_UP,
            'model': 'm1',
            'id': 'a1',
            'user_message': {**USER_MESSAGE, 'files': [{'type': 'file', 'id': 'f2'}]},
        },
        {'chat_files_raise': True},
    ),
    ('existing_chat_without_user_message', {'chat_id': SAVED_CHAT, 'session_id': 's', 'model': 'm1', 'id': 'a1'}, {}),
    ('existing_chat_update_fails', {**FOLLOW_UP, 'model': 'm1', 'id': 'a1'}, {'update_raise': True}),
    (
        'existing_chat_nothing_to_persist',
        {
            **FOLLOW_UP,
            'model': 'm1',
            'user_message': {'id': 'um1', 'content': 'x'},
            'message_ids': [{'model_id': '', 'message_id': None}],
        },
        {},
    ),
    # --- the pipeline (legacy, single model, synchronous)
    ('legacy_pipeline_ok', {'model': 'm1', 'messages': [{'role': 'user', 'content': 'q'}]}, {}),
    ('legacy_pipeline_paused', {'model': 'm1'}, {'paused': True}),
    ('legacy_pipeline_upstream_error', {'model': 'm1'}, {'response': 'error_json'}),
    ('legacy_pipeline_ok_json', {'model': 'm1'}, {'response': 'ok_json'}),
    ('legacy_pipeline_raises', {'model': 'm1'}, {'payload_raise': 'RuntimeError'}),
    ('legacy_pipeline_http_error', {'model': 'm1'}, {'payload_raise': 'HTTPException'}),
    (
        'legacy_pipeline_cancelled',
        {'model': 'm1', 'chat_id': 'local:abc', 'id': 'a1'},
        {'payload_raise': 'CancelledError'},
    ),
    (
        'legacy_pipeline_error_local_chat',
        {'model': 'm1', 'chat_id': 'local:abc', 'id': 'a1'},
        {'payload_raise': 'RuntimeError'},
    ),
    (
        'legacy_pipeline_error_saved_chat_no_emitter',
        {'model': 'm1', 'chat_id': SAVED_CHAT, 'id': 'a1'},
        {'payload_raise': 'RuntimeError', 'no_emitter': True},
    ),
    (
        'legacy_pipeline_error_report_fails',
        {'model': 'm1', 'chat_id': SAVED_CHAT, 'id': 'a1'},
        {'payload_raise': 'RuntimeError', 'upsert_raise': True},
    ),
    ('legacy_pipeline_mcp_cleanup', {'model': 'm1'}, {'mcp_clients': ['ok', 'raise', 'ok']}),
    ('legacy_pipeline_mcp_clients_malformed', {'model': 'm1'}, {'mcp_clients': 'malformed'}),
    # --- fan-out (session + chat id)
    (
        'fanout_two_models',
        {
            **FOLLOW_UP,
            'model': 'm1',
            'message_ids': [
                {'model_id': 'm1', 'message_id': 'a1'},
                {'model_id': 'm2', 'message_id': 'a2'},
                {'model_id': 'gone', 'message_id': 'a3'},
            ],
            'background_tasks': {
                TASKS.TITLE_GENERATION: True,
                TASKS.TAGS_GENERATION: True,
                TASKS.FOLLOW_UP_GENERATION: True,
            },
        },
        {'active_after': [True, False, False, False]},
    ),
    ('fanout_no_emitter', {**FOLLOW_UP, 'model': 'm1', 'id': 'a1'}, {'no_emitter': True}),
    ('fanout_no_assistant_ids', {**FOLLOW_UP, 'model': 'm1', 'id': None}, {}),
    (
        'fanout_only_title_tasks_for_second',
        {
            **FOLLOW_UP,
            'model': 'm1',
            'message_ids': [{'model_id': 'm1', 'message_id': 'a1'}, {'model_id': 'm2', 'message_id': 'a2'}],
            'background_tasks': {TASKS.TITLE_GENERATION: True},
        },
        {},
    ),
    (
        'fanout_internal_runs_inline',
        {
            **FOLLOW_UP,
            'model': 'm1',
            'message_ids': [{'model_id': 'm1', 'message_id': 'a1'}, {'model_id': 'm2', 'message_id': 'a2'}],
        },
        {'internal': True},
    ),
    ('fanout_folder_lookup', {**FOLLOW_UP, 'model': 'm1', 'id': 'a1'}, {'chat_folder': 'fold-9'}),
    ('fanout_cleanup_errors', {**FOLLOW_UP, 'model': 'm1', 'id': 'a1'}, {'cleanup_raise': True, 'pending_raise': True}),
    ('fanout_active_emit_cancelled', {**FOLLOW_UP, 'model': 'm1', 'id': 'a1'}, {'active_emit_cancelled': True}),
    ('fanout_task_cancelled', {**FOLLOW_UP, 'model': 'm1', 'id': 'a1'}, {'payload_raise': 'CancelledError'}),
    (
        'fanout_task_cancelled_no_emitter',
        {**FOLLOW_UP, 'model': 'm1', 'id': 'a1'},
        {'payload_raise': 'CancelledError', 'no_emitter': True},
    ),
    (
        'fanout_cancel_emit_fails',
        {**FOLLOW_UP, 'model': 'm1', 'id': 'a1'},
        {'payload_raise': 'CancelledError', 'emit_raise': True},
    ),
]

CASES = [(f'chat:{c}', form, world) for c, form, world in CASES_SPEC]
ENV = {
    'ENABLE_CUSTOM_MODEL_FALLBACK': False,
    'BYPASS_MODEL_ACCESS_CONTROL': False,
    'BYPASS_ADMIN_ACCESS_CONTROL': False,
}


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, 'model_dump'):
        return _plain(value.model_dump())
    if isinstance(value, JSONResponse):
        return {'json_response': value.status_code, 'body': json.loads(value.body)}
    if isinstance(value, _MCPClient):
        return f'<mcp {value.name}>'
    return repr(value)


class _MCPClient:
    def __init__(self, name, behaviour, log):
        self.name, self.behaviour, self.log = name, behaviour, log

    async def disconnect(self):
        self.log.append(['mcp_disconnect', self.name])
        if self.behaviour == 'raise':
            raise RuntimeError('transport closed')


class _World:
    """The fakes chat_completion talks to, sharing one call log."""

    def __init__(self, world, log):
        self.world, self.log = world, log
        self.background = []
        self.uuids = iter(f'uuid-{n}' for n in range(1, 100))
        self.active = list(world.get('active_after', []))

    def emitter_for(self, metadata, update_db=True):
        if self.world.get('no_emitter'):
            return None
        log, world = self.log, self.world
        message_id = metadata.get('message_id')

        async def emit(event):
            log.append(['emit', message_id, update_db, event])
            if world.get('emit_raise') and event.get('type') == 'chat:tasks:cancel':
                raise RuntimeError('socket gone')
            if world.get('active_emit_cancelled') and event.get('type') == 'chat:active':
                raise asyncio.CancelledError()

        return emit


def _install_models(m, w, request):
    world, log = w.world, w.log
    infos = {**MODEL_INFOS, **world.get('model_infos', {})}

    async def get_all_models(request_, user=None):
        log.append(['get_all_models', user.id])
        request.app.state.MODELS = copy.deepcopy(MODELS)

    async def get_model_by_id(model_id):
        log.append(['model_info', model_id])
        info = infos.get(model_id)
        return _ModelInfo(model_id, info.get('base_model_id'), info.get('params')) if info else None

    async def check_model_access(user, model, model_info=None):
        log.append(['check_model_access', model['id'], getattr(model_info, 'base_model_id', 'no-info')])
        if model['id'] in world.get('denied', []):
            raise HTTPException(status_code=403, detail='Model access denied')

    async def set_direct_model(request_, model_item, user):
        log.append(['set_direct_model', model_item['id'], user.id])

    m.setattr(main, 'get_all_models', get_all_models)
    m.setattr(main.Models, 'get_model_by_id', get_model_by_id)
    m.setattr(main, 'check_model_access', check_model_access)
    m.setattr(main, '_set_direct_model', set_direct_model)


def _install_chats(m, w):
    world, log = w.world, w.log

    def chats_fake(name, result=None, raise_flag=None):
        async def fake(*args, **kwargs):
            log.append([f'chats.{name}', _plain(args), _plain(kwargs)])
            if raise_flag and world.get(raise_flag):
                raise RuntimeError(f'{name} failed')
            return result(*args) if callable(result) else copy.deepcopy(result)

        return fake

    def chat_by_id(chat_id):
        if world.get('chat_missing'):
            return None
        return SimpleNamespace(variables=copy.deepcopy(world.get('saved_variables', {'saved': 'yes'})))

    def message_by_id(chat_id, message_id):
        if message_id == 'a0':
            return copy.deepcopy(world.get('grandparent'))
        return copy.deepcopy(world.get('existing_user_message'))

    def upsert_result(chat_id, message_id, data):
        # Only the write that records a failed reply fails, so the error report itself is what breaks.
        if world.get('upsert_raise') and 'error' in data:
            raise RuntimeError('database is locked')
        return None

    for name, fake in [
        ('get_chat_by_id', chats_fake('get_chat_by_id', chat_by_id)),
        ('insert_new_chat', chats_fake('insert_new_chat')),
        ('insert_chat_files', chats_fake('insert_chat_files', raise_flag='chat_files_raise')),
        ('is_chat_owner', chats_fake('is_chat_owner', lambda *a: not world.get('not_owner'))),
        ('update_chat_by_id', chats_fake('update_chat_by_id', raise_flag='update_raise')),
        ('update_chat_variables_by_id', chats_fake('update_chat_variables_by_id')),
        ('upsert_message_to_chat_by_id_and_message_id', chats_fake('upsert_message', upsert_result)),
        ('get_message_by_id_and_message_id', chats_fake('get_message', message_by_id)),
        ('get_chat_folder_id', chats_fake('get_chat_folder_id', lambda *a: world.get('chat_folder'))),
    ]:
        m.setattr(main.Chats, name, fake)


def _install_access(m, w):
    world, log = w.world, w.log
    channels = {
        'ch-group': SimpleNamespace(id='ch-group', type='group'),
        'ch-std': SimpleNamespace(id='ch-std', type=None),
    }
    messages = {
        'cm1': SimpleNamespace(channel_id='ch-group', user_id='u1'),
        'cm-other': SimpleNamespace(channel_id='ch-std', user_id='someone'),
        'cm-elsewhere': SimpleNamespace(channel_id='ch-far', user_id='u1'),
    }

    async def get_channel_by_id(channel_id):
        log.append(['channel', channel_id])
        return channels.get(channel_id)

    async def is_user_channel_member(channel_id, user_id):
        log.append(['channel_member', channel_id, user_id])
        return user_id in world.get('channel_members', [])

    async def has_access(user_id, resource_type, resource_id, permission, user_group_ids=None):
        log.append(['has_access', user_id, resource_type, resource_id, permission])
        return bool(world.get('channel_grants'))

    async def get_message_by_id(message_id):
        log.append(['channel_message', message_id])
        return messages.get(message_id)

    async def has_folder_write_access(user_id, folder_id):
        log.append(['folder_write_access', user_id, folder_id])
        return bool(world.get('folder_write'))

    async def has_permission(user_id, key, default_permissions=None):
        log.append(['has_permission', user_id, key])
        return bool(world.get('permitted'))

    m.setattr(main.Channels, 'get_channel_by_id', get_channel_by_id)
    m.setattr(main.Channels, 'is_user_channel_member', is_user_channel_member)
    m.setattr(main.AccessGrants, 'has_access', has_access)
    m.setattr(main.Messages, 'get_message_by_id', get_message_by_id)
    m.setattr(main, 'has_folder_write_access', has_folder_write_access)
    m.setattr(main, 'has_permission', has_permission)


def _install_pipeline(m, w):
    world, log = w.world, w.log

    async def process_chat_payload(request, form_data, user, metadata, model):
        log.append(['process_chat_payload', _plain(form_data), model.get('id')])
        failure = world.get('payload_raise')
        if failure == 'RuntimeError':
            raise RuntimeError('payload exploded')
        if failure == 'HTTPException':
            raise HTTPException(status_code=422, detail='bad payload')
        if failure == 'CancelledError':
            raise asyncio.CancelledError()
        metadata = {**metadata}
        if world.get('mcp_clients') == 'malformed':
            metadata['mcp_clients'] = ['not', 'a', 'mapping']
        elif world.get('mcp_clients'):
            metadata['mcp_clients'] = {
                f'c{i}': _MCPClient(f'c{i}', behaviour, log) for i, behaviour in enumerate(world['mcp_clients'])
            }
        return {**form_data, 'processed': True}, metadata, ['event-1']

    async def drain_approved_tool_calls(request, form_data, user, model, metadata):
        log.append(['drain_approved_tool_calls', metadata.get('message_id')])
        return bool(world.get('paused'))

    async def chat_completion_handler(request, form_data, user):
        log.append(['chat_completion_handler', form_data.get('model'), form_data['metadata'].get('message_id')])
        response = world.get('response')
        if response == 'error_json':
            return JSONResponse(status_code=400, content={'error': {'message': 'content filtered'}})
        if response == 'ok_json':
            return JSONResponse(status_code=200, content={'ok': True})
        return {'choices': [{'message': {'content': 'answer'}}]}

    async def build_chat_response_context(request, form_data, user, model, metadata, tasks, events):
        log.append(['build_chat_response_context', model.get('id'), metadata.get('message_id'), _plain(tasks), events])
        return {'ctx_for': metadata.get('message_id')}

    async def process_chat_response(response, ctx):
        log.append(['process_chat_response', _plain(response), ctx])
        return {'processed_response': ctx['ctx_for']}

    async def get_event_emitter(metadata, update_db=True):
        log.append(['get_event_emitter', metadata.get('message_id'), update_db])
        return w.emitter_for(metadata, update_db)

    async def emit_chat_list_event(metadata, chat_id):
        log.append(['emit_chat_list_event', metadata.get('message_id'), chat_id])

    async def publish_event(request, event, actor=None, subject_id=None, **kwargs):
        log.append(
            ['event', str(event.name if hasattr(event, 'name') else event), actor.id, subject_id, _plain(kwargs)]
        )

    async def background_tasks_handler(ctx):
        log.append(['background_tasks_handler', ctx['metadata'].get('message_id'), _plain(ctx['tasks'])])
        if world.get('title_raise'):
            raise RuntimeError('title model down')

    m.setattr(main, 'process_chat_payload', process_chat_payload)
    m.setattr(main, 'drain_approved_tool_calls', drain_approved_tool_calls)
    m.setattr(main, 'chat_completion_handler', chat_completion_handler)
    m.setattr(main, 'build_chat_response_context', build_chat_response_context)
    m.setattr(main, 'process_chat_response', process_chat_response)
    m.setattr(main, 'get_event_emitter', get_event_emitter)
    m.setattr(main, 'emit_chat_list_event', emit_chat_list_event)
    m.setattr(main, 'publish_event', publish_event)
    m.setattr(main, 'background_tasks_handler', background_tasks_handler)


def _install_tasks(m, w):
    world, log = w.world, w.log

    async def create_task(redis, coroutine, id=None, task_id=None):
        log.append(['create_task', redis, id, task_id])
        w.background.append(('task', task_id, coroutine))
        return task_id, None

    def run_in_background(coroutine):
        # Logged under its name from before utils.background, so the golden
        # file did not change when main stopped calling asyncio directly.
        log.append(['asyncio.create_task', coroutine.__name__])
        w.background.append(('asyncio', coroutine.__name__, coroutine))
        return SimpleNamespace(name='<task>')

    async def cleanup_task(redis, task_id, chat_id):
        log.append(['cleanup_task', task_id, chat_id])
        if world.get('cleanup_raise'):
            raise RuntimeError('redis down')

    async def has_active_tasks(redis, chat_id):
        result = w.active.pop(0) if w.active else False
        log.append(['has_active_tasks', chat_id, result])
        return result

    async def cancel_timers_for_chat(chat_id, reason, user_id):
        log.append(['cancel_timers_for_chat', chat_id, reason, user_id])
        if world.get('timers_raise'):
            raise RuntimeError('timers table missing')

    async def process_pending_internal_messages(request, chat_id, user_id, context):
        log.append(['process_pending_internal_messages', chat_id, user_id, _plain(context)])
        if world.get('pending_raise'):
            raise RuntimeError('pending failed')

    m.setattr(main, 'create_task', create_task)
    m.setattr(main, 'run_in_background', run_in_background)
    m.setattr(main, 'cleanup_task', cleanup_task)
    m.setattr(main, 'has_active_tasks', has_active_tasks)
    m.setattr(timers, 'cancel_timers_for_chat', cancel_timers_for_chat)
    m.setattr(subagents, 'process_pending_internal_messages', process_pending_internal_messages)


def _install_config(m, w):
    world, log = w.world, w.log
    config = {'models.default_params': {}, **world.get('config', {})}

    async def config_get(key, default=None):
        log.append(['config_get', key])
        return copy.deepcopy(config.get(key, default))

    m.setattr(main.Config, 'get', config_get)
    m.setattr(main, 'uuid4', lambda: next(w.uuids))
    m.setattr(main.time, 'time', lambda: 1_700_000_000.5)
    for name, value in {**ENV, **world.get('env', {})}.items():
        m.setattr(main, name, value)


async def _call(form, world, w, request, user):
    try:
        outcome = {'returned': _plain(await main.chat_completion(request, copy.deepcopy(form), user=user))}
    except HTTPException as e:
        outcome = {'http_error': [e.status_code, e.detail]}
    except asyncio.CancelledError:
        outcome = {'cancelled': True}
    except Exception as e:
        outcome = {'raised': [type(e).__name__, str(e)]}
    background = []
    while w.background:
        kind, name, coroutine = w.background.pop(0)
        try:
            background.append([kind, name, _plain(await coroutine)])
        except HTTPException as e:
            background.append([kind, name, {'http_error': [e.status_code, e.detail]}])
        except asyncio.CancelledError:
            background.append([kind, name, 'cancelled'])
    return {**outcome, 'background': background}


def _run(form, world, monkeypatch):
    log = []
    w = _World(world, log)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(MODELS={} if world.get('models_unloaded') else copy.deepcopy(MODELS), redis='<redis>')
        ),
        state=SimpleNamespace(**({'internal': True} if world.get('internal') else {})),
        headers={'user-agent': 'pytest-agent'},
    )
    user = SimpleNamespace(id='u1', role='admin' if world.get('admin') else 'user')
    _install_models(monkeypatch, w, request)
    _install_chats(monkeypatch, w)
    _install_access(monkeypatch, w)
    _install_pipeline(monkeypatch, w)
    _install_tasks(monkeypatch, w)
    _install_config(monkeypatch, w)
    outcome = asyncio.run(_call(form, world, w, request, user))
    request_metadata = getattr(request.state, 'metadata', None)
    return json.loads(json.dumps({**outcome, 'request_metadata': _plain(request_metadata), 'calls': log}))


def _golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding='utf-8'))


def test_case_ids_are_unique():
    ids = [case[0] for case in CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(('case_id', 'form', 'world'), CASES, ids=[c[0] for c in CASES])
def test_chat_completion_matches_golden(monkeypatch, case_id, form, world):
    observed = _run(form, world, monkeypatch)
    if RECORD:
        pytest.skip('recording')
    assert observed == _golden()[case_id]


def test_custom_model_fallback_is_preserved_in_session_fanout(monkeypatch):
    form = {
        'model': 'custom',
        'messages': [{'role': 'user', 'content': 'hi'}],
        'session_id': 'sid',
        'chat_id': 'local:sid',
        'id': 'assistant-1',
    }
    world = {
        'env': {'ENABLE_CUSTOM_MODEL_FALLBACK': True},
        'config': {'ui.default_models': 'fallback'},
    }

    observed = _run(form, world, monkeypatch)
    payload_calls = [call for call in observed['calls'] if call[0] == 'process_chat_payload']

    assert [(call[1]['model'], call[2]) for call in payload_calls] == [('fallback', 'fallback')]


def test_custom_model_fallback_remaps_only_its_multi_model_target(monkeypatch):
    form = {
        'model': 'custom',
        'messages': [{'role': 'user', 'content': 'hi'}],
        'session_id': 'sid',
        'chat_id': 'local:sid',
        'message_ids': [
            {'model_id': 'custom', 'message_id': 'assistant-1'},
            {'model_id': 'm2', 'message_id': 'assistant-2'},
        ],
    }
    world = {
        'env': {'ENABLE_CUSTOM_MODEL_FALLBACK': True},
        'config': {'ui.default_models': 'fallback'},
    }

    observed = _run(form, world, monkeypatch)
    payload_calls = [call for call in observed['calls'] if call[0] == 'process_chat_payload']

    assert [(call[1]['model'], call[2]) for call in payload_calls] == [('fallback', 'fallback'), ('m2', 'm2')]


def test_record_golden(monkeypatch):
    """Writes the golden file, only when CHAT_COMPLETION_GOLDEN_RECORD=1."""
    if not RECORD:
        pytest.skip('set CHAT_COMPLETION_GOLDEN_RECORD=1 to record the golden file')
    golden = {}
    for case_id, form, world in CASES:
        with monkeypatch.context() as m:
            golden[case_id] = _run(form, world, m)
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(golden, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
