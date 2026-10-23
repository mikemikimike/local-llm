from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from local_llm.config import RAG_EMBEDDING_CONTENT_PREFIX, RAG_EMBEDDING_QUERY_PREFIX
from local_llm.constants import ERROR_MESSAGES
from local_llm.events import EVENTS, publish_event
from local_llm.internal.db import get_async_session
from local_llm.models.config import Config
from local_llm.models.memories import Memories, MemoryModel
from local_llm.models.users import Users
from local_llm.retrieval.vector.async_client import ASYNC_VECTOR_DB_CLIENT
from local_llm.retrieval.vector.main import SearchResult
from local_llm.utils.access_control import has_permission
from local_llm.utils.auth import get_admin_user, get_verified_user
from local_llm.utils.memory import (
    clean_memory_content,
    clean_memory_path,
    list_memory_path_groups,
    memory_vector_text,
    read_memory_path_rows,
    search_memory_rows,
    validate_memory_operations,
)

log = logging.getLogger(__name__)

router = APIRouter()


async def check_memories_permission(user):
    config = await Config.get_many('memories.enable', 'user.permissions')
    if not config.get('memories.enable'):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if user.role != 'admin' and not await has_permission(user.id, 'features.memories', config.get('user.permissions')):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )


############################
# GetMemories
# Let what is remembered here spare someone the cost
# of learning it twice.
############################


@router.get('/', response_model=list[MemoryModel])
async def get_memories(
    request: Request,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    await check_memories_permission(user)

    return await Memories.get_memories_by_user_id(user.id, db=db)


############################
# AddMemory
############################


class AddMemoryForm(BaseModel):
    content: str
    type: Literal['user', 'context'] = 'context'
    path: str | None = None


class MemoryUpdateModel(BaseModel):
    content: str | None = None
    type: Literal['user', 'context'] | None = None
    path: str | None = None


class MemoryOperationModel(BaseModel):
    action: Literal['add', 'replace', 'remove', 'move']
    id: str | None = None
    content: str | None = None
    type: Literal['user', 'context'] | None = None
    path: str | None = None


class UpdateMemoriesForm(BaseModel):
    operations: list[MemoryOperationModel]
    source: Literal['tool', 'background_review'] | None = None


class SearchMemoriesForm(BaseModel):
    query: str | None = None
    type: Literal['user', 'context', 'all'] = 'all'
    path: str | None = None
    memory_id: str | None = None
    limit: int = 20


class ListMemoryPathsForm(BaseModel):
    query: str | None = None
    type: Literal['user', 'context', 'all'] = 'all'
    limit: int = 100


class ReadMemoryPathForm(BaseModel):
    path: str
    type: Literal['user', 'context', 'all'] = 'all'
    include_children: bool = True
    limit: int = 50


def _memory_metadata(memory: MemoryModel) -> dict:
    return {
        'created_at': memory.created_at,
        'updated_at': memory.updated_at,
        'type': memory.type,
        'path': memory.path,
    }


async def reindex_memory_vectors_for_user(
    request: Request,
    user_id: str,
    memories: list[MemoryModel] | None = None,
    user=None,
) -> int:
    collection_name = f'user-memory-{user_id}'
    if memories is None:
        memories = await Memories.get_memories_by_user_id(user_id)
        if memories is None:
            raise RuntimeError(f'Unable to load memories for user {user_id}')
    if not memories:
        try:
            await ASYNC_VECTOR_DB_CLIENT.delete_collection(collection_name)
        except Exception as e:
            log.debug(e)
        return 0

    # Build every embedding before touching the existing collection. A failed
    # model load must leave the last known-good vectors available for recall.

    vectors = await asyncio.gather(
        *[
            request.app.state.EMBEDDING_FUNCTION(
                memory_vector_text(memory.content, memory.path),
                prefix=RAG_EMBEDDING_CONTENT_PREFIX,
                user=user,
            )
            for memory in memories
        ]
    )

    items = [
        {
            'id': memory.id,
            'text': memory_vector_text(memory.content, memory.path),
            'vector': vectors[idx],
            'metadata': _memory_metadata(memory),
        }
        for idx, memory in enumerate(memories)
    ]
    existing_ids: set[str] = set()
    try:
        existing = await ASYNC_VECTOR_DB_CLIENT.get(collection_name)
        if existing and existing.ids and existing.ids[0]:
            existing_ids = set(existing.ids[0])
    except Exception as e:
        log.debug('Unable to inspect existing memory vectors for %s: %s', user_id, e)

    replaced_collection = False
    try:
        await ASYNC_VECTOR_DB_CLIENT.upsert(collection_name=collection_name, items=items)
    except Exception as e:
        if not _is_dimension_mismatch(e):
            raise

        # A collection created with a previous embedding model cannot accept
        # vectors with the new dimension.  All embeddings are ready now, so it
        # is safe to replace the collection after this point.
        await ASYNC_VECTOR_DB_CLIENT.delete_collection(collection_name)
        await ASYNC_VECTOR_DB_CLIENT.upsert(collection_name=collection_name, items=items)
        replaced_collection = True

    if not replaced_collection:
        stale_ids = existing_ids - {memory.id for memory in memories}
        if stale_ids:
            try:
                await ASYNC_VECTOR_DB_CLIENT.delete(collection_name=collection_name, ids=sorted(stale_ids))
            except Exception as e:
                # Keeping an old vector is safer than losing the complete collection.
                log.warning('Unable to remove stale memory vectors for %s: %s', user_id, e)
    return len(memories)


async def upsert_memory_vectors_or_reindex(
    request: Request,
    user,
    items: list[dict],
    memories: list[MemoryModel] | None = None,
) -> None:
    try:
        await ASYNC_VECTOR_DB_CLIENT.upsert(collection_name=f'user-memory-{user.id}', items=items)
    except Exception as e:
        if not _is_dimension_mismatch(e):
            raise

        log.warning('Memory vector dimension mismatch for user %s; reindexing memory vectors.', user.id)
        await reindex_memory_vectors_for_user(request, user.id, memories=memories, user=user)


def _is_dimension_mismatch(error: Exception) -> bool:
    message = str(error).lower()
    return 'dimension' in message and 'embedding' in message


def _memory_vector_item(memory: MemoryModel, vector: list[float | int]) -> dict:
    return {
        'id': memory.id,
        'text': memory_vector_text(memory.content, memory.path),
        'vector': vector,
        'metadata': _memory_metadata(memory),
    }


def _authoritative_memory_results(results, memories: list[MemoryModel]) -> SearchResult:
    """Replace vector payloads with current database rows and drop orphan IDs."""
    memory_by_id = {memory.id: memory for memory in memories}
    ids = results.ids[0] if results and results.ids and results.ids[0] else []
    distances = results.distances[0] if results and results.distances and results.distances[0] else []
    resolved_ids = []
    documents = []
    metadatas = []
    resolved_distances = []
    for idx, memory_id in enumerate(ids):
        memory = memory_by_id.get(memory_id)
        if memory is None:
            continue
        resolved_ids.append(memory.id)
        documents.append(memory_vector_text(memory.content, memory.path))
        metadatas.append(_memory_metadata(memory))
        if results.distances is not None:
            resolved_distances.append(distances[idx] if idx < len(distances) else 0)

    return SearchResult(
        ids=[resolved_ids],
        documents=[documents],
        metadatas=[metadatas],
        distances=[resolved_distances] if results.distances is not None else None,
    )


def _memory_candidate(
    *,
    user_id: str,
    content: str,
    memory_type: str,
    path: str | None,
    meta: dict | None,
    existing: MemoryModel | None = None,
) -> MemoryModel:
    now = int(time.time())
    return MemoryModel(
        id=existing.id if existing else str(uuid4()),
        user_id=user_id,
        type=memory_type,
        path=path,
        content=content,
        meta=meta,
        created_at=existing.created_at if existing else now,
        updated_at=now,
    )


async def _prepare_memory_operations(
    request: Request,
    user,
    operations: list[dict],
    *,
    deduplicate_adds: bool = True,
) -> list[dict]:
    """Resolve rows and embeddings before the first mutation in a batch."""
    existing_memories = await Memories.get_memories_by_user_id(user.id)
    if existing_memories is None:
        raise RuntimeError(f'Unable to load memories for user {user.id}')
    working_memories = list(existing_memories)
    prepared = []
    for operation in operations:
        action = operation.get('action')
        if action == 'add':
            content = operation.get('content', '').strip()
            memory_type = Memories.normalize_memory_type(operation.get('type'))
            path = operation.get('path')
            if deduplicate_adds:
                duplicate = next(
                    (
                        memory
                        for memory in working_memories
                        if memory.content == content and memory.type == memory_type and memory.path == path
                    ),
                    None,
                )
                if duplicate:
                    prepared.append({'operation': operation, 'duplicate': duplicate})
                    continue

            memory = _memory_candidate(
                user_id=user.id,
                content=content,
                memory_type=memory_type,
                path=path,
                meta=operation.get('meta'),
            )
            vector = await request.app.state.EMBEDDING_FUNCTION(
                memory_vector_text(memory.content, memory.path),
                prefix=RAG_EMBEDDING_CONTENT_PREFIX,
                user=user,
            )
            previous_memories = list(working_memories)
            working_memories.append(memory)
            prepared.append(
                {
                    'operation': operation,
                    'memory': memory,
                    'new_vector': vector,
                    'previous_memories': previous_memories,
                    'vector_memories': list(working_memories),
                }
            )
            continue

        memory = next((item for item in working_memories if item.id == operation.get('id')), None)
        if memory is None:
            raise ValueError(f'Memory not found: {operation.get("id")}')
        if action in {'replace', 'move'}:
            content = operation.get('content', memory.content).strip() if action == 'replace' else memory.content
            memory_type = (
                Memories.normalize_memory_type(operation.get('type'))
                if operation.get('type') is not None
                else memory.type
            )
            path = operation.get('path') if 'path' in operation else memory.path
            meta = memory.meta
            if operation.get('meta') is not None:
                meta = {**(memory.meta or {}), **operation['meta']}
            candidate = _memory_candidate(
                user_id=user.id,
                content=content,
                memory_type=memory_type,
                path=path,
                meta=meta,
                existing=memory,
            )
            prepared_item = {'operation': operation, 'old_memory': memory, 'memory': candidate}
            old_text = memory_vector_text(memory.content, memory.path)
            new_text = memory_vector_text(candidate.content, candidate.path)
            if old_text != new_text:
                prepared_item['old_vector'] = await request.app.state.EMBEDDING_FUNCTION(
                    old_text,
                    prefix=RAG_EMBEDDING_CONTENT_PREFIX,
                    user=user,
                )
                prepared_item['new_vector'] = await request.app.state.EMBEDDING_FUNCTION(
                    new_text,
                    prefix=RAG_EMBEDDING_CONTENT_PREFIX,
                    user=user,
                )
            prepared_item['previous_memories'] = list(working_memories)
            working_memories = [candidate if item.id == candidate.id else item for item in working_memories]
            prepared_item['vector_memories'] = list(working_memories)
            prepared.append(prepared_item)
            continue

        if action == 'remove':
            old_vector = await request.app.state.EMBEDDING_FUNCTION(
                memory_vector_text(memory.content, memory.path),
                prefix=RAG_EMBEDDING_CONTENT_PREFIX,
                user=user,
            )
            previous_memories = list(working_memories)
            working_memories = [item for item in working_memories if item.id != memory.id]
            prepared.append(
                {
                    'operation': operation,
                    'old_memory': memory,
                    'old_vector': old_vector,
                    'previous_memories': previous_memories,
                    'vector_memories': list(working_memories),
                }
            )
            continue

        raise ValueError(f'Unsupported memory operation: {action}')

    return prepared


async def _best_effort_delete_vector(user_id: str, memory_id: str) -> None:
    try:
        await ASYNC_VECTOR_DB_CLIENT.delete(collection_name=f'user-memory-{user_id}', ids=[memory_id])
    except Exception:
        log.exception('Unable to compensate vector delete for memory %s', memory_id)


async def _best_effort_restore_vector(
    request: Request,
    user,
    memory: MemoryModel,
    vector: list[float | int] | None,
    memories: list[MemoryModel],
) -> None:
    if vector is None:
        return
    try:
        await upsert_memory_vectors_or_reindex(
            request,
            user,
            [_memory_vector_item(memory, vector)],
            memories=memories,
        )
    except Exception:
        log.exception('Unable to compensate vector change for memory %s', memory.id)


async def _apply_prepared_memory_operation(
    request: Request,
    user,
    prepared: dict,
) -> tuple[dict, Callable[[], Awaitable[None]] | None]:
    operation = prepared['operation']
    action = operation.get('action')
    duplicate = prepared.get('duplicate')
    if duplicate:
        return {
            'action': action,
            'status': 'skipped',
            'memory': duplicate,
            'reason': 'duplicate',
        }, None

    if action == 'add':
        memory = prepared['memory']
        item = _memory_vector_item(memory, prepared['new_vector'])
        try:
            await upsert_memory_vectors_or_reindex(
                request,
                user,
                [item],
                memories=prepared['vector_memories'],
            )
        except Exception:
            await _best_effort_delete_vector(user.id, memory.id)
            raise
        try:
            saved = await Memories.insert_new_memory(
                user.id,
                memory.content,
                memory_type=memory.type,
                path=memory.path,
                meta=memory.meta,
                memory_id=memory.id,
                created_at=memory.created_at,
                updated_at=memory.updated_at,
            )
            if saved is None:
                raise RuntimeError(f'Unable to create memory: {memory.id}')
        except Exception:
            await _best_effort_delete_vector(user.id, memory.id)
            raise

        async def undo_add() -> None:
            deleted = await Memories.delete_memory_by_id_and_user_id(memory.id, user.id)
            if deleted:
                await ASYNC_VECTOR_DB_CLIENT.delete(collection_name=f'user-memory-{user.id}', ids=[memory.id])

        return {'action': action, 'status': 'created', 'memory': saved}, undo_add

    old_memory = prepared['old_memory']
    if action in {'replace', 'move'}:
        memory = prepared['memory']
        new_vector = prepared.get('new_vector')
        old_vector = prepared.get('old_vector')
        if new_vector is not None:
            try:
                await upsert_memory_vectors_or_reindex(
                    request,
                    user,
                    [_memory_vector_item(memory, new_vector)],
                    memories=prepared['vector_memories'],
                )
            except Exception:
                await _best_effort_restore_vector(
                    request,
                    user,
                    old_memory,
                    old_vector,
                    prepared['previous_memories'],
                )
                raise
        try:
            updated = await Memories.update_memory_by_id_and_user_id(
                memory.id,
                user.id,
                memory.content,
                memory_type=memory.type,
                path=memory.path,
                update_path=True,
                meta=operation.get('meta'),
            )
            if updated is None:
                raise ValueError(f'Memory not found: {memory.id}')
        except Exception:
            await _best_effort_restore_vector(
                request,
                user,
                old_memory,
                old_vector,
                prepared['previous_memories'],
            )
            raise

        async def undo_update() -> None:
            await _best_effort_restore_vector(
                request,
                user,
                old_memory,
                old_vector,
                prepared['previous_memories'],
            )
            await Memories.restore_memory(old_memory)

        return {'action': action, 'status': 'updated', 'memory': updated}, undo_update

    if action == 'remove':
        old_vector = prepared['old_vector']
        try:
            await ASYNC_VECTOR_DB_CLIENT.delete(collection_name=f'user-memory-{user.id}', ids=[old_memory.id])
        except Exception:
            await _best_effort_restore_vector(
                request,
                user,
                old_memory,
                old_vector,
                prepared['previous_memories'],
            )
            raise
        try:
            deleted = await Memories.delete_memory_by_id_and_user_id(old_memory.id, user.id)
            if not deleted:
                raise ValueError(f'Memory not found: {old_memory.id}')
        except Exception:
            await _best_effort_restore_vector(
                request,
                user,
                old_memory,
                old_vector,
                prepared['previous_memories'],
            )
            raise

        async def undo_remove() -> None:
            await Memories.restore_memory(old_memory)
            await _best_effort_restore_vector(
                request,
                user,
                old_memory,
                old_vector,
                prepared['previous_memories'],
            )

        return {'action': action, 'status': 'deleted', 'id': old_memory.id}, undo_remove

    raise ValueError(f'Unsupported memory operation: {action}')


async def _apply_prepared_memory_operations(request: Request, user, prepared_operations: list[dict]) -> list[dict]:
    results = []
    undo_operations = []
    try:
        for prepared in prepared_operations:
            result, undo = await _apply_prepared_memory_operation(request, user, prepared)
            results.append(result)
            if undo is not None:
                undo_operations.append(undo)
    except Exception:
        for undo in reversed(undo_operations):
            try:
                await undo()
            except Exception:
                log.exception('Unable to roll back a memory operation')
        raise
    return results


@router.post('/add', response_model=MemoryModel | None)
async def add_memory(
    request: Request,
    form_data: AddMemoryForm,
    user=Depends(get_verified_user),
):
    """Persist a new memory and embed it into the user's vector collection.

    Does NOT use ``Depends(get_async_session)`` — database operations manage their
    own short-lived sessions so a connection is not held during the external
    embedding API call (``EMBEDDING_FUNCTION``), which can take 1-5+ seconds.
    """
    await check_memories_permission(user)

    content = clean_memory_content(form_data.content)
    path = clean_memory_path(form_data.path)
    prepared = await _prepare_memory_operations(
        request,
        user,
        [
            {
                'action': 'add',
                'content': content,
                'type': form_data.type,
                'path': path,
                'meta': {'created_by': 'manual'},
            }
        ],
        deduplicate_adds=False,
    )
    results = await _apply_prepared_memory_operations(request, user, prepared)
    memory = results[0]['memory']

    await publish_event(
        request,
        EVENTS.MEMORY_CREATED,
        actor=user,
        subject_id=memory.id,
        data={'content_preview': memory.content[:300], 'type': memory.type, 'path': memory.path},
    )
    return memory


@router.post('/update', response_model=list[dict])
async def update_memories(
    request: Request,
    form_data: UpdateMemoriesForm,
    user=Depends(get_verified_user),
):
    await check_memories_permission(user)

    operations = validate_memory_operations(form_data)
    metadata = getattr(request.state, 'metadata', {}) or {}
    source = form_data.source or 'tool'
    for operation in operations:
        if operation.get('action') in {'add', 'replace', 'move'}:
            operation['meta'] = {
                'created_by': source,
                'chat_id': metadata.get('chat_id'),
                'message_id': metadata.get('message_id'),
                'model': metadata.get('model'),
            }

    try:
        prepared = await _prepare_memory_operations(request, user, operations)
        results = await _apply_prepared_memory_operations(request, user, prepared)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    response = []
    for result in results:
        memory = result.get('memory')
        if isinstance(memory, MemoryModel):
            result = {**result, 'memory': memory.model_dump()}
        response.append(result)

    for result in response:
        status_value = result.get('status')
        memory = result.get('memory') or {}
        memory_id = memory.get('id') or result.get('id')

        if status_value == 'created':
            event = EVENTS.MEMORY_CREATED
        elif status_value == 'updated':
            event = EVENTS.MEMORY_UPDATED
        elif status_value == 'deleted':
            event = EVENTS.MEMORY_DELETED
        else:
            continue

        await publish_event(
            request,
            event,
            actor=user,
            subject_id=memory_id,
            data={
                'content_preview': (memory.get('content') or '')[:300],
                'type': memory.get('type'),
                'path': memory.get('path'),
                'operation': result.get('action'),
            },
        )

    return response


############################
# QueryMemory
############################


class QueryMemoryForm(BaseModel):
    content: str
    k: int | None = 1


@router.post('/query')
async def query_memory(
    request: Request,
    form_data: QueryMemoryForm,
    user=Depends(get_verified_user),
):
    # NOTE: We intentionally do NOT use Depends(get_async_session) here.
    # Database operations (get_memories_by_user_id) manage their own short-lived sessions.
    # This prevents holding a connection during EMBEDDING_FUNCTION()
    # which makes external embedding API calls (1-5+ seconds).
    await check_memories_permission(user)

    memories = await Memories.get_memories_by_user_id(user.id)
    if not memories:
        raise HTTPException(status_code=404, detail='No memories found for user')

    vector = await request.app.state.EMBEDDING_FUNCTION(form_data.content, prefix=RAG_EMBEDDING_QUERY_PREFIX, user=user)

    results = await ASYNC_VECTOR_DB_CLIENT.search(
        collection_name=f'user-memory-{user.id}',
        vectors=[vector],
        limit=form_data.k,
    )

    # Filter results by relevance threshold to avoid returning unrelated
    # memories.  Vector similarity search always returns the top-K nearest
    # neighbours even when they are completely irrelevant; applying the
    # same RELEVANCE_THRESHOLD used by RAG ensures only genuinely matching
    # memories are surfaced (distances are normalised to 0→1, higher is
    # better).
    relevance_threshold = await Config.get('rag.relevance_threshold', 0.0)
    if results and relevance_threshold > 0.0 and results.distances and results.distances[0]:
        filtered_ids = []
        filtered_docs = []
        filtered_metas = []
        filtered_dists = []

        for idx, score in enumerate(results.distances[0]):
            if score >= relevance_threshold:
                if results.ids and results.ids[0]:
                    filtered_ids.append(results.ids[0][idx])
                if results.documents and results.documents[0]:
                    filtered_docs.append(results.documents[0][idx])
                if results.metadatas and results.metadatas[0]:
                    filtered_metas.append(results.metadatas[0][idx])
                filtered_dists.append(score)

        results = SearchResult(
            ids=[filtered_ids] if filtered_ids else [[]],
            documents=[filtered_docs] if filtered_docs else [[]],
            metadatas=[filtered_metas] if filtered_metas else [[]],
            distances=[filtered_dists] if filtered_dists else [[]],
        )

    return _authoritative_memory_results(results, memories) if results is not None else None


@router.post('/search', response_model=list[MemoryModel])
async def search_memories(
    form_data: SearchMemoriesForm,
    user=Depends(get_verified_user),
):
    await check_memories_permission(user)

    memories = await Memories.get_memories_by_user_id(user.id)
    return search_memory_rows(
        memories,
        query=form_data.query,
        path=form_data.path,
        memory_id=form_data.memory_id,
        memory_type=form_data.type,
        limit=form_data.limit,
    )


@router.post('/paths')
async def list_memory_paths(
    form_data: ListMemoryPathsForm,
    user=Depends(get_verified_user),
):
    await check_memories_permission(user)

    memories = await Memories.get_memories_by_user_id(user.id)
    return list_memory_path_groups(
        memories,
        query=form_data.query or '',
        memory_type=form_data.type,
        limit=form_data.limit,
    )


@router.post('/path')
async def read_memory_path(
    form_data: ReadMemoryPathForm,
    user=Depends(get_verified_user),
):
    await check_memories_permission(user)

    memories = await Memories.get_memories_by_user_id(user.id)
    result = read_memory_path_rows(
        memories,
        path=form_data.path,
        memory_type=form_data.type,
        include_children=form_data.include_children,
        limit=form_data.limit,
    )
    return {
        **result,
        'memories': [memory.model_dump() for memory in result['memories']],
    }


############################
# ReindexMemoryVectorDB
############################
@router.post('/reindex')
async def reindex_memories_from_vector_db(
    request: Request,
    user=Depends(get_admin_user),
):
    memories = await Memories.get_memories()
    if memories is None:
        raise RuntimeError('Unable to load memories for reindex')
    memories_by_user_id = {}
    for memory in memories:
        memories_by_user_id.setdefault(memory.user_id, []).append(memory)

    users_result = await Users.get_users()
    users = users_result.get('users', []) if users_result else []
    total_memories = 0
    failed_users = []

    for memory_user in users:
        try:
            total_memories += await reindex_memory_vectors_for_user(
                request,
                memory_user.id,
                memories=memories_by_user_id.get(memory_user.id, []),
                user=memory_user,
            )
        except Exception:
            failed_users.append(memory_user.id)
            log.exception('Unable to reindex memory vectors for user %s', memory_user.id)

    await publish_event(
        request,
        EVENTS.MEMORY_RESET,
        actor=user,
        subject_id='all',
        subject_type='user',
        data={
            'count': total_memories,
            'user_count': len(users),
            'failed_user_count': len(failed_users),
            'reindex': True,
        },
    )
    return {
        'status': not failed_users,
        'total_users': len(users),
        'total_memories': total_memories,
        'failed_users': failed_users,
    }


@router.post('/reset', response_model=bool)
async def reset_memory_from_vector_db(
    request: Request,
    user=Depends(get_verified_user),
):
    """Reset user's memory vector embeddings.

    CRITICAL: We intentionally do NOT use Depends(get_async_session) here.
    This endpoint generates embeddings for ALL user memories in parallel using
    asyncio.gather(). A user with 100 memories would trigger 100 embedding API
    calls simultaneously. With a session held, this could block a connection
    for MINUTES, completely exhausting the connection pool.
    """
    await check_memories_permission(user)

    count = await reindex_memory_vectors_for_user(request, user.id, user=user)

    await publish_event(
        request,
        EVENTS.MEMORY_RESET,
        actor=user,
        subject_id=user.id,
        subject_type='user',
        data={'count': count, 'reindex': True},
    )
    return True


############################
# DeleteMemoriesByUserId
############################


@router.delete('/delete/user', response_model=bool)
async def delete_memory_by_user_id(
    request: Request,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    await check_memories_permission(user)

    memories = await Memories.get_memories_by_user_id(user.id, db=db)
    if memories is None:
        return False
    try:
        await ASYNC_VECTOR_DB_CLIENT.delete_collection(f'user-memory-{user.id}')
    except Exception as e:
        log.error('Unable to delete memory vectors for user %s: %s', user.id, e)
        return False

    try:
        result = await Memories.delete_memories_by_user_id(user.id, db=db)
    except Exception:
        if memories:
            try:
                await reindex_memory_vectors_for_user(request, user.id, memories=memories, user=user)
            except Exception:
                log.exception('Unable to restore memory vectors after database deletion failure for user %s', user.id)
        raise

    if result:
        await publish_event(
            request,
            EVENTS.MEMORY_DELETED,
            actor=user,
            subject_id=user.id,
            subject_type='user',
        )
        return True

    if memories:
        try:
            await reindex_memory_vectors_for_user(request, user.id, memories=memories, user=user)
        except Exception:
            log.exception('Unable to restore memory vectors after database deletion failure for user %s', user.id)

    return False


############################
# UpdateMemoryById
############################


@router.post('/{memory_id}/update', response_model=MemoryModel | None)
async def update_memory_by_id(
    memory_id: str,
    request: Request,
    form_data: MemoryUpdateModel,
    user=Depends(get_verified_user),
):
    # NOTE: We intentionally do NOT use Depends(get_async_session) here.
    # Database operations (update_memory_by_id_and_user_id) manage their own
    # short-lived sessions. This prevents holding a connection during
    # EMBEDDING_FUNCTION() which makes external API calls (1-5+ seconds).
    await check_memories_permission(user)

    content = clean_memory_content(form_data.content) if form_data.content is not None else None
    path = clean_memory_path(form_data.path)
    if content is None and form_data.type is None and form_data.path is None:
        raise HTTPException(status_code=400, detail='No memory update provided')
    operation = {
        'action': 'replace',
        'id': memory_id,
        'meta': {'created_by': 'manual'},
    }
    if content is not None:
        operation['content'] = content
    if form_data.type is not None:
        operation['type'] = form_data.type
    if form_data.path is not None:
        operation['path'] = path

    try:
        prepared = await _prepare_memory_operations(request, user, [operation])
        results = await _apply_prepared_memory_operations(request, user, prepared)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=ERROR_MESSAGES.NOT_FOUND) from e

    memory = results[0].get('memory')
    if not isinstance(memory, MemoryModel):
        raise HTTPException(status_code=404, detail=ERROR_MESSAGES.NOT_FOUND)

    await publish_event(
        request,
        EVENTS.MEMORY_UPDATED,
        actor=user,
        subject_id=memory.id,
        data={'content_preview': memory.content[:300], 'type': memory.type, 'path': memory.path},
    )
    return memory


############################
# DeleteMemoryById
############################


@router.delete('/{memory_id}', response_model=bool)
async def delete_memory_by_id(
    memory_id: str,
    request: Request,
    user=Depends(get_verified_user),
):
    await check_memories_permission(user)

    try:
        prepared = await _prepare_memory_operations(
            request,
            user,
            [{'action': 'remove', 'id': memory_id}],
        )
        results = await _apply_prepared_memory_operations(request, user, prepared)
    except ValueError:
        return False

    if results and results[0].get('status') == 'deleted':
        await publish_event(
            request,
            EVENTS.MEMORY_DELETED,
            actor=user,
            subject_id=memory_id,
        )
        return True

    return False
