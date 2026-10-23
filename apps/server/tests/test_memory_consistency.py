import asyncio
from types import SimpleNamespace

import pytest

from local_llm.models.memories import MemoryModel
from local_llm.retrieval.vector.main import SearchResult
from local_llm.routers import memories as memories_router
from local_llm.utils import memory as memory_utils
from local_llm.utils.memory import validate_memory_operations

USER = SimpleNamespace(id='u1', role='user')


def _memory(memory_id='m1', content='old content', path='notes', memory_type='context'):
    return MemoryModel(
        id=memory_id,
        user_id=USER.id,
        type=memory_type,
        path=path,
        content=content,
        meta={'created_by': 'test'},
        created_at=1,
        updated_at=1,
    )


class FakeMemoryStore:
    def __init__(self, rows=None):
        self.rows = {row.id: row for row in rows or []}
        self.insert_calls = 0
        self.update_calls = 0
        self.delete_calls = 0

    async def get_memories_by_user_id(self, user_id, db=None):
        return [row for row in self.rows.values() if row.user_id == user_id]

    async def insert_new_memory(
        self,
        user_id,
        content,
        memory_type=None,
        path=None,
        meta=None,
        *,
        memory_id=None,
        created_at=None,
        updated_at=None,
        db=None,
    ):
        self.insert_calls += 1
        row = MemoryModel(
            id=memory_id or f'm{len(self.rows) + 1}',
            user_id=user_id,
            type=memory_type or 'context',
            path=path,
            content=content,
            meta=meta,
            created_at=created_at or 1,
            updated_at=updated_at or 1,
        )
        self.rows[row.id] = row
        return row

    async def update_memory_by_id_and_user_id(
        self,
        memory_id,
        user_id,
        content,
        memory_type=None,
        path=None,
        update_path=False,
        meta=None,
        db=None,
    ):
        self.update_calls += 1
        row = self.rows.get(memory_id)
        if row is None or row.user_id != user_id:
            return None
        updated = row.model_copy(
            update={
                'content': content if content is not None else row.content,
                'type': memory_type or row.type,
                'path': path if update_path else row.path,
                'meta': {**(row.meta or {}), **(meta or {})},
                'updated_at': row.updated_at + 1,
            }
        )
        self.rows[memory_id] = updated
        return updated

    async def delete_memory_by_id_and_user_id(self, memory_id, user_id, db=None):
        self.delete_calls += 1
        row = self.rows.get(memory_id)
        if row is None or row.user_id != user_id:
            return False
        del self.rows[memory_id]
        return True

    async def restore_memory(self, row, db=None):
        self.rows[row.id] = row
        return row


class FakeVectors:
    def __init__(self):
        self.items = {}
        self.upsert_calls = 0
        self.deleted_collections = []
        self.fail_upsert_call = None
        self.fail_upsert_calls = set()
        self.fail_upsert_message = 'vector upsert failed'
        self.partial_fail_upsert_call = None
        self.fail_delete = False
        self.partial_fail_delete = False
        self.fail_delete_collection = False

    async def upsert(self, collection_name, items):
        self.upsert_calls += 1
        if self.fail_upsert_call == self.upsert_calls or self.upsert_calls in self.fail_upsert_calls:
            raise RuntimeError(self.fail_upsert_message)
        for item in items:
            self.items[item['id']] = item
        if self.partial_fail_upsert_call == self.upsert_calls:
            raise RuntimeError(self.fail_upsert_message)

    async def delete(self, collection_name, ids=None, filter=None):
        if self.fail_delete:
            raise RuntimeError('vector delete failed')
        for memory_id in ids or []:
            self.items.pop(memory_id, None)
        if self.partial_fail_delete:
            raise RuntimeError('vector delete failed after mutation')

    async def get(self, collection_name):
        return SimpleNamespace(ids=[list(self.items)], documents=[[]], metadatas=[[]])

    async def delete_collection(self, collection_name):
        if self.fail_delete_collection:
            raise RuntimeError('vector collection delete failed')
        self.deleted_collections.append(collection_name)
        self.items.clear()


def _request(embedding):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(EMBEDDING_FUNCTION=embedding)),
        state=SimpleNamespace(metadata={}),
    )


def _install(monkeypatch, rows=None, vectors=None):
    store = FakeMemoryStore(rows)
    vectors = vectors or FakeVectors()

    async def allow(*args, **kwargs):
        return None

    async def publish(*args, **kwargs):
        return None

    monkeypatch.setattr(memories_router.Memories, 'get_memories_by_user_id', store.get_memories_by_user_id)
    monkeypatch.setattr(memories_router.Memories, 'insert_new_memory', store.insert_new_memory)
    monkeypatch.setattr(
        memories_router.Memories, 'update_memory_by_id_and_user_id', store.update_memory_by_id_and_user_id
    )
    monkeypatch.setattr(
        memories_router.Memories, 'delete_memory_by_id_and_user_id', store.delete_memory_by_id_and_user_id
    )
    monkeypatch.setattr(memories_router.Memories, 'restore_memory', store.restore_memory)
    monkeypatch.setattr(memories_router, 'ASYNC_VECTOR_DB_CLIENT', vectors)
    monkeypatch.setattr(memories_router, 'check_memories_permission', allow)
    monkeypatch.setattr(memories_router, 'publish_event', publish)
    return store, vectors


async def _embedding(text, prefix=None, user=None):
    return [0.1, 0.2]


def test_add_embedding_failure_does_not_persist(monkeypatch):
    store, vectors = _install(monkeypatch)

    async def broken(*args, **kwargs):
        raise RuntimeError('embedding unavailable')

    with pytest.raises(RuntimeError, match='embedding unavailable'):
        asyncio.run(
            memories_router.add_memory(_request(broken), memories_router.AddMemoryForm(content='new fact'), user=USER)
        )

    assert store.rows == {}
    assert vectors.items == {}


def test_add_vector_failure_does_not_persist(monkeypatch):
    vectors = FakeVectors()
    vectors.fail_upsert_call = 1
    store, _ = _install(monkeypatch, vectors=vectors)

    with pytest.raises(RuntimeError, match='vector upsert failed'):
        asyncio.run(
            memories_router.add_memory(
                _request(_embedding), memories_router.AddMemoryForm(content='new fact'), user=USER
            )
        )

    assert store.rows == {}


def test_add_partial_vector_failure_removes_orphan(monkeypatch):
    vectors = FakeVectors()
    vectors.partial_fail_upsert_call = 1
    store, _ = _install(monkeypatch, vectors=vectors)

    with pytest.raises(RuntimeError, match='vector upsert failed'):
        asyncio.run(
            memories_router.add_memory(
                _request(_embedding), memories_router.AddMemoryForm(content='new fact'), user=USER
            )
        )

    assert store.rows == {}
    assert vectors.items == {}


def test_dimension_mismatch_reindex_keeps_existing_memories(monkeypatch):
    existing = _memory(memory_id='existing')
    vectors = FakeVectors()
    vectors.fail_upsert_calls = {1, 2}
    vectors.fail_upsert_message = 'embedding dimension mismatch'
    store, _ = _install(monkeypatch, rows=[existing], vectors=vectors)

    result = asyncio.run(
        memories_router.add_memory(_request(_embedding), memories_router.AddMemoryForm(content='new fact'), user=USER)
    )

    assert result.id in store.rows
    assert set(vectors.items) == {'existing', result.id}
    assert vectors.deleted_collections == ['user-memory-u1']


def test_update_embedding_failure_preserves_database_and_vectors(monkeypatch):
    row = _memory()
    store, vectors = _install(monkeypatch, rows=[row])
    vectors.items[row.id] = {'id': row.id, 'text': 'notes\nold content', 'vector': [0.1, 0.2]}

    async def broken(*args, **kwargs):
        raise RuntimeError('embedding unavailable')

    with pytest.raises(RuntimeError, match='embedding unavailable'):
        asyncio.run(
            memories_router.update_memory_by_id(
                row.id,
                _request(broken),
                memories_router.MemoryUpdateModel(content='new content'),
                user=USER,
            )
        )

    assert store.rows[row.id] == row
    assert vectors.items[row.id]['text'] == 'notes\nold content'
    assert store.update_calls == 0


def test_batch_embedding_failure_has_no_partial_mutation(monkeypatch):
    store, vectors = _install(monkeypatch)
    calls = 0

    async def fails_on_second(text, prefix=None, user=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('embedding unavailable')
        return [0.1, 0.2]

    form = memories_router.UpdateMemoriesForm(
        operations=[
            {'action': 'add', 'content': 'first'},
            {'action': 'add', 'content': 'second'},
        ]
    )
    with pytest.raises(RuntimeError, match='embedding unavailable'):
        asyncio.run(memories_router.update_memories(_request(fails_on_second), form, user=USER))

    assert store.rows == {}
    assert vectors.items == {}
    assert vectors.upsert_calls == 0


def test_batch_vector_failure_rolls_back_completed_operations(monkeypatch):
    vectors = FakeVectors()
    vectors.fail_upsert_call = 2
    store, _ = _install(monkeypatch, vectors=vectors)
    form = memories_router.UpdateMemoriesForm(
        operations=[
            {'action': 'add', 'content': 'first'},
            {'action': 'add', 'content': 'second'},
        ]
    )

    with pytest.raises(RuntimeError, match='vector upsert failed'):
        asyncio.run(memories_router.update_memories(_request(_embedding), form, user=USER))

    assert store.rows == {}
    assert vectors.items == {}


def test_reindex_embedding_failure_keeps_existing_collection(monkeypatch):
    row = _memory()
    _, vectors = _install(monkeypatch, rows=[row])
    vectors.items[row.id] = {'id': row.id, 'text': 'notes\nold content', 'vector': [0.1, 0.2]}

    async def broken(*args, **kwargs):
        raise RuntimeError('embedding unavailable')

    with pytest.raises(RuntimeError, match='embedding unavailable'):
        asyncio.run(
            memories_router.reindex_memory_vectors_for_user(_request(broken), USER.id, memories=[row], user=USER)
        )

    assert vectors.deleted_collections == []
    assert vectors.items[row.id]['text'] == 'notes\nold content'


def test_database_read_failure_does_not_clear_vectors(monkeypatch):
    row = _memory()
    _, vectors = _install(monkeypatch, rows=[row])
    vectors.items[row.id] = {'id': row.id, 'text': 'notes\nold content', 'vector': [0.1, 0.2]}

    async def failed_read(*args, **kwargs):
        return None

    monkeypatch.setattr(memories_router.Memories, 'get_memories_by_user_id', failed_read)

    with pytest.raises(RuntimeError, match='Unable to load memories'):
        asyncio.run(memories_router.reindex_memory_vectors_for_user(_request(_embedding), USER.id, user=USER))

    assert vectors.deleted_collections == []
    assert vectors.items[row.id]['text'] == 'notes\nold content'


def test_vector_context_uses_current_database_row_and_drops_orphans():
    row = SimpleNamespace(id='m1', type='context', path='notes', content='current content')
    results = SearchResult(
        ids=[['m1', 'orphan']],
        documents=[['notes\nstale content', 'orphan content']],
        metadatas=[[{'path': 'notes', 'type': 'context'}, {'type': 'context'}]],
        distances=[[0.9, 0.8]],
    )

    sections = memory_utils._memory_sections([row], 'unrelated', results)

    assert sections == {'user': [], 'neighborhood': [], 'context': ['notes: current content']}


def test_query_results_resolve_current_text_and_metadata():
    row = _memory(content='current content')
    results = SearchResult(
        ids=[['m1', 'orphan']],
        documents=[['notes\nstale content', 'orphan content']],
        metadatas=[[{'path': 'notes'}, {'path': 'orphan'}]],
        distances=[[0.9, 0.8]],
    )

    resolved = memories_router._authoritative_memory_results(results, [row])

    assert resolved.ids == [['m1']]
    assert resolved.documents == [['notes\ncurrent content']]
    assert resolved.metadatas == [[memories_router._memory_metadata(row)]]
    assert resolved.distances == [[0.9]]


def test_update_database_failure_restores_previous_vector(monkeypatch):
    row = _memory()
    _, vectors = _install(monkeypatch, rows=[row])
    vectors.items[row.id] = {'id': row.id, 'text': 'notes\nold content', 'vector': [0.1, 0.2]}

    async def failed_update(*args, **kwargs):
        return None

    monkeypatch.setattr(memories_router.Memories, 'update_memory_by_id_and_user_id', failed_update)

    with pytest.raises(Exception):
        asyncio.run(
            memories_router.update_memory_by_id(
                row.id,
                _request(_embedding),
                memories_router.MemoryUpdateModel(content='new content'),
                user=USER,
            )
        )

    assert vectors.items[row.id]['text'] == 'notes\nold content'


def test_update_partial_vector_failure_restores_previous_vector(monkeypatch):
    row = _memory()
    store, vectors = _install(monkeypatch, rows=[row])
    vectors.items[row.id] = {'id': row.id, 'text': 'notes\nold content', 'vector': [0.1, 0.2]}
    vectors.partial_fail_upsert_call = 1

    with pytest.raises(RuntimeError, match='vector upsert failed'):
        asyncio.run(
            memories_router.update_memory_by_id(
                row.id,
                _request(_embedding),
                memories_router.MemoryUpdateModel(content='new content'),
                user=USER,
            )
        )

    assert store.rows[row.id] == row
    assert vectors.items[row.id]['text'] == 'notes\nold content'


def test_remove_partial_vector_failure_restores_previous_vector(monkeypatch):
    row = _memory()
    store, vectors = _install(monkeypatch, rows=[row])
    vectors.items[row.id] = {'id': row.id, 'text': 'notes\nold content', 'vector': [0.1, 0.2]}
    vectors.partial_fail_delete = True

    with pytest.raises(RuntimeError, match='vector delete failed'):
        asyncio.run(memories_router.delete_memory_by_id(row.id, _request(_embedding), user=USER))

    assert store.rows[row.id] == row
    assert vectors.items[row.id]['text'] == 'notes\nold content'


def test_user_delete_database_failure_restores_vectors(monkeypatch):
    row = _memory()
    store, vectors = _install(monkeypatch, rows=[row])
    vectors.items[row.id] = {'id': row.id, 'text': 'notes\nold content', 'vector': [0.1, 0.2]}

    async def failed_delete(*args, **kwargs):
        return False

    monkeypatch.setattr(memories_router.Memories, 'delete_memories_by_user_id', failed_delete)

    result = asyncio.run(memories_router.delete_memory_by_user_id(_request(_embedding), user=USER, db=None))

    assert result is False
    assert store.rows[row.id] == row
    assert vectors.items[row.id]['text'] == 'notes\nold content'


def test_admin_reindex_continues_after_one_user_fails(monkeypatch):
    first = _memory(memory_id='m1')
    second = _memory(memory_id='m2').model_copy(update={'user_id': 'u2'})
    _, vectors = _install(monkeypatch, rows=[first, second])
    admin = SimpleNamespace(id='admin', role='admin')
    users = [SimpleNamespace(id='u1'), SimpleNamespace(id='u2')]

    async def get_all_memories(db=None):
        return [first, second]

    async def get_users(*args, **kwargs):
        return {'users': users}

    async def embedding(text, prefix=None, user=None):
        if user.id == 'u1':
            raise RuntimeError('embedding unavailable')
        return [0.1, 0.2]

    published = []

    async def publish(*args, **kwargs):
        published.append(kwargs)

    monkeypatch.setattr(memories_router.Memories, 'get_memories', get_all_memories)
    monkeypatch.setattr(memories_router.Users, 'get_users', get_users)
    monkeypatch.setattr(memories_router, 'publish_event', publish)
    request = _request(embedding)

    result = asyncio.run(memories_router.reindex_memories_from_vector_db(request, user=admin))

    assert result == {'status': False, 'total_users': 2, 'total_memories': 1, 'failed_users': ['u1']}
    assert set(vectors.items) == {'m2'}
    assert published and published[0]['data']['failed_user_count'] == 1


def test_replace_without_path_preserves_existing_path():
    form = memories_router.UpdateMemoriesForm(operations=[{'action': 'replace', 'id': 'm1', 'content': 'new content'}])

    operation = validate_memory_operations(form)[0]

    assert 'path' not in operation
