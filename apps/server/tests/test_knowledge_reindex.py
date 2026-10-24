from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from local_llm.internal.db import get_async_session
from local_llm.routers import knowledge as knowledge_router
from local_llm.routers import retrieval as retrieval_router
from local_llm.utils.auth import get_verified_user

ADMIN = SimpleNamespace(id='admin', role='admin')


class FakeVectorDB:
    def __init__(self, rows: dict[str, list[dict]]):
        self.rows = rows

    async def has_collection(self, collection_name: str) -> bool:
        return collection_name in self.rows

    async def query(self, collection_name: str, filter: dict) -> SimpleNamespace | None:
        if collection_name not in self.rows:
            return None

        matches = [
            row for row in self.rows[collection_name] if all(row['metadata'].get(k) == v for k, v in filter.items())
        ]
        return SimpleNamespace(ids=[[row['id'] for row in matches]])

    async def delete(self, collection_name: str, ids: list[str]) -> None:
        self.rows[collection_name] = [row for row in self.rows[collection_name] if row['id'] not in ids]


def vector(vector_id: str, file_id: str) -> dict:
    return {'id': vector_id, 'metadata': {'file_id': file_id}}


def configure_reindex(monkeypatch, files: list[SimpleNamespace], vectors: FakeVectorDB, process_file):
    async def get_knowledge_bases(db=None):
        return [SimpleNamespace(id='kb1')]

    async def get_files_by_id(knowledge_base_id: str, db=None):
        assert knowledge_base_id == 'kb1'
        return files

    async def publish_event(*_args, **_kwargs):
        return None

    async def get_retrieval_config():
        return SimpleNamespace(BYPASS_EMBEDDING_AND_RETRIEVAL=False)

    monkeypatch.setattr(knowledge_router.Knowledges, 'get_knowledge_bases', get_knowledge_bases)
    monkeypatch.setattr(knowledge_router.Knowledges, 'get_files_by_id', get_files_by_id)
    monkeypatch.setattr(knowledge_router, 'get_retrieval_config', get_retrieval_config)
    monkeypatch.setattr(knowledge_router, 'ASYNC_VECTOR_DB_CLIENT', vectors)
    monkeypatch.setattr(knowledge_router, 'process_file', process_file)
    monkeypatch.setattr(knowledge_router, 'publish_event', publish_event)

    app = FastAPI()
    app.include_router(knowledge_router.router, prefix='/knowledge')
    app.dependency_overrides[get_verified_user] = lambda: ADMIN
    app.dependency_overrides[get_async_session] = lambda: None
    return TestClient(app)


@pytest.mark.asyncio
async def test_reindex_process_file_uses_sql_content_and_refreshes_both_collections(monkeypatch):
    file = SimpleNamespace(
        id='f1',
        user_id='u1',
        filename='one.txt',
        meta={},
        path=None,
        data={'content': 'fresh SQL content'},
    )
    saved = []
    config = SimpleNamespace(BYPASS_EMBEDDING_AND_RETRIEVAL=False)

    async def get_file_by_id(_file_id, db=None):
        return file

    async def validate_collection_access(*_args, **_kwargs):
        return None

    async def get_retrieval_config():
        return config

    async def query_vectors(*_args, **_kwargs):
        return SimpleNamespace(ids=[['stale-id']], documents=[['stale vector content']], metadatas=[[{}]])

    async def run_in_threadpool(_function, *_args, **kwargs):
        saved.append(kwargs)
        return True

    async def no_op(*_args, **_kwargs):
        return None

    class SessionContext:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(retrieval_router.Files, 'get_file_by_id', get_file_by_id)
    monkeypatch.setattr(retrieval_router.Files, 'update_file_data_by_id', no_op)
    monkeypatch.setattr(retrieval_router.Files, 'update_file_metadata_by_id', no_op)
    monkeypatch.setattr(retrieval_router.Files, 'update_file_hash_by_id', no_op)
    monkeypatch.setattr(retrieval_router, '_validate_collection_access', validate_collection_access)
    monkeypatch.setattr(retrieval_router, 'get_retrieval_config', get_retrieval_config)
    monkeypatch.setattr(
        retrieval_router,
        'ASYNC_VECTOR_DB_CLIENT',
        SimpleNamespace(query=query_vectors),
    )
    monkeypatch.setattr(retrieval_router, 'run_in_threadpool', run_in_threadpool)
    monkeypatch.setattr(retrieval_router, 'get_async_db', SessionContext)
    monkeypatch.setattr(retrieval_router, 'publish_event', no_op)

    class DatabaseSession:
        async def commit(self):
            return None

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    result = await retrieval_router.process_file(
        request,
        retrieval_router.ReindexProcessFileForm(file_id='f1', collection_name='kb1'),
        user=ADMIN,
        db=DatabaseSession(),
    )

    assert result['status'] is True
    assert [call['collection_name'] for call in saved] == ['kb1', 'file-f1']
    assert all(call['docs'][0].page_content == 'fresh SQL content' for call in saved)


@pytest.mark.parametrize('failure', ['embedding', 'write'])
def test_failed_file_reindex_preserves_existing_vectors_and_returns_error(monkeypatch, failure):
    files = [SimpleNamespace(id='f1', filename='one.txt')]
    vectors = FakeVectorDB({'kb1': [vector('kb-old', 'f1')], 'file-f1': [vector('file-old', 'f1')]})
    initial_rows = {name: list(rows) for name, rows in vectors.rows.items()}

    async def process_file(_request, form_data, user, db):
        assert isinstance(form_data, knowledge_router.ReindexProcessFileForm)
        assert user is ADMIN
        assert db is None
        if failure == 'write':
            vectors.rows['kb1'].append(vector('kb-partial', 'f1'))
        raise HTTPException(status_code=400, detail=f'{failure} failed')

    client = configure_reindex(monkeypatch, files, vectors, process_file)
    response = client.post('/knowledge/reindex')

    assert response.status_code == 500
    assert response.json()['detail'] == 'Failed to reindex 1 of 1 files'
    assert vectors.rows['file-f1'] == initial_rows['file-f1']
    assert vector('kb-old', 'f1') in vectors.rows['kb1']
    assert vector('kb-partial', 'f1') not in vectors.rows['kb1']
    if failure == 'embedding':
        assert vectors.rows == initial_rows


def test_success_replaces_old_vectors_and_repeated_reindex_does_not_duplicate(monkeypatch):
    files = [SimpleNamespace(id='f1', filename='one.txt')]
    vectors = FakeVectorDB({'kb1': [vector('kb-old', 'f1')], 'file-f1': [vector('file-old', 'f1')]})
    calls = 0

    async def process_file(_request, form_data, user, db):
        nonlocal calls
        calls += 1
        assert user is ADMIN
        assert isinstance(form_data, knowledge_router.ReindexProcessFileForm)
        for collection_name in (form_data.collection_name, f'file-{form_data.file_id}'):
            vectors.rows.setdefault(collection_name, []).append(vector(f'{collection_name}-{calls}', form_data.file_id))
        return {'status': True}

    client = configure_reindex(monkeypatch, files, vectors, process_file)

    assert client.post('/knowledge/reindex').status_code == 200
    assert [row['id'] for row in vectors.rows['kb1']] == ['kb1-1']
    assert [row['id'] for row in vectors.rows['file-f1']] == ['file-f1-1']

    assert client.post('/knowledge/reindex').status_code == 200
    assert [row['id'] for row in vectors.rows['kb1']] == ['kb1-2']
    assert [row['id'] for row in vectors.rows['file-f1']] == ['file-f1-2']


def test_mixed_reindex_replaces_success_and_preserves_failed_file(monkeypatch):
    files = [
        SimpleNamespace(id='f1', filename='one.txt'),
        SimpleNamespace(id='f2', filename='two.txt'),
    ]
    vectors = FakeVectorDB(
        {
            'kb1': [vector('kb-old-1', 'f1'), vector('kb-old-2', 'f2')],
            'file-f1': [vector('file-old-1', 'f1')],
            'file-f2': [vector('file-old-2', 'f2')],
        }
    )

    async def process_file(_request, form_data, user, db):
        assert isinstance(form_data, knowledge_router.ReindexProcessFileForm)
        assert user is ADMIN
        if form_data.file_id == 'f2':
            raise HTTPException(status_code=400, detail='embedding failed')
        for collection_name in (form_data.collection_name, f'file-{form_data.file_id}'):
            vectors.rows[collection_name].append(vector(f'{collection_name}-new', form_data.file_id))
        return {'status': True}

    client = configure_reindex(monkeypatch, files, vectors, process_file)
    response = client.post('/knowledge/reindex')

    assert response.status_code == 500
    assert {row['id'] for row in vectors.rows['kb1']} == {'kb1-new', 'kb-old-2'}
    assert [row['id'] for row in vectors.rows['file-f1']] == ['file-f1-new']
    assert [row['id'] for row in vectors.rows['file-f2']] == ['file-old-2']


def test_reindex_is_rejected_when_embedding_and_retrieval_are_bypassed(monkeypatch):
    files = [SimpleNamespace(id='f1', filename='one.txt')]
    vectors = FakeVectorDB({'kb1': [vector('kb-old', 'f1')], 'file-f1': [vector('file-old', 'f1')]})

    async def process_file(*_args, **_kwargs):
        pytest.fail('process_file must not run while embeddings are bypassed')

    client = configure_reindex(monkeypatch, files, vectors, process_file)

    async def get_retrieval_config():
        return SimpleNamespace(BYPASS_EMBEDDING_AND_RETRIEVAL=True)

    monkeypatch.setattr(knowledge_router, 'get_retrieval_config', get_retrieval_config)
    response = client.post('/knowledge/reindex')

    assert response.status_code == 503
    assert vectors.rows == {'kb1': [vector('kb-old', 'f1')], 'file-f1': [vector('file-old', 'f1')]}


@pytest.mark.asyncio
async def test_concurrent_reindex_requests_serialize_replacement(monkeypatch):
    files = [SimpleNamespace(id='f1', filename='one.txt')]
    vectors = FakeVectorDB({'kb1': [vector('kb-old', 'f1')], 'file-f1': [vector('file-old', 'f1')]})
    active_calls = 0
    max_active_calls = 0
    calls = 0

    async def process_file(_request, form_data, user, db):
        nonlocal active_calls, max_active_calls, calls
        assert user is ADMIN
        assert db is None
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)
        await asyncio.sleep(0)
        calls += 1
        for collection_name in (form_data.collection_name, f'file-{form_data.file_id}'):
            vectors.rows[collection_name].append(vector(f'{collection_name}-{calls}', form_data.file_id))
        active_calls -= 1
        return {'status': True}

    client = configure_reindex(monkeypatch, files, vectors, process_file)
    request = SimpleNamespace(app=client.app)

    results = await asyncio.gather(
        knowledge_router.reindex_knowledge_files(request, ADMIN, None),
        knowledge_router.reindex_knowledge_files(request, ADMIN, None),
    )

    assert results == [True, True]
    assert calls == 2
    assert max_active_calls == 1
    assert len(vectors.rows['kb1']) == 1
    assert len(vectors.rows['file-f1']) == 1
