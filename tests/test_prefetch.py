"""CPU planner and consumer-scope contracts; GPU state machine checked separately."""
from contextlib import contextmanager
import pytest
import torch
from ram_vram_finetune.prefetch import plan_prefetch_pool,PrefetchSession
from ram_vram_finetune.config import TrainConfig
from ram_vram_finetune.training.recompute import _borrow_weights

MiB=1024**2

def plan(**kw):
    values=dict(capacity=48*MiB,block_count=28,pool_budget_bytes=512*MiB,max_slots=8,device_budget_bytes=1536*MiB,live_allocated_bytes=400*MiB,reserve_bytes=256*MiB);values.update(kw);return plan_prefetch_pool(**values)

@pytest.mark.parametrize('budget,count',[(100*MiB,2),(200*MiB,4),(400*MiB,8),(512*MiB,8)])
def test_fixed_pool_scale_resolved_from_budget(budget,count):
    result=plan(pool_budget_bytes=budget);assert result['resolved_slots']==count
    assert result['estimated_pool_charge_bytes']<=result['available_pool_budget_bytes']

def test_live_allocated_and_reserve_limit_pool_scale():
    assert plan(device_budget_bytes=800*MiB)['resolved_slots']==2
    with pytest.raises(MemoryError,match='two admitted'):plan(device_budget_bytes=700*MiB)

def test_single_layer_explicitly_resolves_one_slot_without_lookahead():
    result=plan(block_count=1,pool_budget_bytes=50*MiB,max_slots=1)
    assert result['resolved_slots']==1 and result['lookahead_blocks']==0 and result['single_block_resolution']

@pytest.mark.parametrize('kw',[{'pool_budget_bytes':99*MiB},{'max_slots':1},{'device_budget_bytes':655*MiB}])
def test_insufficient_multilayer_budget_refuses_no_silent_fallback(kw):
    with pytest.raises(MemoryError,match='resource_budget'):plan(**kw)

@pytest.mark.parametrize('kw',[{'reserve_bytes':0},{'max_slots':True},{'live_allocated_bytes':-1},{'capacity':0}])
def test_invalid_admission_limits(kw):
    with pytest.raises(ValueError):plan(**kw)

def test_prefetch_cuda_only_checked_before_any_device_allocation():
    with pytest.raises(ValueError,match='requires cuda:0'):PrefetchSession([], 'cpu',host_budget_bytes=1,device_budget_bytes=1)

def test_configuration_execution_policies_do_not_change_checkpoint_identity():
    left=TrainConfig('model','data','out',device='cuda:0',transfer_mode='batched')
    right=TrainConfig('model','data','out2',device='cuda:0',transfer_mode='prefetch',prefetch_max_slots=8,prefetch_pool_budget_bytes=512*MiB)
    assert left.semantics()==right.semantics()

@pytest.mark.parametrize('kw',[{'prefetch_pool_budget_bytes':0},{'prefetch_reserve_bytes':-1},{'prefetch_max_slots':False},{'transfer_mode':'prefetch'},{'device':'cuda:0','transfer_mode':'prefetch','quantization':'q6'}])
def test_configuration_refuses_unsupported_or_invalid_options(kw):
    with pytest.raises(ValueError):TrainConfig('model','data','out',**kw)

def test_borrow_scope_releases_only_after_full_consumer_and_on_exception():
    events=[]
    class Loader:
        @contextmanager
        def borrow(self,phase):
            events.append(('acquire',phase))
            try:yield {'weights':torch.ones(1)}
            finally:events.append(('release',phase))
    with _borrow_weights(Loader(),'backward'):
        events.extend(['recompute','autograd.grad','grad_A.add','grad_B.add'])
    assert events==[('acquire','backward'),'recompute','autograd.grad','grad_A.add','grad_B.add',('release','backward')]
    events.clear()
    with pytest.raises(ValueError):
        with _borrow_weights(Loader(),'forward'):raise ValueError('functional failure')
    assert events==[('acquire','forward'),('release','forward')]

def test_same_slot_layout_changes_keep_shape_dtype_alignment_and_aliasing():
    # Metadata/view construction uses the same function on CPU and CUDA. Test
    # schema changes without allocating any CUDA context or assuming parity.
    from types import SimpleNamespace
    session=object.__new__(PrefetchSession)
    slot={'buffer':torch.empty(4096,dtype=torch.uint8),'templates':{}}
    left=SimpleNamespace(host_weights={'matrix':torch.arange(12,dtype=torch.float32).view(3,4),'norm':torch.arange(4,dtype=torch.bfloat16)})
    right=SimpleNamespace(host_weights={'matrix':torch.arange(10,dtype=torch.bfloat16).view(2,5),'bias':torch.arange(5,dtype=torch.float32)})
    first=session._views(slot,left);second=session._views(slot,right)
    assert len(slot['templates'])==2 and session._views(slot,left) is first
    for block,views in [(left,first),(right,second),(left,first)]:
        for name,source in block.host_weights.items():
            target=views[name];assert target.dtype==source.dtype and target.shape==source.shape
            assert slot['buffer'].data_ptr()<=target.data_ptr()<slot['buffer'].data_ptr()+slot['buffer'].numel()
            assert (target.data_ptr()-slot['buffer'].data_ptr())%256==0
            target.copy_(source);assert torch.equal(target,source)

@pytest.mark.parametrize('failure',['drain','finalize'])
def test_run_finalization_failures_poison_and_release_capability(monkeypatch,failure):
    import threading
    session=object.__new__(PrefetchSession);session.closed=False;session.poisoned=False
    session.device=torch.device('cuda:0');session._run_lock=threading.Lock();session._metadata_lock=threading.Lock();session._owner=None;session._token=None
    def fail():raise RuntimeError('injected '+failure)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *_:fail() if failure=='drain' else None)
    session._finalize_copies=fail if failure=='finalize' else lambda:None
    with pytest.raises(RuntimeError,match='injected'):
        with session.run():pass
    assert session.poisoned and session._token is None and session._owner is None and not session._run_lock.locked()
    with pytest.raises(RuntimeError,match='poisoned'):
        with session.run():pass

def test_constructor_error_cleanup_even_if_drain_also_fails(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import ram_vram_finetune.prefetch as prefetch
    session=object.__new__(PrefetchSession);block=SimpleNamespace(index=0,storage_specs=None,host_weights={'w':torch.ones(8)})
    old_empty=torch.empty
    monkeypatch.setattr(torch,'empty',lambda *args,**kw:old_empty(*args,**(kw|{'device':'cpu'})))
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda *_:0)
    monkeypatch.setattr(torch.cuda,'device',lambda *_:nullcontext())
    monkeypatch.setattr(torch.cuda,'current_stream',lambda *_:object())
    monkeypatch.setattr(torch.cuda,'Stream',lambda **_:SimpleNamespace(wait_stream=lambda _:None))
    def fail_pin(*_):raise ValueError('pin initialization failure')
    def fail_drain(*_):raise RuntimeError('drain initialization failure')
    monkeypatch.setattr(prefetch,'pin_weights',fail_pin);monkeypatch.setattr(torch.cuda,'synchronize',fail_drain)
    with pytest.raises(RuntimeError,match='drain initialization failure') as error:
        session.__init__([block],'cuda:0',host_budget_bytes=1024,device_budget_bytes=64*MiB,pool_budget_bytes=2*MiB,max_slots=1,reserve_bytes=16*MiB)
    assert isinstance(error.value.__context__,ValueError)
    assert session.poisoned and session.closed and not session.slots and not session.pinned_arenas and session.copy_stream is None

def test_close_drain_failure_poisons_keeps_owners_then_can_retry_cleanup(monkeypatch):
    import threading
    from types import SimpleNamespace
    session=object.__new__(PrefetchSession);session.closed=False;session.poisoned=False;session.device=torch.device('cuda:0')
    session._run_lock=threading.Lock();session._metadata_lock=threading.Lock();session.slots=[object()];session.pinned_arenas=[object()]
    session.copy_records=[];session._consumed_owners=[];session.blocks=[SimpleNamespace(host_weights={'w':object()})];session.copy_stream=object()
    def fail(*_):raise RuntimeError('close drain failure')
    monkeypatch.setattr(torch.cuda,'synchronize',fail)
    with pytest.raises(RuntimeError,match='close drain failure'):session.close()
    assert session.poisoned and not session.closed and len(session.slots)==len(session.pinned_arenas)==1
    with pytest.raises(RuntimeError,match='poisoned'):
        with session.run():pass
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *_:None);session.close()
    assert session.closed and not session.slots and not session.pinned_arenas

@pytest.mark.parametrize('failure',['create','record'])
def test_consumed_event_failure_poisons_and_releases_metadata_guard(monkeypatch,failure):
    import threading
    from types import SimpleNamespace
    block=SimpleNamespace(index=0);session=object.__new__(PrefetchSession);session.closed=False;session.poisoned=False;session.device=torch.device('cuda:0')
    session._run_lock=threading.Lock();session._run_lock.acquire();session._metadata_lock=threading.Lock();session._token=object();session._owner=threading.get_ident()
    session.positions={id(block):0};session.blocks=[block];session._source_versions={id(block):()};session._source_identity=lambda _:();session._clock=0;session.reloads=0;session.hits=0;session.trace=[];session._consumed_owners=[]
    slot={'id':0,'block':block,'generation':1,'active':False,'last_use':0,'ready':object(),'pending':False};session._find=lambda _:slot;session._views=lambda *_:{'w':torch.ones(1)}
    monkeypatch.setattr(torch.cuda,'current_stream',lambda *_:SimpleNamespace(wait_event=lambda _:None,cuda_stream=0))
    def fail(*_):raise RuntimeError('consumed '+failure+' failure')
    monkeypatch.setattr(torch.cuda,'Event',fail if failure=='create' else lambda:SimpleNamespace(record=fail))
    try:
        with pytest.raises(RuntimeError,match='consumed'):
            with session.borrow(block,'forward',token=session._token):pass
        assert session.poisoned and not session._metadata_lock.locked() and slot['active'] and not session._consumed_owners
        with pytest.raises(RuntimeError,match='poisoned'):
            with session.borrow(block,'forward',token=session._token):pass
    finally:session._run_lock.release()
