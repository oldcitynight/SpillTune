"""Budgeted raw-weight cache with one-layer lookahead and explicit leases.

A copy-ready event admits use on the actual compute stream. Only the consumer
lease's exit event admits overwriting the slot, including full backward VJP and
gradient accumulation. Python slot/pinned owners live until a device drain.
"""
import threading
import time
from contextlib import contextmanager

import torch

from .memory import packed_size, pin_weights, pinned_layout


def plan_prefetch_pool(capacity, block_count, *, pool_budget_bytes, max_slots,
                       device_budget_bytes, live_allocated_bytes, reserve_bytes):
    """CPU-only admission estimate; allocation is measured again by the caller.

    2MiB rounding is deliberately conservative, not an allocator guarantee.
    The configured pool budget caps measured allocated deltas as well.
    """
    values=(capacity,block_count,pool_budget_bytes,max_slots,device_budget_bytes,reserve_bytes)
    if any(type(x) is not int or x<1 for x in values) or type(live_allocated_bytes) is not int or live_allocated_bytes<0:
        raise ValueError('prefetch resource limits must be positive integers')
    charge=((capacity+255+2*1024**2-1)//(2*1024**2))*(2*1024**2)
    available=min(pool_budget_bytes,device_budget_bytes-live_allocated_bytes-reserve_bytes)
    count=min(block_count,max_slots,max(0,available//charge))
    minimum=min(2,block_count)
    if count<minimum:
        raise MemoryError('resource_budget: prefetch requires two admitted slots for a multi-block plan')
    return {'resolved_slots':count,'lookahead_blocks':min(1,block_count-1),
            'capacity_per_slot_bytes':capacity,'estimated_charge_per_slot_bytes':charge,
            'estimated_pool_charge_bytes':charge*count,'pool_budget_bytes':pool_budget_bytes,
            'available_pool_budget_bytes':available,'live_allocated_before_pool_bytes':live_allocated_bytes,
            'execution_reserve_bytes':reserve_bytes,'device_budget_bytes':device_budget_bytes,
            'pool_scale':'resolved at initialization and fixed for session lifetime',
            'single_block_resolution':block_count==1}


class _Lease:
    def __init__(self,session,block,token):self.session,self.block,self.token=session,block,token
    def __call__(self):raise RuntimeError('prefetch weights require an explicit consumer borrow context')
    def borrow(self,phase):return self.session.borrow(self.block,phase,token=self.token)


class PrefetchSession:
    asynchronous=True
    def __init__(self,blocks,device,*,host_budget_bytes,device_budget_bytes,
                 pool_budget_bytes=128*1024**2,max_slots=2,reserve_bytes=256*1024**2):
        self.device=torch.device(device)
        if self.device.type!='cuda' or self.device.index not in (None,0):
            raise ValueError('unsupported_capability: prefetch requires cuda:0')
        self.device=torch.device('cuda:0');self.blocks=list(blocks)
        if not self.blocks or len({b.index for b in self.blocks})!=len(self.blocks):
            raise ValueError('prefetch needs a nonempty unique ordered block plan')
        if any(getattr(b,'buffer_contract',None) is not None for b in self.blocks):
            raise ValueError('unsupported_capability: typed-buffer contracts require synchronous transfer')
        if any(b.storage_specs is not None for b in self.blocks):
            raise ValueError('unsupported_capability: prefetch requires uncompressed weights')
        if any(t.device.type!='cpu' or t.requires_grad or not t.is_contiguous() for b in self.blocks for t in b.host_weights.values()):
            raise ValueError('host weights must be frozen contiguous CPU tensors')
        self.host_bytes=sum(t.numel()*t.element_size() for b in self.blocks for t in b.host_weights.values())
        self.pinned_layout=pinned_layout(self.blocks);self.pinned_capacity_bytes=sum(x['capacity'] for x in self.pinned_layout)
        if max(self.host_bytes,self.pinned_capacity_bytes)>host_budget_bytes:
            raise MemoryError('resource_budget: pinned source pool exceeds host limit')
        self.capacity=max(packed_size(b.host_weights) for b in self.blocks)
        self.device_budget_bytes=device_budget_bytes;live=torch.cuda.memory_allocated(self.device)
        self.plan=plan_prefetch_pool(self.capacity,len(self.blocks),pool_budget_bytes=pool_budget_bytes,
             max_slots=max_slots,device_budget_bytes=device_budget_bytes,live_allocated_bytes=live,reserve_bytes=reserve_bytes)
        self.positions={id(b):i for i,b in enumerate(self.blocks)}
        self._run_lock=threading.Lock();self._metadata_lock=threading.Lock();self._owner=None;self._token=None
        self.closed=False;self.poisoned=False;self.slots=[];self.pinned_arenas=[];self.copy_stream=None
        self.trace=[];self.copy_records=[];self._completed_copies=[];self._consumed_owners=[];self.generation=0;self.reloads=0;self.hits=0
        self.transferred_bytes=0;self.actual_h2d_bytes=0;self.upload_capacity=0;self._clock=0
        try:
            with torch.cuda.device(self.device):
                self.copy_stream=torch.cuda.Stream(device=self.device)
                for i in range(self.plan['resolved_slots']):
                    raw=torch.empty(self.capacity+255,dtype=torch.uint8,device=self.device)
                    buffer=raw.narrow(0,(-raw.data_ptr())%256,self.capacity)
                    self.slots.append({'id':i,'buffer':buffer,'block':None,'generation':0,'ready':None,
                                       'consumed':None,'active':False,'last_use':-1,'templates':{},'pending':False})
                # Even a write-only side-stream first use must wait the
                # allocation stream: cached storage may have older work there.
                self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
            delta=torch.cuda.memory_allocated(self.device)-live
            if delta>self.plan['available_pool_budget_bytes'] or torch.cuda.memory_allocated(self.device)+reserve_bytes>device_budget_bytes:
                raise MemoryError('resource_budget: actual pool allocation exceeds admitted budget')
            self.plan['actual_pool_allocated_bytes']=delta
            self.plan['logical_pool_capacity_bytes']=self.capacity*len(self.slots)
            self.pinned_arenas=pin_weights(self.blocks,self.pinned_layout)
            self._source_versions={id(b):self._source_identity(b) for b in self.blocks}
        except BaseException:
            self.poisoned=True
            try:
                torch.cuda.synchronize(self.device)
            finally:
                # No upload can occur before construction publishes a session.
                self.slots.clear();self.pinned_arenas.clear();self.copy_stream=None;self.closed=True
            raise

    def _authorize(self,token):
        if self.closed or self.poisoned:raise RuntimeError('prefetch session is closed or poisoned')
        if token is not self._token or token is None or not self._run_lock.locked():
            raise RuntimeError('prefetch borrow requires the active run capability')

    @contextmanager
    def run(self):
        if self.closed or self.poisoned:raise RuntimeError('prefetch session is closed or poisoned')
        if not self._run_lock.acquire(blocking=False):raise RuntimeError('concurrent/reentrant session use is unsupported')
        self._owner=threading.get_ident();self._token=object()
        try:
            yield
        except BaseException:
            self.poisoned=True
            raise
        finally:
            try:
                # Includes speculative uploads and every compute stream. Normal
                # engine steps use this in place of their old final sync.
                torch.cuda.synchronize(self.device)
                if self._metadata_lock.locked():
                    self.poisoned=True
                    raise RuntimeError('prefetch consumer lease outlived its run')
                self._finalize_copies()
            except BaseException:
                self.poisoned=True
                raise
            finally:
                self._owner=None;self._token=None;self._run_lock.release()

    def reloader(self,block):
        if self._owner!=threading.get_ident() or self._token is None:
            raise RuntimeError('reloader creation requires the active session owner')
        if id(block) not in self.positions:raise ValueError('block does not belong to this session')
        return _Lease(self,block,self._token)

    @staticmethod
    def _source_identity(block):
        if any(t.device.type!='cpu' or t.requires_grad or not t.is_contiguous() or not t.is_pinned() for t in block.host_weights.values()):
            raise ValueError('prefetch sources must remain frozen contiguous pinned CPU tensors')
        return tuple((k,t.data_ptr(),tuple(t.shape),t.dtype,t._version) for k,t in block.host_weights.items())

    def _find(self,block):return next((s for s in self.slots if s['block'] is block),None)

    def _victim(self):
        available=[s for s in self.slots if not s['active']]
        if not available:raise RuntimeError('no inactive prefetch slot available')
        return min(available,key=lambda s:(s['block'] is not None,s['last_use'],s['id']))

    def _views(self,slot,block):
        signature=tuple((name,tuple(t.shape),t.dtype) for name,t in block.host_weights.items())
        if signature not in slot['templates']:
            views={};offset=0
            for name,source in block.host_weights.items():
                offset=(offset+255)//256*256;size=source.numel()*source.element_size()
                views[name]=slot['buffer'].narrow(0,offset,size).view(source.dtype).view(source.shape);offset+=size
            slot['templates'][signature]=views
        return slot['templates'][signature]

    def _enqueue(self,block,slot):
        if slot['active']:raise RuntimeError('cannot overwrite an active consumer')
        if any(t.device.type!='cpu' or t.requires_grad or not t.is_contiguous() or not t.is_pinned() for t in block.host_weights.values()):
            raise ValueError('prefetch sources must remain frozen contiguous pinned CPU tensors')
        views=self._views(slot,block);started=time.perf_counter();self.generation+=1
        record={'block':block.index,'slot':slot['id'],'generation':self.generation,
                'overwrote_block':slot['block'].index if slot['block'] is not None else None,
                'waited_consumed_generation':slot['consumed_generation'] if slot['consumed'] is not None else None,
                'bytes':sum(t.numel()*t.element_size() for t in block.host_weights.values())}
        # Mark the new tag pending as soon as it is enqueued. Cache hits also
        # wait its ready event; pending data can never be used as ready data.
        slot.update(block=block,generation=self.generation,pending=True)
        with torch.cuda.stream(self.copy_stream):
            if slot['consumed'] is not None:self.copy_stream.wait_event(slot['consumed'])
            start=torch.cuda.Event(enable_timing=True);ready=torch.cuda.Event(enable_timing=True);start.record(self.copy_stream)
            for name,source in block.host_weights.items():views[name].copy_(source,non_blocking=True)
            ready.record(self.copy_stream)
        slot['ready']=ready
        record.update(start_event=start,ready_event=ready,waited_consumed_event=slot['consumed'],host_enqueue_seconds=time.perf_counter()-started)
        self.copy_records.append(record);self.transferred_bytes+=record['bytes'];self.actual_h2d_bytes+=record['bytes']
        return views

    @contextmanager
    def borrow(self,block,phase,*,token):
        self._authorize(token)
        if phase not in ('forward','backward'):raise ValueError('unknown prefetch consumer phase')
        if id(block) not in self.positions:raise ValueError('block does not belong to this session')
        if not self._metadata_lock.acquire(blocking=False):raise RuntimeError('concurrent/reentrant prefetch borrow is unsupported')
        slot=None
        try:
            if self._source_identity(block)!=self._source_versions[id(block)]:
                raise ValueError('frozen prefetch source changed during session lifetime')
            slot=self._find(block);hit=slot is not None
            if not hit:slot=self._victim();self._enqueue(block,slot)
            slot['active']=True;self._clock+=1;slot['last_use']=self._clock
            compute=torch.cuda.current_stream(self.device);compute.wait_event(slot['ready'])
            self.reloads+=1;self.hits+=int(hit)
            record={'block':block.index,'phase':phase,'slot':slot['id'],'generation':slot['generation'],
                    'cache_hit':hit,'copy_pending_confirmation':slot['pending'],'ready_wait_enqueued':True,'compute_stream':compute.cuda_stream,
                    'next_prefetched':None,'source_pinned':True,'slot_alias':True}
            self.trace.append(record)
            next_pos=self.positions[id(block)]+(1 if phase=='forward' else -1)
            if 0<=next_pos<len(self.blocks):
                next_block=self.blocks[next_pos]
                if self._find(next_block) is None:
                    self._enqueue(next_block,self._victim());record['next_prefetched']=next_block.index
            yield self._views(slot,block)
        except BaseException:
            self.poisoned=True
            raise
        finally:
            try:
                if slot is not None and slot['active']:
                    # This must run after the entire consumer scope, including
                    # autograd.grad and LoRA gradient-buffer additions.
                    consumed=torch.cuda.Event();consumed.record(torch.cuda.current_stream(self.device))
                    slot['consumed']=consumed;slot['consumed_generation']=slot['generation']
                    slot['active']=False;self._consumed_owners.append(consumed)
            except BaseException:
                self.poisoned=True
                raise
            finally:self._metadata_lock.release()

    def _finalize_copies(self):
        # The enclosing run has drained every stream. Retain small CPU records,
        # not an ever-growing collection of CUDA event handles.
        for row in self.copy_records:
            self._completed_copies.append({k:v for k,v in row.items() if not k.endswith('_event')} |
                         {'copy_cuda_ms':row['start_event'].elapsed_time(row['ready_event'])})
        self.copy_records.clear();self._consumed_owners.clear()
        for slot in self.slots:slot['pending']=False

    def runtime_metadata(self):
        if self._run_lock.locked():raise RuntimeError('prefetch timing requires a completed run')
        return {'mode':'budgeted_one_layer_prefetch_lru','plan':self.plan,'cache_hits':self.hits,
                'uploads':len(self._completed_copies),'borrow_count':self.reloads,'copy_trace':self._completed_copies,
                'lifetime':'ready before use; consumed after complete functional/VJP+gradaccum before overwrite',
                'timing_scope':'enqueue CPU duration and copy-stream CUDA events; no host load-completion timing',
                'process_allocator_cap':'caller/harness; pool admission does not bound non-PyTorch driver allocations'}

    def close(self):
        if self._run_lock.locked() or self._metadata_lock.locked():raise RuntimeError('cannot close an active prefetch session/lease')
        if not self.closed:
            try:
                torch.cuda.synchronize(self.device)
            except BaseException:
                self.poisoned=True
                raise
            self.slots.clear();self.copy_records.clear();self._consumed_owners.clear()
            for block in self.blocks:block.host_weights.clear()
            self.pinned_arenas.clear();self.blocks=[];self.copy_stream=None;self.closed=True
