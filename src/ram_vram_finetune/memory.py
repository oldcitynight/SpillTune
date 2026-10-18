"""One synchronous aligned slot, reloaded for both forward and recomputation.

The session is serial and non-reentrant. Base weights remain on the host; a
reload invalidates every view of the previous block. No asynchronous fallback.
"""
import threading
import time
import math
from contextlib import contextmanager

import torch
from .quantization.codecs import DTYPES,decode_into,validate_spec
from .buffers import raw_buffer_metadata


def packed_size(weights):
    offset = 0
    for tensor in weights.values():
        offset = (offset + 255) // 256 * 256
        offset += tensor.numel() * tensor.element_size()
    return offset


def decoded_layout(block):
    if block.storage_specs is None:
        raw_buffer_metadata(block)
        specs={k:{'spec':{'shape':list(t.shape),'dtype':str(t.dtype).split('.')[-1]},'parts':{'value':k}}
               for k,t in block.host_weights.items()}
    else:specs=block.storage_specs
    offset=0;result=[]
    for name,value in specs.items():
        spec=value['spec'];dtype=block.host_weights[name].dtype if block.storage_specs is None else DTYPES[spec['dtype']]
        offset=(offset+255)//256*256
        size=math.prod(spec['shape'])*torch.empty((),dtype=dtype).element_size()
        result.append((name,offset,size,dtype,value))
        offset+=size
    return offset,result


def pinned_layout(blocks, arena_limit=128 * 1024**2):
    """Metadata-only best-fit decreasing packing; never retain source tensors.

    The host caching allocator rounds each allocation to a power of two. Pack
    tensors into aligned shared allocations rather than rounding every matrix.
    A tensor larger than arena_limit gets a larger arena, never a split view.
    """
    items = []
    for index, block in enumerate(blocks):
        for name, tensor in block.host_weights.items():
            if tensor.device.type != 'cpu' or tensor.requires_grad or not tensor.is_contiguous():
                raise ValueError('host weights must be frozen contiguous CPU tensors')
            size = tensor.numel() * tensor.element_size()
            if size < 1:
                raise ValueError('empty host weight is unsupported')
            items.append((size, index, name))
    return pinned_layout_items(items, arena_limit)


def pinned_layout_items(items, arena_limit=128*1024**2):
    """Plan byte-sized (size, owner-key, part-key) items without tensors."""
    arenas = []
    for size, index, name in sorted(items, key=lambda item: (-item[0], item[1], item[2])):
        candidates = [(arena['limit'] - ((arena['used'] + 255) // 256 * 256) - size, i)
                      for i, arena in enumerate(arenas)
                      if ((arena['used'] + 255) // 256 * 256) + size <= arena['limit']]
        if candidates:
            _, selected = min(candidates)
        else:
            selected = len(arenas)
            arenas.append({'used': 0, 'limit': max(arena_limit, 1 << (size - 1).bit_length()), 'items': []})
        arena = arenas[selected]
        offset = (arena['used'] + 255) // 256 * 256
        arena['items'].append((index, name, offset, size))
        arena['used'] = offset + size
    for arena in arenas:
        arena['capacity'] = 1 << (arena['used'] - 1).bit_length()
    return arenas


def pin_weights(blocks, layout):
    """Transfer ownership incrementally: replace each pageable source promptly.

    Initialization failure invalidates the plan and clears partial views. The
    caller must discard that adapter. No private CUDA allocator API is used.
    """
    owners = []
    try:
        for arena in layout:
            owner = torch.empty(arena['capacity'], dtype=torch.uint8, pin_memory=True)
            owners.append(owner)
            for index, name, offset, size in arena['items']:
                source = blocks[index].host_weights[name]
                view = owner.narrow(0, offset, size).view(source.dtype).view(source.shape)
                view.copy_(source)
                blocks[index].host_weights[name] = view
                del source, view
        return owners
    except BaseException:
        for block in blocks:
            block.host_weights.clear()
        owners.clear()
        raise


class SlotSession:
    def __init__(self, blocks, device, *, host_budget_bytes, device_budget_bytes, transfer_mode='synchronous'):
        if transfer_mode not in ('synchronous', 'batched'):
            raise ValueError('unsupported_capability: unknown transfer mode')
        if transfer_mode != 'synchronous' and any(getattr(b,'buffer_contract',None) is not None for b in blocks):
            raise ValueError('unsupported_capability: typed-buffer contracts require synchronous transfer')
        if transfer_mode == 'batched' and (torch.device(device).type != 'cuda' or any(b.storage_specs is not None for b in blocks)):
            raise ValueError('unsupported_capability: batched transfer requires CUDA and uncompressed weights')
        self.transfer_mode = transfer_mode
        if any(t.device.type!='cpu' or t.requires_grad or not t.is_contiguous() for b in blocks for t in b.host_weights.values()):
            raise ValueError('host weights must be frozen contiguous CPU tensors')
        self.blocks = blocks
        self.buffer_metadata = {id(b):raw_buffer_metadata(b) for b in blocks if b.storage_specs is None or getattr(b,'buffer_contract',None) is not None}
        self.decode_workspace_budget_bytes = max((m['decode_workspace_bytes'] for m in self.buffer_metadata.values() if m is not None),default=0)
        self.device = torch.device(device)
        host_bytes = sum(t.numel() * t.element_size() for b in blocks for t in b.host_weights.values())
        capacity = max(decoded_layout(b)[0] for b in blocks)
        upload_capacity=max((packed_size(b.host_weights) for b in blocks if b.storage_specs is not None),default=0)
        for block in blocks:
            if block.storage_specs is not None:
                used=[]
                for value in block.storage_specs.values():
                    used.extend(value['parts'].values())
                    validate_spec(value['spec'],{p:block.host_weights[k] for p,k in value['parts'].items()})
                if len(used)!=len(set(used)) or set(used)!=set(block.host_weights):raise ValueError('invalid aliased/incomplete stored block inventory')
        workspace=self.decode_workspace_budget_bytes
        explicit_buffers=any(m is not None for m in self.buffer_metadata.values())
        slot_charge=capacity+upload_capacity+(255 if explicit_buffers else 0)
        host_charge=host_bytes+(slot_charge+workspace if self.device.type=='cpu' and explicit_buffers else 0)
        if host_charge > host_budget_bytes or slot_charge+workspace > device_budget_bytes:
            raise MemoryError("resource_budget: block weights exceed configured memory limit")
        self.host_bytes = host_bytes
        self.pinned_arenas = []
        reused = self.device.type == 'cuda' and all(t.is_pinned() for b in blocks for t in b.host_weights.values())
        self.pinned_layout = pinned_layout(blocks) if self.device.type == 'cuda' else []
        self.pinned_capacity_bytes = sum(a['capacity'] for a in self.pinned_layout)
        if reused:
            owners = {t.untyped_storage().data_ptr():t.untyped_storage() for b in blocks for t in b.host_weights.values()}
            self.pinned_arenas = list(owners.values())
            self.pinned_capacity_bytes = sum(s.nbytes() for s in self.pinned_arenas)
        if self.pinned_capacity_bytes > host_budget_bytes:
            raise MemoryError('resource_budget: pinned arena capacity exceeds configured host limit')
        self.capacity = capacity
        self.upload_capacity=upload_capacity
        self.upload_slot=torch.empty(upload_capacity,dtype=torch.uint8,device=self.device) if upload_capacity else None
        self.slot = torch.empty(capacity + 255, dtype=torch.uint8, device=self.device)
        self.slot = self.slot.narrow(0, (-self.slot.data_ptr()) % 256, capacity)
        self.closed = False
        self.generation = 0
        self.reloads = 0
        self.transferred_bytes = 0
        self.actual_h2d_bytes=0
        self._run_lock = threading.Lock()
        self._owner = None
        self._token = None
        self.trace = []
        if self.device.type == "cuda" and not reused:
            self.pinned_arenas = pin_weights(blocks, self.pinned_layout)

    @contextmanager
    def run(self):
        if self.closed:
            raise RuntimeError("session is closed")
        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError("concurrent/reentrant session use is unsupported")
        self._owner = threading.get_ident()
        self._token = object()
        try:
            yield
        finally:
            self._owner = None
            self._token = None
            self._run_lock.release()

    def reloader(self, block):
        if self._owner != threading.get_ident() or self._token is None:
            raise RuntimeError("reloader creation requires the active session owner")
        token = self._token
        # CUDA autograd executes backward on a worker thread. A run-scoped
        # capability admits that callback without allowing stale callbacks
        # from a completed run or concurrent unowned loading.
        return lambda: self.load(block, token=token)

    def load(self, block, *, token=None):
        if self.closed:
            raise RuntimeError("session is closed")
        if token is not None:
            authorized = token is self._token and self._run_lock.locked()
        else:
            authorized = self._owner == threading.get_ident() and self._run_lock.locked()
        if not authorized:
            raise RuntimeError("slot loading requires the active session owner")
        if id(block) not in {id(b) for b in self.blocks}:
            raise ValueError('slot plan is not owned by this session')
        if id(block) in self.buffer_metadata and raw_buffer_metadata(block)!=self.buffer_metadata[id(block)]:
            raise ValueError('typed_buffer: contract or layout changed during session')
        if any(t.device.type!='cpu' or t.requires_grad or not t.is_contiguous() for t in block.host_weights.values()):
            raise ValueError('host weights must be frozen contiguous CPU tensors')
        wait_started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        wait_seconds = time.perf_counter() - wait_started
        started = time.perf_counter()
        start_event = end_event = None
        if self.device.type == "cuda":
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(self.device))
        offset, views = 0, {}
        staged={}
        if block.storage_specs is not None:
            for name,source in block.host_weights.items():
                offset=(offset+255)//256*256;size=source.numel()*source.element_size()
                view=self.upload_slot.narrow(0,offset,size).view(source.dtype).view(source.shape)
                view.copy_(source,non_blocking=False);staged[name]=view;offset+=size
        copy_end_event=None
        if block.storage_specs is not None and self.device.type=='cuda':
            copy_end_event=torch.cuda.Event(enable_timing=True);copy_end_event.record(torch.cuda.current_stream(self.device));copy_end_event.synchronize()
        copy_ended=time.perf_counter()
        decode_started=time.perf_counter()
        if block.storage_specs is not None:
            for name,offset,size,dtype,entry in decoded_layout(block)[1]:
                view=self.slot.narrow(0,offset,size).view(dtype).view(entry['spec']['shape'])
                decode_into(view,entry['spec'],{p:staged[k] for p,k in entry['parts'].items()})
                views[name]=view
            offset=self.capacity
        else:
            offset=0
        for name, source in block.host_weights.items():
            if block.storage_specs is not None:break
            if source.requires_grad or source.device.type != "cpu" or not source.is_contiguous():
                raise ValueError("host weights must be frozen contiguous CPU tensors")
            offset = (offset + 255) // 256 * 256
            size = source.numel() * source.element_size()
            view = self.slot.narrow(0, offset, size).view(source.dtype).view(source.shape)
            # One current-stream batch, with the existing pre-copy device
            # barrier and block-end event wait. No overlap or second slot.
            if self.transfer_mode == 'batched' and not source.is_pinned():
                raise ValueError('batched transfer requires pinned CPU sources')
            view.copy_(source, non_blocking=self.transfer_mode == 'batched')
            views[name] = view
            offset += size
        self.reloads += 1
        self.generation += 1
        self.transferred_bytes += sum(t.numel() * t.element_size() for t in block.host_weights.values())
        stored_bytes=sum(t.numel()*t.element_size() for t in block.host_weights.values())
        if self.device.type=='cuda':self.actual_h2d_bytes+=stored_bytes
        if end_event is not None:
            end_event.record(torch.cuda.current_stream(self.device)); end_event.synchronize()
        self.trace.append({"block": block.index, "generation": self.generation,
                           "device": str(self.device), "slot_capacity": self.capacity,
                           "transfer_mode": self.transfer_mode,
                           "decode_workspace_budget_bytes": self.decode_workspace_budget_bytes,
                           "typed_buffer_contract": self.buffer_metadata.get(id(block)),
                           "logical_bytes": sum(t.numel() * t.element_size() for t in block.host_weights.values()),
                           "stored_bytes":stored_bytes,"decoded_bytes":sum(v.numel()*v.element_size() for v in views.values()),
                           "actual_h2d_bytes":stored_bytes if self.device.type=='cuda' else 0,
                           "upload_slot_capacity":self.upload_capacity,
                           "upload_slot_alias":all(self.upload_slot.data_ptr()<=t.data_ptr()<self.upload_slot.data_ptr()+self.upload_capacity for t in staged.values()) if staged else None,
                           "dequantization":"chunked_fp32_to_bf16" if block.storage_specs is not None else 'none',
                           "decode_wall_seconds":time.perf_counter()-decode_started if block.storage_specs is not None else 0,
                           "compressed_h2d_wall_seconds":copy_ended-started if block.storage_specs is not None else None,
                           "compressed_h2d_cuda_ms":start_event.elapsed_time(copy_end_event) if copy_end_event is not None else None,
                           "dequant_cuda_ms":copy_end_event.elapsed_time(end_event) if copy_end_event is not None else None,
                           "copy_metric_scope":"combined_load_including_dequant" if block.storage_specs is not None else 'weight_copy',
                           "source_cpu": all(t.device.type == "cpu" for t in block.host_weights.values()),
                           "source_pinned": all(t.is_pinned() for t in block.host_weights.values()),
                           "slot_alias": all(self.slot.data_ptr() <= v.data_ptr() < self.slot.data_ptr() + self.capacity for v in views.values()),
                           "precopy_wait_seconds": wait_seconds,
                           "copy_wall_seconds": time.perf_counter() - started,
                           "copy_cuda_ms": start_event.elapsed_time(end_event) if start_event is not None else None})
        return views

    def close(self):
        if self._run_lock.locked():
            raise RuntimeError("cannot close an active session")
        if not self.closed:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.slot = None
            self.upload_slot=None
            if self.device.type == 'cuda':
                for block in self.blocks:
                    block.host_weights.clear()
                self.pinned_arenas.clear()
            self.blocks = []
            self.closed = True
