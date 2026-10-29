"""Opt-in, bounded debug accounting. Importing this module never imports Torch.

Host spans are wall intervals on each Python thread. Exclusive means excluding
direct nested spans on that same thread, not CPU self time. Device samples are
separate stream intervals, never components to add to host update wall time.
"""
from __future__ import annotations
import contextlib,functools,json,math,os,threading,time
from pathlib import Path

SCHEMA='rvft.nf4.full_chain_timer_options.v1'
_ACTIVE=None
_REGISTRY_LOCK=threading.Lock()
_NOOP=contextlib.nullcontext()

def validate_options(value):
    if value is None:return None
    if (type(value)is not dict or set(value)!={'schema','enabled','gradient_callbacks','device_event_sample_limit'}
            or value.get('schema')!=SCHEMA or type(value.get('enabled'))is not bool
            or type(value.get('gradient_callbacks'))is not bool
            or type(value.get('device_event_sample_limit'))is not int
            or value['device_event_sample_limit']!=(128 if value['enabled'] else 0)):
        raise ValueError('unsupported exact full-chain timer options')
    return dict(value)

def active_timer():return _ACTIVE

@contextlib.contextmanager
def installed(timer):
    global _ACTIVE
    with _REGISTRY_LOCK:
        if _ACTIVE is not None:raise RuntimeError('nested/concurrent timer installation is forbidden')
        _ACTIVE=timer
    try:yield timer
    finally:
        with _REGISTRY_LOCK:
            if _ACTIVE is not timer:raise RuntimeError('timer registry ownership changed')
            _ACTIVE=None

def span(name,**metadata):
    timer=_ACTIVE
    return _NOOP if timer is None else timer.span(name,metadata)

def device_span(name,stream=None):
    timer=_ACTIVE
    return _NOOP if timer is None else timer.device_span(name,stream)

def forward_device_span():
    timer=_ACTIVE
    if timer is None or timer.context_copy().get('phase')!='forward':return _NOOP
    return timer.device_span('layer.forward.stream_interval')

def call(name,function,*args,**kwargs):
    if _ACTIVE is None:return function(*args,**kwargs)
    with _ACTIVE.span(name):return function(*args,**kwargs)

def timed(name,tag=None):
    """OFF calls the original immediately: no clock/span/event/tag construction."""
    def decorate(function):
        @functools.wraps(function)
        def invoke(*args,**kwargs):
            timer=_ACTIVE
            if timer is None:return function(*args,**kwargs)
            metadata={} if tag is None else tag(*args,**kwargs)
            with timer.span(name,metadata):return function(*args,**kwargs)
        return invoke
    return decorate

def bind(function):
    """Select an undecorated OFF callable once during object construction.

    ON must be installed before constructing the model. This does not mutate
    a class or a global registry and never removes resource/lifetime guards.
    """
    if _ACTIVE is not None:return function
    raw=getattr(function,'__wrapped__',function)
    owner=getattr(function,'__self__',None)
    return raw.__get__(owner,type(owner)) if raw is not function and owner is not None else raw

def record_tag(self,*args,**kwargs):
    names=getattr(self,'_names',None) or tuple(getattr(self,'records',{}))
    name=names[0] if names else getattr(self,'record_name','unknown')
    parts=str(name).split('.layers.',1)
    layer=parts[-1].split('.',1)[0]if len(parts)==2 else 'unknown'
    return {'layer':layer}

def bounded_metadata(value,depth=0):
    """Only primitive specialization metadata; no repr/tensor contents/pointers."""
    if depth>5:return {'omitted':'depth_limit'}
    if value is None or type(value)in(bool,int):return value
    if type(value)is float:return value if math.isfinite(value)else {'omitted':'nonfinite'}
    if type(value)is str:return value[:256]
    if type(value)in(tuple,list):return[bounded_metadata(x,depth+1)for x in value[:64]]
    if type(value)is dict:
        return {str(k)[:80]:bounded_metadata(v,depth+1)for k,v in list(value.items())[:64]if type(k)in(str,int,tuple)}
    return {'omitted_type':type(value).__name__}

class _Span:
    def __init__(self,timer,name,metadata):self.t=timer;self.name=name;self.metadata=metadata;self.child=0
    def __enter__(self):
        t=self.t;self.thread=threading.get_ident();self.context=t.context_copy()
        self.stack=getattr(t.local,'stack',None)
        if self.stack is None:self.stack=[];t.local.stack=self.stack
        self.parent=self.stack[-1]if self.stack else None
        self.started=t.clock();self.stack.append(self)
        with t.lock:t.open_spans+=1
        return self
    def __exit__(self,kind,value,tb):
        end=self.t.clock();duration=end-self.started
        if duration<0 or not self.stack or self.stack[-1]is not self:raise RuntimeError('invalid timer span stack/clock')
        self.stack.pop()
        if self.parent is not None:self.parent.child+=duration
        exclusive=duration-self.child
        if exclusive<0:raise RuntimeError('negative nested timer exclusive interval')
        t=self.t;thread_role='main'if self.thread==t.main_thread else'worker'
        key=(self.context['update'],self.context.get('microbatch'),self.context.get('row_id'),self.context.get('phase'),thread_role,self.name,tuple(sorted(self.metadata.items())))
        with t.lock:
            if key not in t.stats and len(t.stats)>=t.max_buckets:raise RuntimeError('timer aggregation bucket budget exceeded')
            stat=t.stats.setdefault(key,{'count':0,'inclusive_ns':0,'same_thread_exclusive_ns':0,'min_ns':duration,'max_ns':duration,'errors':0})
            stat['count']+=1;stat['inclusive_ns']+=duration;stat['same_thread_exclusive_ns']+=exclusive
            stat['min_ns']=min(stat['min_ns'],duration);stat['max_ns']=max(stat['max_ns'],duration);stat['errors']+=int(kind is not None)
            t.open_spans-=1;t.span_count+=1
            if self.parent is None and thread_role=='main':t.main_intervals.append((self.started,end))
            if self.name=='transfer.D2H.blocking' and kind is None:t.d2h_bytes+=self.metadata['bytes'];t.d2h_copies+=1
            if self.name=='transfer.H2D.submit' and kind is None:t.h2d_scheduled_bytes+=self.metadata['bytes'];t.h2d_submissions+=1
        return False

class _DeviceSpan:
    def __init__(self,timer,name,stream,context):self.t=timer;self.name=name;self.stream=stream;self.context=context
    def __enter__(self):
        self.start=self.t.event_factory();self.end=self.t.event_factory()
        self.start.record(self.stream);return self
    def __exit__(self,kind,value,tb):
        self.end.record(self.stream)
        with self.t.lock:self.t.pending.append((self.name,self.context,self.start,self.end,kind is not None))
        return False

class DiagnosticTimer:
    def __init__(self,*,event_factory=None,event_limit=128,clock=time.perf_counter_ns,max_buckets=100000,binding=None):
        if type(event_limit)is not int or not 0<=event_limit<=128:raise ValueError('event sample budget invalid')
        self.binding={}if binding is None else dict(binding)
        if self.binding and (set(self.binding)!={'plan_sha256','source_snapshot_sha256','review_sha256'}or any(type(v)is not str or len(v)!=64 or any(c not in '0123456789abcdef'for c in v)for v in self.binding.values())):raise ValueError('invalid timer lineage binding')
        self.event_factory=event_factory;self.event_limit=event_limit;self.clock=clock;self.max_buckets=max_buckets
        self.main_thread=threading.get_ident();self.lock=threading.RLock();self.local=threading.local()
        self.context={'update':0,'microbatch':None,'row_id':None,'phase':'setup'};self.stats={};self.main_intervals=[];self.window_start_ns=None;self.updates=[]
        self.row_metadata=[];self.span_count=0;self.open_spans=0;self.pending=[];self.device_samples=[];self.sample_counts={};self.reserved_pairs=0
        self.d2h_bytes=0;self.d2h_copies=0;self.h2d_scheduled_bytes=0;self.h2d_submissions=0;self.device_unsampled=0;self.timer_write_ns=0
    def context_copy(self):
        with self.lock:return dict(self.context)
    def set_microbatch(self,*,update,microbatch,row_id,sequence_length,padded_length,supervised_tokens):
        if type(update)is not int or not 1<=update<=4 or type(microbatch)is not int or not 0<=microbatch<6:raise ValueError('timer update/microbatch out of contract')
        if type(row_id)is not str or any(type(x)is not int or x<0 for x in (sequence_length,padded_length,supervised_tokens))or padded_length<sequence_length:raise ValueError('timer row metadata invalid')
        with self.lock:
            self.context={'update':update,'microbatch':microbatch,'row_id':row_id,'phase':'inputs'}
            self.row_metadata.append({**self.context,'sequence_length':sequence_length,'padded_length':padded_length,'padding_tokens':padded_length-sequence_length,'supervised_shifted_tokens':supervised_tokens})
    def set_optimizer(self,update):
        with self.lock:self.context={'update':update,'microbatch':None,'row_id':None,'phase':'optimizer'if update else'cleanup'}
    def set_phase(self,phase):
        with self.lock:self.context['phase']=phase
    def start_window(self):self.window_start_ns=self.clock()
    def span(self,name,metadata=None):
        if type(name)is not str or not name:raise ValueError('timer name required')
        metadata={}if metadata is None else metadata
        if any(type(k)is not str or type(v)not in(str,int,bool)for k,v in metadata.items()):raise ValueError('timer metadata must be bounded scalars')
        return _Span(self,name,dict(metadata))
    def device_span(self,name,stream=None):
        if self.event_factory is None:return _NOOP
        context=self.context_copy();key=(context['update'],name)
        with self.lock:
            if self.reserved_pairs>=self.event_limit or self.sample_counts.get(key,0)>=8:
                self.device_unsampled+=1;return _NOOP
            self.reserved_pairs+=1;self.sample_counts[key]=self.sample_counts.get(key,0)+1
        return _DeviceSpan(self,name,stream,context)
    def collect_device_ready(self):
        # Query only. Never call Event.synchronize, cuda.synchronize or wait here.
        with self.lock:
            remaining=[]
            for name,context,start,end,error in self.pending:
                if not start.query()or not end.query():remaining.append((name,context,start,end,error));continue
                ms=float(start.elapsed_time(end))
                if not math.isfinite(ms)or ms<0:raise RuntimeError('invalid CUDA timer sample')
                self.device_samples.append({'name':name,**context,'elapsed_ms':ms,'operation_raised':error})
            self.pending=remaining
    def finish_update(self,cursor,wall_seconds,path,boundary_ns):
        self.collect_device_ready()
        if type(wall_seconds)not in(int,float)or not math.isfinite(wall_seconds)or wall_seconds<=0:raise ValueError('invalid update wall')
        with self.lock:
            if self.window_start_ns is None or boundary_ns<=self.window_start_ns:raise RuntimeError('missing/inverted update timer window')
            covered=sum(max(0,min(end,boundary_ns)-max(begin,self.window_start_ns))for begin,end in self.main_intervals)/1e9
            self.window_start_ns=boundary_ns
            if covered>wall_seconds+1e-3:raise RuntimeError('main-thread timed roots exceed synchronized update wall')
            record={'cursor_updates':cursor,'synchronized_wall_seconds':wall_seconds,'main_thread_root_covered_seconds':covered,'main_thread_unattributed_wall_seconds':max(0.0,wall_seconds-covered),'same_thread_exclusive_is_cpu_self_time':False,'all_thread_sum_is_wall':False}
            self.updates.append(record)
        started=self.clock()
        with Path(path).open('a')as f:f.write(json.dumps(record,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
        self._write_update_snapshot(cursor,Path(path).parent)
        self.timer_write_ns+=self.clock()-started
    def _write_update_snapshot(self,cursor,parent):
        """Preserve completed timing work across SIGTERM/monitor loss.

        Snapshot IO follows the existing synchronized boundary, so it remains
        charged to the next wall interval. Never a model/optimizer checkpoint.
        """
        target=parent/f'FULL_CHAIN_TIMER_UPDATE_{cursor:02d}.json'
        partial=target.with_name(target.name+'.partial')
        value={'schema':'rvft.nf4.full_chain_timer_update_snapshot.v1',
          'binding':dict(self.binding),'completed_cursor_updates':cursor,
          'observer_snapshot':self.summary(),'terminal_qualification':False,
          'scope':'Cumulative completed host spans and ready device samples at this boundary. Background writer may still be active. Not a training-state checkpoint or terminal acceptance.',
          'IO_scope':'This snapshot IO occurs after the boundary and belongs to the following synchronized wall interval. Its cost appears in later snapshots/final timer IO total; no subtraction.'}
        with partial.open('x')as stream:
            json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        # Atomic exclusive publication: never overwrite an existing snapshot.
        os.link(partial,target,follow_symlinks=False)
        partial.unlink()
        fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:os.fsync(fd)
        finally:os.close(fd)
    def summary(self):
        self.collect_device_ready()
        with self.lock:
            rows=[]
            for (update,micro,row,phase,thread,name,metadata),stat in self.stats.items():rows.append({'update':update,'microbatch':micro,'row_id':row,'phase':phase,'thread_role':thread,'name':name,'metadata':dict(metadata),**stat})
            return {'schema':'rvft.nf4.full_chain_timer.v1','enabled':True,'span_count':self.span_count,'open_spans':self.open_spans,'host_spans':rows,'updates':list(self.updates),'row_metadata':list(self.row_metadata),'D2H_actual_successful_payload_bytes':self.d2h_bytes,'D2H_actual_successful_copy_count':self.d2h_copies,'H2D_scheduled_payload_bytes':self.h2d_scheduled_bytes,'H2D_record_submissions':self.h2d_submissions,'device_samples':list(self.device_samples),'device_pending_pairs':len(self.pending),'device_pairs_reserved':self.reserved_pairs,'device_pairs_limit':self.event_limit,'device_unsampled_intervals':self.device_unsampled,'timer_update_log_write_ns':self.timer_write_ns,
             'semantics':{'host_clock':'perf_counter_ns; spans are elapsed wall, not CPU cycles','exclusive':'inclusive minus directly nested same-thread spans; includes waits/OS scheduling and cross-thread work','cross_thread':'May overlap main BW/other spans. Never sum threads to reconstruct update wall. Main root coverage and unattributed wall are separate.','device':'Bounded first8 per category/update, at most128 pairs per job; stream elapsed includes possible host submission gaps and overlaps. Query after existing boundary sync; no new device synchronization. Unsampled work stays unknown.','overhead':'Instrumentation perturbs CPU/event scheduling. No overhead subtraction. Timer log IO charged to following total-wall interval; final log/model-release/setup separately reported. Compare ON/OFF with callbacks fixed; 2x2 needed for interactions. No global profiler.','not_reconstruction':'New diagnostic only, cannot recover R37 historical311second internals.'},'formal_qualified':False}
