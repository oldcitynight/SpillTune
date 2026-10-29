"""Run-owned immutable rows. No accelerator imports or training-time hashing."""
from collections.abc import Mapping
from types import MappingProxyType
from functools import wraps
import json,hashlib

POLICY='startup_full_no_training_content_checks_v1'

class PreparedTrainingRows:
    __slots__=('_rows','_chunks','_order','_binding','_owner','_closed')
    def __init__(self,rows,*,binding,vocab_size,order):
        if type(vocab_size)is not int or vocab_size<=0:raise ValueError('positive preflight vocabulary required')
        owned={};chunks={}
        for key,value in rows.items():
            ids,labels,count=value
            ids=tuple(ids);labels=tuple(labels)
            if type(key)is not str or not key or len(ids)<2 or len(ids)!=len(labels):raise ValueError('invalid prepared row')
            if any(type(n)is not int or n<0 or n>=vocab_size for n in ids):raise ValueError('preflight input outside vocabulary')
            if any(type(n)is not int or(n!=-100 and(n<0 or n>=vocab_size))for n in labels):raise ValueError('preflight label outside vocabulary')
            first=next((i for i,n in enumerate(labels)if n!=-100),None)
            if first is None or first<1 or labels[first:]!=ids[first:] or any(n!=-100 for n in labels[:first]):raise ValueError('preflight supervised suffix mismatch')
            if type(count)is not int or count!=sum(n!=-100 for n in labels[1:]) or count<=0:raise ValueError('preflight supervised count mismatch')
            owned[key]=(ids,labels,count)
            # The supported training path fixes token_chunk=2. Derive masks
            # from owned CPU integers once; never reduce GPU labels in forward.
            chunks[key]=tuple((start,min(start+2,len(ids)-1))
                for start in range(0,len(ids)-1,2)
                if any(n!=-100 for n in labels[start+1:min(start+2,len(ids)-1)+1]))
        if not owned:raise ValueError('empty preflight rows')
        # Fresh immutable tuples/ints; no input tensors, files or mutable lists
        # are borrowed. Input mapping replacement cannot alter this snapshot.
        order=tuple(order)
        count=binding.get('target_updates',0)*binding.get('accumulation',0)
        if not count or count>len(order)or any(key not in owned for key in order[:count]):raise ValueError('preflight order/range mismatch')
        def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        rows_digest=digest([{'id':key,'input_ids':owned[key][0],'labels':owned[key][1]}for key in order[:count]])
        if binding.get('training_rows_sha256')!=rows_digest or binding.get('training_order_sha256')!=digest(list(order)):
            raise ValueError('preflight rows/order do not match actual execution binding')
        self._rows=MappingProxyType(owned);self._chunks=MappingProxyType(chunks);self._order=order
        self._binding=json.dumps(binding,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
        self._owner=None;self._closed=False

    @property
    def binding(self):return json.loads(self._binding)

    def claim(self,owner):
        if self._closed or self._owner is not None:raise RuntimeError('row snapshot already claimed/closed')
        self._owner=owner

    def row(self,owner,key):
        if self._closed or owner is not self._owner:raise RuntimeError('row snapshot has no live matching owner')
        return self._rows[key]

    def row_with_chunks(self,owner,key,*,token_chunk):
        if token_chunk!=2:raise ValueError('prepared training requires startup token_chunk=2')
        return self.row(owner,key),self._chunks[key]

    def close(self,owner):
        if owner is not self._owner:raise RuntimeError('foreign row snapshot close')
        self._closed=True


def training_lifetime(function):
    """Finish only an explicitly prepared model session, also on exceptions."""
    @wraps(function)
    def run(model,*args,**kwargs):
        primary=None
        try:return function(model,*args,**kwargs)
        except BaseException as error:primary=error;raise
        finally:
            finish=getattr(model,'finish_preflight_training',None)
            try:
                if callable(finish):finish()
            except BaseException as cleanup_error:
                if primary is None:raise
                primary.add_note('preflight session close also failed: '+repr(cleanup_error))
    return run
