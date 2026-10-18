"""Corrupt real stored bytes across bounded validation chunk boundaries."""
import pytest
import torch
from ram_vram_finetune.quantization.codecs import CHUNK_GROUPS,VALIDATION_CHUNK_GROUPS,FORMATS,GROUP,encode,validate_spec,stride_for

GROUPS=CHUNK_GROUPS+3
POSITIONS=(0,VALIDATION_CHUNK_GROUPS-1,VALIDATION_CHUNK_GROUPS,GROUPS-1)

def record(mode,tail=False):
    torch.set_num_threads(1)
    return encode(torch.zeros(GROUPS*GROUP-(1 if tail else 0)),mode)

@pytest.mark.parametrize('mode',FORMATS)
@pytest.mark.parametrize('group',POSITIONS)
def test_invalid_payload_at_beginning_chunk_boundary_and_end(mode,group):
    r=record(mode);stride=stride_for(mode)
    r.parts['payload'][group*stride]=({'int4-sym-g128':0,'int6-sym-g128':0,'int8-sym-g128':128}.get(mode,127))
    with pytest.raises(ValueError,match='reserved|nonfinite'):
        validate_spec(r.spec,r.parts)

@pytest.mark.parametrize('mode',FORMATS)
@pytest.mark.parametrize('group',POSITIONS)
@pytest.mark.parametrize('bad',[0.,-1.,float('nan'),float('inf')])
def test_invalid_scale_at_beginning_chunk_boundary_and_end(mode,group,bad):
    r=record(mode);r.parts['scales'][group]=bad
    with pytest.raises(ValueError,match='scales'):validate_spec(r.spec,r.parts)

@pytest.mark.parametrize('mode',FORMATS)
@pytest.mark.parametrize('group',POSITIONS)
def test_finite_fp32_product_but_bf16_overflow_is_rejected(mode,group):
    r=record(mode);start=group*stride_for(mode)
    # Set exactly the first element of this group to +1 by format bits.
    p=r.parts['payload']
    if mode=='int4-sym-g128':p[start]=(int(p[start])&240)|9
    elif mode=='int6-sym-g128':p[start]=(int(p[start])&192)|33
    else:p[start]=1 if mode=='int8-sym-g128' else (0x38 if 'e4m3fn' in mode else 0x3c)
    r.parts['scales'][group]=3.4e38
    assert torch.isfinite(r.parts['scales'][group]) and not torch.isfinite(r.parts['scales'][group].bfloat16())
    with pytest.raises(ValueError,match='BF16'):validate_spec(r.spec,r.parts)

@pytest.mark.parametrize('mode',FORMATS)
def test_tail_of_last_chunk_is_checked(mode):
    r=record(mode,tail=True)
    if mode=='int4-sym-g128':r.parts['payload'][-1]^=16
    elif mode=='int6-sym-g128':r.parts['payload'][-1]|=4
    else:r.parts['payload'][-1]=1
    with pytest.raises(ValueError,match='padding'):validate_spec(r.spec,r.parts)

@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
@pytest.mark.parametrize('offset',[0,CHUNK_GROUPS*GROUP-1,CHUNK_GROUPS*GROUP, GROUPS*GROUP-1])
@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_raw_nonfinite_values_across_chunks(dtype,offset,bad):
    r=encode(torch.zeros(GROUPS*GROUP,dtype=dtype),'none');r.parts['value'][offset]=bad
    with pytest.raises(ValueError,match='nonfinite raw'):validate_spec(r.spec,r.parts)

@pytest.mark.parametrize('mode',FORMATS)
def test_valid_multichunk_all_storage_bytes_remain_frozen(mode):
    r=record(mode,tail=True);saved={k:t.clone() for k,t in r.parts.items()}
    validate_spec(r.spec,r.parts)
    assert all(torch.equal(t,saved[k]) for k,t in r.parts.items())

@pytest.mark.parametrize('mode',FORMATS)
def test_fp32_multiply_overflow_remains_rejected(mode):
    torch.set_num_threads(1)
    r=encode(torch.ones(128),mode);r.parts['scales'].fill_(1e38)
    limit={'int4-sym-g128':7,'int6-sym-g128':31,'int8-sym-g128':127,'fp8-e4m3fn-g128':448,'fp8-e5m2-g128':57344}[mode]
    assert torch.isinf(r.parts['scales'][0]*limit)
    with pytest.raises(ValueError,match='BF16'):validate_spec(r.spec,r.parts)

@pytest.mark.parametrize('mode,code',[
    ('fp8-e5m2-g128',0x7c),('fp8-e5m2-g128',0xfc),
    ('fp8-e5m2-g128',0x7f),('fp8-e5m2-g128',0xff),
    ('fp8-e4m3fn-g128',0x7f),('fp8-e4m3fn-g128',0xff)])
def test_fp8_infinity_and_nan_code_classification(mode,code):
    r=encode(torch.zeros(128),mode);r.parts['payload'][0]=code
    with pytest.raises(ValueError,match='nonfinite FP8 code'):validate_spec(r.spec,r.parts)
