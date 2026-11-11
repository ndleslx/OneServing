"""Each column tile executes the entire copy plan sequentially, using only local integer storage."""
import pypto.language as pl

BATCH=8
CONV_WORDS=48*4*10240//2
RECURRENT_WORDS=48*48*128*128
TILE=8192


@pl.jit
def compact_states_local_tile(conv:pl.InOut[pl.Tensor],recurrent:pl.InOut[pl.Tensor],
        plan:pl.Tensor,step_count:pl.Scalar[pl.INT32]):
    conv_rows=conv
    recurrent_rows=recurrent
    with pl.spmd(CONV_WORDS//TILE,name_hint='qwen_compact_local_conv_words'):
        conv_column=pl.tile.get_block_idx()*TILE
        conv_temporary=pl.full([1,TILE],dtype=pl.INT32,value=0)
        for conv_step in pl.range(step_count):
            conv_mode=pl.read(plan,[conv_step,0])
            conv_source=pl.read(plan,[conv_step,1])
            conv_destination=pl.read(plan,[conv_step,2])
            if conv_mode==0:
                conv_rows[conv_destination:conv_destination+1,conv_column:conv_column+TILE]=pl.slice(conv_rows,[1,TILE],[conv_source,conv_column])
            elif conv_mode==1:
                conv_temporary[0:1,0:TILE]=pl.slice(conv_rows,[1,TILE],[conv_source,conv_column])
            else:
                conv_rows[conv_destination:conv_destination+1,conv_column:conv_column+TILE]=conv_temporary
    with pl.spmd(RECURRENT_WORDS//TILE,name_hint='qwen_compact_local_recurrent_words'):
        recurrent_column=pl.tile.get_block_idx()*TILE
        recurrent_temporary=pl.full([1,TILE],dtype=pl.INT32,value=0)
        for recurrent_step in pl.range(step_count):
            recurrent_mode=pl.read(plan,[recurrent_step,0])
            recurrent_source=pl.read(plan,[recurrent_step,1])
            recurrent_destination=pl.read(plan,[recurrent_step,2])
            if recurrent_mode==0:
                recurrent_rows[recurrent_destination:recurrent_destination+1,recurrent_column:recurrent_column+TILE]=pl.slice(recurrent_rows,[1,TILE],[recurrent_source,recurrent_column])
            elif recurrent_mode==1:
                recurrent_temporary[0:1,0:TILE]=pl.slice(recurrent_rows,[1,TILE],[recurrent_source,recurrent_column])
            else:
                recurrent_rows[recurrent_destination:recurrent_destination+1,recurrent_column:recurrent_column+TILE]=recurrent_temporary
    return conv,recurrent


@pl.jit.host
def compact_states_local_tile_fwd(conv:pl.InOut[pl.Tensor],recurrent:pl.InOut[pl.Tensor],
        plan:pl.Tensor,step_count:pl.Scalar[pl.INT32]):
    compact_states_local_tile(conv,recurrent,plan,step_count,device=0)
