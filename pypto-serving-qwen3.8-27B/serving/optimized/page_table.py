"""Fill the existing shared row while retaining every original page validation."""
import torch
from .kv_cache import BLOCK_SIZE,CONTEXT,POOL_BLOCKS


def fill_table(cache,pages,destination,previous):
    if pages.released or len(pages.pages)>CONTEXT//BLOCK_SIZE:
        raise ValueError('Invalid KV page table')
    if pages.pages and any(page<=0 or page>=POOL_BLOCKS or cache.refs[page]<=0 for page in pages.pages):
        raise ValueError('KV page table references a free or zero page')
    signature=tuple(pages.pages)
    if previous is None or previous[0] is not pages or previous[1]!=signature:
        payload=torch.tensor(signature,dtype=torch.int32)
        previous=(pages,signature,payload)
    destination[:len(signature)]=previous[2]
    return previous
