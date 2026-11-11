"""Simultaneous injective slot copies, using one temporary slot for cycles."""
BATCH=8
MAX_STEPS=12
COPY=0
SAVE=1
RESTORE=2


def make_plan(sources):
    """Return mode/source/destination steps for new[i] = old[sources[i]]."""
    sources=list(sources)
    if len(sources)>BATCH or len(set(sources))!=len(sources) or any(type(s) is not int or not 0<=s<BATCH for s in sources):
        raise ValueError('Invalid compaction sources')
    pending={dest:source for dest,source in enumerate(sources) if dest!=source}
    plan=[]
    while pending:
        needed=set(pending.values())
        ready=[dest for dest in pending if dest not in needed]
        if ready:
            for dest in sorted(ready):plan.append((COPY,pending.pop(dest),dest))
            continue
        first=min(pending)
        plan.append((SAVE,first,0))
        dest=first
        while True:
            source=pending.pop(dest)
            if source==first:
                plan.append((RESTORE,0,dest))
                break
            plan.append((COPY,source,dest))
            dest=source
    assert len(plan)<=MAX_STEPS
    return plan
