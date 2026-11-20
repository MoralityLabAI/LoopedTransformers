"""Interpret the pinned SDK's raw message and stripped last_reply accessor."""


def raw_reply_bound(record,trace):
    nodes=trace.get('nodes',[])
    message=nodes[-1].get('message',{}) if nodes else {}
    return (not record['mock_only'] and message.get('role')=='assistant' and message.get('content')==record['text']
        and trace.get('last_reply')==record['text'].strip())


def successful_episode(episode):
    if not episode.get('ok') or episode.get('is_timeout') or len(episode.get('traces',[]))!=1:return False
    trace=episode['traces'][0]
    return bool(trace.get('ok') and trace.get('is_completed') and not trace.get('is_timeout') and not trace.get('errors'))
