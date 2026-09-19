"""Minimal mcap reader for the docking trial bags, no ROS needed.

The trial runner killed the recorder mid-finalise, so every bag's tail (message
indexes, summary, footer) is truncated. The chunks themselves are intact, so this
reader walks records sequentially, breaks each chunk open, and decodes messages
with the schemas embedded in the bag (mcap-ros2-support). The custom
interfaces/DockPoseMeasurement type therefore works without a built workspace.
"""
from __future__ import annotations

import sys

import numpy as np
from mcap.records import Channel, Chunk, Message, Schema
from mcap.stream_reader import StreamReader, breakup_chunk
from mcap_ros2.decoder import DecoderFactory


def read_topics(path, topics):
    """Return {topic: [(log_time_s, msg), ...]} for the requested topics."""
    want = set(topics)
    out = {t: [] for t in topics}
    schemas, channels, decoders = {}, {}, {}
    factory = DecoderFactory()

    def handle(rec):
        if isinstance(rec, Schema):
            schemas[rec.id] = rec
        elif isinstance(rec, Channel):
            channels[rec.id] = rec
        elif isinstance(rec, Message):
            ch = channels.get(rec.channel_id)
            if ch is None or ch.topic not in want:
                return
            dec = decoders.get(ch.id)
            if dec is None:
                dec = factory.decoder_for(ch.message_encoding, schemas[ch.schema_id])
                decoders[ch.id] = dec
            out[ch.topic].append((rec.log_time * 1e-9, dec(rec.data)))

    try:
        with open(path, "rb") as f:
            for rec in StreamReader(f, emit_chunks=True).records:
                if isinstance(rec, Chunk):
                    for inner in breakup_chunk(rec):
                        handle(inner)
                else:
                    handle(rec)
    except Exception as e:  # noqa: BLE001
        print(f"[bagio] stopped at truncated tail: {type(e).__name__}: {e}", file=sys.stderr)
    return out


def stamp(msg):
    h = msg.header.stamp
    return h.sec + h.nanosec * 1e-9


def quat_xyzw(q):
    return np.array([q.x, q.y, q.z, q.w], dtype=float)


def vec3(v):
    return np.array([v.x, v.y, v.z], dtype=float)
