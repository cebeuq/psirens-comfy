import json
import wave

from psirens_voice.events import EventBus
from psirens_voice.recorder import SessionRecorder


def test_recorder_writes_wav_and_events(tmp_path):
    rec = SessionRecorder(tmp_path, "testsession", 24000, meta={"a": 1})
    rec.write_audio(b"\x00\x01" * 1920)

    bus = EventBus()
    ev = bus.publish("audio_level", {"dbfs": -20.0}, t_session_ms=80)
    rec.write_event(ev)
    rec.close()

    d = tmp_path / "testsession"
    with wave.open(str(d / "audio.wav")) as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getnframes() == 1920

    lines = (d / "events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["type"] == "audio_level"
    assert parsed["t_session_ms"] == 80

    meta = json.loads((d / "meta.json").read_text())
    assert meta == {"a": 1}
