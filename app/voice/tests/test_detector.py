from psirens_voice.commit.detector import CommitDetector, DetectorConfig

CFG = DetectorConfig(
    eot_enter=0.6, eot_exit=0.4, pause_min_ms=500, confirm_ms=550,
    hold_timeout_ms=2500, min_content_words=2, pitch_rise_hz_per_s=40,
    delay_ms=500, fillers=frozenset({"um", "uh", "umm", "like"}),
)

FRAME = 80  # ms per tick


def run_ticks(det, t_start, n, p_eot):
    """Advance the detector n frames; return first non-None decision."""
    t = t_start
    for _ in range(n):
        t += FRAME
        d = det.tick(t, p_eot)
        if d:
            return d, t
    return None, t


def say(det, words, t0):
    """Feed words ending at successive times; returns end time of last word."""
    t = t0
    for w in words:
        det.on_word({"w": w, "t_start_ms": t, "t_end_ms": t + 200})
        t += 250
    return t - 250 + 200  # last word end


def test_basic_commit_flow():
    det = CommitDetector(CFG)
    last_end = say(det, ["make", "the", "sky", "darker"], 1000)
    assert det.state == "SPEAKING"
    # silence with confident end-of-turn: needs pause_min (observed) + dwell
    decision, t = run_ticks(det, last_end + 500, 40, p_eot=0.9)
    assert decision is not None and decision.kind == "commit"
    assert decision.data["utterance_clean"] == "make the sky darker"
    assert det.state == "LISTENING"
    assert any(r.startswith("via=gate") for r in decision.data["reasons"])


def test_no_commit_while_pending_partial():
    det = CommitDetector(CFG)
    say(det, ["make", "it"], 1000)
    det.on_partial("blu")
    decision, _ = run_ticks(det, 3000, 30, p_eot=0.95)
    assert decision is None  # pending partial keeps it in SPEAKING


def test_trailing_filler_holds_then_timeout_commits():
    det = CommitDetector(CFG)
    last_end = say(det, ["make", "it", "bigger", "um"], 1000)
    decision, t = run_ticks(det, last_end + 500, 30, p_eot=0.9)
    assert decision is not None and decision.kind == "hold"
    assert decision.data["reason"] == "trailing_filler"
    assert det.state == "HOLDING"
    # silence continues -> timeout commit
    decision, _ = run_ticks(det, t, 40, p_eot=0.9)
    assert decision is not None and decision.kind == "commit"
    assert any("via=timeout" in r for r in decision.data["reasons"])


def test_hold_then_continuation_returns_to_speaking():
    det = CommitDetector(CFG)
    last_end = say(det, ["make", "it", "um"], 1000)
    decision, t = run_ticks(det, last_end + 500, 30, p_eot=0.9)
    assert decision is not None and decision.kind == "hold"
    say(det, ["bigger"], t + 100)
    assert det.state == "SPEAKING"


def test_rising_pitch_holds():
    det = CommitDetector(CFG, pitch_slope_fn=lambda: 80.0)
    last_end = say(det, ["make", "it", "blue"], 1000)
    decision, _ = run_ticks(det, last_end + 500, 30, p_eot=0.9)
    assert decision is not None and decision.kind == "hold"
    assert decision.data["reason"] == "rising_pitch"


def test_low_eot_never_enters_maybe_done():
    det = CommitDetector(CFG)
    last_end = say(det, ["make", "it", "blue"], 1000)
    decision, _ = run_ticks(det, last_end + 500, 60, p_eot=0.3)
    assert decision is None
    assert det.state == "SPEAKING"


def test_min_content_discard():
    det = CommitDetector(CFG)
    say(det, ["um", "yeah"], 1000)  # 1 content word only
    decision, _ = run_ticks(det, 2500, 40, p_eot=0.9)
    assert decision is not None and decision.kind == "discard"
    assert det.state == "LISTENING"


def test_self_repair_applied_on_commit():
    det = CommitDetector(CFG)
    last_end = say(det, ["make", "it", "blue", "no,", "green"], 1000)
    decision, _ = run_ticks(det, last_end + 500, 40, p_eot=0.9)
    assert decision is not None and decision.kind == "commit"
    assert decision.data["utterance_clean"] == "make it green"
    assert len(decision.data["repairs"]) == 1


def test_observed_pause_accounts_for_stt_delay():
    det = CommitDetector(CFG)
    last_end = say(det, ["make", "it", "blue"], 1000)
    # audio clock just past word end + delay: observed pause is ~0, no commit
    t = last_end + CFG.delay_ms + FRAME
    assert det.tick(t, 0.9) is None
    assert det.state == "SPEAKING"
