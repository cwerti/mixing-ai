import numpy as np

from app.eval.tail_decay import FLOOR_DB, decay_profile, phrase_ends, probe_segment, profile_at, profile_distance

SR = 22050


def _phrases(tail_s: float, n: int = 3, seed: int = 0) -> tuple[np.ndarray, list[int]]:
    """Стерео: фразы шума по 2 с с экспоненциальным хвостом (−60 дБ за tail_s) и паузами по 2 с."""
    rng = np.random.default_rng(seed)
    phrase, pause = 2 * SR, 2 * SR
    y = np.zeros((2, n * (phrase + pause) + SR))
    ends = []
    for i in range(n):
        a = i * (phrase + pause)
        y[:, a:a + phrase] = rng.standard_normal((2, phrase)) * 0.1
        if tail_s > 0:
            t = np.arange(pause) / SR
            y[:, a + phrase:a + phrase + pause] = rng.standard_normal((2, pause)) * 0.1 * 10 ** (-3 * t / tail_s)
        ends.append(a + phrase)
    return y, ends


def test_phrase_ends_found_at_phrase_boundaries():
    y, ends = _phrases(tail_s=0.0)
    found = phrase_ends(y, SR)
    assert len(found) == 3
    for (frame, _), end in zip(found, ends):
        assert abs(frame * 0.01 * SR - end) < 0.1 * SR


def test_no_pauses_no_profile():
    y = np.random.default_rng(0).standard_normal((2, 5 * SR)) * 0.1
    assert phrase_ends(y, SR) == []
    assert decay_profile(y, SR) is None
    assert probe_segment(y, SR) is None


def test_longer_tail_gives_higher_profile():
    dry = decay_profile(_phrases(0.0)[0], SR)
    short = decay_profile(_phrases(0.3)[0], SR)
    long = decay_profile(_phrases(1.5)[0], SR)
    assert dry is not None and short is not None and long is not None
    assert np.all(dry <= FLOOR_DB + 1)
    assert long[1] > short[1] > dry[1]  # Mid через 0.3 с
    assert profile_distance(long, short) > profile_distance(short, short)


def test_profile_at_measures_tail_that_never_pauses():
    y, ends = _phrases(tail_s=1.5, n=1)
    end_frame = int(round(ends[0] / (0.01 * SR)))
    prof = profile_at(y, SR, end_frame)
    assert prof[0] > -10  # через 0.1 с хвост еще громкий
    assert prof[3] < prof[0]  # и спадает
