import time
import random


def live_can_stream(callback, fps=20):
    """
    Software CAN stream generator.
    Emits CAN-like frames at real-time speed.
    """

    interval = 1.0 / fps

    while True:
        frame = {
            "ts": time.time(),
            "can_id": random.randint(0x100, 0x7FF),
            "dlc": 8,
            "data": [random.randint(0, 255) for _ in range(8)],
            "source": "live"
        }

        # Simulated attack pattern
        if random.random() < 0.15:
            frame["can_id"] = 0x666
            frame["data"][0] = 255

        callback(frame)
        time.sleep(interval)
