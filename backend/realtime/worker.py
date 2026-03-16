import threading

from backend.realtime.can_sources import live_can_stream
from backend.realtime.pipeline import process_frame
from backend.ws.socket import emit_event


def start_worker():
    """
    Starts the realtime IDS worker in a background thread.
    """

    def on_frame(frame):
        event = process_frame(frame)
        emit_event(event)

    thread = threading.Thread(
        target=live_can_stream,
        args=(on_frame,),
        daemon=True
    )
    thread.start()
