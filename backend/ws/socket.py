from flask_socketio import SocketIO

socketio = SocketIO(cors_allowed_origins="*")


def emit_event(event):
    """
    Emits attack events to frontend via WebSocket.
    """
    if event.get("attack"):
        socketio.emit("attack", event)
