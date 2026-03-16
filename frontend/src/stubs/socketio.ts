// frontend/src/stubs/socketio.ts
// Socket.IO stub: prevents legacy socket.io-client code from connecting to port 5000
// All existing imports of "socket.io-client" or "@/socket" will resolve to this file.

export type SocketLike = {
  on: (event: string, cb: (...args: any[]) => void) => void;
  off: (event: string, cb?: (...args: any[]) => void) => void;
  emit: (event: string, payload?: unknown) => void;
  connect: () => void;
  disconnect: () => void;
};

export function io(_url?: string, _opts?: unknown): SocketLike {
  const noop = () => {};
  return {
    on: noop,
    off: noop,
    emit: noop,
    connect: noop,
    disconnect: noop,
  };
}

// alias for code expecting { socket } or { Socket } types
export const socket: SocketLike = io();
export type Socket = SocketLike;

export default { io, socket };
