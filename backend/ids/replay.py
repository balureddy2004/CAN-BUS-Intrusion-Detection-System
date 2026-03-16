# Stubs for reading CSV/PCAP and yielding CAN frames.
# Replace with your actual parser later.

def iterate_csv(file_like):
    """
    Yield dicts shaped like packets the app expects:
    { "ts": ms_epoch, "can_id": "0x123", "dlc": 8, "data": "AA BB ..." }
    """
    for _ in range(10):
        yield {
            "ts": 0,
            "can_id": "0x123",
            "dlc": 8,
            "data": "00 11 22 33 44 55 66 77",
        }
