"""
collect_data.py — serial logging tool for labeled vibration + current windows.

Reads the raw sample stream produced by firmware/Drivers/BSP/uart_stream.c
(UART_SendSample, active when STREAM_RAW_SAMPLES=1 in main.h) and writes
labeled runs to data/raw/<class>/<run_id>/.

Wire format (must match firmware/Drivers/BSP/uart_stream.h exactly):
    [0]     0xAA        sync byte
    [1..2]  int16       vib_x, little-endian
    [3..6]  float32     current_mA, little-endian
    [7]     uint8       checksum = XOR of bytes [1..6]
    8 bytes total, one packet per sample tick.

If you change SAMPLE_RATE_HZ, WINDOW_SIZE_SAMPLES, or the packet layout in
firmware, update PACKET_STRUCT_FMT / PACKET_LEN / baud rate here too — there
is no shared source of truth between the two right now, this is a real
footgun (see README "Open items").

Usage:
    python collect_data.py --port /dev/tty.usbmodemXXXX --class healthy --run-id remount01
    python collect_data.py --port /dev/tty.usbmodemXXXX --class healthy --run-id remount01 --duration 30
"""

import argparse
import csv
import struct
import sys
import time
from pathlib import Path

try:
    import serial
except ImportError:
    print("Missing dependency: pip install pyserial", file=sys.stderr)
    raise

SYNC_BYTE = 0xAA
PACKET_STRUCT_FMT = "<hf"   # int16 vib_x, float32 current_mA (checksum handled separately)
PACKET_BODY_LEN = struct.calcsize(PACKET_STRUCT_FMT)  # 6 bytes
PACKET_LEN = 1 + PACKET_BODY_LEN + 1  # sync + body + checksum = 8 bytes

BAUD_RATE = 115200  # must match firmware CubeMX USART2 config

VALID_CLASSES = ("healthy", "imbalance", "looseness", "overload")


def read_one_packet(ser: "serial.Serial") -> tuple[int, float] | None:
    """Blocking read + resync on one packet. Returns (vib_x, current_mA) or
    None if the checksum failed (caller should just try again)."""
    b = ser.read(1)
    if not b or b[0] != SYNC_BYTE:
        return None  # not synced yet — caller loops and tries the next byte

    body = ser.read(PACKET_BODY_LEN)
    checksum_byte = ser.read(1)
    if len(body) != PACKET_BODY_LEN or len(checksum_byte) != 1:
        return None  # short read — likely a dropped connection, caller handles it

    checksum = 0
    for byte in body:
        checksum ^= byte
    if checksum != checksum_byte[0]:
        return None  # corrupted packet, drop it and resync on the next 0xAA

    vib_x, current_mA = struct.unpack(PACKET_STRUCT_FMT, body)
    return vib_x, current_mA


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True, help="Serial port, e.g. /dev/tty.usbmodemXXXX or COM3")
    parser.add_argument("--class", dest="fault_class", required=True, choices=VALID_CLASSES)
    parser.add_argument("--run-id", required=True, help="Identifier for this physical remount/run, e.g. remount01")
    parser.add_argument("--duration", type=float, default=None,
                         help="Seconds to log before stopping automatically. Omit to log until Ctrl+C.")
    parser.add_argument("--data-dir", default="../../data/raw", help="Root of data/raw/ relative to this script")
    args = parser.parse_args()

    out_dir = Path(args.data_dir) / args.fault_class / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{int(time.time())}.csv"

    print(f"Opening {args.port} @ {BAUD_RATE} baud...")
    ser = serial.Serial(args.port, BAUD_RATE, timeout=1.0)
    time.sleep(2.0)  # let the board finish resetting after the port opens

    print(f"Logging class={args.fault_class} run_id={args.run_id} -> {out_path}")
    print("Ctrl+C to stop." if args.duration is None else f"Stopping after {args.duration}s.")

    n_written = 0
    n_dropped = 0
    start = time.monotonic()

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "vib_x", "current_mA"])
        try:
            while True:
                if args.duration is not None and (time.monotonic() - start) >= args.duration:
                    break
                pkt = read_one_packet(ser)
                if pkt is None:
                    n_dropped += 1
                    continue
                vib_x, current_mA = pkt
                writer.writerow([n_written, vib_x, f"{current_mA:.3f}"])
                n_written += 1
                if n_written % 800 == 0:  # ~1x/sec at the default 800Hz sample rate
                    print(f"  {n_written} samples logged, {n_dropped} dropped/corrupted")
        except KeyboardInterrupt:
            pass

    ser.close()
    elapsed = time.monotonic() - start
    print(f"Done. {n_written} samples written to {out_path} in {elapsed:.1f}s "
          f"({n_written / elapsed if elapsed > 0 else 0:.0f} Hz effective). {n_dropped} packets dropped.")
    if n_dropped > n_written * 0.05:
        print("WARNING: >5% packet drop rate — check baud rate match, cable, and USB port before trusting this run.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
