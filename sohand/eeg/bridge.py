"""Bridge the Arduino's serial stream onto an LSL outlet.

    python -m sohand.eeg.bridge                       # /dev/ttyACM0 at 115200
    python -m sohand.eeg.bridge --port /dev/ttyUSB0
    python -m sohand.eeg.bridge --list                # show available ports

The sketch in `firmware/bioamp_eeg/` prints one microvolt reading per line at
500 Hz. This republishes them as the LSL stream `BioAmp_EXG`, which is what
`sohand.eeg.classify` and `sohand.eeg.visualizer` both subscribe to -- LSL
gives timestamping and lets several consumers read one device at once.

The reported rate is worth watching: if it drifts below the nominal 500 Hz the
FFT bin centres are wrong and every band power shifts with them.
"""

import argparse
import sys
import time

import pylsl
import serial
import serial.tools.list_ports

DEFAULT_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200
SAMPLING_RATE = 500
NUM_CHANNELS = 1
STREAM_NAME = "BioAmp_EXG"


def list_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
    for port in ports:
        print(f"  {port.device}  {port.description}")
    return ports


def make_outlet(rate, channels):
    info = pylsl.StreamInfo(name=STREAM_NAME, type="EEG", channel_count=channels,
                            nominal_srate=rate, channel_format="float32",
                            source_id="bioamp_r3")
    desc = info.desc().append_child("channels")
    for i in range(channels):
        ch = desc.append_child("channel")
        ch.append_child_value("label", f"Ch{i + 1}")
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")
    return pylsl.StreamOutlet(info)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default=DEFAULT_PORT)
    p.add_argument("--baud", type=int, default=BAUD_RATE)
    p.add_argument("--rate", type=int, default=SAMPLING_RATE,
                   help="nominal sample rate advertised to LSL, Hz")
    p.add_argument("--list", action="store_true", help="list serial ports and exit")
    args = p.parse_args()

    if args.list:
        list_ports()
        return

    try:
        port = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as exc:
        print(f"Failed to open {args.port}: {exc}\n\nAvailable ports:")
        list_ports()
        sys.exit(1)
    # Opening the port toggles DTR, which resets the Uno; anything sent before
    # the bootloader finishes is lost.
    time.sleep(2)
    print(f"Connected to {args.port} at {args.baud} baud")

    outlet = make_outlet(args.rate, NUM_CHANNELS)
    print(f"LSL stream '{STREAM_NAME}' open "
          f"({NUM_CHANNELS} ch, nominal {args.rate} Hz)")
    print("Streaming. Ctrl-C to stop.\n")

    count, malformed, t0 = 0, 0, time.time()
    try:
        while True:
            try:
                line = port.readline().decode("utf-8").strip()
            except UnicodeDecodeError:
                malformed += 1
                continue
            if not line:
                continue
            try:
                value = float(line)
            except ValueError:
                malformed += 1
                continue

            outlet.push_sample([value])
            count += 1
            if count % args.rate == 0:
                elapsed = time.time() - t0
                print(f"\rsamples {count:>9}   measured {count / elapsed:6.1f} Hz"
                      f"   last {value:+9.2f} uV   dropped {malformed}",
                      end="", flush=True)
    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print(f"\n\nStopped. {count} samples in {elapsed:.1f}s "
              f"({count / max(elapsed, 1e-9):.1f} Hz), {malformed} unparsable lines.")
    finally:
        port.close()


if __name__ == "__main__":
    main()
