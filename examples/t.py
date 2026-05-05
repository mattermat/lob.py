import gzip
import struct

with gzip.open("test_data/S101819-v50.txt.gz", "rb") as f:
    head = f.read(64)

print("hex:", head.hex())
print("bytes:", list(head))
# if SoupBinTCP: first 2 bytes = length, 3rd byte = packet type (e.g. 0x53 = 'S')
length = struct.unpack(">H", head[:2])[0]
print(f"length={length}, packet_type={chr(head[2])!r}, msg_type={chr(head[3])!r}")
