import struct
import sys
print("Bitness:", struct.calcsize('P') * 8)
print("Executable:", sys.executable)
