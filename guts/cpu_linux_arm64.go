package guts

import (
	"encoding/binary"
	"os"
)

// haveSVE2 reports whether the kernel advertises SVE2. It is a variable so
// that package tests can force the NEON kernels.
var haveSVE2 = detectSVE2()

// detectSVE2 reads the auxiliary vector the kernel handed this process. The
// kernel only sets HWCAP2_SVE2 when every CPU supports SVE2 and SVE has not
// been disabled.
func detectSVE2() bool {
	const (
		atHWCAP2   = 26
		hwcap2SVE2 = 1 << 1
	)
	auxv, err := os.ReadFile("/proc/self/auxv")
	if err != nil {
		return false
	}
	for ; len(auxv) >= 16; auxv = auxv[16:] {
		if binary.LittleEndian.Uint64(auxv) == atHWCAP2 {
			return binary.LittleEndian.Uint64(auxv[8:])&hwcap2SVE2 != 0
		}
	}
	return false
}
