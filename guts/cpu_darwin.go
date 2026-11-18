package guts

import (
	"syscall"

	"github.com/klauspost/cpuid/v2"
)

var (
	haveAVX2   bool
	haveAVX512 bool
)

func init() {
	haveAVX2 = cpuid.CPU.Supports(cpuid.AVX2)
	// The AVX-512 kernels use EVEX encodings of 128- and 256-bit operations
	// as well as 512-bit ones, so they need AVX512VL in addition to AVX512F.
	// The single-block kernels also rely on AVX2 instructions.
	haveAVX512 = cpuid.CPU.Supports(cpuid.AVX512F, cpuid.AVX512VL)
	if !haveAVX512 {
		// On some Macs, AVX512 detection is buggy, so fallback to sysctl
		haveAVX512 = sysctlEnabled("hw.optional.avx512f") && sysctlEnabled("hw.optional.avx512vl")
	}
	haveAVX512 = haveAVX512 && haveAVX2
}

func sysctlEnabled(name string) bool {
	b, _ := syscall.Sysctl(name)
	return len(b) > 0 && b[0] == 1
}
