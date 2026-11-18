//go:build !darwin
// +build !darwin

package guts

import "github.com/klauspost/cpuid/v2"

var (
	haveAVX2 = cpuid.CPU.Supports(cpuid.AVX2)
	// The AVX-512 kernels use EVEX encodings of 128- and 256-bit operations
	// as well as 512-bit ones, so they need AVX512VL in addition to AVX512F.
	// The single-block kernels also rely on AVX2 instructions.
	haveAVX512 = haveAVX2 && cpuid.CPU.Supports(cpuid.AVX512F, cpuid.AVX512VL)
)
