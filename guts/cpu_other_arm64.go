//go:build !linux

package guts

// haveSVE2 is only detected on Linux. No other arm64 platform Go supports
// exposes SVE2 to user space at the time of writing.
var haveSVE2 = false
