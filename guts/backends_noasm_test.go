//go:build !amd64 && !arm64

package guts

func backends() []backend {
	return []backend{{"generic", func() func() { return func() {} }}}
}
