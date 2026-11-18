package guts

func backends() []backend {
	force := func(neon bool) func() func() {
		return func() func() {
			old := haveNEON
			haveNEON = neon
			return func() { haveNEON = old }
		}
	}
	return []backend{
		{"generic", force(false)},
		{"NEON", force(true)},
	}
}
