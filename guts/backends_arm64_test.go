package guts

func backends() []backend {
	force := func(neon, sve2 bool) func() func() {
		return func() func() {
			oldNEON, oldSVE2 := haveNEON, haveSVE2
			haveNEON, haveSVE2 = neon, sve2
			return func() { haveNEON, haveSVE2 = oldNEON, oldSVE2 }
		}
	}
	bs := []backend{
		{"generic", force(false, false)},
		{"NEON", force(true, false)},
	}
	if haveSVE2 {
		bs = append(bs, backend{"SVE2", force(true, true)})
	}
	return bs
}
