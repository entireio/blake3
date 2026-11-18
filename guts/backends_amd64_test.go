package guts

func backends() []backend {
	force := func(avx512, avx2 bool) func() func() {
		return func() func() {
			old512, old2 := haveAVX512, haveAVX2
			haveAVX512, haveAVX2 = avx512, avx2
			return func() { haveAVX512, haveAVX2 = old512, old2 }
		}
	}
	bs := []backend{{"generic", force(false, false)}}
	if haveAVX2 {
		bs = append(bs, backend{"AVX2", force(false, true)})
	}
	if haveAVX512 {
		bs = append(bs, backend{"AVX512", force(true, haveAVX2)})
	}
	return bs
}
