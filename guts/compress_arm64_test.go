package guts

import (
	"math/rand"
	"testing"
)

func TestCompressNodeNEONMatchesGeneric(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	blockLens := []uint32{0, 1, 31, 63, 64}
	for i := 0; i < 1000; i++ {
		var n Node
		n.CV = testKey(rng)
		for j := range n.Block {
			n.Block[j] = rng.Uint32()
		}
		n.Counter = rng.Uint64()
		n.BlockLen = blockLens[i%len(blockLens)]
		n.Flags = uint32(i) & (FlagDeriveKeyMaterial<<1 - 1)

		want := compressNodeGeneric(n)
		var got [16]uint32
		compressNodeNEON(&n, &got)
		if got != want {
			t.Fatalf("case %d: NEON compression mismatch\nwant %08x\n got %08x", i, want, got)
		}
	}
}
