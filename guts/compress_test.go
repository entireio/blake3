package guts

import (
	"fmt"
	"math/rand"
	"testing"
)

// A backend is one of the implementations the architecture-specific entry
// points can dispatch to. enable forces that implementation and returns a
// function that restores the detected configuration.
type backend struct {
	name   string
	enable func() (restore func())
}

// forEachBackend runs f once for every backend this machine supports, so the
// differential tests below cover the fallback paths as well as the fastest one.
func forEachBackend(t *testing.T, f func(t *testing.T)) {
	for _, b := range backends() {
		t.Run(b.name, func(t *testing.T) {
			defer b.enable()()
			f(t)
		})
	}
}

func testKey(rng *rand.Rand) (key [8]uint32) {
	for i := range key {
		key[i] = rng.Uint32()
	}
	return
}

var testFlags = []uint32{0, FlagKeyedHash, FlagDeriveKeyContext, FlagDeriveKeyMaterial}

// testCounters straddle the 32-bit boundaries that vectorized counters must
// carry across.
var testCounters = []uint64{0, 37, 1<<32 - 9, 1<<32 - 1, 1 << 32, 0xfedcba98fffffff5, 1<<64 - 70}

func TestCompressNodeMatchesGeneric(t *testing.T) {
	forEachBackend(t, func(t *testing.T) {
		rng := rand.New(rand.NewSource(1))
		flags := []uint32{
			0,
			FlagChunkStart,
			FlagChunkEnd,
			FlagChunkStart | FlagChunkEnd | FlagRoot,
			FlagParent,
			FlagKeyedHash | FlagChunkStart | FlagChunkEnd,
			FlagDeriveKeyContext | FlagRoot,
			FlagDeriveKeyMaterial | FlagRoot,
		}
		blockLens := []uint32{0, 1, 31, 63, 64}
		for i := 0; i < 1000; i++ {
			var n Node
			n.CV = testKey(rng)
			for j := range n.Block {
				n.Block[j] = rng.Uint32()
			}
			n.Counter = rng.Uint64()
			n.BlockLen = blockLens[i%len(blockLens)]
			n.Flags = flags[i%len(flags)]

			if got, want := CompressNode(n), compressNodeGeneric(n); got != want {
				t.Fatalf("case %d: compression mismatch\nwant %08x\n got %08x", i, want, got)
			}
			want := compressNodeGeneric(n)
			if got := ChainingValue(n); got != [8]uint32(want[:8]) {
				t.Fatalf("case %d: chaining value mismatch\nwant %08x\n got %08x", i, want[:8], got)
			}
		}
	})
}

func TestCompressChunkMatchesGeneric(t *testing.T) {
	forEachBackend(t, func(t *testing.T) {
		rng := rand.New(rand.NewSource(2))
		// Place the chunk at the end of an allocation so that reading past it
		// is more likely to be noticed.
		backing := make([]byte, 4*ChunkSize)
		rng.Read(backing)
		key := testKey(rng)
		for length := 0; length <= ChunkSize; length++ {
			chunk := backing[len(backing)-length:]
			flags := testFlags[length%len(testFlags)]
			counter := testCounters[length%len(testCounters)]
			got := CompressChunk(chunk, &key, counter, flags)
			want := compressChunkGeneric(chunk, &key, counter, flags)
			if got != want {
				t.Fatalf("length %d: chunk mismatch\nwant %+v\n got %+v", length, want, got)
			}
		}
	})
}

func TestChunkCVsMatchesGeneric(t *testing.T) {
	forEachBackend(t, func(t *testing.T) {
		rng := rand.New(rand.NewSource(3))
		const maxChunks = 4*MaxSIMD + 3
		backing := make([]byte, maxChunks*ChunkSize)
		rng.Read(backing)
		key := testKey(rng)
		for n := 0; n <= maxChunks; n++ {
			// Use the tail of the allocation so that reading past the last
			// chunk is more likely to be noticed.
			buf := backing[len(backing)-n*ChunkSize:]
			flags := testFlags[n%len(testFlags)]
			for _, counter := range testCounters {
				// Guard words catch writes past the requested chaining values.
				got := make([][8]uint32, n+1)
				guard := testKey(rng)
				got[n] = guard
				want := make([][8]uint32, n)
				ChunkCVs(got[:n], buf, &key, counter, flags)
				compressChunksGeneric(want, buf, &key, counter, flags)
				for i := range want {
					if got[i] != want[i] {
						t.Fatalf("%d chunks, counter %#x, chunk %d: mismatch\nwant %08x\n got %08x", n, counter, i, want[i], got[i])
					}
				}
				if got[n] != guard {
					t.Fatalf("%d chunks: wrote past the last chaining value", n)
				}
			}
		}
	})
}

func TestMergeCVsMatchesGeneric(t *testing.T) {
	// mergeGeneric is the reference tree reduction: pair neighbours, carry an
	// odd trailing value.
	mergeGeneric := func(cvs [][8]uint32, key *[8]uint32, flags uint32) Node {
		n := len(cvs)
		for n > 2 {
			half := n / 2
			compressParentsGeneric(cvs[:half], cvs[:2*half], key, flags)
			if n%2 != 0 {
				cvs[half] = cvs[n-1]
				half++
			}
			n = half
		}
		return ParentNode(cvs[0], cvs[1], key, flags)
	}
	forEachBackend(t, func(t *testing.T) {
		rng := rand.New(rand.NewSource(4))
		key := testKey(rng)
		for n := 2; n <= 8*MaxSIMD+3; n++ {
			input := make([][8]uint32, n)
			for i := range input {
				input[i] = testKey(rng)
			}
			flags := testFlags[n%len(testFlags)]
			got := MergeCVs(append([][8]uint32(nil), input...), &key, flags)
			want := mergeGeneric(append([][8]uint32(nil), input...), &key, flags)
			if got != want {
				t.Fatalf("%d chaining values: merge mismatch\nwant %+v\n got %+v", n, want, got)
			}
		}
	})
}

func TestCompressParentsDoesNotOverrun(t *testing.T) {
	forEachBackend(t, func(t *testing.T) {
		rng := rand.New(rand.NewSource(5))
		key := testKey(rng)
		for n := 0; n <= 2*MaxSIMD+1; n++ {
			cvs := make([][8]uint32, 2*n)
			for i := range cvs {
				cvs[i] = testKey(rng)
			}
			got := make([][8]uint32, n+1)
			guard := testKey(rng)
			got[n] = guard
			want := make([][8]uint32, n)
			compressParents(got[:n], cvs, &key, FlagKeyedHash)
			compressParentsGeneric(want, cvs, &key, FlagKeyedHash)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%d parents, parent %d: mismatch\nwant %08x\n got %08x", n, i, want[i], got[i])
				}
			}
			if got[n] != guard {
				t.Fatalf("%d parents: wrote past the last parent", n)
			}
		}
	})
}

func TestCompressBufferMatchesGeneric(t *testing.T) {
	// compressBufferGeneric hashes a buffer one node at a time.
	compressBufferGeneric := func(buf []byte, key *[8]uint32, counter uint64, flags uint32) Node {
		if len(buf) <= ChunkSize {
			return compressChunkGeneric(buf, key, counter, flags)
		}
		var cvs [][8]uint32
		for i := 0; i < len(buf); i += ChunkSize {
			chunk := buf[i:min(i+ChunkSize, len(buf))]
			full := compressNodeGeneric(compressChunkGeneric(chunk, key, counter+uint64(len(cvs)), flags))
			cvs = append(cvs, [8]uint32(full[:8]))
		}
		for len(cvs) > 2 {
			half := len(cvs) / 2
			compressParentsGeneric(cvs[:half], cvs[:2*half], key, flags)
			if len(cvs)%2 != 0 {
				cvs[half] = cvs[len(cvs)-1]
				half++
			}
			cvs = cvs[:half]
		}
		return ParentNode(cvs[0], cvs[1], key, flags)
	}
	forEachBackend(t, func(t *testing.T) {
		rng := rand.New(rand.NewSource(6))
		var buf [MaxSIMD * ChunkSize]byte
		rng.Read(buf[:])
		key := testKey(rng)
		lengths := []int{
			0, 1, BlockSize - 1, BlockSize, BlockSize + 1,
			ChunkSize - 1, ChunkSize, ChunkSize + 1,
		}
		for chunks := 2; chunks <= MaxSIMD; chunks++ {
			lengths = append(lengths, chunks*ChunkSize-17, chunks*ChunkSize)
			if chunks < MaxSIMD {
				lengths = append(lengths, chunks*ChunkSize+1)
			}
		}
		for _, length := range lengths {
			for _, flags := range testFlags {
				got := CompressBuffer(&buf, length, &key, 37, flags)
				want := compressBufferGeneric(buf[:length], &key, 37, flags)
				if got != want {
					t.Fatalf("length %d flags %d: buffer mismatch\nwant %+v\n got %+v", length, flags, want, got)
				}
			}
		}
	})
}

func TestCompressEigentreeMatchesGeneric(t *testing.T) {
	forEachBackend(t, func(t *testing.T) {
		rng := rand.New(rand.NewSource(7))
		buf := make([]byte, 32*MaxSIMD*ChunkSize)
		rng.Read(buf)
		key := testKey(rng)
		for chunks := 1; chunks*ChunkSize <= len(buf); chunks *= 2 {
			tree := buf[:chunks*ChunkSize]
			want := compressChunkGeneric(tree, &key, 3, FlagKeyedHash)
			if chunks > 1 {
				cvs := make([][8]uint32, chunks)
				compressChunksGeneric(cvs, tree, &key, 3, FlagKeyedHash)
				for n := chunks; n > 2; n /= 2 {
					compressParentsGeneric(cvs[:n/2], cvs[:n], &key, FlagKeyedHash)
				}
				want = ParentNode(cvs[0], cvs[1], &key, FlagKeyedHash)
			}
			for name, compress := range map[string]func([]byte, *[8]uint32, uint64, uint32) Node{
				"parallel": CompressEigentree,
				"serial":   CompressEigentreeSerial,
			} {
				if got := compress(tree, &key, 3, FlagKeyedHash); got != want {
					t.Fatalf("%d chunks (%s): eigentree mismatch\nwant %+v\n got %+v", chunks, name, want, got)
				}
			}
		}
	})
}

func TestCompressBlocksMatchesGeneric(t *testing.T) {
	forEachBackend(t, func(t *testing.T) {
		rng := rand.New(rand.NewSource(8))
		for _, counter := range testCounters {
			for _, blockLen := range []uint32{0, 1, BlockSize - 1, BlockSize} {
				var n Node
				n.CV = testKey(rng)
				for i := range n.Block {
					n.Block[i] = rng.Uint32()
				}
				n.Counter = counter
				n.BlockLen = blockLen
				n.Flags = FlagKeyedHash | FlagRoot

				// Guard bytes catch writes outside the output buffer.
				var got struct {
					before [BlockSize]byte
					out    [MaxSIMD * BlockSize]byte
					after  [BlockSize]byte
				}
				var want [MaxSIMD * BlockSize]byte
				CompressBlocks(&got.out, n)
				compressBlocksGeneric(&want, n)
				if got.out != want {
					t.Fatalf("counter %#x: output-block compression mismatch", counter)
				}
				if got.before != [BlockSize]byte{} || got.after != [BlockSize]byte{} {
					t.Fatalf("counter %#x: wrote outside the output buffer", counter)
				}
			}
		}
	})
}

// benchmarkBackends runs f once for every backend this machine supports, so
// that one run compares the implementations on the same hardware.
func benchmarkBackends(b *testing.B, f func(b *testing.B)) {
	for _, be := range backends() {
		b.Run(be.name, func(b *testing.B) {
			defer be.enable()()
			f(b)
		})
	}
}

func BenchmarkChunkCVs(b *testing.B) {
	benchmarkBackends(b, func(b *testing.B) {
		var key [8]uint32
		for _, chunks := range []int{1, 2, 3, 4, 8, 15, 16, 64} {
			b.Run(fmt.Sprint(chunks), func(b *testing.B) {
				buf := make([]byte, chunks*ChunkSize)
				cvs := make([][8]uint32, chunks)
				b.SetBytes(int64(len(buf)))
				for i := 0; i < b.N; i++ {
					ChunkCVs(cvs, buf, &key, 0, 0)
				}
			})
		}
	})
}

func BenchmarkMergeCVs(b *testing.B) {
	benchmarkBackends(b, func(b *testing.B) {
		var key [8]uint32
		for _, n := range []int{2, 4, 8, 16, 64} {
			b.Run(fmt.Sprint(n), func(b *testing.B) {
				src := make([][8]uint32, n)
				cvs := make([][8]uint32, n)
				// Report throughput in terms of the input the tree covers.
				b.SetBytes(int64(n * ChunkSize))
				for i := 0; i < b.N; i++ {
					copy(cvs, src)
					MergeCVs(cvs, &key, 0)
				}
			})
		}
	})
}

func BenchmarkCompressNode(b *testing.B) {
	benchmarkBackends(b, func(b *testing.B) {
		n := Node{CV: IV, BlockLen: BlockSize}
		b.SetBytes(BlockSize)
		for i := 0; i < b.N; i++ {
			out := CompressNode(n)
			n.Block[0] = out[0]
		}
	})
}

func BenchmarkCompressBlocks(b *testing.B) {
	benchmarkBackends(b, func(b *testing.B) {
		n := Node{CV: IV, BlockLen: BlockSize, Flags: FlagRoot}
		var out [MaxSIMD * BlockSize]byte
		b.SetBytes(int64(len(out)))
		for i := 0; i < b.N; i++ {
			CompressBlocks(&out, n)
			n.Counter += MaxSIMD
		}
	})
}
