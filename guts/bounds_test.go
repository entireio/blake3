//go:build linux || darwin

package guts

import (
	"math/rand"
	"os"
	"syscall"
	"testing"
	"unsafe"
)

// guardedTail maps size bytes that end exactly where an inaccessible page
// begins, so that touching even one byte too many faults.
func guardedTail(t *testing.T, size int) []byte {
	t.Helper()
	page := os.Getpagesize()
	mapped := (size + page - 1) / page * page
	mem, err := syscall.Mmap(-1, 0, mapped+page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Skipf("mmap: %v", err)
	}
	t.Cleanup(func() { syscall.Munmap(mem) })
	if err := syscall.Mprotect(mem[mapped:], syscall.PROT_NONE); err != nil {
		t.Skipf("mprotect: %v", err)
	}
	return mem[mapped-size : mapped]
}

// guardedCVs is guardedTail for n chaining values.
func guardedCVs(t *testing.T, n int) [][8]uint32 {
	t.Helper()
	if n == 0 {
		return nil
	}
	mem := guardedTail(t, n*32)
	return unsafe.Slice((*[8]uint32)(unsafe.Pointer(&mem[0])), n)
}

// SIMD kernels hash a partial group of inputs by pointing their spare lanes
// at an input that exists, and may compute more results than were asked for.
// These tests fault if a kernel reads past its last input or writes past its
// last output.

func TestCompressChunksStaysInBounds(t *testing.T) {
	forEachBackend(t, func(t *testing.T) {
		rng := rand.New(rand.NewSource(1))
		key := testKey(rng)
		for n := 1; n <= 2*MaxSIMD+1; n++ {
			buf := guardedTail(t, n*ChunkSize)
			rng.Read(buf)
			got := guardedCVs(t, n)
			want := make([][8]uint32, n)
			compressChunks(got, buf, &key, 1<<32-2, FlagKeyedHash)
			compressChunksGeneric(want, buf, &key, 1<<32-2, FlagKeyedHash)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%d chunks, chunk %d: mismatch\nwant %08x\n got %08x", n, i, want[i], got[i])
				}
			}
		}
	})
}

func TestCompressParentsStaysInBounds(t *testing.T) {
	forEachBackend(t, func(t *testing.T) {
		rng := rand.New(rand.NewSource(2))
		key := testKey(rng)
		for n := 1; n <= 2*MaxSIMD+1; n++ {
			cvs := guardedCVs(t, 2*n)
			for i := range cvs {
				cvs[i] = testKey(rng)
			}
			want := make([][8]uint32, n)
			compressParentsGeneric(want, cvs, &key, FlagKeyedHash)

			got := guardedCVs(t, n)
			compressParents(got, cvs, &key, FlagKeyedHash)
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("%d parents, parent %d: mismatch\nwant %08x\n got %08x", n, i, want[i], got[i])
				}
			}

			// Reduce in place, as MergeCVs does.
			compressParents(cvs[:n], cvs, &key, FlagKeyedHash)
			for i := range want {
				if cvs[i] != want[i] {
					t.Fatalf("%d parents in place, parent %d: mismatch\nwant %08x\n got %08x", n, i, want[i], cvs[i])
				}
			}
		}
	})
}

func TestCompressChunkStaysInBounds(t *testing.T) {
	forEachBackend(t, func(t *testing.T) {
		rng := rand.New(rand.NewSource(3))
		key := testKey(rng)
		for length := 0; length <= ChunkSize; length++ {
			var chunk []byte
			if length > 0 {
				chunk = guardedTail(t, length)
				rng.Read(chunk)
			}
			got := CompressChunk(chunk, &key, 5, FlagKeyedHash)
			want := compressChunkGeneric(chunk, &key, 5, FlagKeyedHash)
			if got != want {
				t.Fatalf("length %d: chunk mismatch\nwant %+v\n got %+v", length, want, got)
			}
		}
	})
}

func TestCompressBlocksStaysInBounds(t *testing.T) {
	forEachBackend(t, func(t *testing.T) {
		rng := rand.New(rand.NewSource(4))
		n := Node{CV: testKey(rng), Counter: 1<<32 - 3, BlockLen: BlockSize, Flags: FlagRoot}
		for i := range n.Block {
			n.Block[i] = rng.Uint32()
		}
		out := (*[MaxSIMD * BlockSize]byte)(guardedTail(t, MaxSIMD*BlockSize))
		var want [MaxSIMD * BlockSize]byte
		CompressBlocks(out, n)
		compressBlocksGeneric(&want, n)
		if *out != want {
			t.Fatal("output-block compression mismatch")
		}
	})
}
