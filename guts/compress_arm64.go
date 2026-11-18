//go:build arm64

package guts

import (
	"encoding/binary"
	"unsafe"
)

//go:generate go run gen_arm64.go -out compress_arm64.s

// compress_arm64.s loads Node fields by fixed byte offset. Fail the build if a
// future Node edit changes that ABI without updating the assembly.
var (
	_ [0]byte   = [unsafe.Offsetof(Node{}.CV)]byte{}
	_ [32]byte  = [unsafe.Offsetof(Node{}.Block)]byte{}
	_ [96]byte  = [unsafe.Offsetof(Node{}.Counter)]byte{}
	_ [104]byte = [unsafe.Offsetof(Node{}.BlockLen)]byte{}
	_ [108]byte = [unsafe.Offsetof(Node{}.Flags)]byte{}
	_ [112]byte = [unsafe.Sizeof(Node{})]byte{}
)

// ARMv8-A requires Advanced SIMD, so the NEON path is always available on
// GOARCH=arm64. Keeping this as a variable lets package tests force the
// portable implementation for differential checks.
var haveNEON = true

// hash4NEON hashes the same number of 64-byte blocks from each of four
// inputs, and stores the resulting chaining values in out. Every lane
// starts from key; lane i uses counter+i*inc. Each block is compressed with
// flags, plus flagsStart for the first block and flagsEnd for the last.
//
// The inputs are independent pointers so that a caller with fewer than four
// inputs can aim the spare lanes at memory it knows to be readable.
//
//go:noescape
func hash4NEON(out *[4][8]uint32, p0, p1, p2, p3 *byte, blocks int, key *[8]uint32, counter, inc uint64, flags, flagsStart, flagsEnd uint32)

// minLanes is the smallest number of inputs worth a four-lane call. A call
// costs about as much as 1.4 scalar compressions per block on the cores this
// was measured on (Neoverse N2, Apple M2), so it pays from two inputs up.
const minLanes = 2

// CompressNode compresses a node into a 16-word output.
func CompressNode(n Node) [16]uint32 {
	return compressNodeGeneric(n)
}

// CompressChunk compresses a single chunk, returning its final uncompressed
// node.
func CompressChunk(chunk []byte, key *[8]uint32, counter uint64, flags uint32) Node {
	return compressChunkGeneric(chunk, key, counter, flags)
}

// CompressBlocks compresses MaxSIMD copies of n with successive counters.
func CompressBlocks(out *[MaxSIMD * BlockSize]byte, n Node) {
	compressBlocksGeneric(out, n)
}

// compressChunks stores the chaining values of len(cvs) full chunks in cvs.
func compressChunks(cvs [][8]uint32, buf []byte, key *[8]uint32, counter uint64, flags uint32) {
	if !haveNEON {
		compressChunksGeneric(cvs, buf, key, counter, flags)
		return
	}
	const blocks = ChunkSize / BlockSize
	for len(cvs) >= 4 {
		hash4NEON((*[4][8]uint32)(cvs),
			&buf[0], &buf[ChunkSize], &buf[2*ChunkSize], &buf[3*ChunkSize],
			blocks, key, counter, 1, flags, FlagChunkStart, FlagChunkEnd)
		cvs, buf, counter = cvs[4:], buf[4*ChunkSize:], counter+4
	}
	if n := len(cvs); n >= minLanes {
		// Hash a partial group by pointing the spare lanes at the first
		// chunk, so that nothing past the end of buf is read, and dropping
		// their output.
		var out [4][8]uint32
		lanes := [4]*byte{&buf[0], &buf[0], &buf[0], &buf[0]}
		for i := 1; i < n; i++ {
			lanes[i] = &buf[i*ChunkSize]
		}
		hash4NEON(&out, lanes[0], lanes[1], lanes[2], lanes[3],
			blocks, key, counter, 1, flags, FlagChunkStart, FlagChunkEnd)
		copy(cvs, out[:])
		return
	}
	for i := range cvs {
		cvs[i] = ChainingValue(CompressChunk(buf[i*ChunkSize:(i+1)*ChunkSize], key, counter+uint64(i), flags))
	}
}

// compressParents stores the chaining value of the parent of cvs[2i] and
// cvs[2i+1] in parents[i]. parents may alias the front of cvs.
func compressParents(parents, cvs [][8]uint32, key *[8]uint32, flags uint32) {
	compressParentsGeneric(parents, cvs, key, flags)
}

// BytesToWords converts 64 little-endian bytes to 16 words.
func BytesToWords(block [BlockSize]byte) (words [16]uint32) {
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(block[4*i:])
	}
	return words
}

// WordsToBytes converts 16 words to 64 little-endian bytes.
func WordsToBytes(words [16]uint32) (block [BlockSize]byte) {
	for i, word := range words {
		binary.LittleEndian.PutUint32(block[4*i:], word)
	}
	return block
}
