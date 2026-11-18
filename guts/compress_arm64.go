//go:build arm64

package guts

// The NEON kernels in compress_arm64.s were ported from
// goforge.dev/blake3sum v1.0.0. Copyright 2026 brain-fuel, MIT license; see
// LICENSE-GOFORGE and THIRD_PARTY_NOTICES.md in the module root.

import (
	"encoding/binary"
	"unsafe"
)

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

//go:noescape
func compressNodeNEON(n *Node, out *[16]uint32)

// compressChunksNEON compresses four full chunks in parallel.
//
//go:noescape
func compressChunksNEON(cvs *[4][8]uint32, buf *[4 * ChunkSize]byte, key *[8]uint32, counter uint64, flags uint32)

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
	if haveNEON {
		for len(cvs) >= 4 {
			compressChunksNEON((*[4][8]uint32)(cvs), (*[4 * ChunkSize]byte)(buf), key, counter, flags)
			cvs, buf, counter = cvs[4:], buf[4*ChunkSize:], counter+4
		}
	}
	compressChunksGeneric(cvs, buf, key, counter, flags)
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
