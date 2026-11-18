//go:build !amd64 && !arm64
// +build !amd64,!arm64

package guts

import (
	"encoding/binary"
)

// CompressNode compresses a node into a 16-word output.
func CompressNode(n Node) [16]uint32 {
	return compressNodeGeneric(n)
}

// CompressChunk compresses a single chunk, returning its final (uncompressed)
// node.
func CompressChunk(chunk []byte, key *[8]uint32, counter uint64, flags uint32) Node {
	return compressChunkGeneric(chunk, key, counter, flags)
}

// CompressBlocks compresses MaxSIMD copies of n with successive counter values,
// storing the results in out.
func CompressBlocks(out *[MaxSIMD * BlockSize]byte, n Node) {
	compressBlocksGeneric(out, n)
}

// compressChunks stores the chaining values of len(cvs) full chunks in cvs.
func compressChunks(cvs [][8]uint32, buf []byte, key *[8]uint32, counter uint64, flags uint32) {
	compressChunksGeneric(cvs, buf, key, counter, flags)
}

// compressParents stores the chaining value of the parent of cvs[2i] and
// cvs[2i+1] in parents[i]. parents may alias the front of cvs.
func compressParents(parents, cvs [][8]uint32, key *[8]uint32, flags uint32) {
	compressParentsGeneric(parents, cvs, key, flags)
}

// BytesToWords converts an array of 64 bytes to an array of 16 bytes.
func BytesToWords(bytes [64]byte) (words [16]uint32) {
	for i := range words {
		words[i] = binary.LittleEndian.Uint32(bytes[4*i:])
	}
	return
}

// WordsToBytes converts an array of 16 words to an array of 64 bytes.
func WordsToBytes(words [16]uint32) (block [64]byte) {
	for i, w := range words {
		binary.LittleEndian.PutUint32(block[4*i:], w)
	}
	return
}
