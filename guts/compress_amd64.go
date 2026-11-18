package guts

import (
	"unsafe"
)

//go:generate go run -C ../avo . -out ../guts/compress_amd64.s

//go:noescape
func compressChunksAVX512(cvs *[16][8]uint32, buf *[16 * ChunkSize]byte, key *[8]uint32, counter uint64, flags uint32)

//go:noescape
func compressChunksAVX2(cvs *[8][8]uint32, buf *[8 * ChunkSize]byte, key *[8]uint32, counter uint64, flags uint32)

//go:noescape
func compressBlocksAVX512(out *[1024]byte, block *[16]uint32, cv *[8]uint32, counter uint64, blockLen uint32, flags uint32)

//go:noescape
func compressBlocksAVX2(out *[512]byte, msgs *[16]uint32, cv *[8]uint32, counter uint64, blockLen uint32, flags uint32)

//go:noescape
func compressParentsAVX2(parents *[8][8]uint32, cvs *[16][8]uint32, key *[8]uint32, flags uint32)

// CompressNode compresses a node into a 16-word output.
func CompressNode(n Node) [16]uint32 {
	return compressNodeGeneric(n)
}

// compressChunks stores the chaining values of len(cvs) full chunks in cvs.
func compressChunks(cvs [][8]uint32, buf []byte, key *[8]uint32, counter uint64, flags uint32) {
	if haveAVX512 {
		for len(cvs) >= 16 {
			compressChunksAVX512((*[16][8]uint32)(cvs), (*[16 * ChunkSize]byte)(buf), key, counter, flags)
			cvs, buf, counter = cvs[16:], buf[16*ChunkSize:], counter+16
		}
	}
	if haveAVX2 {
		for len(cvs) >= 8 {
			compressChunksAVX2((*[8][8]uint32)(cvs), (*[8 * ChunkSize]byte)(buf), key, counter, flags)
			cvs, buf, counter = cvs[8:], buf[8*ChunkSize:], counter+8
		}
		if len(cvs) >= 2 {
			// The kernel reads eight whole chunks, so pad a partial group.
			var padded [8 * ChunkSize]byte
			var out [8][8]uint32
			copy(padded[:], buf)
			compressChunksAVX2(&out, &padded, key, counter, flags)
			copy(cvs, out[:])
			return
		}
	}
	compressChunksGeneric(cvs, buf, key, counter, flags)
}

// compressParents stores the chaining value of the parent of cvs[2i] and
// cvs[2i+1] in parents[i]. parents may alias the front of cvs.
func compressParents(parents, cvs [][8]uint32, key *[8]uint32, flags uint32) {
	if !haveAVX2 {
		compressParentsGeneric(parents, cvs, key, flags)
		return
	}
	for len(parents) >= 8 {
		compressParentsAVX2((*[8][8]uint32)(parents), (*[16][8]uint32)(cvs), key, flags)
		parents, cvs = parents[8:], cvs[16:]
	}
	if len(parents) > 0 {
		var in [16][8]uint32
		var out [8][8]uint32
		copy(in[:], cvs)
		compressParentsAVX2(&out, &in, key, flags)
		copy(parents, out[:])
	}
}

// CompressChunk compresses a single chunk, returning its final (uncompressed)
// node.
func CompressChunk(chunk []byte, key *[8]uint32, counter uint64, flags uint32) Node {
	n := Node{
		CV:       *key,
		Counter:  counter,
		BlockLen: BlockSize,
		Flags:    flags | FlagChunkStart,
	}
	blockBytes := (*[64]byte)(unsafe.Pointer(&n.Block))[:]
	for len(chunk) > BlockSize {
		copy(blockBytes, chunk)
		chunk = chunk[BlockSize:]
		n.CV = ChainingValue(n)
		n.Flags &^= FlagChunkStart
	}
	// pad last block with zeros
	n.Block = [16]uint32{}
	copy(blockBytes, chunk)
	n.BlockLen = uint32(len(chunk))
	n.Flags |= FlagChunkEnd
	return n
}

// CompressBlocks compresses MaxSIMD copies of n with successive counter values,
// storing the results in out.
func CompressBlocks(out *[MaxSIMD * BlockSize]byte, n Node) {
	switch {
	case haveAVX512:
		compressBlocksAVX512(out, &n.Block, &n.CV, n.Counter, n.BlockLen, n.Flags)
	case haveAVX2:
		outs := (*[2][512]byte)(unsafe.Pointer(out))
		compressBlocksAVX2(&outs[0], &n.Block, &n.CV, n.Counter, n.BlockLen, n.Flags)
		compressBlocksAVX2(&outs[1], &n.Block, &n.CV, n.Counter+8, n.BlockLen, n.Flags)
	default:
		compressBlocksGeneric(out, n)
	}
}

// BytesToWords converts an array of 64 bytes to an array of 16 bytes.
func BytesToWords(bytes [64]byte) [16]uint32 {
	return *(*[16]uint32)(unsafe.Pointer(&bytes))
}

// WordsToBytes converts an array of 16 words to an array of 64 bytes.
func WordsToBytes(words [16]uint32) [64]byte {
	return *(*[64]byte)(unsafe.Pointer(&words))
}
