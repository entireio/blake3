package guts

import "unsafe"

//go:generate go run -C ../avo . -out ../guts/compress_amd64.s

// The wide kernels hash one input per SIMD lane. n is the number of inputs
// actually present, which are contiguous at buf (or cvs); lanes beyond n
// re-read the last input, so nothing past it is touched. Every lane's result
// is stored, which is why the output is a full-width array.

//go:noescape
func compressChunks16AVX512(cvs *[16][8]uint32, buf *byte, n uint64, key *[8]uint32, counter uint64, flags uint32)

//go:noescape
func compressChunks8AVX512(cvs *[8][8]uint32, buf *byte, n uint64, key *[8]uint32, counter uint64, flags uint32)

//go:noescape
func compressChunks8AVX2(cvs *[8][8]uint32, buf *byte, n uint64, key *[8]uint32, counter uint64, flags uint32)

//go:noescape
func compressParents16AVX512(parents *[16][8]uint32, cvs *byte, n uint64, key *[8]uint32, flags uint32)

//go:noescape
func compressParents8AVX512(parents *[8][8]uint32, cvs *byte, n uint64, key *[8]uint32, flags uint32)

//go:noescape
func compressParents8AVX2(parents *[8][8]uint32, cvs *byte, n uint64, key *[8]uint32, flags uint32)

//go:noescape
func compressBlocksAVX512(out *[1024]byte, block *[16]uint32, cv *[8]uint32, counter uint64, blockLen uint32, flags uint32)

//go:noescape
func compressBlocksAVX2(out *[512]byte, block *[16]uint32, cv *[8]uint32, counter uint64, blockLen uint32, flags uint32)

// CompressNode compresses a node into a 16-word output.
func CompressNode(n Node) [16]uint32 {
	return compressNodeGeneric(n)
}

// CompressChunk compresses a single chunk, returning its final (uncompressed)
// node.
func CompressChunk(chunk []byte, key *[8]uint32, counter uint64, flags uint32) Node {
	return compressChunkGeneric(chunk, key, counter, flags)
}

// A wide kernel costs the same however many of its lanes carry real input,
// so each one has a break-even point below which narrower code is faster.
// These were measured rather than derived; see BenchmarkChunkCVs and
// BenchmarkMergeCVs.
const (
	// minChunks8 is the fewest chunks worth an eight-lane kernel rather than
	// one single-chunk pass each.
	minChunks8 = 2
	// minChunks16 is the fewest chunks worth the sixteen-lane kernel rather
	// than the eight-lane one plus whatever is left over.
	minChunks16 = 9
	// minParents8 and minParents16 are the same thresholds for parents.
	minParents8  = 2
	minParents16 = 9
)

// compressChunks stores the chaining values of len(cvs) full chunks in cvs.
func compressChunks(cvs [][8]uint32, buf []byte, key *[8]uint32, counter uint64, flags uint32) {
	if !haveAVX2 {
		compressChunksGeneric(cvs, buf, key, counter, flags)
		return
	}
	if haveAVX512 {
		for len(cvs) >= 16 {
			compressChunks16AVX512((*[16][8]uint32)(cvs), &buf[0], 16, key, counter, flags)
			cvs, buf, counter = cvs[16:], buf[16*ChunkSize:], counter+16
		}
		if len(cvs) >= minChunks16 {
			var out [16][8]uint32
			compressChunks16AVX512(&out, &buf[0], uint64(len(cvs)), key, counter, flags)
			copy(cvs, out[:])
			return
		}
	}
	for len(cvs) >= minChunks8 {
		n := min(len(cvs), 8)
		var out [8][8]uint32
		if haveAVX512 {
			compressChunks8AVX512(&out, &buf[0], uint64(n), key, counter, flags)
		} else {
			compressChunks8AVX2(&out, &buf[0], uint64(n), key, counter, flags)
		}
		copy(cvs, out[:])
		cvs, buf, counter = cvs[n:], buf[n*ChunkSize:], counter+uint64(n)
	}
	for i := range cvs {
		cvs[i] = ChainingValue(CompressChunk(buf[i*ChunkSize:(i+1)*ChunkSize], key, counter+uint64(i), flags))
	}
}

// compressParents stores the chaining value of the parent of cvs[2i] and
// cvs[2i+1] in parents[i]. parents may alias the front of cvs: every kernel
// loads all of its inputs before it stores anything.
func compressParents(parents, cvs [][8]uint32, key *[8]uint32, flags uint32) {
	if !haveAVX2 {
		compressParentsGeneric(parents, cvs, key, flags)
		return
	}
	flags |= FlagParent
	if haveAVX512 {
		for len(parents) >= 16 {
			compressParents16AVX512((*[16][8]uint32)(parents), (*byte)(unsafe.Pointer(&cvs[0])), 16, key, flags)
			parents, cvs = parents[16:], cvs[32:]
		}
		if len(parents) >= minParents16 {
			var out [16][8]uint32
			compressParents16AVX512(&out, (*byte)(unsafe.Pointer(&cvs[0])), uint64(len(parents)), key, flags)
			copy(parents, out[:])
			return
		}
	}
	for len(parents) >= minParents8 {
		n := min(len(parents), 8)
		var out [8][8]uint32
		if haveAVX512 {
			compressParents8AVX512(&out, (*byte)(unsafe.Pointer(&cvs[0])), uint64(n), key, flags)
		} else {
			compressParents8AVX2(&out, (*byte)(unsafe.Pointer(&cvs[0])), uint64(n), key, flags)
		}
		copy(parents, out[:])
		parents, cvs = parents[n:], cvs[2*n:]
	}
	for i := range parents {
		parents[i] = ChainingValue(ParentNode(cvs[2*i], cvs[2*i+1], key, flags&^FlagParent))
	}
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
