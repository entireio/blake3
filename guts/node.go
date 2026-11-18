// Package guts provides a low-level interface to the BLAKE3 cryptographic hash
// function.
package guts

import (
	"math/bits"
	"runtime"
	"sync"
)

// Various constants.
const (
	FlagChunkStart = 1 << iota
	FlagChunkEnd
	FlagParent
	FlagRoot
	FlagKeyedHash
	FlagDeriveKeyContext
	FlagDeriveKeyMaterial

	BlockSize = 64
	ChunkSize = 1024

	MaxSIMD = 16 // AVX-512 vectors can store 16 words
)

// IV is the BLAKE3 initialization vector.
var IV = [8]uint32{
	0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
	0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
}

// A Node represents a chunk or parent in the BLAKE3 Merkle tree.
type Node struct {
	CV       [8]uint32 // chaining value from previous node
	Block    [16]uint32
	Counter  uint64
	BlockLen uint32
	Flags    uint32
}

// ParentNode returns a Node that incorporates the chaining values of two child
// nodes.
func ParentNode(left, right [8]uint32, key *[8]uint32, flags uint32) Node {
	n := Node{
		CV:       *key,
		Counter:  0,         // counter is reset for parents
		BlockLen: BlockSize, // block is full
		Flags:    flags | FlagParent,
	}
	copy(n.Block[:8], left[:])
	copy(n.Block[8:], right[:])
	return n
}

// Eigentrees returns the sequence of eigentree heights that increment counter
// to counter+chunks.
func Eigentrees(counter uint64, chunks uint64) (trees []int) {
	for i := counter; i < counter+chunks; {
		bite := min(bits.TrailingZeros64(i), bits.Len64(counter+chunks-i)-1)
		trees = append(trees, bite)
		i += 1 << bite
	}
	return
}

// CompressEigentree compresses a buffer of 2^n chunks in parallel, returning
// their root node.
func CompressEigentree(buf []byte, key *[8]uint32, counter uint64, flags uint32) Node {
	return compressEigentree(buf, key, counter, flags, true)
}

// CompressEigentreeSerial is equivalent to CompressEigentree but does not
// create goroutines. It is useful when the caller already parallelizes
// independent messages.
func CompressEigentreeSerial(buf []byte, key *[8]uint32, counter uint64, flags uint32) Node {
	return compressEigentree(buf, key, counter, flags, false)
}

// ChunkCVs computes the chaining values of len(cvs) consecutive full chunks.
// buf must hold exactly len(cvs)*ChunkSize bytes; counter is the index of the
// first chunk.
func ChunkCVs(cvs [][8]uint32, buf []byte, key *[8]uint32, counter uint64, flags uint32) {
	if len(buf) != len(cvs)*ChunkSize {
		panic("guts: buffer does not match chaining value count")
	}
	compressChunks(cvs, buf, key, counter, flags)
}

// MergeCVs returns the root node of the subtree whose leaves, left to right,
// have the chaining values cvs. It panics if len(cvs) < 2 and overwrites cvs.
//
// The number of leaves need not be a power of two; BLAKE3 subtrees are
// left-complete, so pairing neighbours and carrying an odd trailing value
// reproduces the tree layout.
func MergeCVs(cvs [][8]uint32, key *[8]uint32, flags uint32) Node {
	n := len(cvs)
	if n < 2 {
		panic("guts: cannot merge fewer than two chaining values")
	}
	for n > 2 {
		half := n / 2
		compressParents(cvs[:half], cvs[:2*half], key, flags)
		if n%2 != 0 {
			cvs[half] = cvs[n-1]
			half++
		}
		n = half
	}
	return ParentNode(cvs[0], cvs[1], key, flags)
}

// CompressBuffer compresses up to MaxSIMD chunks in parallel and returns their
// root node.
func CompressBuffer(buf *[MaxSIMD * ChunkSize]byte, buflen int, key *[8]uint32, counter uint64, flags uint32) Node {
	if buflen <= ChunkSize {
		return CompressChunk(buf[:buflen], key, counter, flags)
	}
	var cvs [MaxSIMD][8]uint32
	n := buflen / ChunkSize
	compressChunks(cvs[:n], buf[:n*ChunkSize], key, counter, flags)
	if buflen%ChunkSize != 0 {
		cvs[n] = ChainingValue(CompressChunk(buf[n*ChunkSize:buflen], key, counter+uint64(n), flags))
		n++
	}
	return MergeCVs(cvs[:n], key, flags)
}

// compressSmallSubtree returns the root node of a subtree of at most MaxSIMD
// full chunks.
func compressSmallSubtree(buf []byte, key *[8]uint32, counter uint64, flags uint32) Node {
	var cvs [MaxSIMD][8]uint32
	n := len(buf) / ChunkSize
	compressChunks(cvs[:n], buf, key, counter, flags)
	return MergeCVs(cvs[:n], key, flags)
}

// subtreeCVs reduces a subtree of MaxSIMD<<k chunks to the MaxSIMD chaining
// values k levels above its leaves.
//
// Reducing both halves to MaxSIMD values before merging them means that every
// parent compression on the way up is one full batch of MaxSIMD, however large
// the subtree is, and that the working memory is a couple of batches per
// level rather than a chaining value per chunk.
func subtreeCVs(out *[MaxSIMD][8]uint32, buf []byte, key *[8]uint32, counter uint64, flags uint32) {
	if len(buf) == MaxSIMD*ChunkSize {
		compressChunks(out[:], buf, key, counter, flags)
		return
	}
	var cvs [2 * MaxSIMD][8]uint32
	half := len(buf) / 2
	subtreeCVs((*[MaxSIMD][8]uint32)(cvs[:MaxSIMD]), buf[:half], key, counter, flags)
	subtreeCVs((*[MaxSIMD][8]uint32)(cvs[MaxSIMD:]), buf[half:], key, counter+uint64(half/ChunkSize), flags)
	compressParents(out[:], cvs[:], key, flags)
}

// compressSubtree returns the root node of a subtree of MaxSIMD<<k chunks
// without allocating or starting goroutines.
func compressSubtree(buf []byte, key *[8]uint32, counter uint64, flags uint32) Node {
	var cvs [MaxSIMD][8]uint32
	subtreeCVs(&cvs, buf, key, counter, flags)
	return MergeCVs(cvs[:], key, flags)
}

func compressEigentree(buf []byte, key *[8]uint32, counter uint64, flags uint32, parallel bool) Node {
	numChunks := len(buf) / ChunkSize
	if bits.OnesCount64(uint64(numChunks)) != 1 {
		panic("non-power-of-two eigentree size")
	} else if numChunks == 1 {
		return CompressChunk(buf, key, counter, flags)
	} else if numChunks < MaxSIMD {
		return compressSmallSubtree(buf, key, counter, flags)
	}

	workers := 1
	if parallel {
		workers = runtime.GOMAXPROCS(0)
	}
	// Split the tree into equal subtrees for the workers to claim. A few per
	// worker lets the faster cores of an uneven machine take more of them.
	unit := numChunks
	for unit > MaxSIMD && numChunks/unit < 4*workers {
		unit /= 2
	}
	if workers == 1 || unit == numChunks {
		return compressSubtree(buf, key, counter, flags)
	}

	// The workers capture the key. Hand them a copy, so that the caller's key
	// (typically a field of a Hasher) need not escape to the heap just
	// because a large tree might be hashed concurrently.
	return compressEigentreeParallel(buf, *key, counter, flags, unit, workers)
}

func compressEigentreeParallel(buf []byte, key [8]uint32, counter uint64, flags uint32, unit, workers int) Node {
	cvs := make([][8]uint32, len(buf)/(unit*ChunkSize))
	compressUnit := func(i int) {
		start := i * unit * ChunkSize
		cvs[i] = ChainingValue(compressSubtree(
			buf[start:start+unit*ChunkSize],
			&key,
			counter+uint64(unit*i),
			flags,
		))
	}
	if len(cvs) <= workers {
		// For a small tree, keep the compression call directly in each
		// goroutine and skip the shared work counter.
		var wg sync.WaitGroup
		wg.Add(len(cvs))
		for i := range cvs {
			go func(i int) {
				defer wg.Done()
				compressUnit(i)
			}(i)
		}
		wg.Wait()
	} else {
		parallelFor(len(cvs), compressUnit)
	}
	return MergeCVs(cvs, &key, flags)
}
